"""SQL 持久化任务队列:claim / lease / reaper —— 崩溃可续跑的批量执行。

**为什么不用内存队列**(ThreadPoolExecutor / queue.Queue):

内存队列里「谁在跑」只活在进程内。进程一挂(关窗口、OOM、断电),在跑的任务
既没有结果、也没人知道它们跑过 —— 重启只能整批重来,LLM token 白烧一遍。

把「谁在跑 / 跑到哪 / 租约何时到期」也写进库之后,崩溃的后果变成三档:

    已完成        → 直接读结果,不重跑
    租约过期      → reaper 捞回 pending,换个 worker 重跑
    超出重试上限  → 落到 dead,不再无限重试

这三件事正是内存队列给不了的。租约(lease)是关键:进程死了没法主动归还任务,
所以「占用」必须带一个过期时间,由**别人**来判断它是否已经死了。

并发正确性靠 SQLite 的 `BEGIN IMMEDIATE` + 条件 UPDATE 保证:同一时刻只有一个
worker 能拿到写锁,抢不到的行不会被重复领取。
"""
import json
import random
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "jobs.db"

PENDING, RUNNING, DONE, DEAD = "pending", "running", "done", "dead"

#: 指数退避:第 n 次失败后等 `base * 2^(n-1)` 秒,上限 cap。
#:
#: **为什么要退避**:失败了立刻重试等于没重试 —— 尤其是 429,同一堵墙会连撞
#: 三次,三次额度全烧掉。接第二个厂商(真实跑 Kimi)时就是这么被打的。
BACKOFF_BASE = 2.0
BACKOFF_CAP = 60.0


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def backoff_delay(attempts: int, base: float = BACKOFF_BASE,
                  cap: float = BACKOFF_CAP, rand=random.random) -> float:
    """第 `attempts` 次失败之后,该等多少秒再试。

    **为什么要抖动**:多个 worker 同时被限流,如果都精确等 `2^n` 秒,它们会
    **同时醒来再一起撞上去**(thundering herd)。所以用「等比例抖动」——
    保证至少等一半,另一半随机,把重试时间摊开。

    `rand` 可注入,为了让测试能拿确定性结果。
    """
    nominal = min(cap, base * (2 ** max(0, attempts - 1)))
    return nominal / 2 + rand() * nominal / 2


class JobQueue:
    """SQLite 上的持久化任务队列。一个实例对应一个库文件(测试可传临时路径)。"""

    def __init__(self, db_path: Path | str | None = None,
                 backoff_base: float = BACKOFF_BASE,
                 backoff_cap: float = BACKOFF_CAP) -> None:
        self.db_path = Path(db_path) if db_path else DB_PATH
        #: 退避参数做成实例属性,测试才能把它压到 0.01 秒 ——
        #: 否则每跑一次重试测试就要真等好几秒。
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None:关掉隐式事务,由我们自己 BEGIN IMMEDIATE,
        # 否则 Python 会在 UPDATE 前偷偷开启一个延迟事务,拿不到写锁。
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id        TEXT    NOT NULL,
                    key           TEXT    NOT NULL,
                    payload_json  TEXT    NOT NULL,
                    state         TEXT    NOT NULL DEFAULT 'pending',
                    attempts      INTEGER NOT NULL DEFAULT 0,
                    max_attempts  INTEGER NOT NULL DEFAULT 3,
                    lease_owner   TEXT,
                    lease_expires REAL,
                    next_attempt_at REAL,
                    result_json   TEXT,
                    error         TEXT,
                    created_at    TEXT    NOT NULL,
                    updated_at    TEXT    NOT NULL,
                    UNIQUE(run_id, key)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(state, run_id)")
            # 老库补列:CREATE TABLE IF NOT EXISTS 不会给已存在的表加字段。
            cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
            if "next_attempt_at" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN next_attempt_at REAL")
        finally:
            conn.close()

    # ------------------------------------------------------------ 入队

    def enqueue(self, run_id: str, key: str, payload: dict, max_attempts: int = 3) -> None:
        """入队一个任务。同 (run_id, key) 重复入队是**幂等**的 —— 不会覆盖已有状态。

        这一点是「重启续跑」的前提:重新跑同一批任务时,已完成的行不会被重置回
        pending,于是不会被重跑。
        """
        self.init_db()
        ts = _now_iso()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO jobs "
                "(run_id, key, payload_json, state, max_attempts, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, key, json.dumps(payload, ensure_ascii=False),
                 PENDING, max_attempts, ts, ts),
            )
        finally:
            conn.close()

    # ------------------------------------------------------------ 领取

    def claim(self, owner: str, run_id: str | None = None,
              lease_seconds: float = 300, now: float | None = None) -> dict | None:
        """原子领取一个 pending 任务,并给它打上租约。没有可领的返回 None。

        **退避中的任务领不到** —— 这是退避能生效的关键:失败了立刻重试等于没重试。
        """
        self.init_db()
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")  # 拿到写锁,杜绝两人领同一条
            row = conn.execute(
                "SELECT * FROM jobs WHERE state = ? AND (? IS NULL OR run_id = ?) "
                "AND (next_attempt_at IS NULL OR next_attempt_at <= ?) "
                "ORDER BY id LIMIT 1",
                (PENDING, run_id, run_id, now),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            cur = conn.execute(
                "UPDATE jobs SET state = ?, lease_owner = ?, lease_expires = ?, "
                "attempts = attempts + 1, updated_at = ? WHERE id = ? AND state = ?",
                (RUNNING, owner, now + lease_seconds, _now_iso(), row["id"], PENDING),
            )
            if cur.rowcount != 1:  # 理论上不会发生,防御性写法
                conn.execute("COMMIT")
                return None
            conn.execute("COMMIT")
            job = dict(row)
        finally:
            conn.close()
        job.update(state=RUNNING, lease_owner=owner, lease_expires=now + lease_seconds,
                   attempts=job["attempts"] + 1)
        job["payload"] = json.loads(job.pop("payload_json"))
        return job

    def next_attempt_in(self, run_id: str, now: float | None = None) -> float | None:
        """还有 pending 任务的话,距离最近一个变成可领取还有多少秒。

        为什么需要它:有了退避,"领不到任务"就有两种含义 ——
        **活干完了**,或者**剩下的都在退避中**。分不清的话 worker 会提前收工,
        把还在退避的任务丢在那儿不管。返回 None 才表示真的没活了。
        """
        self.init_db()
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT MIN(COALESCE(next_attempt_at, 0)) AS t, COUNT(*) AS n "
                "FROM jobs WHERE run_id = ? AND state = ?",
                (run_id, PENDING),
            ).fetchone()
        finally:
            conn.close()
        if row is None or not row["n"]:
            return None
        return max(0.0, float(row["t"] or 0) - now)

    # ------------------------------------------------------ 完成 / 失败

    def complete(self, job_id: int, result: dict) -> None:
        self.init_db()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE jobs SET state = ?, result_json = ?, error = NULL, "
                "lease_owner = NULL, lease_expires = NULL, next_attempt_at = NULL, "
                "updated_at = ? WHERE id = ?",
                (DONE, json.dumps(result, ensure_ascii=False), _now_iso(), job_id),
            )
        finally:
            conn.close()

    def fail(self, job_id: int, error: str, now: float | None = None) -> str:
        """记一次失败:还有重试额度就退回 pending(**但要等一会儿**),否则落 dead。

        返回该任务的新状态,便于调用方与测试断言。

        退避多久由 `backoff_delay(attempts)` 决定:2s、4s、8s… 上限 60s,
        并带等比例抖动。退避时间写在 `next_attempt_at` 上,`claim()` 会跳过
        还没到点的任务 —— 所以「退回 pending」不再等于「马上被重领」。
        """
        self.init_db()
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT attempts, max_attempts FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return DEAD
            state = PENDING if row["attempts"] < row["max_attempts"] else DEAD
            ready_at = None
            if state == PENDING:
                ready_at = now + backoff_delay(row["attempts"],
                                               base=self.backoff_base,
                                               cap=self.backoff_cap)
            conn.execute(
                "UPDATE jobs SET state = ?, error = ?, lease_owner = NULL, "
                "lease_expires = NULL, next_attempt_at = ?, updated_at = ? WHERE id = ?",
                (state, error, ready_at, _now_iso(), job_id),
            )
            return state
        finally:
            conn.close()

    # ------------------------------------------------------------ 回收

    def reap_expired(self, now: float | None = None) -> list[str]:
        """把租约过期的 running 任务捞回来。返回被回收的 key 列表。

        进程崩溃后没人主动归还任务,所以必须由下一个启动者做这件事 ——
        判断依据只能是「租约到期了」,而不是「进程还在不在」。

        这里**清掉退避**(`next_attempt_at = NULL`),让回收后能立刻重跑:
        崩溃恢复要的是尽快接上,而崩溃循环本身已经被租约时长节流了。
        """
        self.init_db()
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, key, attempts, max_attempts FROM jobs "
                "WHERE state = ? AND lease_expires IS NOT NULL AND lease_expires < ?",
                (RUNNING, now),
            ).fetchall()
            reaped = []
            for r in rows:
                state = PENDING if r["attempts"] < r["max_attempts"] else DEAD
                conn.execute(
                    "UPDATE jobs SET state = ?, lease_owner = NULL, lease_expires = NULL, "
                    "error = COALESCE(error, '租约过期,已回收'), next_attempt_at = NULL, "
                    "updated_at = ? WHERE id = ?",
                    (state, _now_iso(), r["id"]),
                )
                reaped.append(r["key"])
            return reaped
        finally:
            conn.close()

    # ------------------------------------------------------------ 查询

    def stats(self, run_id: str) -> dict[str, int]:
        self.init_db()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT state, COUNT(*) AS n FROM jobs WHERE run_id = ? GROUP BY state",
                (run_id,),
            ).fetchall()
            out = {PENDING: 0, RUNNING: 0, DONE: 0, DEAD: 0}
            for r in rows:
                out[r["state"]] = r["n"]
            return out
        finally:
            conn.close()

    def results(self, run_id: str) -> dict[str, dict]:
        """已完成任务的结果,按 key 索引 —— 续跑时直接拿它跳过重算。"""
        self.init_db()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT key, result_json FROM jobs WHERE run_id = ? AND state = ?",
                (run_id, DONE),
            ).fetchall()
            return {r["key"]: json.loads(r["result_json"]) for r in rows}
        finally:
            conn.close()

    def failures(self, run_id: str) -> dict[str, str]:
        """彻底失败(dead)的任务与最后一次错误。"""
        self.init_db()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT key, error FROM jobs WHERE run_id = ? AND state = ?",
                (run_id, DEAD),
            ).fetchall()
            return {r["key"]: r["error"] for r in rows}
        finally:
            conn.close()


DEFAULT_QUEUE = JobQueue()


def run_durable(
    queue: JobQueue,
    run_id: str,
    tasks: list[tuple[str, dict]],
    run_one,
    max_workers: int = 4,
    lease_seconds: float = 300,
    max_attempts: int = 3,
    progress=None,
) -> dict:
    """在持久化队列上跑一批任务。

    参数:
        tasks:    [(key, payload), ...]
        run_one:  payload -> dict(结果)
        progress: (done, total) 回调

    返回:
        {"results": {key: result}, "failed": {key: error}, "stats": {...}}

    幂等:对同一个 run_id 重复调用,已完成的任务不会再执行一遍。
    """
    queue.reap_expired()  # 先收拾上一次崩溃留下的在跑任务
    for key, payload in tasks:
        queue.enqueue(run_id, key, payload, max_attempts=max_attempts)

    total = len(tasks)
    results = queue.results(run_id)
    failures = queue.failures(run_id)
    done_count = len(results) + len(failures)

    if done_count >= total:
        if progress:
            progress(total, total)
        return {"results": results, "failed": failures, "stats": queue.stats(run_id)}

    lock = threading.Lock()
    stop = threading.Event()

    def worker(idx: int) -> None:
        nonlocal done_count
        owner = f"w{idx}-{threading.get_ident()}"
        while not stop.is_set():
            job = queue.claim(owner, run_id=run_id, lease_seconds=lease_seconds)
            if job is None:
                # "领不到"现在有两种含义:**活干完了**,或者**剩下的都在退避中**。
                # 分不清的话 worker 会提前收工,把还在退避的任务丢在那儿没人管
                # —— 加了退避之后这是最容易漏的一处。
                delay = queue.next_attempt_in(run_id)
                if delay is None:
                    return  # 真的没活了
                time.sleep(min(max(delay, 0.05), 2.0))
                continue

            key = job["key"]
            try:
                result = run_one(job["payload"])
            except Exception as exc:  # noqa: BLE001 - 交给队列决定重试还是落 dead
                detail = f"{type(exc).__name__}: {exc}"
                if queue.fail(job["id"], detail) != DEAD:
                    continue  # 还有重试额度,退回 pending(队列已给它排了退避)
                with lock:
                    failures[key] = detail
                    done_count += 1
            else:
                queue.complete(job["id"], result)
                with lock:
                    results[key] = result
                    done_count += 1

            # 进度回调必须放在**任务状态落库之后**、且**在 try 之外**。
            #
            # 之前它写在 try 里面,于是回调一抛异常,except 分支就会去 fail() 一个
            # 刚刚 complete() 过的任务 —— 已完成的任务被翻回 pending(白跑一遍),
            # 而且 done_count 会被重复 +1,进度冲到 116%。
            # 进度是**展示**,任何情况下都不该反过来改任务状态。
            if progress:
                with lock:
                    snapshot = done_count
                try:
                    progress(snapshot, total)
                except Exception:  # noqa: BLE001 - 回调坏了也不能影响任务
                    pass

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(max(1, min(max_workers, total)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return {"results": results, "failed": failures, "stats": queue.stats(run_id)}
