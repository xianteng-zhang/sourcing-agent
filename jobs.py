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
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "jobs.db"

PENDING, RUNNING, DONE, DEAD = "pending", "running", "done", "dead"


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class JobQueue:
    """SQLite 上的持久化任务队列。一个实例对应一个库文件(测试可传临时路径)。"""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DB_PATH

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
                    result_json   TEXT,
                    error         TEXT,
                    created_at    TEXT    NOT NULL,
                    updated_at    TEXT    NOT NULL,
                    UNIQUE(run_id, key)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(state, run_id)")
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
        """原子领取一个 pending 任务,并给它打上租约。没有可领的返回 None。"""
        self.init_db()
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")  # 拿到写锁,杜绝两人领同一条
            row = conn.execute(
                "SELECT * FROM jobs WHERE state = ? AND (? IS NULL OR run_id = ?) "
                "ORDER BY id LIMIT 1",
                (PENDING, run_id, run_id),
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

    # ------------------------------------------------------ 完成 / 失败

    def complete(self, job_id: int, result: dict) -> None:
        self.init_db()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE jobs SET state = ?, result_json = ?, error = NULL, "
                "lease_owner = NULL, lease_expires = NULL, updated_at = ? WHERE id = ?",
                (DONE, json.dumps(result, ensure_ascii=False), _now_iso(), job_id),
            )
        finally:
            conn.close()

    def fail(self, job_id: int, error: str) -> str:
        """记一次失败:还有重试额度就退回 pending(换个 worker 再来),否则落 dead。

        返回该任务的新状态,便于调用方与测试断言。
        """
        self.init_db()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT attempts, max_attempts FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return DEAD
            state = PENDING if row["attempts"] < row["max_attempts"] else DEAD
            conn.execute(
                "UPDATE jobs SET state = ?, error = ?, lease_owner = NULL, "
                "lease_expires = NULL, updated_at = ? WHERE id = ?",
                (state, error, _now_iso(), job_id),
            )
            return state
        finally:
            conn.close()

    # ------------------------------------------------------------ 回收

    def reap_expired(self, now: float | None = None) -> list[str]:
        """把租约过期的 running 任务捞回来。返回被回收的 key 列表。

        进程崩溃后没人主动归还任务,所以必须由下一个启动者做这件事 ——
        判断依据只能是「租约到期了」,而不是「进程还在不在」。
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
                    "error = COALESCE(error, '租约过期,已回收'), updated_at = ? WHERE id = ?",
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
                return
            key = job["key"]
            try:
                result = run_one(job["payload"])
            except Exception as exc:  # noqa: BLE001 - 交给队列决定重试还是落 dead
                detail = f"{type(exc).__name__}: {exc}"
                if queue.fail(job["id"], detail) != DEAD:
                    continue  # 还有重试额度,退回 pending 等下一次领
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
