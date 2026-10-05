"""运行事件日志 + 断线回放(Last-Event-ID 语义)。

**问题**

Agent 跑一次会产出几十个事件:开始调工具、工具返回、判定、最终答案……
现在这些只活在 Streamlit 这一次渲染里。用户刷新页面、切走一会儿再回来、
或者网络抖一下,前面发生过什么就没了,只剩一个最终结果 —— 想排查「它刚才
到底调了哪几个工具、顺序对不对」只能重跑一遍。

**做法**

每个 run 的每个事件都带一个**单调递增的 seq** 落库。客户端记住自己收到的
最后一个 seq,断线后带上它重新连,服务端只回补**比它大**的那些:

    断线期间产生的事件 → 补上,不丢
    已经收到过的事件   → 不再发,不重复

**两个容易写错的地方,都用测试钉住了**

1. 回放必须**严格大于** last_event_id。写成 `>=` 的话客户端每次重连都会把
   最后一条重渲染一遍。
2. seq 在同一 run 内必须**连续且唯一**。客户端就是靠它判断「有没有丢事件」的,
   跳号意味着真的丢了东西,不能当没看见。
"""
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "events.db"


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class EventLog:
    """按 run 追加事件,并支持从任意 seq 之后回放。"""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DB_PATH
        self._lock = threading.Lock()  # 进程内串行化 seq 分配

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id     TEXT    NOT NULL,
                    seq        INTEGER NOT NULL,
                    type       TEXT    NOT NULL,
                    data_json  TEXT    NOT NULL,
                    created_at TEXT    NOT NULL,
                    UNIQUE(run_id, seq)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON run_events(run_id, seq)")
        finally:
            conn.close()

    def append(self, run_id: str, type: str, data: dict | None = None) -> dict:
        """追加一个事件,自动分配下一个 seq。"""
        self.init_db()
        payload = json.dumps(data or {}, ensure_ascii=False, default=str)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")  # 拿写锁,避免两个线程分到同一个 seq
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 AS next FROM run_events WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                seq = int(row["next"])
                conn.execute(
                    "INSERT INTO run_events (run_id, seq, type, data_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (run_id, seq, type, payload, _now_iso()),
                )
                conn.execute("COMMIT")
            finally:
                conn.close()
        return {"run_id": run_id, "seq": seq, "type": type, "data": data or {}}

    def events_since(self, run_id: str, last_event_id: int = 0, limit: int | None = None) -> list[dict]:
        """回放 seq **严格大于** last_event_id 的事件。

        对应 SSE 的 `Last-Event-ID`:客户端带上它收到的最后一个 id,只拿新的。
        """
        self.init_db()  # 读也要保证表在,否则「先读后写」的库会直接崩
        conn = self._connect()
        try:
            sql = ("SELECT run_id, seq, type, data_json, created_at FROM run_events "
                   "WHERE run_id = ? AND seq > ? ORDER BY seq")
            args: tuple = (run_id, last_event_id)
            if limit is not None:
                sql += " LIMIT ?"
                args = (*args, limit)
            rows = conn.execute(sql, args).fetchall()
            return [
                {"run_id": r["run_id"], "seq": r["seq"], "type": r["type"],
                 "data": json.loads(r["data_json"]), "created_at": r["created_at"]}
                for r in rows
            ]
        finally:
            conn.close()

    def last_seq(self, run_id: str) -> int:
        """这个 run 当前的最大 seq(客户端下次重连就带它)。"""
        self.init_db()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS s FROM run_events WHERE run_id = ?", (run_id,)
            ).fetchone()
            return int(row["s"])
        finally:
            conn.close()

    def missing_seqs(self, run_id: str) -> list[int]:
        """检查 seq 有没有跳号。客户端靠 seq 判断丢没丢事件,所以这里必须能自检。"""
        self.init_db()
        conn = self._connect()
        try:
            got = [int(r["seq"]) for r in conn.execute(
                "SELECT seq FROM run_events WHERE run_id = ? ORDER BY seq", (run_id,)
            ).fetchall()]
        finally:
            conn.close()
        if not got:
            return []
        return [n for n in range(got[0], got[-1] + 1) if n not in set(got)]

    def runs(self, limit: int = 20) -> list[dict]:
        """最近有事件记录的 run,带事件数与起止时间。"""
        self.init_db()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT run_id, COUNT(*) AS n, MIN(seq) AS first_seq, MAX(seq) AS last_seq, "
                "MIN(created_at) AS started_at, MAX(created_at) AS updated_at "
                "FROM run_events GROUP BY run_id ORDER BY MAX(id) DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


DEFAULT_LOG = EventLog()
