"""原始答案留存 —— 「可溯源」的全部含义。

指标是**派生数据**，原文是**一手数据**。如果只存指标，「提及率 64.8%」就是一
句无法核对的断言：面试官、客户、或者三个月后的你自己问「这数怎么来的」，
答不上来。

所以这一层很薄，但职责很硬：**每个采样点一行，一字不改**。任何指标都能靠
`answer_id` 回到产生它的那条原文。

幂等性：`UNIQUE(run_id, engine, question, sample)` + `INSERT OR IGNORE`。
这让「同一个 run 重跑」变成纯读操作 —— 和任务队列的续跑语义对齐：崩了重来，
已经采到的不重采（不重花钱）。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from engines.base import Answer, FailureKind

DB_PATH = Path(__file__).parent.parent / "data" / "answers.db"


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class AnswerStore:
    """答案库。一个实例对应一个库文件（测试传临时路径）。"""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DB_PATH

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS answers (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id        TEXT    NOT NULL,
                    engine        TEXT    NOT NULL,
                    question      TEXT    NOT NULL,
                    sample        INTEGER NOT NULL,
                    text          TEXT    NOT NULL DEFAULT '',
                    citations_json TEXT   NOT NULL DEFAULT '[]',
                    model         TEXT    NOT NULL DEFAULT '',
                    latency_ms    INTEGER NOT NULL DEFAULT 0,
                    error         TEXT,
                    error_detail  TEXT    NOT NULL DEFAULT '',
                    created_at    TEXT    NOT NULL,
                    UNIQUE(run_id, engine, question, sample)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_answers_run ON answers(run_id, engine)"
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------ 写

    def save_many(self, run_id: str, answers: list[Answer]) -> int:
        """批量落库。同 (run_id, engine, question, sample) 已存在则跳过。

        返回真正新写入的行数 —— 重跑时它会明显小于 len(answers)，这正是
        「没有重采」的证据。
        """
        self.init_db()
        ts = _now_iso()
        conn = self._connect()
        try:
            cur = conn.executemany(
                "INSERT OR IGNORE INTO answers "
                "(run_id, engine, question, sample, text, citations_json, model, "
                " latency_ms, error, error_detail, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (run_id, a.engine, a.question, a.sample, a.text,
                     json.dumps(a.citations, ensure_ascii=False), a.model,
                     a.latency_ms,
                     a.error.value if a.error else None, a.error_detail, ts)
                    for a in answers
                ],
            )
            conn.commit()
            return cur.rowcount if cur.rowcount >= 0 else 0
        finally:
            conn.close()

    # ------------------------------------------------------------ 读

    def load(self, run_id: str) -> list[Answer]:
        """按 (引擎, 问题, 采样序号) 稳定排序取出。顺序稳定，指标才可复现。"""
        self.init_db()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM answers WHERE run_id = ? "
                "ORDER BY engine, question, sample, id",
                (run_id,),
            ).fetchall()
        finally:
            conn.close()
        return [
            Answer(
                engine=r["engine"],
                question=r["question"],
                sample=r["sample"],
                text=r["text"],
                citations=json.loads(r["citations_json"]),
                model=r["model"],
                latency_ms=r["latency_ms"],
                error=FailureKind(r["error"]) if r["error"] else None,
                error_detail=r["error_detail"],
                asked_at=r["created_at"],
                id=r["id"],
            )
            for r in rows
        ]

    def get(self, answer_id: int) -> Answer | None:
        """按 id 取一条 —— 「这个指标凭什么这么说」的最后一跳。"""
        self.init_db()
        conn = self._connect()
        try:
            r = conn.execute("SELECT * FROM answers WHERE id = ?", (answer_id,)).fetchone()
        finally:
            conn.close()
        if r is None:
            return None
        return Answer(
            engine=r["engine"], question=r["question"], sample=r["sample"],
            text=r["text"], citations=json.loads(r["citations_json"]),
            model=r["model"], latency_ms=r["latency_ms"],
            error=FailureKind(r["error"]) if r["error"] else None,
            error_detail=r["error_detail"], asked_at=r["created_at"], id=r["id"],
        )

    def runs(self, limit: int = 20) -> list[dict]:
        """每次采集批次的概况，用于历史列表。"""
        self.init_db()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT run_id, COUNT(*) AS total, "
                "SUM(CASE WHEN error IS NULL THEN 1 ELSE 0 END) AS ok, "
                "MIN(created_at) AS started_at, "
                "COUNT(DISTINCT engine) AS engines, "
                "COUNT(DISTINCT question) AS questions "
                "FROM answers GROUP BY run_id ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [
            {"run_id": r["run_id"], "total": r["total"], "ok": r["ok"],
             "failed": r["total"] - r["ok"], "started_at": r["started_at"],
             "engines": r["engines"], "questions": r["questions"]}
            for r in rows
        ]


DEFAULT_STORE = AnswerStore()
