"""历史分析记录的本地存储(SQLite 单文件,零配置)。"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "history.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """建表(不存在才建)。"""
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                review_count INTEGER,
                avg_rating REAL,
                stats_json TEXT,
                report TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_analysis(
    title: str,
    review_count: int,
    avg_rating: float | None,
    stats: dict,
    report: str,
) -> int:
    """保存一次分析,返回记录 id。"""
    init_db()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO analysis_history (title, review_count, avg_rating, stats_json, report, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (title, review_count, avg_rating, json.dumps(stats, ensure_ascii=False), report, created_at),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def list_history(limit: int = 100) -> list[dict]:
    """列出历史记录(新的在前),不含完整报告正文。"""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, title, review_count, avg_rating, created_at "
            "FROM analysis_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_analysis(analysis_id: int) -> dict | None:
    """取单条完整记录(含报告)。"""
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM analysis_history WHERE id = ?", (analysis_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_analysis(analysis_id: int) -> None:
    """删除一条记录。"""
    init_db()
    conn = _connect()
    try:
        conn.execute("DELETE FROM analysis_history WHERE id = ?", (analysis_id,))
        conn.commit()
    finally:
        conn.close()
