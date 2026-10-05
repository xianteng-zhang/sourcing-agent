"""历史记录存储的测试。

重点不在 CRUD,而在**迁移**:历史记录是用户数据,不能因为加了一列就让人
把库删掉重来。所以「老库加 run_id 列」的路径必须有测试盯着。
"""
import sqlite3

import pytest

import storage


@pytest.fixture()
def db(tmp_path, monkeypatch):
    path = tmp_path / "history.db"
    monkeypatch.setattr(storage, "DB_PATH", path)
    return path


def _columns(path) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [r[1] for r in conn.execute("PRAGMA table_info(analysis_history)")]
    finally:
        conn.close()


# ------------------------------------------------------------ 基本读写

def test_save_and_get_round_trips_run_id(db):
    rid = storage.save_analysis("标题", 10, 4.5, {"a": 1}, "报告",
                                type="选品Agent", run_id="run-abc")
    got = storage.get_analysis(rid)
    assert got["run_id"] == "run-abc"
    assert got["type"] == "选品Agent"
    assert got["report"] == "报告"


def test_run_id_is_optional(db):
    """评论挖掘没有 run_id,不能因此写不进去。"""
    rid = storage.save_analysis("标题", 10, None, {}, "报告")
    assert storage.get_analysis(rid)["run_id"] is None


def test_list_history_exposes_run_id(db):
    storage.save_analysis("有回放", 1, None, {}, "r", run_id="run-1")
    storage.save_analysis("没回放", 1, None, {}, "r")

    by_title = {r["title"]: r for r in storage.list_history()}
    assert by_title["有回放"]["run_id"] == "run-1"
    assert by_title["没回放"]["run_id"] is None


def test_delete_removes_only_that_row(db):
    a = storage.save_analysis("A", 1, None, {}, "r")
    storage.save_analysis("B", 1, None, {}, "r")
    storage.delete_analysis(a)
    titles = [r["title"] for r in storage.list_history()]
    assert titles == ["B"]


# ------------------------------------------------------------ 迁移

_LEGACY_WITH_TYPE = """
CREATE TABLE analysis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL DEFAULT '评论挖掘',
    title TEXT NOT NULL,
    review_count INTEGER,
    avg_rating REAL,
    stats_json TEXT,
    report TEXT,
    created_at TEXT NOT NULL
)
"""

_LEGACY_WITHOUT_TYPE = """
CREATE TABLE analysis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    review_count INTEGER,
    avg_rating REAL,
    stats_json TEXT,
    report TEXT,
    created_at TEXT NOT NULL
)
"""


def _make_legacy(path, ddl: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(ddl)
    conn.execute("INSERT INTO analysis_history (title, created_at) VALUES ('老记录', '2026-01-01')")
    conn.commit()
    conn.close()


def test_migration_adds_run_id_to_a_legacy_table(db):
    """老库(有 type、没 run_id)必须被自动补列。"""
    _make_legacy(db, _LEGACY_WITH_TYPE)

    storage.init_db()

    assert "run_id" in _columns(db)
    assert len(storage.list_history()) == 1, "迁移不该丢数据"
    rid = storage.save_analysis("新记录", 1, None, {}, "r", run_id="run-x")
    assert storage.get_analysis(rid)["run_id"] == "run-x"


def test_migration_on_truly_old_table_handles_both_columns(db):
    """更老的库连 type 都没有 —— 两条迁移要能一起跑完。"""
    _make_legacy(db, _LEGACY_WITHOUT_TYPE)

    storage.init_db()

    cols = _columns(db)
    assert "type" in cols and "run_id" in cols
    assert len(storage.list_history()) == 1


def test_init_db_is_idempotent(db):
    """反复调用不能报错,也不能把数据清掉 —— app 每次交互都会调到它。"""
    storage.save_analysis("A", 1, None, {}, "r")
    storage.init_db()
    storage.init_db()
    assert len(storage.list_history()) == 1
