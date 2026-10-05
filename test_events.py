"""事件日志与断线回放的测试。

核心断言只有两条,其余都在给它们加固:
    1. 回放取的是 **seq > last_event_id**,不是 >= —— 否则重连必重复渲染
    2. seq 在同一 run 内连续唯一 —— 客户端靠它判断有没有丢事件
"""
import sqlite3
import threading

import pytest

from events import EventLog


@pytest.fixture()
def log(tmp_path):
    return EventLog(tmp_path / "events.db")


# ------------------------------------------------------------ 追加与 seq

def test_append_assigns_consecutive_seqs(log):
    for i in range(5):
        ev = log.append("r1", "tool_call", {"i": i})
        assert ev["seq"] == i + 1
    assert log.last_seq("r1") == 5
    assert log.missing_seqs("r1") == []


def test_first_seq_is_one(log):
    assert log.append("r1", "run_started")["seq"] == 1
    assert log.last_seq("r1") == 1


def test_runs_are_isolated(log):
    log.append("r1", "a")
    log.append("r2", "b")
    assert log.last_seq("r1") == 1
    assert log.last_seq("r2") == 1


def test_data_round_trips_unchanged(log):
    log.append("r1", "tool_result", {"tool": "generate_listing", "args": {"x": [1, 2]}})
    got = log.events_since("r1")[0]
    assert got["type"] == "tool_result"
    assert got["data"] == {"tool": "generate_listing", "args": {"x": [1, 2]}}


def test_concurrent_appends_never_share_a_seq(log):
    """并发写不能分到同一个 seq,否则 UNIQUE 冲突或事件被覆盖。"""
    total = 60
    lock = threading.Lock()
    seqs: list[int] = []

    def writer(tid: int):
        for i in range(total // 6):
            ev = log.append("r1", "tick", {"t": tid, "i": i})
            with lock:
                seqs.append(ev["seq"])

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seqs) == total
    assert len(set(seqs)) == total, "有重复 seq"
    assert sorted(seqs) == list(range(1, total + 1)), "seq 不连续"
    assert log.missing_seqs("r1") == []


# ------------------------------------------------------------ 回放

def test_events_since_is_strictly_greater(log):
    """带 >= 的实现在这里就会挂 —— 重连会把最后一条重渲染一遍。"""
    for i in range(3):
        log.append("r1", "e", {"i": i})

    assert [e["seq"] for e in log.events_since("r1", 0)] == [1, 2, 3]
    assert [e["seq"] for e in log.events_since("r1", 1)] == [2, 3]
    assert [e["seq"] for e in log.events_since("r1", 3)] == []
    assert log.events_since("r1", 99) == []


def test_replay_after_disconnect_returns_only_what_was_missed(log):
    """核心场景:客户端收到 3 条后断线,期间又产生 2 条。"""
    for i in range(3):
        log.append("r1", "e", {"i": i})
    cursor = log.last_seq("r1")          # 客户端记住的最后一个 id
    assert cursor == 3

    for i in range(3, 5):                 # 断线期间
        log.append("r1", "e", {"i": i})

    missed = log.events_since("r1", cursor)
    assert [e["seq"] for e in missed] == [4, 5]
    assert [e["data"]["i"] for e in missed] == [3, 4]

    # 再连一次不该拿到任何东西(游标已经是 5)
    assert log.events_since("r1", 5) == []


def test_cursor_advances_to_last_seq(log):
    log.append("r1", "a")
    log.append("r1", "b")
    cursor = log.last_seq("r1")
    log.append("r1", "c")
    assert [e["type"] for e in log.events_since("r1", cursor)] == ["c"]


def test_events_since_limit(log):
    for i in range(10):
        log.append("r1", "e", {"i": i})
    assert len(log.events_since("r1", 0, limit=4)) == 4
    assert [e["seq"] for e in log.events_since("r1", 0, limit=4)] == [1, 2, 3, 4]


def test_empty_run_replays_nothing(log):
    assert log.events_since("never-existed") == []
    assert log.last_seq("never-existed") == 0


# ------------------------------------------------------------ 自检

def test_missing_seqs_detects_a_gap(log):
    """客户端靠 seq 判断丢没丢事件,所以这条自检必须真的能发现问题。"""
    for i in range(5):
        log.append("r1", "e", {"i": i})

    conn = sqlite3.connect(log.db_path)
    conn.execute("DELETE FROM run_events WHERE run_id='r1' AND seq=3")
    conn.commit()
    conn.close()

    assert log.missing_seqs("r1") == [3]


def test_runs_lists_recent_with_counts(log):
    for i in range(3):
        log.append("r1", "e", {"i": i})
    log.append("r2", "e")

    runs = {r["run_id"]: r for r in log.runs()}
    assert runs["r1"]["n"] == 3
    assert runs["r1"]["last_seq"] == 3
    assert runs["r2"]["n"] == 1
