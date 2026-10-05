"""持久化任务队列的测试。

这个文件要证明的是「内存队列给不了的三件事」真的成立:
    1. 已完成的任务不会被重跑(重启续跑)
    2. 租约过期的在跑任务会被回收(崩溃恢复)
    3. 并发领取不会把同一个任务发给两个 worker(不重复扣费)

运行:pytest -q
"""
import threading
import time

import pytest

from jobs import DEAD, DONE, PENDING, RUNNING, JobQueue, run_durable


@pytest.fixture()
def queue(tmp_path):
    return JobQueue(tmp_path / "jobs.db")


# ------------------------------------------------------------------ 入队

def test_enqueue_is_idempotent_per_run_and_key(queue):
    """同 (run_id, key) 重复入队不能覆盖已有状态 —— 否则续跑会把完成的重置回 pending。"""
    queue.enqueue("r1", "a", {"x": 1})
    job = queue.claim("w1", run_id="r1")
    queue.complete(job["id"], {"ok": True})

    queue.enqueue("r1", "a", {"x": 999})  # 再次入队
    assert queue.stats("r1")[DONE] == 1, "重复入队把已完成的任务重置了"
    assert queue.results("r1") == {"a": {"ok": True}}


def test_runs_are_isolated(queue):
    queue.enqueue("r1", "a", {})
    queue.enqueue("r2", "a", {})
    assert queue.stats("r1")[PENDING] == 1
    assert queue.stats("r2")[PENDING] == 1


# ------------------------------------------------------------------ 领取

def test_claim_returns_none_when_empty(queue):
    queue.enqueue("r1", "a", {})
    assert queue.claim("w1", run_id="r1") is not None
    assert queue.claim("w1", run_id="r1") is None, "同一条任务被领了两次"


def test_claim_marks_running_and_sets_lease(queue):
    queue.enqueue("r1", "a", {"k": "v"})
    job = queue.claim("w1", run_id="r1", lease_seconds=60, now=1000.0)

    assert job["state"] == RUNNING
    assert job["lease_owner"] == "w1"
    assert job["lease_expires"] == 1060.0
    assert job["attempts"] == 1
    assert job["payload"] == {"k": "v"}
    assert queue.stats("r1")[RUNNING] == 1


def test_concurrent_claims_never_double_issue(queue):
    """并发领取是这套东西的地基:重复发同一条 = 同一份 token 烧两遍。"""
    total = 40
    for i in range(total):
        queue.enqueue("r1", f"job-{i}", {"i": i})

    got: list[int] = []
    lock = threading.Lock()

    def claimer(wid: int):
        while True:
            job = queue.claim(f"w{wid}", run_id="r1")
            if job is None:
                return
            with lock:
                got.append(job["id"])

    threads = [threading.Thread(target=claimer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(got) == total, f"领取次数 {len(got)} != 任务数 {total}"
    assert len(set(got)) == total, "同一个任务被发给了多个 worker"


# ------------------------------------------------------------ 完成与失败

def test_complete_stores_result(queue):
    queue.enqueue("r1", "a", {})
    job = queue.claim("w1", run_id="r1")
    queue.complete(job["id"], {"score": 7})

    assert queue.results("r1") == {"a": {"score": 7}}
    assert queue.stats("r1")[DONE] == 1
    assert queue.stats("r1")[RUNNING] == 0


def test_fail_requeues_then_goes_dead_after_max_attempts(queue):
    queue.enqueue("r1", "a", {}, max_attempts=3)

    states = []
    for _ in range(3):
        job = queue.claim("w1", run_id="r1")
        assert job is not None, "还有重试额度时应当可再次领取"
        states.append(queue.fail(job["id"], "boom"))

    assert states == [PENDING, PENDING, DEAD]
    assert queue.stats("r1")[DEAD] == 1
    assert queue.failures("r1") == {"a": "boom"}
    assert queue.claim("w1", run_id="r1") is None, "dead 任务不该再被领取"


# ------------------------------------------------------------------ 回收

def test_reaper_returns_expired_lease_to_pending(queue):
    """进程崩溃后没人主动归还任务,所以只能靠租约到期来判定。"""
    queue.enqueue("r1", "a", {})
    queue.claim("dead-worker", run_id="r1", lease_seconds=60, now=1000.0)

    assert queue.reap_expired(now=1030.0) == [], "租约还没到期就不该回收"
    assert queue.stats("r1")[RUNNING] == 1

    assert queue.reap_expired(now=1061.0) == ["a"]
    assert queue.stats("r1")[PENDING] == 1
    assert queue.claim("w2", run_id="r1") is not None, "回收后应当能被重新领取"


def test_reaper_marks_dead_when_attempts_exhausted(queue):
    queue.enqueue("r1", "a", {}, max_attempts=1)
    queue.claim("dead-worker", run_id="r1", lease_seconds=1, now=1000.0)

    assert queue.reap_expired(now=2000.0) == ["a"]
    assert queue.stats("r1")[DEAD] == 1


def test_reaper_leaves_live_lease_alone(queue):
    queue.enqueue("r1", "a", {})
    queue.claim("w1", run_id="r1", lease_seconds=600, now=1000.0)
    assert queue.reap_expired(now=1500.0) == []
    assert queue.stats("r1")[RUNNING] == 1


# ------------------------------------------------------- run_durable 端到端

def test_run_durable_executes_everything_once(queue):
    calls: list[str] = []

    def run_one(payload):
        calls.append(payload["k"])
        return {"v": payload["k"].upper()}

    out = run_durable(queue, "r1", [("a", {"k": "a"}), ("b", {"k": "b"})], run_one)

    assert sorted(calls) == ["a", "b"]
    assert out["results"] == {"a": {"v": "A"}, "b": {"v": "B"}}
    assert out["failed"] == {}


def test_run_durable_skips_completed_on_resume(queue):
    """核心场景:跑到一半进程没了,重启后已完成的不该再算一遍。

    这是内存队列做不到的事 —— 它会整批重来,已完成的 token 全白烧。
    """
    tasks = [("a", {"k": "a"}), ("b", {"k": "b"})]

    # 手工复现崩溃现场:a 已完成;b 被领走、进程随即挂掉,租约留在那里过期
    for key, payload in tasks:
        queue.enqueue("r1", key, payload)
    queue.complete(queue.claim("w1", run_id="r1")["id"], {"v": "a"})
    queue.claim("crashed", run_id="r1", lease_seconds=1, now=time.time() - 100)
    assert queue.stats("r1")[RUNNING] == 1

    calls: list[str] = []
    out = run_durable(queue, "r1", tasks,
                      lambda p: calls.append(p["k"]) or {"v": p["k"]})

    assert calls == ["b"], f"重启后只该补跑未完成的 b,实际跑了 {calls}"
    assert out["results"]["a"] == {"v": "a"}, "a 的结果应当从库里直接读出来"
    assert out["results"]["b"] == {"v": "b"}


def test_run_durable_reports_dead_jobs(queue):
    def always_bad(payload):
        raise ValueError("坏 CSV")

    out = run_durable(queue, "r1", [("a", {})], always_bad, max_workers=1, max_attempts=2)

    assert out["results"] == {}
    assert "a" in out["failed"]
    assert "坏 CSV" in out["failed"]["a"]
    assert out["stats"][DEAD] == 1


def test_run_durable_retries_transient_failure(queue):
    """LLM 调用偶发超时应当重试成功,而不是直接判死。"""
    attempts = {"n": 0}

    def flaky(payload):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise TimeoutError("偶发超时")
        return {"ok": True}

    out = run_durable(queue, "r1", [("a", {})], flaky, max_workers=1, max_attempts=3)

    assert out["results"] == {"a": {"ok": True}}
    assert attempts["n"] == 2


def test_run_durable_progress_reaches_total(queue):
    seen: list[tuple[int, int]] = []
    run_durable(
        queue, "r1",
        [(f"j{i}", {}) for i in range(3)],
        lambda p: {"ok": True},
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen[-1] == (3, 3)


def test_run_durable_reaps_orphan_before_starting(queue):
    """上一次崩溃留下的 running 任务,本次启动应当先被回收再执行。"""
    queue.enqueue("r1", "a", {"k": "a"})
    queue.claim("crashed-process", run_id="r1", lease_seconds=1, now=time.time() - 100)

    calls: list[str] = []
    out = run_durable(queue, "r1", [("a", {"k": "a"})],
                      lambda p: calls.append(p["k"]) or {"ok": True})

    assert calls == ["a"], "孤儿任务应当被回收后重新执行"
    assert out["results"] == {"a": {"ok": True}}


# ------------------------------------------------ 竞品对比跑在持久化队列上

class _FakeLLM:
    """记录调用次数;每次都返回结构合法的批次 JSON。"""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1

        class _Resp:
            content = ('{"positives": [], "pain_points": [], '
                       '"purchase_motives": [], "competitor_mentions": []}')

        return _Resp()


def _write_csv(path, rows=("5,很好用",)):
    path.write_text("rating,review\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_run_compare_resumes_without_reanalysing_done_products(queue, tmp_path):
    """同一批竞品重跑,已完成的竞品不该再喂一遍 LLM。"""
    from competitor_compare import run_compare

    a = _write_csv(tmp_path / "a.csv")
    b = _write_csv(tmp_path / "b.csv")
    products = [("A", str(a)), ("B", str(b))]

    llm = _FakeLLM()
    first = run_compare(products, llm, run_id="fixed", queue=queue)
    assert set(first["per_product"]) == {"A", "B"}
    assert first["failed"] == {}
    calls_first = llm.calls  # 2 个竞品 + 1 次汇总

    llm2 = _FakeLLM()
    second = run_compare(products, llm2, run_id="fixed", queue=queue)

    assert set(second["per_product"]) == {"A", "B"}
    assert llm2.calls == 1, f"重跑只该剩 1 次汇总调用,实际 {llm2.calls} 次"
    assert llm2.calls < calls_first


def test_run_compare_default_run_id_tracks_input(queue, tmp_path):
    """输入没变 → run_id 不变 → 可续跑;输入变了 → 换 run_id → 重新分析。"""
    from competitor_compare import default_run_id

    a = _write_csv(tmp_path / "a.csv")
    products = [("A", str(a))]
    first = default_run_id(products)
    assert default_run_id(products) == first

    _write_csv(tmp_path / "a.csv", rows=("5,很好用", "4,还不错"))
    assert default_run_id(products) != first, "文件变了 run_id 却没变,会拿旧结果糊弄"


def test_run_compare_reports_partial_failure(queue, tmp_path):
    """一个竞品挂了,不能连带整批失败 —— 要出报告并注明缺了谁。"""
    from competitor_compare import run_compare

    a = _write_csv(tmp_path / "a.csv")
    products = [("A", str(a)), ("BAD", str(tmp_path / "does-not-exist.csv"))]

    out = run_compare(products, _FakeLLM(), run_id="partial", queue=queue)

    assert set(out["per_product"]) == {"A"}
    assert "BAD" in out["failed"]
    assert "BAD" in out["report"], "失败名单应当写进报告里"
