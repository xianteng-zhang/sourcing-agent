"""多引擎采集的测试。

这个文件要钉死的是**四条口径**，它们全是「不写测试就一定会写错」的地方：

    1. 失败不能进分母 —— 否则引擎越不稳定，指标越难看
    2. 失败不能被算成「没提到品牌」—— 那是把基建问题记成品牌问题
    3. 提到 ≠ 被推荐 —— 顺带提及不该算进榜
    4. 同一个问题样本自相矛盾时，它的比率不可当结论

另外两条工程不变量：
    5. 重跑同一个 run 不重采（省钱）—— 靠队列 + 落库双重幂等
    6. 任务在队列里死掉时，那条采样不能凭空消失（既不在分子也不在分母）

运行：pytest -q
"""
from __future__ import annotations

import pytest

from engines import (
    Answer,
    AnswerStore,
    EngineError,
    FailureKind,
    OpenAICompatEngine,
    ReplayEngine,
    check_comparability,
    collect,
    compute,
    default_run_id,
    dump_jsonl,
    extract_citations,
    find_mention,
    format_report,
    load_recordings,
    plan_tasks,
    safe_ask,
    summary,
)
from jobs import DEAD, DONE, JobQueue

BRAND = "Acme"
SAMPLE_POS2 = "Here are my picks:\n\n1. Other Brand\n2. Acme\n3. Third Brand"
SAMPLE_POS1 = "Top picks:\n\n1. Acme\n2. Other Brand"
SAMPLE_UNRANKED = "You might also look at Acme, though it is not a common recommendation."
SAMPLE_ABSENT = "The two most recommended options are Other Brand and Third Brand."


def ans(engine="e1", question="q1", sample=0, text="", error=None,
        citations=(), answer_id=None, model="m1") -> Answer:
    return Answer(engine=engine, question=question, sample=sample, text=text,
                  error=error, citations=list(citations), model=model, id=answer_id)


@pytest.fixture()
def store(tmp_path) -> AnswerStore:
    return AnswerStore(tmp_path / "answers.db")


@pytest.fixture()
def queue(tmp_path) -> JobQueue:
    return JobQueue(tmp_path / "jobs.db")


@pytest.fixture()
def demo_engines() -> list[ReplayEngine]:
    engines = load_recordings()
    assert engines, "engines/recordings/*.jsonl 录制文件缺失"
    return engines


# ============================================================ 契约层

def test_safe_ask_turns_a_raise_into_a_failed_answer():
    """采集失败必须变成**数据**，不能变成异常 —— 异常会让整批任务中断。"""

    class Boom:
        name = "boom"

        def ask(self, question, sample=0):
            raise EngineError(FailureKind.RATE_LIMIT, "429")

    got = safe_ask(Boom(), "q1", 0)
    assert got.error is FailureKind.RATE_LIMIT
    assert got.error_detail == "429"
    assert not got.ok


def test_safe_ask_maps_unknown_exceptions_to_unknown():
    class Boom:
        name = "boom"

        def ask(self, question, sample=0):
            raise ValueError("谁知道呢")

    got = safe_ask(Boom(), "q1", 0)
    assert got.error is FailureKind.UNKNOWN
    assert "ValueError" in got.error_detail


def test_extract_citations_dedupes_and_keeps_first_seen_order():
    text = "see https://a.com/x, then https://b.com/y. again https://a.com/x"
    assert extract_citations(text) == ["https://a.com/x", "https://b.com/y"]


def test_answer_round_trips_through_json_without_losing_the_failure_kind():
    a = ans(text="t", error=FailureKind.PARSE, citations=["u"], answer_id=7)
    back = Answer.from_dict(a.to_dict(), id=7)
    assert back.error is FailureKind.PARSE
    assert back.citations == ["u"]
    assert back.id == 7
    assert "id" not in a.to_dict(), "id 由库生成，不该混进待落库的字典"


# ============================================================ 口径一 / 二：失败的归属

def test_failures_are_excluded_from_the_denominator():
    """口径一：10 个采样里 2 个失败 → 分母是 8，不是 10。"""
    answers = (
        [ans(question="q1", sample=i, text=SAMPLE_POS1) for i in range(4)]
        + [ans(question="q2", sample=i, text=SAMPLE_ABSENT) for i in range(4)]
        + [ans(question="q3", sample=0, error=FailureKind.TIMEOUT),
           ans(question="q3", sample=1, error=FailureKind.RATE_LIMIT)]
    )
    m = compute(answers, BRAND, planned={"e1": 10})[0]

    assert m.ok == 8 and m.failed == 2
    assert m.mentions == 4
    assert m.mention_rate == pytest.approx(4 / 8), "分母用了计划数而不是有效样本数"
    assert m.coverage == pytest.approx(0.8)
    assert m.failures == {"timeout": 1, "rate_limit": 1}


def test_a_failed_sample_is_never_counted_as_not_mentioned():
    """口径二：失败 ≠ 没提到。两种记法算出来的提及率必须不同，且失败那种更高。"""
    ok_mentions = [ans(question="q1", sample=i, text=SAMPLE_POS1) for i in range(4)]
    ok_absent = [ans(question="q2", sample=i, text=SAMPLE_ABSENT) for i in range(4)]
    bad = [ans(question="q3", sample=i, error=FailureKind.TIMEOUT) for i in range(2)]

    correct = compute(ok_mentions + ok_absent + bad, BRAND, planned={"e1": 10})[0]
    wrong = 4 / 10  # 把失败当成「未提及」的错误算法

    assert correct.mention_rate == pytest.approx(0.5)
    assert correct.mention_rate > wrong, "剔除失败后提及率应当**更高**，否则说明失败被计成了未提及"


# ============================================================ 口径三：提到 ≠ 被推荐

def test_mention_position_is_extracted_from_numbered_lists():
    assert find_mention(SAMPLE_POS2, BRAND).position == 2
    assert find_mention(SAMPLE_POS1, BRAND).position == 1


def test_bulleted_lists_use_running_order_as_position():
    text = "Recommendations:\n\n- Other Brand\n- Acme\n- Third Brand"
    assert find_mention(text, BRAND).position == 2


def test_chinese_style_numbering_is_supported():
    """中文写法「1、xxx」「1）xxx」分隔符后面**没有空格**，正则不能用 \\s+。"""
    assert find_mention("推荐：\n1、别的牌子\n2、Acme", BRAND).position == 2
    assert find_mention("推荐：\n1）别的牌子\n2）Acme", BRAND).position == 2


def test_a_decimal_number_is_not_mistaken_for_a_list_item():
    """「1.5 kg」不是列表第 1 项 —— 否则位次会被凭空造出来。"""
    assert find_mention("Weight is 1.5 kg. Acme is fine.", BRAND).position is None


def test_mention_outside_a_list_counts_for_mention_but_not_for_rank():
    """口径三：顺带提及进提及率，但**不进**进榜率。"""
    answers = [
        ans(question="q1", sample=0, text=SAMPLE_UNRANKED),
        ans(question="q2", sample=0, text=SAMPLE_ABSENT),
    ]
    m = compute(answers, BRAND)[0]

    assert m.mentions == 1 and m.ranked == 0
    assert m.mention_rate == pytest.approx(0.5)
    assert m.rank_rate == pytest.approx(0.0), "没进推荐列表却算成了进榜"
    assert m.avg_position is None


def test_aliases_are_matched_too():
    answers = [ans(question="q1", sample=0, text="1. Acme Coffee Co\n2. Other")]
    assert compute(answers, "Acme", aliases=["Acme Coffee"])[0].mentions == 1


def test_word_boundary_is_not_required_but_case_is_ignored():
    assert find_mention("1. ACME\n2. Other", "acme") is not None


# ============================================================ 口径四：不稳定性

def test_a_question_with_inconsistent_samples_is_flagged():
    """口径四：同一个问题问 3 遍，2 遍提到 1 遍没提到 → 这个比率不可当结论。"""
    answers = [
        ans(question="q1", sample=0, text=SAMPLE_POS1),
        ans(question="q1", sample=1, text=SAMPLE_POS1),
        ans(question="q1", sample=2, text=SAMPLE_ABSENT),
    ]
    m = compute(answers, BRAND)[0]
    assert m.unstable_questions == ["q1"]

    stable = [ans(question="q1", sample=i, text=SAMPLE_POS1) for i in range(3)]
    assert compute(stable, BRAND)[0].unstable_questions == []


# ============================================================ 可溯源

def test_evidence_ids_point_back_to_the_raw_answer(store):
    """指标必须能回到原文 —— 这是「提及率 64.8%」从断言变成结论的唯一途径。"""
    store.save_many("r1", [
        ans(engine="alpha", question="q1", sample=0, text=SAMPLE_POS1),
        ans(engine="alpha", question="q1", sample=1, text=SAMPLE_ABSENT),
    ])
    loaded = store.load("r1")
    m = compute(loaded, BRAND, planned={"alpha": 2})[0]

    assert len(m.evidence_ids) == 1
    raw = store.get(m.evidence_ids[0])
    assert raw is not None
    assert BRAND in raw.text, "evidence_id 指向的原文里应当真的有这个品牌"
    assert raw.sample == 0


def test_summary_aggregates_across_engines_on_valid_samples_only():
    answers = [
        ans(engine="a", question="q1", sample=0, text=SAMPLE_POS1),
        ans(engine="a", question="q1", sample=1, text=SAMPLE_ABSENT),
        ans(engine="b", question="q1", sample=0, text=SAMPLE_POS1),
        ans(engine="b", question="q1", sample=1, error=FailureKind.BLOCKED),
    ]
    s = summary(compute(answers, BRAND, planned={"a": 2, "b": 2}), BRAND)
    assert s["ok"] == 3 and s["failed"] == 1
    assert s["mention_rate"] == pytest.approx(2 / 3)
    assert s["coverage"] == pytest.approx(0.75)
    assert s["failures"] == {"blocked": 1}


# ============================================================ 可比性

def test_model_drift_makes_two_runs_incomparable():
    before = compute([ans(engine="a", text=SAMPLE_POS1, model="v1")], BRAND, planned={"a": 1})
    after = compute([ans(engine="a", text=SAMPLE_POS1, model="v2")], BRAND, planned={"a": 1})
    warnings = check_comparability(before, after)
    assert any("模型变了" in w for w in warnings), "换了模型还直接比增长率"


def test_low_coverage_is_flagged_as_untrustworthy():
    answers = [ans(engine="a", question=f"q{i}", sample=0, text=SAMPLE_ABSENT) for i in range(5)]
    answers.append(ans(engine="a", question="q9", sample=0, error=FailureKind.TIMEOUT))
    m = compute(answers, BRAND, planned={"a": 20})
    assert any("覆盖率" in w for w in check_comparability(m, m))


def test_format_report_shows_coverage_before_the_rates():
    answers = [ans(engine="a", question="q1", sample=0, text=SAMPLE_POS1),
               ans(engine="a", question="q1", sample=1, error=FailureKind.PARSE)]
    text = format_report(compute(answers, BRAND, planned={"a": 2}), BRAND)
    assert "覆盖率" in text
    assert "parse" in text, "失败分布必须报出来，否则看不出是基建还是品牌的问题"
    assert text.index("覆盖率") < text.index("提及率")


# ============================================================ 存储

def test_store_read_methods_work_on_a_fresh_db(store):
    """回归：读方法自己不建表的话，第一次就读会 `no such table`。"""
    assert store.load("nope") == []
    assert store.get(1) is None
    assert store.runs() == []


def test_store_is_idempotent_per_sample(store):
    """重跑同一个 run 不能重复落库 —— 否则分母会被自己灌水。"""
    rows = [ans(question="q1", sample=0, text=SAMPLE_POS1),
            ans(question="q1", sample=1, text=SAMPLE_ABSENT)]
    assert store.save_many("r1", rows) == 2
    store.save_many("r1", rows)          # 再存一遍
    assert len(store.load("r1")) == 2, "同一样本被写入了两次"


def test_store_keeps_the_raw_text_verbatim(store):
    weird = "  line1\n\nline2  \t trailing  "
    store.save_many("r1", [ans(question="q1", sample=0, text=weird)])
    assert store.load("r1")[0].text == weird


def test_store_runs_lists_batches_with_failure_counts(store):
    store.save_many("r1", [ans(question="q1", sample=0, text="x"),
                           ans(question="q1", sample=1, error=FailureKind.PARSE)])
    run = store.runs()[0]
    assert (run["run_id"], run["total"], run["ok"], run["failed"]) == ("r1", 2, 1, 1)


# ============================================================ 编排

def test_plan_tasks_expands_every_engine_question_and_sample(demo_engines):
    questions = sorted({q for e in demo_engines for q in e.questions})
    tasks = plan_tasks(demo_engines, questions, samples=3)
    assert len(tasks) == len(demo_engines) * len(questions) * 3
    assert len({k for k, _ in tasks}) == len(tasks), "任务 key 有重复，队列会互相顶掉"


def test_default_run_id_is_stable_and_changes_with_the_spec(demo_engines):
    qs = ["q1", "q2"]
    a = default_run_id(demo_engines, qs, 3)
    assert a == default_run_id(demo_engines, qs, 3), "同样的输入应当得到同样的 run_id"
    assert a != default_run_id(demo_engines, qs, 5)
    assert a != default_run_id(demo_engines, qs[:1], 3)


class _Counting:
    """统计真实调用次数，用来证明第二轮没有重采。"""

    def __init__(self, inner):
        self._inner, self.name, self.calls = inner, inner.name, 0

    def ask(self, question, sample=0):
        self.calls += 1
        return self._inner.ask(question, sample)


def test_second_run_with_same_id_recollects_nothing(demo_engines, queue, store):
    """靠队列 + 落库双重幂等：重跑不重采，不重花钱。"""
    questions = sorted({q for e in demo_engines for q in e.questions})

    first = collect(demo_engines, questions, samples=3, queue=queue, store=store)
    assert first["stats"][DONE] == 12
    assert len(first["answers"]) == 12

    counted = [_Counting(e) for e in demo_engines]
    again = collect(counted, questions, samples=3, run_id=first["run_id"],
                    queue=queue, store=store)

    assert sum(e.calls for e in counted) == 0, "第二轮又调了引擎 —— 白花钱"
    assert len(again["answers"]) == 12
    assert len(store.runs()) == 1


def test_a_dead_job_becomes_a_failed_answer_not_a_silent_disappearance(
        demo_engines, queue, store):
    """队列里死掉的任务必须补一条失败记录 —— 否则那条采样凭空消失，
    既不在分子也不在分母，指标会悄悄变好看。"""
    questions = ["q1"]
    run_id = "geo-dead"
    tasks = plan_tasks(demo_engines, questions, samples=1)

    # 人为制造一个 dead 任务：max_attempts=1，失败一次就落 dead
    queue.enqueue(run_id, tasks[0][0], tasks[0][1], max_attempts=1)
    job = queue.claim("w1", run_id=run_id)
    assert queue.fail(job["id"], "凑个死人") == DEAD

    out = collect(demo_engines, questions, samples=1, run_id=run_id,
                  queue=queue, store=store)

    assert tasks[0][0] in out["dead"]
    dead_answers = [a for a in out["answers"]
                    if a.question == tasks[0][1]["question"]
                    and a.engine == tasks[0][1]["engine"]
                    and a.sample == tasks[0][1]["sample"]]
    assert len(dead_answers) == 1, "死任务的那条采样不见了"
    assert dead_answers[0].error is FailureKind.UNKNOWN

    # 而且它只能进失败计数，不能进分母
    metrics = {m.engine: m for m in compute(out["answers"], BRAND)}
    m = metrics[tasks[0][1]["engine"]]
    assert m.failed == 1
    assert m.mention_rate == 0.0 or m.ok > 0


def test_collect_isolates_runs_by_run_id(demo_engines, queue, store):
    questions = sorted({q for e in demo_engines for q in e.questions})
    collect(demo_engines, questions, samples=1, run_id="runA", queue=queue, store=store)
    collect(demo_engines, questions, samples=1, run_id="runB", queue=queue, store=store)
    assert {r["run_id"] for r in store.runs()} == {"runA", "runB"}


# ============================================================ 录制回放

def test_replay_is_deterministic(demo_engines, queue, store, tmp_path):
    """同一份录制必须跑出同一组指标 —— 否则「可复核」是句空话。"""
    questions = sorted({q for e in demo_engines for q in e.questions})
    a = collect(demo_engines, questions, samples=3, run_id="ra",
                queue=JobQueue(tmp_path / "j1.db"), store=AnswerStore(tmp_path / "a1.db"))
    b = collect(demo_engines, questions, samples=3, run_id="rb",
                queue=JobQueue(tmp_path / "j2.db"), store=AnswerStore(tmp_path / "a2.db"))

    sa = summary(compute(a["answers"], BRAND, planned={"replay-alpha": 6, "replay-beta": 6}), BRAND)
    sb = summary(compute(b["answers"], BRAND, planned={"replay-alpha": 6, "replay-beta": 6}), BRAND)
    for key in ("ok", "failed", "mentions", "mention_rate", "rank_rate", "avg_position"):
        assert sa[key] == sb[key], f"{key} 两次回放不一致"


def test_replay_reproduces_recorded_failures(demo_engines, queue, store):
    """录制里的失败要精确重现，否则「失败不进分母」这条规则测不到。"""
    questions = sorted({q for e in demo_engines for q in e.questions})
    out = collect(demo_engines, questions, samples=3, queue=queue, store=store)
    kinds = sorted(a.error.value for a in out["answers"] if a.error)
    assert kinds == ["parse", "rate_limit"]


def test_replay_raises_instead_of_guessing_when_the_recording_misses_a_sample():
    engine = ReplayEngine("r", [{"engine": "r", "question": "q1", "sample": 0, "text": "x"}])
    with pytest.raises(EngineError) as ei:
        engine.ask("q1", sample=9)
    assert ei.value.kind is FailureKind.UNKNOWN


def test_dump_jsonl_round_trips_including_failures(tmp_path):
    path = tmp_path / "rec.jsonl"
    src = [ans(engine="r", question="q1", sample=0, text="hello"),
           ans(engine="r", question="q1", sample=1, error=FailureKind.TIMEOUT)]
    assert dump_jsonl(src, path) == 2

    back = ReplayEngine.from_jsonl("r", path)
    assert back.ask("q1", 0).text == "hello"
    assert back.ask("q1", 1).error is FailureKind.TIMEOUT


def test_demo_recordings_produce_the_documented_numbers(demo_engines, queue, store):
    """演示数据的数字是写死在文档里的，改动录制必须同步改文档。"""
    questions = sorted({q for e in demo_engines for q in e.questions})
    out = collect(demo_engines, questions, samples=3, queue=queue, store=store)
    metrics = compute(out["answers"], "NovaBrew",
                      planned={"replay-alpha": 6, "replay-beta": 6})
    s = summary(metrics, "NovaBrew")

    assert (s["ok"], s["failed"]) == (10, 2)
    assert s["mention_rate"] == pytest.approx(0.7)
    assert s["rank_rate"] == pytest.approx(0.6)
    assert s["avg_position"] == pytest.approx(2.0)
    assert s["failures"] == {"parse": 1, "rate_limit": 1}

    by = {m.engine: m for m in metrics}
    assert by["replay-alpha"].coverage == pytest.approx(1.0)
    assert by["replay-beta"].coverage == pytest.approx(4 / 6)
    assert by["replay-beta"].rank_rate == pytest.approx(0.25), "顺带提及被算成了进榜"


# ============================================================ 真实引擎适配器（不联网）

def test_openai_compat_maps_sdk_errors_to_our_failure_kinds(monkeypatch):
    """把 SDK 异常翻译成我们的分类 —— 分类错了，重试策略就会跟着错。"""
    import httpx
    import openai

    engine = OpenAICompatEngine("x", "https://example.invalid", "k", "m")
    resp = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid/v1"))

    def boom(**kwargs):
        raise openai.RateLimitError("429 Too Many Requests", response=resp, body=None)

    monkeypatch.setattr(engine, "_client", lambda: type(
        "C", (), {"chat": type("Ch", (), {"completions": type(
            "Co", (), {"create": staticmethod(boom)})()})()})())

    got = safe_ask(engine, "q1")
    assert got.error is FailureKind.RATE_LIMIT


def test_empty_response_body_is_a_parse_failure_not_an_absence(monkeypatch):
    """拿到 200 但正文是空的 —— 最危险的一类，绝不能记成「品牌没被提到」。"""
    engine = OpenAICompatEngine("x", "https://example.invalid", "k", "m")

    fake = type("C", (), {"chat": type("Ch", (), {"completions": type(
        "Co", (), {"create": staticmethod(lambda **kw: type(
            "R", (), {"choices": [], "model": "m"})())})()})()})()
    monkeypatch.setattr(engine, "_client", lambda: fake)

    got = safe_ask(engine, "q1")
    assert got.error is FailureKind.PARSE


def test_from_env_reports_missing_config_as_an_auth_failure(monkeypatch):
    from engines.openai_compat import from_env

    for suffix in ("BASE_URL", "API_KEY", "MODEL"):
        monkeypatch.delenv(f"NOPE_{suffix}", raising=False)
    with pytest.raises(EngineError) as ei:
        from_env("NOPE")
    assert ei.value.kind is FailureKind.AUTH
