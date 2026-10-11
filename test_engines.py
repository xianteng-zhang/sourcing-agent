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
    discover_prefixes,
    dump_jsonl,
    extract_citations,
    find_mention,
    format_occupancy,
    format_report,
    from_env,
    known_questions,
    list_items,
    load_recordings,
    occupancy,
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
    # 退避压到 0.01 秒:真实值会让每个涉及重试的用例真等好几秒
    return JobQueue(tmp_path / "jobs.db", backoff_base=0.01, backoff_cap=0.02)


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
    """靠队列 + 落库双重幂等：重跑不重采，不重花钱。

    注意 `DONE=11 / DEAD=1`:录制里那条 429 是**可重试**的失败,现在会被真的
    重试(重试上限次),用尽后落 dead。这正是想要的 —— 以前它被 safe_ask 吞成
    一条失败 Answer,队列看到的是「成功」,重试机制形同虚设。
    """
    questions = sorted({q for e in demo_engines for q in e.questions})

    first = collect(demo_engines, questions, samples=3, queue=queue, store=store)
    assert first["stats"][DONE] == 11
    assert first["stats"][DEAD] == 1, "可重试的 429 应当被真的重试到落 dead"
    assert len(first["answers"]) == 12, "落 dead 的那条也要补成失败记录,不能消失"

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
    metrics = compute(out["answers"], "北辰",
                      planned={"replay-alpha": 6, "replay-beta": 6})
    s = summary(metrics, "北辰")

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


# ============================================================ 类目占位分析
#
# metrics 回答「我这个品牌表现如何」，occupancy 回答「这个类目的推荐位被谁占了」。
# 后者才是选品能用的那个，所以它的口径也要一样硬。

def test_list_items_extracts_position_and_text():
    """编号项用自己写的号，项目符号按出现顺序数 —— 位次口径只有这一处。"""
    items = list_items("介绍:\n\n1. A — 说明\n2、B\n- C\n\n尾注")
    assert items == [(1, "A — 说明"), (2, "B"), (3, "C")]


def test_occupancy_maps_each_position_to_a_brand():
    answers = [ans(text="Picks:\n\n1. Alpha\n2. Beta\n3. Gamma")]
    q = occupancy(answers, ["Alpha", "Beta", "Gamma"]).by_question[0]

    assert q.slot(1).holders == {"Alpha": 1}
    assert q.slot(2).holders == {"Beta": 1}
    assert q.slot(3).holders == {"Gamma": 1}
    assert q.samples == 1


def test_occupancy_surfaces_brands_you_did_not_put_on_the_list():
    """最有价值的一条:你只能列出你已经知道的对手,AI 会告诉你漏掉的那些。"""
    answers = [ans(text="1. Alpha\n2. **Stranger** — 没听过这个")]
    r = occupancy(answers, ["Alpha"])

    assert r.by_question[0].slot(2).holders == {}
    assert r.by_question[0].slot(2).others == {"Stranger": 1}
    assert dict(r.unknown) == {"Stranger": 1}


def test_a_prose_mention_counts_as_mentioned_but_not_ranked():
    answers = [ans(text="Alpha is decent, though I would not put it on a list.")]
    share = occupancy(answers, ["Alpha"]).brands[0]

    assert (share.mentioned, share.ranked) == (1, 0)
    assert share.avg_position is None


def test_occupancy_skips_failed_samples():
    """和 metrics 同一条口径:引擎超时不等于「这个位次不存在」。"""
    answers = [
        ans(sample=0, text="1. Alpha", error=FailureKind.TIMEOUT),
        ans(sample=1, text="1. Alpha"),
    ]
    r = occupancy(answers, ["Alpha"])

    assert r.samples == 1, "失败样本混进了占位统计"
    assert r.brands[0].ranked == 1


def test_a_longer_brand_name_is_not_stolen_by_its_prefix():
    """先匹配长的:否则「Wacaco」会把「Wacaco Nanopresso」抢走,一个品牌被拆成两半。"""
    answers = [ans(text="1. Wacaco Nanopresso")]
    by = {s.brand: s for s in occupancy(answers, ["Wacaco", "Wacaco Nanopresso"]).brands}

    assert by["Wacaco Nanopresso"].ranked == 1
    assert by["Wacaco"].ranked == 0


def test_an_unstable_head_slot_is_flagged():
    swapped = [ans(sample=0, text="1. Alpha\n2. Beta"),
               ans(sample=1, text="1. Beta\n2. Alpha")]
    q = occupancy(swapped, ["Alpha", "Beta"]).by_question[0]
    assert q.unstable_head is True
    assert q.head.holders == {"Alpha": 1, "Beta": 1}

    steady = [ans(sample=i, text="1. Alpha\n2. Beta") for i in range(3)]
    assert occupancy(steady, ["Alpha", "Beta"]).by_question[0].unstable_head is False


def test_an_unrecognised_brand_at_slot_one_also_makes_the_head_unstable():
    """没认出来的对手占过第 1 位,头部位次同样是易手的 —— 不能因为认不出就当它不存在。"""
    answers = [ans(sample=0, text="1. Alpha"), ans(sample=1, text="1. Stranger")]
    assert occupancy(answers, ["Alpha"]).by_question[0].unstable_head is True


def test_a_brand_listed_twice_in_one_answer_is_ranked_once():
    answers = [ans(text="1. Alpha — 首选\n2. Beta\n3. Alpha — 备选")]
    r = occupancy(answers, ["Alpha", "Beta"])
    alpha = {s.brand: s for s in r.brands}["Alpha"]

    assert alpha.ranked == 1, "同一条答案里出现两次,进榜只能算一次"
    assert alpha.positions == [1]
    assert r.by_question[0].slot(3).holders == {"Alpha": 1}, "但占位表两个位次都要算"


def test_occupancy_on_the_demo_recordings(demo_engines, queue, store):
    """演示数据的数字写在文档里,改录制必须同步改文档。"""
    questions = sorted({q for e in demo_engines for q in e.questions})
    out = collect(demo_engines, questions, samples=3, queue=queue, store=store)

    rivals = ["北辰", "Wacaco Nanopresso", "Staresso MiNi"]
    r = occupancy(out["answers"], rivals)

    assert r.samples == 10
    by = {s.brand: s for s in r.brands}
    assert (by["Wacaco Nanopresso"].mentioned, by["Wacaco Nanopresso"].ranked) == (10, 6)
    assert (by["北辰"].mentioned, by["北辰"].ranked, by["北辰"].top1) == (7, 6, 2)
    assert by["Staresso MiNi"].ranked == 1
    assert dict(r.unknown) == {"AeroPress Go": 3, "Outin Nano": 2}

    # 两个问题的头部位次都在样本间易手 —— 演示数据本身就该是个「有缝」的类目
    assert len(r.unstable_questions) == 2

    text = format_occupancy(r, rivals)
    assert "AeroPress Go" in text, "漏看的对手必须出现在报告里"
    assert "没有稳定偏好" in text
    assert "?Outin Nano" in text, "没认出来的项在占位表里要带 ? 标出来"


# ============================================================ 真实引擎的配置发现

def test_discover_prefixes_needs_all_three_vars(monkeypatch):
    """只配了 key 不算数。

    环境里带 `_API_KEY` 的变量太多了(各种 SDK、云服务),只认 key 会误判一堆
    出来。所以要求 BASE_URL / API_KEY / MODEL 三项齐全。
    """
    monkeypatch.setenv("AAA_BASE_URL", "https://a.example")
    monkeypatch.setenv("AAA_API_KEY", "sk-a")
    monkeypatch.setenv("AAA_MODEL", "m-a")

    monkeypatch.setenv("BBB_API_KEY", "sk-b")          # 只配了 key
    for k in ("BBB_BASE_URL", "BBB_MODEL"):
        monkeypatch.delenv(k, raising=False)

    monkeypatch.setenv("CCC_BASE_URL", "https://c.example")
    monkeypatch.setenv("CCC_API_KEY", "")
    monkeypatch.setenv("CCC_MODEL", "m-c")             # key 是空串

    found = discover_prefixes()
    assert "AAA" in found
    assert "BBB" not in found, "只配了 key 的前缀被误认成引擎"
    assert "CCC" not in found, "空字符串的配置被当成了有效配置"


def test_a_discovered_prefix_can_actually_build_an_engine(monkeypatch):
    """自动发现的名单必须真的能建出引擎 —— 否则「配了就可用」是句空话。"""
    monkeypatch.setenv("ZZZ_BASE_URL", "https://z.example/v1")
    monkeypatch.setenv("ZZZ_API_KEY", "sk-z")
    monkeypatch.setenv("ZZZ_MODEL", "z-model")

    assert "ZZZ" in discover_prefixes()
    engine = from_env("ZZZ")
    assert engine.name == "zzz"
    assert engine.base_url == "https://z.example/v1"
    assert engine.model == "z-model"


def test_known_questions_tolerates_engines_that_have_no_questions():
    """回归。`.questions` 只有录制回放有,真实引擎没有。

    这里漏了兜底的话,界面一切到「真实引擎」就 AttributeError 把整个 app
    打崩 —— 而**只测回放模式是永远发现不了的**。
    """
    class Real:
        name = "real"

        def ask(self, question, sample=0):  # pragma: no cover - 本用例不调用
            raise NotImplementedError

    replay = ReplayEngine("r", [
        {"engine": "r", "question": "q2", "sample": 0, "text": "x"},
        {"engine": "r", "question": "q1", "sample": 0, "text": "y"},
    ])

    assert known_questions([Real()]) == [], "真实引擎不该被假设有 .questions"
    assert known_questions([replay, Real()]) == ["q1", "q2"], "应当取并集并排序"
    assert known_questions([]) == []


def test_the_app_survives_switching_to_the_real_engine_mode(demo_engines):
    """端到端回归:切到真实引擎模式不能崩。

    用 AppTest 真的把界面跑一遍 —— 纯函数测试全绿也不代表分支切过去没事,
    上一个 bug 就是这么漏掉的。
    """
    import os

    from streamlit.testing.v1 import AppTest

    # 造一个「配齐了」的假引擎前缀,让自动发现能认出它(不会真的发请求)
    for k, v in (("FAKE_BASE_URL", "https://fake.example/v1"),
                 ("FAKE_API_KEY", "sk-fake"), ("FAKE_MODEL", "fake-1")):
        os.environ[k] = v
    try:
        at = AppTest.from_file("app.py", default_timeout=240)
        at.run()
        at.radio(key="geo_mode").set_value("🌐 真实引擎").run()
        assert not at.exception, [e.value for e in at.exception]
    finally:
        for k in ("FAKE_BASE_URL", "FAKE_API_KEY", "FAKE_MODEL"):
            os.environ.pop(k, None)


# ============================================================ temperature 的坑

def _recording_client(sent: dict, text: str = "OK"):
    """假 client:把 create() 收到的 kwargs 记下来,返回一个正常响应。

    注意属性是**在类外挂上去的**:类体不参与闭包,`class X: create = create`
    会 `NameError` —— 类体里的名字只在类命名空间和全局里找,看不到外层的
    函数局部变量。
    """
    message = type("M", (), {"content": text})()
    resp = type("R", (), {"choices": [type("C", (), {"message": message})()],
                          "model": "m", "citations": None})()

    def create(**kwargs):
        sent.update(kwargs)
        return resp

    class _Completions:
        pass

    _Completions.create = staticmethod(create)

    class _Chat:
        pass

    _Chat.completions = _Completions()

    class _Client:
        pass

    _Client.chat = _Chat()
    return _Client()


def test_temperature_is_not_sent_unless_explicitly_set(monkeypatch):
    """回归。真实跑第二个厂商时才炸出来的:

        Kimi 400: invalid temperature: only 1 is allowed for this model

    `kimi-k2.6` 是思考模型,只接受 temperature=1。我原先写死 0.7,
    DeepSeek 收下了,Kimi 直接拒绝 —— **写死任何值都迟早会在某家上炸**。
    所以默认不传,让各家用自己的默认值。
    """
    sent: dict = {}
    engine = OpenAICompatEngine("x", "https://e.invalid", "k", "m")
    monkeypatch.setattr(engine, "_client", lambda: _recording_client(sent))

    got = safe_ask(engine, "q1")
    assert got.ok
    assert "temperature" not in sent, "默认不该把 temperature 发出去"


def test_an_explicit_temperature_is_sent_as_is(monkeypatch):
    sent: dict = {}
    engine = OpenAICompatEngine("x", "https://e.invalid", "k", "m", temperature=1.0)
    monkeypatch.setattr(engine, "_client", lambda: _recording_client(sent))

    safe_ask(engine, "q1")
    assert sent["temperature"] == 1.0


def test_temperature_can_be_configured_per_engine(monkeypatch):
    monkeypatch.setenv("TT_BASE_URL", "https://t.invalid/v1")
    monkeypatch.setenv("TT_API_KEY", "sk-t")
    monkeypatch.setenv("TT_MODEL", "m")

    monkeypatch.delenv("TT_TEMPERATURE", raising=False)
    assert from_env("TT").temperature is None

    monkeypatch.setenv("TT_TEMPERATURE", "1")
    assert from_env("TT").temperature == 1.0

    monkeypatch.setenv("TT_TEMPERATURE", "不是数字")
    with pytest.raises(EngineError) as ei:
        from_env("TT")
    assert "不是数字" in ei.value.detail


# ============================================================ 可重试的失败要真的被重试
#
# 这是**真实跑 Kimi 才暴露**的一组问题:429 被 safe_ask 吞成一条失败 Answer,
# 队列看到的是「成功」—— 状态显示 done:8/dead:0,退避和重试机制一次都没触发过。

class _Flaky:
    """前 `fail_times` 次抛 429,之后就正常。用来证明重试真的发生了。"""

    def __init__(self, name: str, fail_times: int) -> None:
        self.name = name
        self.fail_times = fail_times
        self.calls = 0

    def ask(self, question: str, sample: int = 0) -> Answer:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise EngineError(FailureKind.RATE_LIMIT, "429 Too Many Requests")
        return Answer(engine=self.name, question=question, sample=sample, text="1. Acme")


def test_a_retryable_failure_is_actually_retried(queue, store):
    """回归。原先 429 被 `safe_ask` 吞成数据,队列以为成功了,重试形同虚设。"""
    engine = _Flaky("flaky", fail_times=2)

    out = collect([engine], ["q1"], samples=1, queue=queue, store=store)

    assert engine.calls == 3, f"429 该被真的重试到第 3 次,实际只调了 {engine.calls} 次"
    assert out["stats"][DONE] == 1
    assert out["stats"][DEAD] == 0
    assert out["answers"][0].ok, "重试成功后应当拿到一条正常答案"


def test_retries_exhausted_land_in_dead_and_still_leave_a_record(queue, store):
    """重试用尽必须落 dead,而且**仍要留一条失败记录** —— 不能凭空消失。"""
    engine = _Flaky("flaky", fail_times=99)

    out = collect([engine], ["q1"], samples=1, queue=queue, store=store,
                  max_attempts=2)

    assert engine.calls == 2, "重试上限是 2,就只该调 2 次"
    assert out["stats"][DEAD] == 1
    assert len(out["answers"]) == 1
    assert out["answers"][0].error is FailureKind.RATE_LIMIT, \
        "失败分类要能从队列留下的错误串里还原,不能一律记成 unknown"


def test_a_non_retryable_failure_is_not_retried(queue, store):
    """解析失败重试也没用 —— 它该直接落成数据,不该浪费重试额度。"""

    class _ParseFail:
        name = "parsefail"

        def __init__(self):
            self.calls = 0

        def ask(self, question, sample=0):
            self.calls += 1
            raise EngineError(FailureKind.PARSE, "响应正文为空")

    engine = _ParseFail()
    out = collect([engine], ["q1"], samples=1, queue=queue, store=store)

    assert engine.calls == 1, "不可重试的失败被重试了,白烧额度"
    assert out["stats"][DONE] == 1
    assert out["stats"][DEAD] == 0
    assert out["answers"][0].error is FailureKind.PARSE
