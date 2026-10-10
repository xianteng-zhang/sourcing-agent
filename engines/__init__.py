"""多引擎答案采集 —— 把「同一批问题拿去问 N 个 AI」做成可重试、可归因、可复现的事。

分层，每层只干一件事：

    base.py           契约：Answer / 失败分类 / Engine 协议
    store.py          原文留存：指标能回到产生它的那条答案
    replay.py         录制回放：测试与复核不碰网络
    openai_compat.py  真实引擎：一个类覆盖所有 OpenAI 兼容接口
    collect.py        编排：多引擎 × 多问题 × N 采样，跑在已有的租约队列上
    metrics.py        口径：提及率 / 进榜率 / 覆盖率，以及它们什么时候不该被相信

快速上手：

    from engines import ReplayEngine, collect, compute, format_report

    engines = [ReplayEngine.from_jsonl("alpha", "engines/recordings/replay-alpha.jsonl")]
    out = collect(engines, ["best portable espresso maker?"], samples=3)
    print(format_report(compute(out["answers"], "NovaBrew"), "NovaBrew"))
"""
from engines.base import (
    RETRYABLE,
    Answer,
    Engine,
    EngineError,
    FailureKind,
    extract_citations,
    safe_ask,
)
from engines.collect import analyze, collect, default_run_id, plan_tasks
from engines.metrics import (
    EngineMetrics,
    Mention,
    QuestionMetrics,
    check_comparability,
    compute,
    find_mention,
    format_report,
    summary,
)
from engines.openai_compat import OpenAICompatEngine, from_env
from engines.replay import ReplayEngine, dump_jsonl, load_recordings
from engines.store import AnswerStore

__all__ = [
    "RETRYABLE", "Answer", "AnswerStore", "Engine", "EngineError", "EngineMetrics",
    "FailureKind", "Mention", "OpenAICompatEngine", "QuestionMetrics", "ReplayEngine",
    "analyze", "check_comparability", "collect", "compute", "default_run_id",
    "dump_jsonl", "extract_citations", "find_mention", "format_report", "from_env",
    "load_recordings", "plan_tasks", "safe_ask", "summary",
]
