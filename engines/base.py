"""多引擎答案采集的契约层。

**要解决的问题**：同一个品牌问题，拿去问 10 个 AI（ChatGPT / Gemini / Perplexity /
DeepSeek / 豆包 / ...），把它们的回答收回来。10 个引擎的接口、鉴权、返回结构、
限流策略各不相同，但对上层只应该暴露一件事：

    ask(question, sample) -> Answer

这里定义的就是那个 Answer，和那条调用线。

**三个必须先想清楚的点**（后面所有指标都建立在这上面）：

1. **失败 ≠ 没有提及。** 「引擎超时 8 次、成功 2 次、这 2 次没提到品牌」和
   「10 次全成功、都没提到品牌」是**完全不同**的两件事：前者说明采集基建有问题，
   后者说明品牌在 AI 眼里真的不存在。混在一起算，指标会系统性偏低，而且不可归因。
   所以 Answer 必须带 `error`，指标的分母只能是**有效样本**。

2. **同一个问题要问 N 遍。** AI 回答是随机采样，一次结果没有意义。每个
   (引擎, 问题) 采 `samples` 次，每次是一个独立样本，`sample` 是它的序号。

3. **原始答案一字不改地留着。** 任何指标都要能回到产生它的原文 —— 否则
   「提及率 64.8%」就是一句无法核对的断言。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class FailureKind(str, Enum):
    """采集失败的原因分类。

    分类不是为了好看，是因为**后续处理完全不同**：

        TIMEOUT / RATE_LIMIT / UNKNOWN  -> 可重试，交给任务队列的 max_attempts
        AUTH                            -> 重试没用，得换 key
        BLOCKED                         -> 重试会更糟，要退避 + 换出口
        PARSE                           -> 最危险的一种：响应拿到了，但没解析出内容。
                                           它**不是**「品牌没被提及」。必须单独计数，
                                           否则会静默污染提及率的分母。
    """

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    BLOCKED = "blocked"
    PARSE = "parse"
    UNKNOWN = "unknown"


#: 值得重试的分类。AUTH / BLOCKED / PARSE 重试只会浪费额度。
RETRYABLE = frozenset({FailureKind.TIMEOUT, FailureKind.RATE_LIMIT, FailureKind.UNKNOWN})


class EngineError(Exception):
    """引擎调用失败。带上分类，让采集层能把它记进 Answer 而不是抛掉。"""

    def __init__(self, kind: FailureKind, detail: str = "") -> None:
        super().__init__(f"{kind.value}: {detail}" if detail else kind.value)
        self.kind = kind
        self.detail = detail


@dataclass
class Answer:
    """一个引擎对一个问题的一次回答（或一次失败的记录）。

    `id` 由 AnswerStore 落库时分配，是「指标 → 原文」这条溯源链的外键。
    """

    engine: str
    question: str
    sample: int = 0
    text: str = ""
    citations: list[str] = field(default_factory=list)
    model: str = ""
    latency_ms: int = 0
    error: FailureKind | None = None
    error_detail: str = ""
    asked_at: str = ""
    id: int | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """落库 / 进队列用的纯 JSON 形态。`id` 不进 —— 它由库生成。"""
        return {
            "engine": self.engine,
            "question": self.question,
            "sample": self.sample,
            "text": self.text,
            "citations": list(self.citations),
            "model": self.model,
            "latency_ms": self.latency_ms,
            "error": self.error.value if self.error else None,
            "error_detail": self.error_detail,
            "asked_at": self.asked_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], **extra: Any) -> "Answer":
        err = d.get("error")
        return cls(
            engine=d["engine"],
            question=d["question"],
            sample=int(d.get("sample", 0)),
            text=d.get("text") or "",
            citations=list(d.get("citations") or []),
            model=d.get("model") or "",
            latency_ms=int(d.get("latency_ms") or 0),
            error=FailureKind(err) if err else None,
            error_detail=d.get("error_detail") or "",
            asked_at=d.get("asked_at") or "",
            **extra,
        )


@runtime_checkable
class Engine(Protocol):
    """一个 AI 引擎。实现这个协议就能被 collect() 采集。"""

    name: str

    def ask(self, question: str, sample: int = 0) -> Answer:
        """问一次，返回一次答案。失败请抛 EngineError，不要返回空 Answer。"""
        ...


def safe_ask(engine: Engine, question: str, sample: int = 0) -> Answer:
    """调用引擎，并把任何异常收敛成一个**失败的 Answer**。

    关键在「不抛」：单次采集失败不该中断整批任务。失败要以**数据**的形式进
    AnswerStore，才能被统计、被归因、被重试 —— 抛出去就只剩一行日志了。
    """
    try:
        return engine.ask(question, sample)
    except EngineError as exc:
        return Answer(engine=engine.name, question=question, sample=sample,
                      error=exc.kind, error_detail=exc.detail)
    except Exception as exc:  # noqa: BLE001 - 未知异常也要落成数据
        return Answer(engine=engine.name, question=question, sample=sample,
                      error=FailureKind.UNKNOWN,
                      error_detail=f"{type(exc).__name__}: {exc}")


_URL_RE = re.compile(r"https?://[^\s\)\]\"'<>，。；、]+")


def extract_citations(text: str) -> list[str]:
    """从答案正文里捞 URL，作为引擎不返回 citations 时的兜底。

    去重且保持出现顺序 —— 后面的「引用源数」指标依赖这个顺序的稳定性。
    """
    seen: set[str] = set()
    out: list[str] = []
    for m in _URL_RE.finditer(text or ""):
        url = m.group(0).rstrip(".,;:、）)")
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out
