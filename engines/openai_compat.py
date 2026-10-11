"""真实引擎适配器：OpenAI 兼容接口。

**为什么一个类就够，而不是十个类：** DeepSeek、通义千问、Kimi、豆包、以及
Perplexity 都提供 OpenAI 兼容的 `/chat/completions`。差异只在
`base_url` / `model` / 鉴权头 —— 那是**配置**，不是代码。

真正需要单独写适配器的，是那些**没有稳定公开接口**的（ChatGPT 网页版、
Gemini 网页版）。那一层属于采集基建（出口、限流、反爬、账号池），不属于这里，
硬塞进来只会把这一层搞脏。

**温度：默认不传，而不是写死一个值。** 这是我踩过的坑 —— 一开始写死
`temperature=0.7`，DeepSeek 收下了，换到 Kimi 直接 400：

    invalid temperature: only 1 is allowed for this model

因为 `kimi-k2.6` 是**思考模型**，只接受 `temperature=1`。不同厂商对这个参数的
合法取值根本不一样，**写死任何一个值都迟早会在某家上炸**。所以默认 `None`
= 不传这个参数，让各家用自己的默认值；真要显式指定，用
`<PREFIX>_TEMPERATURE` 环境变量单独配。

顺带一个真话：我们**要**观察方差（同一个问题问十遍得到十个不同答案是这门
生意的核心难题）。但"要方差"不等于"能随便set温度"—— 服务端默认值通常就是
为采样设计的，比我们拍脑袋的 0.7 更合适。
"""
from __future__ import annotations

import os
import time

from engines.base import Answer, EngineError, FailureKind, extract_citations


def _classify(exc: Exception) -> FailureKind:
    """把 SDK 的异常映射成我们的失败分类。

    import 放在函数里：让这个模块在没装 openai 的环境里也能被 import
    （只有真正调用时才需要 SDK）。
    """
    import openai

    if isinstance(exc, openai.APITimeoutError):
        return FailureKind.TIMEOUT
    if isinstance(exc, openai.RateLimitError):
        return FailureKind.RATE_LIMIT
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return FailureKind.AUTH
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        if status == 429:
            return FailureKind.RATE_LIMIT
        if status in (401, 403):
            return FailureKind.BLOCKED
        return FailureKind.UNKNOWN
    return FailureKind.UNKNOWN


class OpenAICompatEngine:
    """任意 OpenAI 兼容接口。"""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        temperature: float | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        #: None = 不传 temperature,让服务端用自己的默认值。
        #: 写死任何值都迟早会在某家上炸(Kimi 的思考模型只接受 1)。
        self.temperature = temperature

    def _client(self):
        import openai  # 延迟 import：调用时才需要 SDK

        return openai.OpenAI(base_url=self.base_url, api_key=self.api_key,
                             timeout=self.timeout)

    def ask(self, question: str, sample: int = 0) -> Answer:
        started = time.perf_counter()
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": question}],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            resp = self._client().chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - 分类后抛给 safe_ask 落成数据
            raise EngineError(_classify(exc), f"{type(exc).__name__}: {exc}") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)

        choices = getattr(resp, "choices", None) or []
        if not choices:
            raise EngineError(FailureKind.PARSE, "响应里没有 choices")

        text = (getattr(choices[0].message, "content", None) or "").strip()
        if not text:
            # 拿到 200 但正文是空的 —— 最危险的一类：不能记成「没提到品牌」。
            raise EngineError(FailureKind.PARSE, "响应正文为空")

        # 引擎自己给的引用优先；没有再退回从正文里捞 URL。
        citations = list(getattr(resp, "citations", None) or []) or extract_citations(text)

        return Answer(
            engine=self.name,
            question=question,
            sample=sample,
            text=text,
            citations=citations,
            model=resp.model or self.model,
            latency_ms=latency_ms,
        )


def from_env(prefix: str, name: str | None = None, **kwargs) -> OpenAICompatEngine:
    """从环境变量建引擎：`<PREFIX>_BASE_URL` / `<PREFIX>_API_KEY` / `<PREFIX>_MODEL`。

    例如 prefix="DEEPSEEK" 会读 DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY /
    DEEPSEEK_MODEL。加一个引擎通常只是加三个环境变量，不用写代码。

    可选第四项 `<PREFIX>_TEMPERATURE`：**不填就不传这个参数**，让服务端用
    自己的默认值。填了会原样发出去 —— 注意各家合法取值不一样，比如 Kimi 的
    思考模型只接受 1，填 0.7 会直接 400。所以除非你确知这一家要多少，别填。

    **一把 key 就能跑。** 不需要「多个 key」才能用这个模块 —— 采几个引擎
    就需要几把 key，但采一个引擎完全够把整条链路(采集 → 落库 → 口径 → 占位)
    跑通。只是那样叫「多引擎」就名不副实了。
    """
    base_url = os.getenv(f"{prefix}_BASE_URL", "")
    api_key = os.getenv(f"{prefix}_API_KEY", "")
    model = os.getenv(f"{prefix}_MODEL", "")
    missing = [k for k, v in (("BASE_URL", base_url), ("API_KEY", api_key),
                              ("MODEL", model)) if not v]
    if missing:
        raise EngineError(
            FailureKind.AUTH,
            f"{prefix}_ 缺少配置: {', '.join(missing)}",
        )

    raw_temp = (os.getenv(f"{prefix}_TEMPERATURE") or "").strip()
    if raw_temp and "temperature" not in kwargs:
        try:
            kwargs["temperature"] = float(raw_temp)
        except ValueError as exc:
            raise EngineError(
                FailureKind.UNKNOWN,
                f"{prefix}_TEMPERATURE 不是数字: {raw_temp!r}",
            ) from exc

    return OpenAICompatEngine(name=name or prefix.lower(), base_url=base_url,
                              api_key=api_key, model=model, **kwargs)


#: 一个引擎要配齐这三项才算数
_SUFFIXES = ("_BASE_URL", "_API_KEY", "_MODEL")


def discover_prefixes() -> list[str]:
    """扫环境变量，找出所有**配齐了三项**的引擎前缀。

    为什么要这个：如果只认 `GEO_ENGINES` 那份手工名单，`.env` 里明明已经
    配好 DeepSeek 了，界面上按钮还是灰的，而且**不说为什么**。
    自动发现让「配了就可用」，少一处会忘记同步的地方。

    要求三项齐全而不是只认 `_API_KEY`：环境里带 `_API_KEY` 的变量太多了
    （各种 SDK、云服务），只认 key 会误判一堆出来。
    """
    candidates = set()
    for name, value in os.environ.items():
        if not (value or "").strip():
            continue
        for suffix in _SUFFIXES:
            if name.endswith(suffix):
                candidates.add(name[: -len(suffix)])
    return sorted(
        p for p in candidates
        if p and all((os.getenv(p + s) or "").strip() for s in _SUFFIXES)
    )
