"""指标与口径 —— 这个文件才是这门生意的难点。

采集不难，**把一堆自然语言的、随机的回答变成可比较的数字**才难。

## 口径一：分母是**有效样本**，不是计划样本

    10 次采样，3 次限流失败 → 分母是 7，不是 10。

把失败算成「没提到」，会让**引擎越不稳定、指标越难看** —— 那是基建问题，
不是品牌问题。所以 `mention_rate = mentions / ok`，同时把 `coverage`
（ok / planned）单独报出来。覆盖率低的时候，提及率本身就不该被相信。

## 口径二：提到 ≠ 被推荐

    「另外，XX 这类小众品牌也做类似产品」        → 提到，但没被推荐
    「推荐如下：1. A  2. B  3. XX」              → 提到，且排第 3

前者只是被顺带提及，后者才是 GEO 卖的东西。所以拆成两个率：
`mention_rate`（提及率）和 `rank_rate`（进榜率）。

## 口径三：单个问题内部可能自相矛盾

同一个问题问 5 遍，3 遍提到、2 遍没提到 —— 这个问题的答案本身就**不稳定**。
把这种问题混进总数，得到的「64.8%」是 5 个稳定问题和 3 个不稳定问题的混合物，
谁也说不清涨跌从哪来。所以按问题拆开，并显式标出 `unstable`。

## 口径四：换模型了就不是增长

`model` 字段在库里存着。如果上一轮的引擎是 `gpt-4o`、这轮变成 `gpt-5`，
那「+21.6%」里有模型的功劳，不全是内容的功劳。`check_comparability()` 专门告警。
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from engines.base import Answer

#: 有序列表项：`1. xxx` / `2、xxx` / `3）xxx` / `- xxx` / `• xxx`
#:
#: 两个容易写错的点：
#:   · **分隔符后面不一定有空格** —— 中文写法「1、别的牌子」就是紧贴的，
#:     所以不能用 `\s+`。
#:   · 要用 `(?!\d)` 排掉小数 —— 否则「1.5 kg」会被当成列表第 1 项。
_ITEM_RE = re.compile(r"^\s*(?:(\d{1,2})\s*[.、)）](?!\d)|[-*•])\s*(.+)$")


@dataclass
class Mention:
    """一次提及。`position` 为 None 表示「提到了，但不在任何列表里」。"""

    position: int | None


def _hit(haystack: str, names: list[str]) -> bool:
    low = haystack.lower()
    return any(n and n.lower() in low for n in names)


def find_mention(text: str, brand: str, aliases: list[str] | None = None) -> Mention | None:
    """在一条答案里找品牌。返回 None 表示完全没提到。

    纯字符串匹配，**不调用模型** —— 和 compliance / spec_anchor 同一条原则：
    判定规则要确定性，否则「提及率」本身就变成概率性的了。
    """
    names = [brand, *(aliases or [])]
    if not _hit(text or "", names):
        return None

    bullet_order = 0
    for line in (text or "").splitlines():
        m = _ITEM_RE.match(line)
        if not m:
            continue
        bullet_order += 1
        if _hit(m.group(2), names):
            return Mention(position=int(m.group(1)) if m.group(1) else bullet_order)
    return Mention(position=None)  # 提到了，但不在推荐列表里


@dataclass
class QuestionMetrics:
    question: str
    ok: int = 0
    mentions: int = 0
    ranked: int = 0
    positions: list[int] = field(default_factory=list)

    @property
    def mention_rate(self) -> float:
        return self.mentions / self.ok if self.ok else 0.0

    @property
    def avg_position(self) -> float | None:
        return sum(self.positions) / len(self.positions) if self.positions else None

    @property
    def unstable(self) -> bool:
        """同一个问题问多遍，结果不一致 —— 这个问题的指标不可当作结论。"""
        return 0 < self.mentions < self.ok


@dataclass
class EngineMetrics:
    engine: str
    brand: str = ""
    planned: int = 0
    ok: int = 0
    failed: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    mentions: int = 0
    ranked: int = 0
    positions: list[int] = field(default_factory=list)
    citations: int = 0
    models: list[str] = field(default_factory=list)
    evidence_ids: list[int] = field(default_factory=list)
    by_question: list[QuestionMetrics] = field(default_factory=list)

    @property
    def mention_rate(self) -> float:
        return self.mentions / self.ok if self.ok else 0.0

    @property
    def rank_rate(self) -> float:
        """进榜率：不但提到，而且进了推荐列表。GEO 真正卖的是这个。"""
        return self.ranked / self.ok if self.ok else 0.0

    @property
    def avg_position(self) -> float | None:
        return sum(self.positions) / len(self.positions) if self.positions else None

    @property
    def coverage(self) -> float:
        """有效样本 / 计划样本。明显小于 1 时，上面那些率都不该被当真。"""
        return self.ok / self.planned if self.planned else 0.0

    @property
    def unstable_questions(self) -> list[str]:
        return [q.question for q in self.by_question if q.unstable]


def compute(
    answers: list[Answer],
    brand: str,
    aliases: list[str] | None = None,
    planned: dict[str, int] | None = None,
) -> list[EngineMetrics]:
    """按引擎汇总。`planned` 是每个引擎**计划**的采样数，用于算覆盖率。"""
    planned = planned or {}
    grouped: dict[str, list[Answer]] = defaultdict(list)
    for a in answers:
        grouped[a.engine].append(a)

    out: list[EngineMetrics] = []
    for engine in sorted(grouped):
        group = grouped[engine]
        ok = [a for a in group if a.ok]
        bad = [a for a in group if not a.ok]

        m = EngineMetrics(
            engine=engine,
            brand=brand,
            planned=planned.get(engine, len(group)),
            ok=len(ok),
            failed=len(bad),
            failures=dict(Counter(a.error.value for a in bad if a.error)),
            models=sorted({a.model for a in ok if a.model}),
        )

        by_q: dict[str, QuestionMetrics] = {}
        cited: set[str] = set()
        for a in ok:
            q = by_q.setdefault(a.question, QuestionMetrics(question=a.question))
            q.ok += 1
            hit = find_mention(a.text, brand, aliases)
            if hit is not None:
                q.mentions += 1
                m.mentions += 1
                if a.id is not None:
                    m.evidence_ids.append(a.id)  # 指标 → 原文的外键
                if hit.position is not None:
                    q.ranked += 1
                    q.positions.append(hit.position)
                    m.ranked += 1
                    m.positions.append(hit.position)
            cited.update(a.citations)

        m.citations = len(cited)
        m.by_question = [by_q[q] for q in sorted(by_q)]
        out.append(m)
    return out


def summary(metrics: list[EngineMetrics], brand: str = "") -> dict:
    """跨引擎汇总。分母同样是**有效样本合计**。"""
    ok = sum(m.ok for m in metrics)
    mentions = sum(m.mentions for m in metrics)
    ranked = sum(m.ranked for m in metrics)
    positions = [p for m in metrics for p in m.positions]
    return {
        "brand": brand or (metrics[0].brand if metrics else ""),
        "engines": len(metrics),
        "planned": sum(m.planned for m in metrics),
        "ok": ok,
        "failed": sum(m.failed for m in metrics),
        "coverage": ok / sum(m.planned for m in metrics) if metrics and sum(m.planned for m in metrics) else 0.0,
        "mentions": mentions,
        "mention_rate": mentions / ok if ok else 0.0,
        "rank_rate": ranked / ok if ok else 0.0,
        "avg_position": sum(positions) / len(positions) if positions else None,
        "citations": sum(m.citations for m in metrics),
        "unstable_questions": sorted({q for m in metrics for q in m.unstable_questions}),
        "failures": dict(Counter(k for m in metrics for k, v in m.failures.items() for _ in range(v))),
    }


def check_comparability(before: list[EngineMetrics], after: list[EngineMetrics]) -> list[str]:
    """对比两轮之前，先检查它们**是否可以比**。

    这是最容易被跳过、也最容易翻车的一步：「这周 +21.6%」听起来很好，
    但如果引擎换了大模型、或者这轮覆盖率只有 40%，这个数字没有任何意义。
    """
    warnings: list[str] = []
    a = {m.engine: m for m in before}
    b = {m.engine: m for m in after}

    for engine in sorted(set(a) | set(b)):
        if engine not in a or engine not in b:
            warnings.append(f"{engine}: 只有一轮有数据，不能比")
            continue
        if a[engine].models != b[engine].models:
            warnings.append(
                f"{engine}: 引擎模型变了 {a[engine].models} → {b[engine].models}，"
                "增长里混了模型升级的功劳"
            )
        for label, m in (("上一轮", a[engine]), ("这一轮", b[engine])):
            if m.coverage < 0.8:
                warnings.append(
                    f"{engine}: {label}覆盖率只有 {m.coverage:.0%}"
                    f"（{m.ok}/{m.planned}），提及率不可信"
                )
    return warnings


def format_report(metrics: list[EngineMetrics], brand: str) -> str:
    """给人看的报告。先报覆盖率，再报率 —— 顺序本身就是口径。"""
    s = summary(metrics, brand)
    lines = [
        f"品牌：{brand}　引擎：{s['engines']} 个　有效样本：{s['ok']}/{s['planned']}"
        f"（覆盖率 {s['coverage']:.0%}）",
        "",
        f"{'引擎':<12}{'有效':>6}{'失败':>6}{'提及率':>9}{'进榜率':>9}{'平均位次':>10}{'引用源':>8}",
        "-" * 62,
    ]
    for m in metrics:
        pos = f"{m.avg_position:.1f}" if m.avg_position is not None else "—"
        lines.append(
            f"{m.engine:<12}{m.ok:>6}{m.failed:>6}"
            f"{m.mention_rate:>8.0%}{m.rank_rate:>9.0%}{pos:>10}{m.citations:>8}"
        )
    lines.append("-" * 62)
    pos = f"{s['avg_position']:.1f}" if s["avg_position"] is not None else "—"
    lines.append(
        f"{'合计':<12}{s['ok']:>6}{s['failed']:>6}"
        f"{s['mention_rate']:>8.0%}{s['rank_rate']:>9.0%}{pos:>10}{s['citations']:>8}"
    )

    if s["failures"]:
        detail = "、".join(f"{k} {v}" for k, v in sorted(s["failures"].items()))
        lines += ["", f"⚠ 失败分布：{detail}（已从分母剔除，不计为「未提及」）"]
    if s["unstable_questions"]:
        lines += ["", f"⚠ 采样不一致的问题（{len(s['unstable_questions'])} 个），其指标不可当结论："]
        lines += [f"    · {q}" for q in s["unstable_questions"]]
    return "\n".join(lines)
