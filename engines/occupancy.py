"""类目占位分析 —— 选品阶段要问的那个问题：**这个类目的 AI 推荐位被谁占了？**

和 `metrics.py` 的分工：

    metrics.py    回答「**我这个品牌**在 AI 答案里表现如何」
    occupancy.py  回答「**这个类目的推荐位**被谁占了、有没有空位」

**后者才是选品能用的那个。** 场景：

    你要决定「便携咖啡机」这个类目做不做。
    评论挖掘告诉你现有商品的痛点，趋势分析告诉你搜索热度，
    但没人回答：买家直接问 AI 的时候，AI 推的是谁？

    · #1 位永远是同一个大牌        → AI 入口锁死，别进
    · 头部位次在不同样本间换来换去  → AI 对这个类目没有稳定偏好，有机会
    · 冒出你没列过的品牌            → 那些是你漏看的对手

最后一条尤其有用：**你只能列出你已经知道的竞品，但 AI 会告诉你你不知道的那些。**

和 compliance / spec_anchor 同一条原则：**这里是纯字符串匹配，不调用模型。**
判定规则一旦交给模型，「占位」也就变成概率性的了。
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from engines.base import Answer
from engines.metrics import list_items

#: 把一条推荐项收拾成品牌名：`**Wacaco Nanopresso** — 最紧凑的手动款` → `Wacaco Nanopresso`
_EMPHASIS = re.compile(r"[*_`]")
_LABEL_SPLIT = re.compile(r"\s*[—–:：]\s*")


def _clean_label(item: str) -> str:
    s = _EMPHASIS.sub("", item).strip()
    s = _LABEL_SPLIT.split(s, maxsplit=1)[0].strip()
    return (s or item.strip())[:60]


def _match_brand(item: str, names: list[str]) -> str | None:
    """这一条推荐项说的是哪个品牌。认不出返回 None。

    `names` 已按长度倒序 —— 否则「Wacaco」会先把「Wacaco Nanopresso」抢走，
    把同一个品牌拆成两个。
    """
    low = item.lower()
    for n in names:
        if n.lower() in low:
            return n
    return None


@dataclass
class Slot:
    """某个位次（#1 / #2 / #3）被谁占过。"""

    position: int
    holders: dict[str, int] = field(default_factory=dict)    # 认得出的品牌 → 次数
    others: Counter[str] = field(default_factory=Counter)    # 认不出的推荐项 → 次数

    @property
    def distinct(self) -> int:
        """这个位次上出现过多少个**不同**的东西（含没认出来的）。"""
        return len(self.holders) + len(self.others)

    @property
    def contested(self) -> bool:
        return self.distinct > 1

    def render(self) -> str:
        parts = [f"{b} {n}" for b, n in
                 sorted(self.holders.items(), key=lambda kv: (-kv[1], kv[0]))]
        parts += [f"?{o} {n}" for o, n in
                  sorted(self.others.items(), key=lambda kv: (-kv[1], kv[0]))]
        return " · ".join(parts) or "—"


@dataclass
class QuestionOccupancy:
    question: str
    samples: int = 0     # 有效样本数
    listed: int = 0      # 其中带推荐列表的样本数
    slots: dict[int, Slot] = field(default_factory=dict)
    unknown: Counter[str] = field(default_factory=Counter)

    def slot(self, position: int) -> Slot:
        return self.slots.setdefault(position, Slot(position=position))

    @property
    def ordered_slots(self) -> list[Slot]:
        return [self.slots[p] for p in sorted(self.slots)]

    @property
    def head(self) -> Slot | None:
        return self.slots.get(1)

    @property
    def unstable_head(self) -> bool:
        """第 1 位在样本之间易手 —— AI 对这个类目没有稳定偏好。

        这条比「提及率」更接近选品要的答案：**头部位次稳不稳，决定这个类目
        是铁板一块还是有缝。**
        """
        return bool(self.head and self.head.contested)


@dataclass
class BrandShare:
    brand: str
    mentioned: int = 0          # 出现在答案正文里的样本数（含散文提及）
    ranked: int = 0             # 进了推荐列表的样本数
    top1: int = 0               # 占住第 1 位的样本数
    positions: list[int] = field(default_factory=list)

    @property
    def avg_position(self) -> float | None:
        return sum(self.positions) / len(self.positions) if self.positions else None

    @property
    def best_position(self) -> int | None:
        return min(self.positions) if self.positions else None


@dataclass
class OccupancyReport:
    brands: list[BrandShare] = field(default_factory=list)
    by_question: list[QuestionOccupancy] = field(default_factory=list)
    unknown: Counter[str] = field(default_factory=Counter)
    samples: int = 0

    @property
    def unstable_questions(self) -> list[str]:
        return [q.question for q in self.by_question if q.unstable_head]


def occupancy(answers: list[Answer], brands: list[str]) -> OccupancyReport:
    """算出「这个类目的推荐位被谁占了」。

    失败的样本直接跳过 —— 和 metrics 同一条口径：引擎超时不等于「没有推荐位」。
    """
    names = sorted({b.strip() for b in brands if b and b.strip()},
                   key=len, reverse=True)
    shares = {b: BrandShare(brand=b) for b in names}
    by_q: dict[str, QuestionOccupancy] = {}
    unknown: Counter[str] = Counter()
    n_ok = 0

    for a in answers:
        if not a.ok:
            continue
        n_ok += 1
        q = by_q.setdefault(a.question, QuestionOccupancy(question=a.question))
        q.samples += 1

        low = a.text.lower()
        for b in names:
            if b.lower() in low:          # 正文任意位置，含散文
                shares[b].mentioned += 1

        items = list_items(a.text)
        if items:
            q.listed += 1

        ranked_here: set[str] = set()
        for position, item in items:
            slot = q.slot(position)
            brand = _match_brand(item, names)
            if brand is None:
                label = _clean_label(item)
                slot.others[label] += 1
                q.unknown[label] += 1
                unknown[label] += 1
                continue
            slot.holders[brand] = slot.holders.get(brand, 0) + 1
            if brand in ranked_here:
                continue                   # 同一个品牌在同一条答案里出现两次，只算一次
            ranked_here.add(brand)
            shares[brand].positions.append(position)
            if position == 1:
                shares[brand].top1 += 1

        for b in ranked_here:
            shares[b].ranked += 1

    return OccupancyReport(
        brands=sorted(shares.values(), key=lambda s: (-s.ranked, -s.mentioned, s.brand)),
        by_question=[by_q[k] for k in sorted(by_q)],
        unknown=unknown,
        samples=n_ok,
    )


def format_occupancy(report: OccupancyReport, brands: list[str]) -> str:
    """给人看的占位报告。"""
    if not report.samples:
        return "没有有效样本，无法做占位分析（检查引擎是否全部失败）。"

    lines = [
        f"类目占位分析　竞品 {len(brands)} 个　问题 {len(report.by_question)} 个"
        f"　有效样本 {report.samples}",
        "",
        "■ 品牌份额（按进榜次数排）",
        f"{'品牌':<20}{'提及':>6}{'进榜':>6}{'占住#1':>8}{'最好':>6}{'平均':>7}",
        "-" * 55,
    ]
    for s in report.brands:
        best = f"#{s.best_position}" if s.best_position else "—"
        avg = f"{s.avg_position:.1f}" if s.avg_position is not None else "—"
        lines.append(f"{s.brand:<20}{s.mentioned:>6}{s.ranked:>6}"
                     f"{s.top1:>8}{best:>6}{avg:>7}")

    lines += ["", "■ 逐问题占位"]
    for q in report.by_question:
        lines.append(f"\n{q.question}")
        lines.append(f"  有效样本 {q.samples} · 有推荐列表 {q.listed}")
        for slot in q.ordered_slots:
            mark = "　← 该位次易手" if slot.contested else ""
            lines.append(f"  #{slot.position}  {slot.render()}{mark}")
        if q.unstable_head:
            lines.append(f"  ⚠ 第 1 位在样本间易手（{q.head.distinct} 个不同品牌）"
                         " → AI 对这个类目没有稳定偏好")

    if report.unknown:
        lines += ["", "■ 未列进竞品表、但被 AI 推荐过的项",
                  "  （你只能列出你已经知道的对手，AI 会告诉你漏掉的那些）"]
        for label, n in report.unknown.most_common():
            lines.append(f"  {label} ×{n}")
    else:
        lines += ["", "■ 没有发现竞品表之外的推荐项"]
    return "\n".join(lines)
