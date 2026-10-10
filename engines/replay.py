"""录制回放引擎 —— 让「多引擎采集」在测试里不碰网络、不花钱、且可复现。

**为什么这个文件不是测试脚手架，而是产品的一部分：**

1. **测试不能依赖网络。** 靠真引擎跑测试，等于把 CI 的绿灯交给别人的服务可用性
   （以及自己的余额）。CI 上跑 10 个引擎 × N 个问题 × N 次采样，是烧钱买不稳定。

2. **口径必须可复核。** 「提及率 64.8%」这句话，别人凭什么信？如果他能拿到同一份
   录制、跑出同一个数，这句话就从「断言」变成了「可验证的结论」。
   录制文件就是那个**可复核的证据**。

3. **失败也要录。** 一次限流、一次解析失败，本身就是有价值的样本。录制里带上
   `error` 字段，回放时它们会精确重现 —— 于是「失败不能算作没有提及」这条规则
   才**测得到**。
"""
from __future__ import annotations

import json
from pathlib import Path

from engines.base import Answer, EngineError, FailureKind

RECORDINGS = Path(__file__).parent / "recordings"


class ReplayEngine:
    """按录制回放。命不中录制就抛 EngineError —— 不猜、不编。"""

    def __init__(self, name: str, records: list[dict]) -> None:
        self.name = name
        self._records = {(r["question"], int(r.get("sample", 0))): r for r in records}

    @classmethod
    def from_jsonl(cls, name: str, path: Path | str) -> "ReplayEngine":
        """从一个 JSONL 文件读录制。每行一条 Answer（或一条失败记录）。"""
        records = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return cls(name, records)

    def ask(self, question: str, sample: int = 0) -> Answer:
        rec = self._records.get((question, sample))
        if rec is None:
            raise EngineError(
                FailureKind.UNKNOWN,
                f"录制里没有 {question!r} 的第 {sample} 次采样",
            )
        return Answer.from_dict({**rec, "engine": self.name})

    @property
    def questions(self) -> list[str]:
        return sorted({q for q, _ in self._records})

    @property
    def samples_per_question(self) -> int:
        return len({s for _, s in self._records}) or 0


def dump_jsonl(answers: list[Answer], path: Path | str) -> int:
    """把一批真实采到的答案录下来。录制 = 未来所有测试和复盘的输入。

    失败记录同样写进去：它是数据，不是噪声。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for a in answers:
            f.write(json.dumps(a.to_dict(), ensure_ascii=False) + "\n")
    return len(answers)


def load_recordings(names: list[str] | None = None) -> list[ReplayEngine]:
    """加载 engines/recordings/*.jsonl，文件名即引擎名。"""
    engines = []
    for path in sorted(RECORDINGS.glob("*.jsonl")):
        if names and path.stem not in names:
            continue
        engines.append(ReplayEngine.from_jsonl(path.stem, path))
    return engines
