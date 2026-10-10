"""采集编排：多引擎 × 多问题 × N 次采样，跑在**已有的持久化队列**上。

一个采集批次的任务量很容易失控：

    10 个引擎 × 500 个问题 × 每天 3 次采样 = 每天 15000 个任务

这种量级下，「跑一半进程挂了」不是异常，是常态。而这里**没有新写一套并发**——
直接复用 `jobs.py` 的 claim / lease / reaper：崩了重来，已经采到的不重采
（不重花钱），租约过期由 reaper 捞回。

**一个必须堵住的漏洞：** 任务队列里的任务死亡（dead）时，如果就这么算了，那条
采样就**凭空消失**了 —— 它既不在分子里，也不在分母里，指标悄悄变好看。
所以死掉的任务要补一条**失败的 Answer**，让它老实待在分母之外、失败计数之内。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from engines.base import Answer, Engine, FailureKind, safe_ask
from engines.metrics import compute, format_report
from engines.store import DEFAULT_STORE, AnswerStore
from jobs import DEFAULT_QUEUE, JobQueue, run_durable


def default_run_id(engines: list[Engine], questions: list[str], samples: int) -> str:
    """同一批引擎 + 同一批问题 + 同样的采样数 → 同一个 run_id。

    这样重跑同一批就是**幂等续跑**，而不是又花一遍钱采一遍。
    """
    spec = "|".join(sorted(e.name for e in engines))
    spec += "#" + "|".join(questions) + f"#{samples}"
    return "geo-" + hashlib.sha1(spec.encode("utf-8")).hexdigest()[:12]


def plan_tasks(engines: list[Engine], questions: list[str],
               samples: int) -> list[tuple[str, dict]]:
    """展开成 (key, payload) 列表。key 里带上问题和采样序号，方便直接看懂。"""
    return [
        (f"{e.name}|{s}|{q}", {"engine": e.name, "question": q, "sample": s})
        for e in engines
        for q in questions
        for s in range(samples)
    ]


def collect(
    engines: list[Engine],
    questions: list[str],
    samples: int = 3,
    run_id: str | None = None,
    queue: JobQueue | None = None,
    store: AnswerStore | None = None,
    max_workers: int = 4,
    progress=None,
) -> dict:
    """采一批答案，落库，返回可复现的结果。

    返回 {"run_id", "answers", "metrics", "summary", "stats", "dead"}。
    """
    queue = queue or DEFAULT_QUEUE
    store = store or DEFAULT_STORE
    run_id = run_id or default_run_id(engines, questions, samples)

    by_name = {e.name: e for e in engines}
    tasks = plan_tasks(engines, questions, samples)
    payload_of = dict(tasks)

    def run_one(payload: dict) -> dict:
        return safe_ask(by_name[payload["engine"]],
                        payload["question"], payload["sample"]).to_dict()

    out = run_durable(queue, run_id, tasks, run_one, max_workers=max_workers,
                      progress=progress)

    answers = [Answer.from_dict(d) for d in out["results"].values()]

    # 队列里彻底失败的任务：补一条失败 Answer，否则它既不进分子也不进分母。
    for key, err in out["failed"].items():
        p = payload_of.get(key)
        if p is None:
            continue
        answers.append(Answer(engine=p["engine"], question=p["question"],
                              sample=p["sample"], error=FailureKind.UNKNOWN,
                              error_detail=f"任务未完成: {err}"))

    store.save_many(run_id, answers)
    stored = store.load(run_id)  # 取回带 id 的版本 —— 指标要靠 id 溯源
    return {
        "run_id": run_id,
        "answers": stored,
        "metrics": compute(stored, brand="", planned=_planned(engines, questions, samples)),
        "stats": out["stats"],
        "dead": out["failed"],
    }


def _planned(engines: list[Engine], questions: list[str], samples: int) -> dict[str, int]:
    return {e.name: len(questions) * samples for e in engines}


def analyze(run_id: str, brand: str, aliases: list[str] | None = None,
            store: AnswerStore | None = None, planned: dict[str, int] | None = None):
    """从库里读一次采集，算指标。

    品牌与别名是**分析期**参数，不是采集期参数 —— 同一批原始答案，换个品牌名
    就能重新分析，不用重采一遍。这是「原文留存」的直接好处。
    """
    answers = (store or DEFAULT_STORE).load(run_id)
    return compute(answers, brand=brand, aliases=aliases, planned=planned)


# ------------------------------------------------------------------ CLI 演示

class _Counting:
    """给引擎套一层计数器，用来证明「第二轮一次都没调用」。"""

    def __init__(self, inner: Engine) -> None:
        self._inner = inner
        self.name = inner.name
        self.calls = 0

    def ask(self, question: str, sample: int = 0) -> Answer:
        self.calls += 1
        return self._inner.ask(question, sample)


def demo() -> None:
    """用录制样本跑一遍 —— 不联网、不花钱、结果可复现。

        python -m engines
    """
    from engines.replay import load_recordings

    engines = load_recordings()
    if not engines:
        print("没有找到 engines/recordings/*.jsonl 录制文件")
        return

    questions = sorted({q for e in engines for q in e.questions})
    samples, brand = 3, "NovaBrew"
    planned = _planned(engines, questions, samples)

    base = Path(__file__).parent.parent / "data"
    qdb, adb = base / "demo-jobs.db", base / "demo-answers.db"
    for p in (qdb, adb):
        p.unlink(missing_ok=True)  # 演示每次从零开始

    queue, store = JobQueue(qdb), AnswerStore(adb)
    print(f"引擎 {len(engines)} 个：{[e.name for e in engines]}")
    print(f"问题 {len(questions)} 个 × 每个采 {samples} 次 → 计划 {sum(planned.values())} 个采样点")

    first = collect(engines, questions, samples=samples, queue=queue, store=store)
    print(f"\n——— 第 1 轮：真实采集　run_id={first['run_id']} ———")
    print(format_report(compute(first["answers"], brand, planned=planned), brand))

    counted = [_Counting(e) for e in engines]
    again = collect(counted, questions, samples=samples, run_id=first["run_id"],
                    queue=queue, store=store)
    print("\n——— 第 2 轮：同一个 run_id 再跑一次 ———")
    print(f"队列状态 {again['stats']}")
    print(f"引擎调用次数 {sum(e.calls for e in counted)} 次"
          f"　落库条数 {len(again['answers'])}（与第 1 轮相同 → 没有重采）")

    print("\n——— 溯源：把任意一个指标拉回原文 ———")
    for m in compute(first["answers"], brand, planned=planned):
        if not m.evidence_ids:
            continue
        raw = store.get(m.evidence_ids[0])
        print(f"[{m.engine}] 提及 {m.mentions}/{m.ok}（进榜 {m.ranked}）"
              f"　→ 证据 answer_id={raw.id}　模型={raw.model}")
        print(f"  问题：{raw.question}")
        print(f"  原文：{raw.text.replace(chr(10), ' ')[:150]}...")
        break


if __name__ == "__main__":
    demo()
