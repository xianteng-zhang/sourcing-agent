"""竞品对比分析:并行分析多个竞品,汇总生成对比报告。

**跑在持久化队列上**(jobs.py),而不是内存线程池。差别在于崩溃之后:

    内存 ThreadPoolExecutor  → 进程一挂,整批重来,已分析完的竞品 token 全白烧
    持久化队列               → 已完成的结果在库里,重启只补跑没做完的那几个

同一个 run_id 重复调用是幂等的,所以「用户关掉窗口再打开、App 崩了重开」
都会续跑而不是重算。run_id 默认由这批输入(名字 + 文件大小 + mtime)算出来,
输入没变就是同一批活。
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

import jobs
from jobs import run_durable
from prompts import COMPARISON_SYSTEM, comparison_prompt
from review_miner import analyze_reviews, compute_stats, read_reviews_csv


def _analyze_one(name: str, csv_path: str, llm) -> dict[str, Any]:
    """分析单个竞品:读 CSV → 统计 → 分批提取卖点/痛点等。"""
    df = read_reviews_csv(csv_path)
    stats = compute_stats(df)
    batches = analyze_reviews(df, llm)

    # 合并所有批次的结果(同类目拼到一起)
    merged: dict[str, list] = {
        "positives": [],
        "pain_points": [],
        "purchase_motives": [],
        "competitor_mentions": [],
    }
    for b in batches:
        for key in merged:
            merged[key].extend(b.get(key, []))

    return {"name": name, "stats": stats, "findings": merged}


def default_run_id(products: list[tuple[str, str]]) -> str:
    """由这批输入算出稳定的 run_id:输入没变 → 同一批活 → 可以续跑。"""
    digest = hashlib.sha1()
    for name, path in sorted(products):
        p = Path(path)
        stamp = f"{p.stat().st_size}:{int(p.stat().st_mtime)}" if p.exists() else "missing"
        digest.update(f"{name}|{path}|{stamp}\n".encode())
    return "compare-" + digest.hexdigest()[:12]


def run_compare(
    products: list[tuple[str, str]],
    llm,
    progress: Callable[[int, int], None] | None = None,
    run_id: str | None = None,
    queue: "jobs.JobQueue | None" = None,
) -> dict[str, Any]:
    """并行分析多个竞品,再汇总生成对比报告。

    参数:
        products: [(竞品名, csv 路径), ...]
        run_id:   不给就按输入算;给了就能跨进程续跑同一批
    返回:
        {"per_product": {...}, "report": ..., "failed": {...}, "run_id": ...}
    """
    if not products:
        raise ValueError("至少需要一个竞品")

    run_id = run_id or default_run_id(products)
    queue = queue or jobs.DEFAULT_QUEUE

    tasks = [(name, {"name": name, "csv_path": path}) for name, path in products]
    out = run_durable(
        queue,
        run_id,
        tasks,
        lambda payload: _analyze_one(payload["name"], payload["csv_path"], llm),
        max_workers=min(len(products), 4),
        progress=progress,
    )

    results: dict[str, dict] = out["results"]
    failed: dict[str, str] = out["failed"]

    if not results:
        return {
            "per_product": {},
            "report": "全部竞品分析都失败了,没有可汇总的内容。\n\n"
                      + "\n".join(f"- {k}:{v}" for k, v in failed.items()),
            "failed": failed,
            "run_id": run_id,
        }

    # 汇总生成对比报告(单次调用,不进队列)
    aggregated = json.dumps(results, ensure_ascii=False, indent=2)
    resp = llm.invoke([
        SystemMessage(content=COMPARISON_SYSTEM),
        HumanMessage(content=comparison_prompt(aggregated)),
    ])
    report = resp.content
    if failed:
        report += "\n\n---\n\n**以下竞品分析失败,未计入上表:**\n" + "\n".join(
            f"- {k}:{v}" for k, v in failed.items()
        )

    return {"per_product": results, "report": report, "failed": failed, "run_id": run_id}
