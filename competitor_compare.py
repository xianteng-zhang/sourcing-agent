"""竞品对比分析:并行分析多个竞品的评论,汇总生成对比报告。"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

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


def run_compare(
    products: list[tuple[str, str]],
    llm,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """并行分析多个竞品,再汇总生成对比报告。

    参数:
        products: [(竞品名, csv 路径), ...]
    返回:
        {"per_product": {...}, "report": "对比报告 Markdown"}
    """
    results: dict[str, dict] = {}
    max_workers = min(len(products), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_analyze_one, name, path, llm): name for name, path in products}
        done = 0
        for fut in as_completed(futures):
            name = futures[fut]
            results[name] = fut.result()
            done += 1
            if progress:
                progress(done, len(products))

    # 汇总生成对比报告
    aggregated = json.dumps(results, ensure_ascii=False, indent=2)
    resp = llm.invoke([
        SystemMessage(content=COMPARISON_SYSTEM),
        HumanMessage(content=comparison_prompt(aggregated)),
    ])
    return {"per_product": results, "report": resp.content}
