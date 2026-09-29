"""命令行验证:不启动界面,直接跑一遍评论挖掘 / 趋势分析。

用法:
    python verify.py              # 评论挖掘(示例数据)
    python verify.py --trend      # 趋势分析
"""
import sys

import config


def verify_review_mining() -> None:
    from review_miner import run_review_mining

    llm = config.get_llm()
    print("开始评论挖掘(示例数据)...\n")
    result = run_review_mining("data/sample_reviews.csv", llm, batch_size=25)

    print("=== 评论概览 ===")
    print(result["stats"])
    print("\n=== 选品报告 ===\n")
    print(result["report"])


def verify_trend() -> None:
    from trend_analyzer import get_trends, trend_summary

    df = get_trends(["wireless earbuds", "noise cancelling headphones"], "today 12-m")
    print("=== 趋势数据(前几行) ===")
    print(df.head())
    print("\n=== 趋势判断 ===")
    print(trend_summary(df))


if __name__ == "__main__":
    if "--trend" in sys.argv:
        verify_trend()
    else:
        verify_review_mining()
