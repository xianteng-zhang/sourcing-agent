"""review_miner 的单元测试 —— 重点钉住「脏 CSV 的列识别与评分解析」。

为什么只测这一层:它是整个工具唯一直接面对不可控外部数据的地方。
上游插件换一版导出格式,识别逻辑会**静默失效**——不报错,只是评分全变成空,
报告照常生成,但概览里的平均分和好评占比悄悄没了。这种失败必须靠测试兜住。

运行:pytest -q
"""
from pathlib import Path

import pandas as pd
import pytest

from review_miner import (
    _detect_review_column,
    _normalize_columns,
    _parse_rating,
    analyze_reviews,
    compute_stats,
    read_reviews_csv,
)

SAMPLE_CSV = Path(__file__).parent / "data" / "sample_reviews.csv"

REVIEW_TEXT = "This is a normal length customer review about the product."


# ------------------------------------------------------------------ 评分解析

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("5.0 out of 5 stars", 5.0),  # 亚马逊 a-icon-alt 列
        ("4.0 颗星", 4.0),  # 中文导出
        ("3", 3.0),  # 纯数字评分列(手工整理 / 示例数据)
        ("2.5", 2.5),
        ("", None),
        ("没有评分", None),
        (None, None),
    ],
)
def test_parse_rating_accepts_star_text_and_bare_numbers(raw, expected):
    assert _parse_rating(raw) == expected


# ---------------------------------------------------------------- 列名标准化

def test_normalize_columns_by_standard_name():
    """第一层:标准列名直接命中。"""
    df = pd.DataFrame({"review": ["很好用", "一般般"], "rating": ["5", "3"]})
    out = _normalize_columns(df)
    assert list(out.columns) == ["review", "rating"]
    assert out["rating"].tolist() == [5.0, 3.0]


def test_normalize_columns_by_platform_classname():
    """第二层:Instant Data Scraper 导出的亚马逊格式,列名是网页类名。"""
    df = pd.DataFrame(
        {
            "a-icon-alt": ["5.0 out of 5 stars", "4.0 out of 5 stars"],
            "cr-original-review-content": [REVIEW_TEXT, REVIEW_TEXT + " Second one is longer."],
        }
    )
    out = _normalize_columns(df)
    assert out["review"].tolist() == [REVIEW_TEXT, REVIEW_TEXT + " Second one is longer."]
    assert out["rating"].tolist() == [5.0, 4.0]


def test_normalize_columns_by_content_feature_when_names_are_opaque():
    """第三层:列名毫无语义时,退到内容特征推断。"""
    df = pd.DataFrame(
        {
            "col_a": ["5.0 out of 5 stars", "3.0 out of 5 stars", "4.0 out of 5 stars"],
            "col_b": [REVIEW_TEXT, REVIEW_TEXT + " ok", REVIEW_TEXT + " good enough"],
        }
    )
    out = _normalize_columns(df)
    assert out["review"].iloc[0] == REVIEW_TEXT
    assert out["rating"].tolist() == [5.0, 3.0, 4.0]


# ------------------------------------------------------------ 干扰列排除

def test_detect_review_column_prefers_coverage_over_length():
    """回归:lightbox 弹窗那种「行数少但单条极长」的列不能被选中。"""
    df = pd.DataFrame(
        {
            "lightbox": ["X" * 3000, "Y" * 3000] + [None] * 20,
            "review": [REVIEW_TEXT] * 22,
        }
    )
    assert _detect_review_column(df) == "review"


def test_detect_review_column_rejects_markup_columns():
    """整列是 HTML / CSS / JS 片段时应被排除。"""
    markup = (
        'window.__DATA__ = function () { return "<div class=\\"a-section\\" '
        'style=\\"margin:0\\"></div>"; };'
    )
    df = pd.DataFrame({"script": [markup] * 10, "review": [REVIEW_TEXT] * 10})
    assert _detect_review_column(df) == "review"


# -------------------------------------------------------------- 端到端回归

def test_sample_csv_keeps_all_ratings():
    """回归:示例数据的 rating 列是纯数字,修复前会被整列解析成空。

    这是使用者的第一条路径(界面上的「使用示例数据」勾选框),
    一旦评分全空,概览卡片就没有平均分和好评占比。
    """
    df = read_reviews_csv(str(SAMPLE_CSV))
    assert len(df) == 37
    assert df["rating"].notna().all()

    stats = compute_stats(df)
    assert stats["total_reviews"] == 37
    assert 1 <= stats["avg_rating"] <= 5
    assert sum(stats["sentiment"].values()) == 37


# ------------------------------------------------------------ 分批与抽样

class _FakeLLM:
    """记录每次调用的输入并返回合法 JSON —— 测分批/抽样,不打真实 API。"""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)

        class _Resp:
            content = (
                '{"positives": [], "pain_points": [], '
                '"purchase_motives": [], "competitor_mentions": []}'
            )

        return _Resp()


def test_analyze_reviews_batches_and_samples_deterministically():
    df = pd.DataFrame({"review": [f"review-{i}" for i in range(500)], "rating": [None] * 500})

    llm = _FakeLLM()
    results = analyze_reviews(df, llm, batch_size=25, max_reviews=300)

    assert len(results) == 12  # 300 条 / 每批 25 条
    assert len(llm.prompts) == 12
    assert "review-0" in llm.prompts[0]  # 等距抽样保留首条
    assert "review-499" in llm.prompts[-1]  # 与末条

    again = _FakeLLM()
    analyze_reviews(df, again, batch_size=25, max_reviews=300)
    assert again.prompts == llm.prompts  # 用索引公式而非随机,抽样可复现
