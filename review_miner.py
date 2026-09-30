"""评论挖掘核心模块。

流程(Map-Reduce):
1. 读取评论 CSV(自动探测编码、灵活匹配列名)
2. 统计概览(总数、评分分布、情感占比)
3. 把评论分成小批,每批让 LLM 提取结构化要点(卖点/痛点/动机/竞品)
4. 汇总所有批次结果,再让 LLM 生成最终 Markdown 选品报告
"""
import json
import re
from typing import Any, Callable

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from prompts import (
    BATCH_ANALYSIS_SYSTEM,
    SUMMARY_SYSTEM,
    batch_analysis_prompt,
    summary_prompt,
)

# 评论列 / 评分列的候选列名(兼容中英文)
REVIEW_COLUMNS = ["review", "comment", "review_text", "content", "text", "body", "评论", "评论文本", "内容", "正文"]
RATING_COLUMNS = ["rating", "star", "stars", "score", "rating_star", "评分", "星级", "打分"]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _parse_rating(value) -> float | None:
    """从 '5.0 颗星' / '5.0 out of 5 stars' / 纯 1~5 数字 里提取评分。

    纯数字分支必须和 _detect_rating_column 的识别规则保持一致:
    否则纯数字评分列会被「识别出来」却「解析成空」——不报错,
    但平均分和好评占比会静默消失(示例数据和手工整理的 CSV 都走这条路径)。
    """
    text = str(value).strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:颗星|out of)", text)
    if m:
        return float(m.group(1))
    if re.fullmatch(r"[1-5](?:\.\d+)?", text):  # 纯数字评分列
        return float(text)
    return None


def _detect_review_column(df: pd.DataFrame) -> str | None:
    """按内容特征识别评论正文列:非空行最多、且平均文本较长、且不是 URL/代码。

    优先看「覆盖了多少行」(评论列通常每行都有),再看「平均长度」。
    这样能避开 lightbox 弹窗那种只有少数几行、但单条很长的列。
    """
    candidates = []
    for col in df.columns:
        s = df[col].astype(str)
        nonnull = int(s.notna().sum())
        avg_len = float(s.str.len().mean())
        if avg_len < 20:  # 评论正文一般较长,太短的列不是
            continue
        bad = s.str.contains(
            r"https?://|href=|</|<img|window\.|function\s*\(|@media|margin:",
            na=False,
        ).mean()
        if bad > 0.4:
            continue
        candidates.append((nonnull, avg_len, col))
    if not candidates:
        return None
    # 优先非空行最多,其次平均长度最长
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def _detect_rating_column(df: pd.DataFrame) -> str | None:
    """按内容特征识别评分列:值匹配 '5.0 out of 5' / '5.0 颗星' / 纯 1~5 数字。

    优先选「匹配数量最多」的列(评分列通常每行都有,避免选中少数几行的弹窗评分列)。
    """
    best_col, best_count = None, 0
    for col in df.columns:
        s = df[col].astype(str)
        count = int(
            s.str.contains(
                r"(?:\d+(?:\.\d+)?)\s*(?:out of|颗星|star|星)|^[1-5](?:\.0)?$",
                na=False,
            ).sum()
        )
        if count > best_count:
            best_count, best_col = count, col
    return best_col if best_count > 0 else None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把任意列名的 DataFrame 标准化为 review / rating 两列。

    识别顺序(越往后越通用):
    1. 列名匹配:标准列名 review/评论 等
    2. Instant Data Scraper 格式:icon-alt / original-review-content 列
    3. 内容特征兜底:评论=最长文本列、评分=评分格式列 —— 列名完全未知也能识别
    """
    review_col = _find_column(df, REVIEW_COLUMNS)
    rating_col = _find_column(df, RATING_COLUMNS)

    # Instant Data Scraper 导出的亚马逊格式
    if review_col is None:
        body_candidates = [c for c in df.columns if "original-review-content" in c]
        if body_candidates:
            review_col = max(body_candidates, key=lambda c: df[c].astype(str).str.len().mean())
    if rating_col is None:
        icon_candidates = [c for c in df.columns if "icon-alt" in c]
        if icon_candidates:
            rating_col = icon_candidates[0]

    # 内容特征兜底:列名完全对不上时,按内容识别
    if review_col is None:
        review_col = _detect_review_column(df)
    if rating_col is None:
        rating_col = _detect_rating_column(df)

    if review_col is None:
        raise ValueError(
            "在这个 CSV 里找不到评论内容列。请确认文件里确实有一列是评论正文文字。"
        )

    out = pd.DataFrame()
    out["review"] = df[review_col].astype(str).str.strip()
    if rating_col:
        out["rating"] = df[rating_col].astype(str).apply(_parse_rating)
    else:
        out["rating"] = pd.Series([None] * len(df))
    return out


def read_reviews_csv(csv_path: str) -> pd.DataFrame:
    """读取评论 CSV,自动探测编码,返回标准化后的 DataFrame(含 review、rating 两列)。

    以 # 开头的行会被当作注释跳过(用于承载官方星级分布等元数据)。
    """
    df = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1"):
        try:
            df = pd.read_csv(csv_path, encoding=encoding, comment="#")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if df is None:
        raise ValueError(f"无法解析 CSV 编码:{csv_path}")

    df = _normalize_columns(df)
    df = df[df["review"].notna() & (df["review"] != "") & (df["review"] != "nan")]
    return df.reset_index(drop=True)


def read_reviews_meta(csv_path: str) -> dict:
    """读取 CSV 里以 # 开头的元数据行(如「# 平均分=4.6 总评数=838 5星=80%」),返回 dict。"""
    meta: dict = {}
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1"):
        try:
            with open(csv_path, "r", encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#"):
                        for part in line.lstrip("#").split():
                            if "=" in part:
                                k, v = part.split("=", 1)
                                meta[k] = v
                        break
                    elif line:
                        break
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    return meta


def compute_stats(df: pd.DataFrame) -> dict[str, Any]:
    """计算评论概览统计。"""
    stats: dict[str, Any] = {"total_reviews": int(len(df))}

    ratings = pd.to_numeric(df["rating"], errors="coerce")
    if ratings.notna().sum() > 0:
        valid = ratings.dropna()
        stats["avg_rating"] = round(float(valid.mean()), 2)
        stats["rating_distribution"] = {
            str(k): int(v) for k, v in valid.value_counts().sort_index().items()
        }
        stats["sentiment"] = {
            "positive": int((valid >= 4).sum()),
            "neutral": int((valid == 3).sum()),
            "negative": int((valid <= 2).sum()),
        }
    return stats


def _parse_json(text: str) -> dict:
    """从 LLM 输出中稳健地提取 JSON。"""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def analyze_reviews(
    df: pd.DataFrame,
    llm,
    batch_size: int = 25,
    max_reviews: int = 300,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """把评论分批喂给 LLM,提取结构化要点。"""
    reviews = df["review"].tolist()

    # 评论过多时均匀抽样,控制 token 成本
    if len(reviews) > max_reviews:
        idx = [round(i * (len(reviews) - 1) / (max_reviews - 1)) for i in range(max_reviews)]
        reviews = [reviews[i] for i in idx]

    results: list[dict] = []
    for start in range(0, len(reviews), batch_size):
        batch = reviews[start : start + batch_size]
        numbered = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(batch))
        resp = llm.invoke([
            SystemMessage(content=BATCH_ANALYSIS_SYSTEM),
            HumanMessage(content=batch_analysis_prompt(numbered)),
        ])
        results.append(_parse_json(resp.content))
        if progress:
            progress(min(start + len(batch), len(reviews)), len(reviews))
    return results


def generate_report(batch_results: list[dict], llm, meta: dict | None = None) -> str:
    """汇总分批结果,生成最终 Markdown 选品报告。"""
    aggregated = json.dumps(batch_results, ensure_ascii=False, indent=2)

    # 若抓取时带回了官方星级分布(基于全部评论),优先采信,拼到最前面
    if meta and (meta.get("平均分") or meta.get("总评数")):
        parts = []
        if meta.get("平均分"):
            parts.append(f"平均分 {meta['平均分']}")
        if meta.get("总评数"):
            parts.append(f"总评数 {meta['总评数']}")
        dist = [f"{k} {v}" for k, v in meta.items() if "星" in k]
        if dist:
            parts.append("星级分布 " + "、".join(dist))
        aggregated = (
            "[官方口碑数据(基于全部评论,比样本更权威,请优先采信)]\n" + "、".join(parts) + "\n\n" + aggregated
        )

    resp = llm.invoke([
        SystemMessage(content=SUMMARY_SYSTEM),
        HumanMessage(content=summary_prompt(aggregated)),
    ])
    return resp.content


def run_review_mining(
    csv_path: str,
    llm,
    batch_size: int = 25,
    max_reviews: int = 300,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """一键运行评论挖掘:读 CSV → 统计 → 分批分析 → 生成报告。"""
    df = read_reviews_csv(csv_path)
    meta = read_reviews_meta(csv_path)
    stats = compute_stats(df)
    batch_results = analyze_reviews(df, llm, batch_size, max_reviews, progress)
    report = generate_report(batch_results, llm, meta)
    return {"stats": stats, "report": report, "review_count": int(len(df)), "meta": meta}
