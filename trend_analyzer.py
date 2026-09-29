"""关键词趋势分析:基于 pytrends(Google 搜索趋势,免费)。"""
import pandas as pd
from pytrends.request import TrendReq


def get_trends(keywords: list[str], timeframe: str = "today 12-m") -> pd.DataFrame:
    """查询一组关键词过去一段时间的热度走势。

    参数:
        keywords: 关键词列表,1~5 个。英文关键词效果最好(中文搜索量可能偏低)。
        timeframe: 如 "today 3-m"(近3个月)、"today 12-m"(近12个月)、"today 5-y"(近5年)。
    返回:
        以日期为索引、关键词为列的 DataFrame(0~100 的相对热度)。
    """
    if not keywords:
        raise ValueError("请至少提供一个关键词")
    if len(keywords) > 5:
        raise ValueError("Google Trends 一次最多比较 5 个关键词")

    pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
    pytrends.build_payload(keywords, timeframe=timeframe)
    df = pytrends.interest_over_time()
    if df is not None and "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    return df


def trend_summary(df: pd.DataFrame) -> str:
    """根据趋势数据,给出每个关键词的「上升/平稳/下降」判断。"""
    if df is None or df.empty:
        return "暂无趋势数据"

    lines = []
    for col in df.columns:
        series = df[col]
        head = float(series.iloc[: len(series) // 3].mean())
        tail = float(series.iloc[-len(series) // 3 :].mean())
        if tail - head > 5:
            direction = "上升趋势 📈"
        elif tail - head < -5:
            direction = "下降趋势 📉"
        else:
            direction = "平稳 ➡️"
        lines.append(f"- **{col}**:{direction}(前期均值 {head:.0f} → 近期均值 {tail:.0f})")
    return "\n".join(lines)
