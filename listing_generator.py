"""Listing 生成、合规审查、关键词挖掘、多语言翻译。

合规是**两层**的,别把两层混为一谈:

  1. 确定性闸门(compliance.scan) —— 命中违禁词/促销语/绝对化用语/未授权认证
     直接判不合格,LLM 无权推翻,也不会被调用。
  2. LLM 审查(本文件的 check_compliance) —— 未命中时再由模型细查参数单位、
     标题格式、关键词堆砌、语言一致性等软问题。

`generate_listing` 用第 1 层做「生成 → 拦截 → 重写」闭环,最多 MAX_COMPLIANCE_RETRIES 轮。
"""
from langchain_core.messages import HumanMessage, SystemMessage

from compliance import format_violations, scan
from prompts import (
    COMPLIANCE_SYSTEM,
    KEYWORD_SYSTEM,
    LISTING_SYSTEM,
    TRANSLATE_SYSTEM,
    compliance_prompt,
    keyword_prompt,
    listing_prompt,
    translate_prompt,
)
from review_miner import read_reviews_csv

MAX_COMPLIANCE_RETRIES = 3


def generate_listing(
    sell_points: str,
    product_category: str = "",
    llm=None,
    max_retries: int = MAX_COMPLIANCE_RETRIES,
) -> dict:
    """生成亚马逊 Listing,并用确定性闸门做「生成 → 拦截 → 重写」闭环。

    返回:
        {"text": 最终文案, "attempts": 实际轮次,
         "violations": 仍残留的违规项(通过时为空), "passed": 是否通过闸门}
    """
    violations: list = []
    text = ""

    for attempt in range(1, max_retries + 1):
        resp = llm.invoke([
            SystemMessage(content=LISTING_SYSTEM),
            HumanMessage(
                content=listing_prompt(
                    sell_points, product_category, format_violations(violations)
                )
            ),
        ])
        text = resp.content
        violations = scan(text)
        if not violations:
            return {"text": text, "attempts": attempt, "violations": [], "passed": True}

    return {
        "text": text,
        "attempts": max_retries,
        "violations": violations,
        "passed": False,
    }


def check_compliance(listing_text: str, llm=None) -> str:
    """合规审查:先跑确定性闸门,命中即直接判不合格;未命中再交给 LLM 细查。

    命中时**不调用模型** —— 红线是硬性的,不需要也不应该让模型再判一次
    (既可能被推翻,也白花一次 token)。
    """
    violations = scan(listing_text)
    if violations:
        return (
            "❌ 未通过确定性合规闸门(一票否决,未经模型判定)\n\n"
            + format_violations(violations)
            + "\n\n以上命中项必须修改后重新生成;LLM 细查已跳过。"
        )

    resp = llm.invoke([
        SystemMessage(content=COMPLIANCE_SYSTEM),
        HumanMessage(content=compliance_prompt(listing_text)),
    ])
    return resp.content


def extract_keywords(csv_path: str, llm=None) -> str:
    """从评论 CSV 提取买家高频关键词(用于 Listing 埋词和 PPC 广告)。"""
    df = read_reviews_csv(csv_path)
    reviews_text = "\n".join(df["review"].head(50).tolist())
    resp = llm.invoke([
        SystemMessage(content=KEYWORD_SYSTEM),
        HumanMessage(content=keyword_prompt(reviews_text)),
    ])
    return resp.content


def translate_listing(listing_text: str, target_language: str, llm=None) -> str:
    """把 Listing 翻译成目标语言(日语 / 西班牙语 / 德语 / 法语等)。"""
    resp = llm.invoke([
        SystemMessage(content=TRANSLATE_SYSTEM),
        HumanMessage(content=translate_prompt(listing_text, target_language)),
    ])
    return resp.content
