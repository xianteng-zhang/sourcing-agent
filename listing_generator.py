"""Listing 生成、合规审查、关键词挖掘、多语言翻译。

确定性闸门有**两类**,都在模型之外先行判定:

  1. 违禁词(compliance.scan) —— 绝对化用语 / 促销语 / 承诺性表述 / 医疗功效 /
     联系方式外链 / 未授权认证。命中即判不合格,LLM 无权推翻,也不会被调用。
  2. 参数锚定(spec_anchor.check_spec_anchor) —— Listing 里的数值型参数必须能在
     卖家填写的规格里找到,查不到即为候选编造。

两类都通过后,才轮到 LLM 细查(本文件的 check_compliance)参数单位、标题格式、
关键词堆砌、语言一致性这类软问题 —— 那些是模型更擅长的部分。

`generate_listing` 用前两类做「生成 → 拦截 → 重写」闭环,最多 MAX_GATE_RETRIES 轮。
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
from spec_anchor import check_spec_anchor

MAX_GATE_RETRIES = 3

# 兼容旧名字
MAX_COMPLIANCE_RETRIES = MAX_GATE_RETRIES


def run_deterministic_gates(text: str, my_product_specs: str = "") -> list:
    """两类确定性闸门合一:违禁词 + 参数锚定。不调用任何模型。"""
    return scan(text) + check_spec_anchor(text, my_product_specs)


def generate_listing(
    competitor_insights: str,
    my_product_specs: str = "",
    llm=None,
    max_retries: int = MAX_GATE_RETRIES,
) -> dict:
    """根据竞品评论洞察 + 卖家产品规格生成 Listing,并用确定性闸门做闭环。

    「生成 → 拦截 → 重写」:命中项连同修改建议回喂给模型重新生成,直到两类闸门都放行。

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
                    competitor_insights, my_product_specs, format_violations(violations)
                )
            ),
        ])
        text = resp.content
        violations = run_deterministic_gates(text, my_product_specs)
        if not violations:
            return {"text": text, "attempts": attempt, "violations": [], "passed": True}

    return {
        "text": text,
        "attempts": max_retries,
        "violations": violations,
        "passed": False,
    }


def check_compliance(listing_text: str, llm=None, my_product_specs: str = "") -> str:
    """合规审查:先跑确定性闸门,命中即直接判不合格;未命中再交给 LLM 细查。

    命中时**不调用模型** —— 红线是硬性的,不需要也不应该让模型再判一次
    (既可能被推翻,也白花一次 token)。
    """
    violations = run_deterministic_gates(listing_text, my_product_specs)
    if violations:
        return (
            "❌ 未通过确定性闸门(一票否决,未经模型判定)\n\n"
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
