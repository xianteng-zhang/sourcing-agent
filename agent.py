"""选品 + 上架 Agent:LangGraph ReAct,LLM 自主规划调用工具。

Agent 绑定五个工具:
1. analyze_reviews     —— 评论挖掘,提取卖点/痛点/动机/竞品
2. extract_keywords    —— 从评论提取买家高频关键词
3. generate_listing    —— 根据卖点生成 Listing 文案
4. translate_listing   —— 把 Listing 翻译成多语言
5. compliance_check    —— 合规审查

LLM 根据用户需求自主决定:调用哪些工具、什么顺序、结果怎么流转。
"""
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from compliance import format_violations
from listing_generator import (
    check_compliance,
    extract_keywords,
    generate_listing,
    translate_listing,
)
from review_miner import analyze_reviews, compute_stats, read_reviews_csv

AGENT_SYSTEM = (
    "你是跨境电商「选品 + 上架」助手。用户会给你商品评论 CSV 的路径和需求。\n"
    "你有五个工具:\n"
    "1. analyze_reviews:分析评论,挖出卖点/痛点/购买动机/竞品线索\n"
    "2. extract_keywords:从评论提取买家高频关键词(用于埋词和广告)\n"
    "3. generate_listing:根据卖点生成 Listing 文案(标题/五点/详情/关键词)\n"
    "4. translate_listing:把 Listing 翻译成目标语言(日语/西语/德语/法语等)\n"
    "5. compliance_check:检查 Listing 文案的合规风险\n"
    "你要自主规划:根据用户需求决定调用哪些工具、什么顺序,最后综合输出一份完整报告。\n"
    "注意:\n"
    "- generate_listing 内置了确定性合规闸门(命中违禁词会自动重写,最多 3 轮),"
    "工具返回开头的 [合规闸门] 一行会告诉你是否通过。\n"
    "- 生成 Listing 后,通常应该再用 compliance_check 检查一遍合规性。\n"
    "- 若 [合规闸门] 显示未通过,必须在最终回答里明确指出残留问题,"
    "不得把它当作可直接上架的文案。\n"
    "- 最终回答必须【完整、原文】贴出工具生成的全部内容,尤其是完整的 Listing 文案"
    "(标题、五点描述、详情页、规格参数表、关键词建议),一字不差地展示出来,"
    "不要只说「已生成,见工具输出」。"
)


def build_agent(llm):
    """构建选品 + 上架 Agent(工具闭包持有 llm)。"""

    @tool
    def analyze_reviews_tool(csv_path: str) -> str:
        """分析商品评论 CSV,提取卖点、痛点、购买动机、竞品线索。
        参数 csv_path:评论 CSV 文件的完整路径。"""
        df = read_reviews_csv(csv_path)
        stats = compute_stats(df)
        batches = analyze_reviews(df, llm)

        merged = {
            "positives": [],
            "pain_points": [],
            "purchase_motives": [],
            "competitor_mentions": [],
        }
        for b in batches:
            for k in merged:
                merged[k].extend(b.get(k, []))

        return (
            f"评论概览:共 {stats.get('total_reviews', 0)} 条,"
            f"平均评分 {stats.get('avg_rating', '未知')}\n"
            f"卖点:{' ; '.join(merged['positives'][:15])}\n"
            f"痛点:{' ; '.join(merged['pain_points'][:15])}\n"
            f"购买动机:{' ; '.join(merged['purchase_motives'][:10])}\n"
            f"竞品提及:{' ; '.join(merged['competitor_mentions'][:10])}"
        )

    @tool
    def extract_keywords_tool(csv_path: str) -> str:
        """从商品评论 CSV 提取买家高频关键词(用于 Listing 埋词和 PPC 广告)。
        参数 csv_path:评论 CSV 文件路径。"""
        return extract_keywords(csv_path, llm=llm)

    @tool
    def generate_listing_tool(sell_points: str) -> str:
        """根据卖点生成亚马逊 Listing 文案(标题、五点描述、详情页、关键词)。
        已内置确定性合规闸门:命中违禁词会自动重写,最多 3 轮。
        参数 sell_points:产品的卖点文字。"""
        result = generate_listing(sell_points, llm=llm)
        if result["passed"]:
            status = f"[合规闸门] 第 {result['attempts']} 轮通过,未命中违禁词。\n\n"
        else:
            status = (
                f"[合规闸门] 已达 {result['attempts']} 轮上限仍未通过,"
                "以下问题**未修复**,不得当作可上架文案:\n"
                + format_violations(result["violations"])
                + "\n\n"
            )
        return status + result["text"]

    @tool
    def translate_listing_tool(listing_text: str, target_language: str) -> str:
        """把 Listing 文案翻译成目标语言(如日语、西班牙语、德语、法语)。
        参数 listing_text:文案全文;target_language:目标语言。"""
        return translate_listing(listing_text, target_language, llm=llm)

    @tool
    def compliance_check_tool(listing_text: str) -> str:
        """检查 Listing 文案的合规风险(违禁词、夸大宣传、侵权)。
        参数 listing_text:需要检查的文案全文。"""
        return check_compliance(listing_text, llm=llm)

    tools = [
        analyze_reviews_tool,
        extract_keywords_tool,
        generate_listing_tool,
        translate_listing_tool,
        compliance_check_tool,
    ]
    return create_react_agent(llm, tools, prompt=AGENT_SYSTEM)


def run_agent(csv_path: str, user_request: str, llm) -> dict:
    """运行选品 + 上架 Agent,返回最终回答 + 工具调用过程。"""
    agent = build_agent(llm)
    default_request = "请分析这个产品的评论(卖点/痛点/关键词),生成完整的 Listing 上架文案,并做合规审查。"
    prompt = f"商品评论 CSV 路径:{csv_path}\n用户需求:{user_request or default_request}"
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})

    messages = result.get("messages", [])

    # 提取工具调用步骤(用于界面可视化)
    steps = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                steps.append({"tool": tc["name"], "args": tc.get("args", {}), "result": ""})
        elif msg.type == "tool":
            if steps:
                steps[-1]["result"] = msg.content

    final = messages[-1].content if messages else "Agent 没有返回结果"
    return {"answer": final, "steps": steps}


def run_agent_stream(csv_path: str, user_request: str, llm, on_step=None) -> dict:
    """流式运行 Agent,每一步通过 on_step(tool, result) 回调通知(用于实时展示)。

    on_step(tool_name, result):result 为 None 表示「开始调用工具」,有值表示「工具完成」。
    """
    agent = build_agent(llm)
    default_request = "请分析这个产品的评论(卖点/痛点/关键词),生成完整的 Listing 上架文案,并做合规审查。"
    prompt = f"商品评论 CSV 路径:{csv_path}\n用户需求:{user_request or default_request}"

    steps_by_id = {}
    order = []
    final_answer = ""

    for chunk in agent.stream(
        {"messages": [HumanMessage(content=prompt)]},
        stream_mode="updates",
    ):
        for node, output in chunk.items():
            msgs = output.get("messages", []) if isinstance(output, dict) else []
            if node == "tools":
                for msg in msgs:
                    if msg.type == "tool":
                        tc_id = msg.tool_call_id
                        if tc_id in steps_by_id:
                            steps_by_id[tc_id]["result"] = msg.content
                            if on_step:
                                on_step(steps_by_id[tc_id]["tool"], msg.content)
            elif node == "agent":
                for msg in msgs:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            steps_by_id[tc["id"]] = {"tool": tc["name"], "args": tc.get("args", {}), "result": ""}
                            order.append(tc["id"])
                            if on_step:
                                on_step(tc["name"], None)
                    else:
                        final_answer = msg.content

    steps = [steps_by_id[tid] for tid in order]
    return {"answer": final_answer, "steps": steps}
