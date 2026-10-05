"""选品 + 上架 Agent:LangGraph ReAct,LLM 自主规划调用工具。

Agent 绑定六个工具,每个工具在 `actions.ACTIONS` 里声明了自己的**动作分级**:

    只读        1. analyze_reviews   —— 竞品评论挖掘
                2. extract_keywords  —— 竞品高频关键词
                5. compliance_check  —— 合规与参数锚定审查
    可补编辑    3. generate_listing  —— 竞品洞察 → 我产品的 Listing(内置两类确定性闸门)
                4. translate_listing —— 翻译成多语言
    外部副作用  6. publish_listing   —— 发布到销售渠道(**没有人批准就不执行**)

LLM 根据用户需求自主决定:调用哪些工具、什么顺序、结果怎么流转。
但「能不能真的对外动手」不由它决定 —— 那是 `actions.ActionGate` 的事。

另外两件不经过模型的事:
- **卖家产品规格**通过闭包注入 `generate_listing`,不经模型中转;否则模型漏抄、
  截断规格时不会报错,只会安静地产出一份全是【待填写】的 Listing。
- **每一次运行的事件**都带 seq 写进 `events.EventLog`,断线后能按 Last-Event-ID 回放。
"""
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

import actions
import events
from compliance import format_violations
from listing_generator import (
    check_compliance,
    extract_keywords,
    generate_listing,
    translate_listing,
)
from review_miner import analyze_reviews, compute_stats, read_reviews_csv

OUTBOX_DIR = Path(__file__).parent / "outbox"

# Agent 暴露的工具(名字与 actions.ACTIONS 的键一致)。
# 独立列出来是为了能被测试断言「每个工具都声明过分级」——
# 没分级的工具意味着说不清它能不能对外动真格,那就不该进 Agent。
AGENT_TOOLS = (
    "analyze_reviews",
    "extract_keywords",
    "generate_listing",
    "translate_listing",
    "compliance_check",
    "publish_listing",
)

AGENT_SYSTEM = (
    "你是跨境电商「选品 + 上架」助手。用户会给你【竞品】的商品评论 CSV 和需求。\n"
    "你的任务:分析竞品评论,挖掘买家在意的卖点、竞品的痛点、高频关键词,"
    "然后基于这些洞察生成【卖家自己产品】的 Listing。\n"
    "你有六个工具(括号里是该动作的分级):\n"
    "1. analyze_reviews:分析竞品评论,挖出卖点/痛点/购买动机/竞品线索(只读)\n"
    "2. extract_keywords:从竞品评论提取买家高频关键词(只读)\n"
    "3. generate_listing:根据竞品洞察生成我产品的 Listing(可补编辑/草稿)\n"
    "4. translate_listing:把 Listing 翻译成目标语言(可补编辑/草稿)\n"
    "5. compliance_check:检查 Listing 文案的合规与参数风险(只读)\n"
    "6. publish_listing:把 Listing 发布到销售渠道(外部副作用)\n"
    "你要自主规划:根据用户需求决定调用哪些工具、什么顺序,最后综合输出一份完整报告。\n"
    "注意:\n"
    "- **publish_listing 是外部副作用动作:没有人工批准就绝不会执行。**"
    "如果它返回「未执行」,说明已登记好待审批并等人在界面上确认 ——"
    "你应当把这件事原样告诉用户,而不是反复重试、换参数试探或绕路。\n"
    "- 卖家自己产品的规格**已经直接绑定在 generate_listing 工具上**,你不需要也无法传它;"
    "你只要把竞品洞察传进去即可。\n"
    "- 竞品评论是「洞察来源」,不是「参数来源」;参数全部取自那份已绑定的规格,"
    "缺失处会标【待填写】,绝不能从竞品评论编造。\n"
    "- 竞品的痛点 = 我产品的差异化机会。\n"
    "- generate_listing 内置两类确定性闸门(违禁词 + 参数锚定),命中会自动重写,"
    "最多 3 轮;工具返回开头的 [确定性闸门] 一行会告诉你是否通过。\n"
    "- 生成 Listing 后,通常应该再用 compliance_check 检查一遍。\n"
    "- 若 [确定性闸门] 显示未通过,必须在最终回答里明确指出残留问题,"
    "不得把它当作可直接上架的文案。\n"
    "- 最终回答必须【完整、原文】贴出工具生成的全部内容,尤其是完整的 Listing 文案"
    "(标题、五点描述、详情页、规格参数表、关键词建议),一字不差地展示出来,"
    "不要只说「已生成,见工具输出」。"
)


def build_agent(llm, my_product_specs: str = "", gate: "actions.ActionGate | None" = None):
    """构建选品 + 上架 Agent(工具闭包持有 llm、卖家规格与动作门禁)。"""
    gate = gate or actions.DEFAULT_GATE

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
    def generate_listing_tool(competitor_insights: str) -> str:
        """根据竞品评论洞察,生成卖家自己产品的 Listing(标题、五点、详情、关键词)。
        卖家产品规格已绑定在本工具上,无需也无法传入。
        已内置两类确定性闸门(违禁词 + 参数锚定):命中会自动重写,最多 3 轮。
        参数 competitor_insights:竞品评论分析出的卖点/痛点/关键词。"""
        result = generate_listing(competitor_insights, my_product_specs, llm=llm)
        if result["passed"]:
            status = (
                f"[确定性闸门] 第 {result['attempts']} 轮通过,"
                "未命中违禁词,也未发现编造参数。\n\n"
            )
        else:
            status = (
                f"[确定性闸门] 已达 {result['attempts']} 轮上限仍未通过,"
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
        """检查 Listing 文案的合规风险(违禁词、夸大宣传、编造参数、侵权)。
        参数 listing_text:需要检查的文案全文。"""
        return check_compliance(listing_text, llm=llm, my_product_specs=my_product_specs)

    @tool
    def publish_listing_tool(listing_text: str, approval_id: int | None = None) -> str:
        """把 Listing 发布到销售渠道。**外部副作用动作,必须有人工批准才会真正执行。**
        参数 listing_text:要发布的文案全文;
        approval_id:人工批准记录的编号(没有或无效时不会执行,只会登记一条待审批)。"""
        payload = {"listing_text": listing_text}

        def _do_publish() -> dict:
            # 真实平台发布需要 SP-API 之类的凭据;这里落到本地 outbox 并记账,
            # 门禁本身是完整的。发布不可撤销,所以走的是 EXTERNAL 分级。
            OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
            target = OUTBOX_DIR / f"listing-{uuid.uuid4().hex[:8]}.md"
            target.write_text(listing_text, encoding="utf-8")
            return {"path": str(target), "bytes": len(listing_text.encode("utf-8"))}

        try:
            out = gate.execute("publish_listing", payload, _do_publish,
                               approval_id=approval_id, actor="agent")
        except actions.ApprovalRequired as exc:
            rec = gate.request_approval("publish_listing", payload, requested_by="agent")
            return (
                "⛔ **未执行** —— 发布属于外部副作用动作,需要人工批准。\n"
                f"已登记待审批 #{rec['id']},请在界面的「待审批动作」里确认后重新发布。\n"
                f"拦截原因:{exc}"
            )
        return f"✅ 已发布:{out['result']['path']}({out['result']['bytes']} 字节,已记账)"

    tools = [
        analyze_reviews_tool,
        extract_keywords_tool,
        generate_listing_tool,
        translate_listing_tool,
        compliance_check_tool,
        publish_listing_tool,
    ]
    return create_react_agent(llm, tools, prompt=AGENT_SYSTEM)


def run_agent(csv_path: str, user_request: str, llm, my_product_specs: str = "") -> dict:
    """运行选品 + 上架 Agent,返回最终回答 + 工具调用过程。"""
    agent = build_agent(llm, my_product_specs)
    default_request = "请分析这些竞品的评论(卖点/痛点/关键词),然后生成我产品的 Listing 上架文案,并做合规审查。"
    prompt = f"竞品评论 CSV 路径:{csv_path}\n用户需求:{user_request or default_request}"
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


def run_agent_stream(
    csv_path: str,
    user_request: str,
    llm,
    on_step=None,
    my_product_specs: str = "",
    gate: "actions.ActionGate | None" = None,
    log: "events.EventLog | None" = None,
    run_id: str | None = None,
) -> dict:
    """流式运行 Agent,每一步通过 on_step(tool, result) 回调通知(用于实时展示)。

    on_step(tool_name, result):result 为 None 表示「开始调用工具」,有值表示「工具完成」。

    每个事件都带 seq 写进事件日志,返回里给回 run_id 与 last_seq —— 客户端刷新或断线后
    带上 last_event_id 重新拉,只会补到没收到的那部分。
    """
    gate = gate or actions.DEFAULT_GATE
    log = log or events.DEFAULT_LOG
    run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"

    agent = build_agent(llm, my_product_specs, gate=gate)
    default_request = "请分析这些竞品的评论(卖点/痛点/关键词),然后生成我产品的 Listing 上架文案,并做合规审查。"
    prompt = f"竞品评论 CSV 路径:{csv_path}\n用户需求:{user_request or default_request}"

    log.append(run_id, "run_started", {"csv_path": csv_path, "request": user_request})
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
                            step = steps_by_id[tc_id]
                            step["result"] = msg.content
                            log.append(run_id, "tool_result",
                                       {"tool": step["tool"], "tier": step["tier"],
                                        "chars": len(msg.content or "")})
                            if on_step:
                                on_step(step["tool"], msg.content)
            elif node == "agent":
                for msg in msgs:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            name = tc["name"]
                            step = {"tool": name, "args": tc.get("args", {}),
                                    "result": "", "tier": _declared_tier(name)}
                            steps_by_id[tc["id"]] = step
                            order.append(tc["id"])
                            log.append(run_id, "tool_call",
                                       {"tool": name, "tier": step["tier"]})
                            if on_step:
                                on_step(name, None)
                    else:
                        final_answer = msg.content

    steps = [steps_by_id[tid] for tid in order]
    log.append(run_id, "run_finished", {"steps": len(steps)})
    return {
        "answer": final_answer,
        "steps": steps,
        "run_id": run_id,
        "last_seq": log.last_seq(run_id),
    }


def _declared_tier(tool_name: str) -> str:
    """把工具名映射回它声明的动作分级;没声明的记 unknown(便于事后审计发现遗漏)。"""
    name = tool_name[:-5] if tool_name.endswith("_tool") else tool_name
    try:
        return actions.tier_of(name)
    except actions.UnknownAction:
        return "unknown"
