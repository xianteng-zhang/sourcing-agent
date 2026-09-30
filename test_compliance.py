"""合规闸门与重写闭环的测试。

这个文件存在的理由:合规是**两层**的,而两层都容易悄悄失效——
确定性闸门失效是「漏拦」(最危险),重写闭环失效是「空转」(最烧钱)。
两类都用测试钉住。

运行:pytest -q
"""
import pytest

from compliance import format_violations, scan, strip_commentary
from listing_generator import MAX_COMPLIANCE_RETRIES, check_compliance, generate_listing

# 一份真实风格、完全合规的 Listing —— 用作「不误报」的基线
CLEAN_LISTING = """# Elihome 木纤维复合砧板 — 亚马逊 Listing

## 一、标题
Elihome 木纤维复合砧板 带汁水槽 防滑橡胶脚 可进洗碗机 中号 38x28cm

## 二、五点描述
1. 不含微塑料:木纤维复合材质,切割食材时不产生塑料碎屑。
2. 防滑设计:四角橡胶脚固定台面,切剁时不易滑动。
3. 汁水槽:边缘导槽可容纳约 60ml 汤汁,减少台面溢洒。
4. 材质与手感:密度 1.1g/cm³,表面经打磨处理,刀具划过不易留深痕。
5. 随附与保养:随附说明书一份,建议每月涂抹食品级木蜡油一次。

## 三、规格参数
尺寸 38x28x1.2cm,重量 620g,材质 木纤维复合。
"""


class _ScriptedLLM:
    """按顺序返回预设文案;记录每轮收到的提示词,用于断言「违规项真的被喂回去了」。"""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        content = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]

        class _Resp:
            pass

        r = _Resp()
        r.content = content
        return r


# ------------------------------------------------------------ 确定性闸门

@pytest.mark.parametrize(
    "text,category",
    [
        ("本产品是全网最好的砧板", "绝对化用语"),
        ("现在下单包邮,限时折扣", "促销信息"),
        ("Top Rated kitchen essential", "主观夸赞"),
        ("提供终身保修,保证满意", "承诺性表述"),
        ("可消炎杀菌,辅助降血压", "医疗功效"),
        ("详情见 https://example.com", "联系方式与外链"),
        ("已通过 FDA 认证", "未授权认证"),
    ],
)
def test_scan_catches_each_category(text, category):
    hits = scan(text)
    assert category in {v.category for v in hits}, f"漏拦:{text!r}"


def test_scan_ignores_clean_listing():
    """闸门必须对正常文案放行 —— 否则重写闭环会无限空转。"""
    assert scan(CLEAN_LISTING) == []


def test_scan_does_not_flag_legitimate_material_spec():
    """刻意收窄的边界:「100% 棉」是合法参数,「100% 有效」才是违规。"""
    assert scan("面料成分为 100% 棉") == []
    assert scan("100% 有效") != []


def test_scan_does_not_flag_ce_inside_english_words():
    """裸写 CE 会误命中 CERTIFIED,所以用了 ASCII 字母边界。"""
    assert scan("CERTIFIED quality") == []
    assert scan("CE 认证") != []


def test_scan_dedups_repeated_hits():
    hits = scan("最好最好最好的砧板")
    assert len([v for v in hits if v.matched == "最好"]) == 1


def test_scan_handles_empty_input():
    assert scan("") == []
    assert scan(None) == []


# ------------------------------------------- 引用违禁词的「自检说明」不算违规

SELF_CHECK_RESPONSE = """# 亚马逊 Listing

## 一、标题
Elihome 木纤维复合砧板 带汁水槽 防滑橡胶脚 38x28cm

## 六、合规自检对照

| 红线项 | 本版处理 |
|---|---|
| 绝对化用语 | 未出现「最好/第一/顶级/唯一/爆款/热销」 |
| 促销信息 | 未出现价格、「限时」「包邮」「买2送1」 |
| 承诺性表述 | 未出现「终身保修」「保证」 |
"""


def test_scan_ignores_rule_quotation_in_self_check_table():
    """回归:模型附带的「合规自检」表会**引用**违禁词,不能当成真实违规。

    这是真机联调踩到的坑 —— 自检表让重写闭环永远无法收敛,
    每轮都白烧 3 倍 token,却始终报「未通过」。
    """
    assert scan(SELF_CHECK_RESPONSE) == []


def test_strip_commentary_keeps_real_listing_lines():
    text = "标题:木纤维砧板\n| 促销信息 | 未出现「包邮」 |\n五点描述:四角防滑橡胶脚"
    stripped = strip_commentary(text)
    assert "木纤维砧板" in stripped
    assert "四角防滑橡胶脚" in stripped
    assert "包邮" not in stripped


def test_scan_still_catches_violation_outside_commentary_lines():
    """剥注释不能把真实违规一起放过。"""
    hits = scan("标题:全网最好的砧板\n| 绝对化用语 | 未出现 |")
    assert {v.category for v in hits} == {"绝对化用语"}


def test_format_violations_is_readable():
    out = format_violations(scan("全网最低价,包邮"))
    assert "绝对化用语" in out and "促销信息" in out
    assert out.count("\n") == 1  # 两条命中,两行


# ------------------------------------------------------- 审查:命中即一票否决

class _ExplodingLLM:
    def invoke(self, messages):  # pragma: no cover - 被调用即失败
        raise AssertionError("确定性闸门命中时不应调用模型")


def test_check_compliance_short_circuits_without_calling_llm():
    verdict = check_compliance("本产品终身保修,绝对是最好的", llm=_ExplodingLLM())
    assert "未通过确定性合规闸门" in verdict
    assert "承诺性表述" in verdict and "绝对化用语" in verdict


def test_check_compliance_delegates_to_llm_when_clean():
    llm = _ScriptedLLM(["LLM 细查结论:未发现违规"])
    assert check_compliance(CLEAN_LISTING, llm=llm) == "LLM 细查结论:未发现违规"
    assert len(llm.prompts) == 1


# ------------------------------------------------------------ 生成重写闭环

def test_generate_listing_passes_on_first_attempt():
    llm = _ScriptedLLM([CLEAN_LISTING])
    result = generate_listing("不含微塑料 / 防滑", llm=llm)
    assert result["passed"] is True
    assert result["attempts"] == 1
    assert result["violations"] == []
    assert result["text"] == CLEAN_LISTING


def test_generate_listing_rewrites_until_gate_passes():
    bad = "# 标题\n全网最低价,包邮,终身保修"
    llm = _ScriptedLLM([bad, CLEAN_LISTING])
    result = generate_listing("防滑", llm=llm)

    assert result["attempts"] == 2
    assert result["passed"] is True
    assert result["text"] == CLEAN_LISTING


def test_generate_listing_feeds_violations_back_into_prompt():
    """闭环的关键不是重试次数,而是把命中项真的告诉模型。

    标记词要选**不在**基础提示词红线清单里的(「最好」「包邮」等本身就在清单里,
    断言「第一轮没有」会假失败)。
    """
    bad = "最佳品质,秒杀同行"
    llm = _ScriptedLLM([bad, CLEAN_LISTING])
    generate_listing("防滑", llm=llm)

    assert "上一版被确定性合规闸门拦下" not in llm.prompts[0]
    assert "上一版被确定性合规闸门拦下" in llm.prompts[1]
    assert "最佳" in llm.prompts[1]  # 命中项被回喂
    assert "秒杀" in llm.prompts[1]


def test_generate_listing_gives_up_after_max_retries():
    llm = _ScriptedLLM(["终身保修,保证满意"])
    result = generate_listing("防滑", llm=llm)

    assert result["attempts"] == MAX_COMPLIANCE_RETRIES
    assert len(llm.prompts) == MAX_COMPLIANCE_RETRIES
    assert result["passed"] is False
    assert {v.category for v in result["violations"]} == {"承诺性表述"}


def test_generate_listing_respects_custom_retry_budget():
    llm = _ScriptedLLM(["包邮"])
    result = generate_listing("防滑", llm=llm, max_retries=2)
    assert result["attempts"] == 2
    assert len(llm.prompts) == 2


# ------------------------------------------------- agent 层:步骤组装

class _Msg:
    def __init__(self, type, content="", tool_calls=None, tool_call_id=None):
        self.type = type
        self.content = content
        self.tool_calls = tool_calls or []
        self.tool_call_id = tool_call_id


class _FakeAgent:
    """伪造节点级事件流,验证 run_agent_stream 的步骤组装(不跑真实 Agent)。"""

    def stream(self, *_args, **_kwargs):
        yield {"agent": {"messages": [
            _Msg("ai", tool_calls=[{"id": "c1", "name": "generate_listing",
                                    "args": {"sell_points": "防滑"}}])
        ]}}
        yield {"tools": {"messages": [_Msg("tool", content="LISTING 文案", tool_call_id="c1")]}}
        yield {"agent": {"messages": [_Msg("ai", content="最终报告")]}}


def test_run_agent_stream_assembles_steps_in_order(monkeypatch):
    import agent as agent_mod

    monkeypatch.setattr(agent_mod, "build_agent", lambda llm: _FakeAgent())
    seen: list[tuple] = []

    result = agent_mod.run_agent_stream(
        "x.csv", "需求", llm=object(), on_step=lambda t, r: seen.append((t, r))
    )

    assert result["answer"] == "最终报告"
    assert len(result["steps"]) == 1
    step = result["steps"][0]
    assert step["tool"] == "generate_listing"
    assert step["args"] == {"sell_points": "防滑"}
    assert step["result"] == "LISTING 文案"
    # 回调:先「开始」(result=None),再「完成」
    assert seen == [("generate_listing", None), ("generate_listing", "LISTING 文案")]
