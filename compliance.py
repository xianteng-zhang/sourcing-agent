"""确定性合规闸门 —— 命中即判不合格,不经过模型。

为什么要有这一层:LLM 审查是**概率性**的,同一份文案多跑几次结论可能不同,
而平台红线是**硬性**的。所以拆成两层叠加:

  1. 本模块(确定性): 命中违禁词 / 促销语 / 绝对化用语 / 未授权认证
     → 直接判不合格,LLM 无权推翻,也**不会被调用**(省一次 token)
  2. compliance_check 里的 LLM 审查(概率性): 未命中时再逐条细查
     参数单位、标题格式、关键词堆砌、语言一致性等「模型更擅长」的软问题

这才是「一票否决 + 细查」——而不是让模型自己判断自己该不该被否决。

规则集刻意**收窄到行业公认的硬红线**,不做宽泛模糊匹配:过宽的规则会让正常
Listing 反复被拦、重写轮次空转。典型取舍见 `_RULES` 里 100% 的处理
(「100% 有效」拦,「100% 棉」不拦)——test_compliance.py 有用例钉住这些边界。
"""
import re
from typing import NamedTuple


class Violation(NamedTuple):
    """一条命中的违规项。"""

    category: str
    matched: str
    advice: str


# (类别, 正则, 修改建议)
_RULES: list[tuple[str, str, str]] = [
    (
        "绝对化用语",
        r"最好|最佳|最强|最优|最先进|第一品牌|全国第一|唯一|独家|顶级|顶尖"
        r"|国家级|世界级|全网最低|史上最低|销量冠军|绝无仅有",
        "删除绝对化表述,改为可核验的具体事实",
    ),
    (
        "促销信息",
        r"包邮|免邮|限时|秒杀|特价|清仓|买\s*\d+\s*送|满\s*\d+\s*减|立减"
        r"|折扣|促销价|￥\s*\d|¥\s*\d",
        "删除价格与促销信息 —— 亚马逊禁止在 Listing 中出现",
    ),
    (
        "主观夸赞",
        r"best\s*seller|hot\s*item|top\s*rated|amazon'?s\s*choice|爆款|热销|畅销|抢购",
        "删除平台主观夸赞词 —— 这由平台数据决定,不应由卖家自述",
    ),
    (
        "承诺性表述",
        r"终身保修|终身质保|永久保修|无效退款|保证|承诺\s*(?:有效|满意)"
        r"|100\s*%\s*(?:有效|保证|满意|安全|无风险)|百分之百",
        "删除承诺性表述,改为客观陈述随附物品与保修条款原文",
    ),
    (
        "医疗功效",
        r"治疗|治愈|根治|消炎|杀菌|降血压|降血糖|抗癌|抗肿瘤|疗效|药用",
        "删除医疗 / 功效声明 —— 非医疗器械不得宣称",
    ),
    (
        "联系方式与外链",
        r"https?://|www\.[a-z0-9-]|[\w.+-]+@[\w-]+\.[a-z]{2,}|二维码|微信号|加\s*V\s*[:：]",
        "删除联系方式、外链与二维码",
    ),
    (
        "未授权认证",
        # 用 ASCII 字母做边界:中文是 Unicode 词字符,`\b` 在「FDA认证」里不成立,
        # 而裸写 CE 会误命中 CERTIFIED 这类单词。
        r"(?<![A-Za-z])(?:FDA|CE|RoHS|FCC|3C|UL|ETL)(?![A-Za-z])",
        "认证宣称需具备对应证书,上架前须核实并留存检测报告",
    ),
]

_COMPILED = [(cat, re.compile(pat, re.IGNORECASE), adv) for cat, pat, adv in _RULES]

# 模型经常在 Listing 后面附一段「合规自检对照」表,里面会**引用**违禁词
# (例如「未出现『最好 / 第一 / 顶级』」)。逐词扫描会把这些**引用**当成真实违规,
# 于是重写闭环永远无法收敛 —— 每一轮都白烧 token,还始终报「不合格」。
# 下面这些词只出现在「讨论规则」的注释里,不会出现在真实的 Listing 文案中。
_COMMENTARY_MARKERS = re.compile(
    r"未出现|不得出现|禁止|红线|自检|对照表|已删除|已修正|修改说明|违规项|一票否决|合规检查"
)


def strip_commentary(text: str) -> str:
    """剥掉「讨论规则」的注释行,只留下真正的 Listing 内容。"""
    return "\n".join(
        line for line in text.splitlines() if not _COMMENTARY_MARKERS.search(line)
    )


def scan(text: str) -> list[Violation]:
    """扫描文案,返回命中的违规项。纯确定性,不调用任何模型。

    会先剥掉注释行(见 strip_commentary)—— 因为输入通常是「模型的完整回复」,
    而回复里可能带有引用违禁词的自检说明,那些不是真实违规。

    同一类别下重复出现的同一个词只报一次(否则「最好」写三遍就刷屏三条)。
    """
    if not text:
        return []

    text = strip_commentary(text)
    found: list[Violation] = []
    seen: set[tuple[str, str]] = set()
    for category, pattern, advice in _COMPILED:
        for m in pattern.finditer(text):
            key = (category, m.group(0).lower())
            if key in seen:
                continue
            seen.add(key)
            found.append(Violation(category, m.group(0), advice))
    return found


def format_violations(violations: list[Violation]) -> str:
    """把命中项渲染成可读清单(用于回喂给模型重写,以及展示给用户)。"""
    return "\n".join(
        f"- [{v.category}] 命中「{v.matched}」:{v.advice}" for v in violations
    )
