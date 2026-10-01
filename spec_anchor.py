"""规格锚点 —— 确定性校验 Listing 里的参数是否来自卖家填写的规格。

为什么需要这一层:让模型「不要编造参数」是一句**提示词**,不是**保证**。
合规那边我们信不过模型,所以做了确定性闸门(命中即短路、不经过模型);
参数这边同理 —— 从 Listing 里抽出所有数值型参数,逐个回查卖家规格,
查不到的直接判不合格,走**同一条**重写闭环。

刻意划清的能力边界(别把它当万能):
  ✅ 数值型参数编造(320g / 38x28x1.2cm / 1.55kg / 1920x1080)—— 最高危也最好核
  ❌ 定性描述编造(凭空多出「防水」「拉链」)—— 正则无能为力,仍只能靠提示词约束

判定用的是「数值」而不是「数值+单位」:规格里写 320 克、文案写 320g 应当放行,
单位同义词(g/克、cm/厘米、ml/毫升)不该造成误报。
"""
import re

from compliance import Violation

# 常见单位。单字母/短单位后面加 (?![A-Za-z]) 挡住英文单词:
# 既能校验 "320g" / "3.6GHz",又不会把 "5 min" 误当成 5m、"8GB" 当成 8g。
_UNITS = (
    r"mm|cm|dm|km|mg|kg|ml|cl|GB|TB|MB|KB"
    r"|mAh|kHz|MHz|GHz|Hz|dB|oz|lb|lbs|inch|ft"
    r"|℃|°C|℉|°F|W|V|A|L|g|m"
    r"|克|千克|公斤|厘米|毫米|毫升|升|英寸|寸|毫安|瓦|伏"
)

# 两种参数形态:
#   1) 数值 + 单位("320g" / "38x28x1.2cm")
#   2) 纯尺寸链,无单位但明显是尺寸("1920x1080" / "38x28x1.2")
_PARAM = re.compile(
    r"\d+(?:\.\d+)?(?:\s*[x×*]\s*\d+(?:\.\d+)?)*\s*(?:%|" + _UNITS + r")(?![A-Za-z])"
    r"|"
    r"\d+(?:\.\d+)?(?:\s*[x×*]\s*\d+(?:\.\d+)?)+",
    re.IGNORECASE,
)

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _normalize(num: str) -> str:
    """归一化数值:320.0 与 320 视为同一个,避免小数写法造成误报。"""
    value = float(num)
    return str(int(value)) if value == int(value) else str(value)


def _numbers_in(text: str) -> set[str]:
    return {_normalize(m.group(0)) for m in _NUMBER.finditer(text or "")}


def check_spec_anchor(listing_text: str, my_specs: str) -> list[Violation]:
    """校验 Listing 里的数值型参数能否在卖家规格里找到。

    返回命中的违规项(空列表 = 通过)。纯确定性,不调用任何模型。
    """
    if not listing_text:
        return []

    allowed = _numbers_in(my_specs)
    found: list[Violation] = []
    seen: set[str] = set()

    for m in _PARAM.finditer(listing_text):
        fragment = m.group(0).strip()
        for raw in _NUMBER.findall(fragment):
            num = _normalize(raw)
            if num in allowed or num in seen:
                continue
            seen.add(num)
            found.append(
                Violation(
                    "参数锚定",
                    fragment,
                    f"Listing 用到数值 {num}(出现在「{fragment}」),"
                    "但卖家规格里找不到 —— 要么是编造,要么是规格没填全,请核对后再上架",
                )
            )
    return found
