"""`python -m engines` —— 用录制样本跑一遍完整流程（不联网、不花钱）。

放在这里而不是 collect.py 的 `__main__`，是为了避免 `python -m engines.collect`
触发的 runpy 重复导入警告（包的 __init__ 已经导入了 collect）。
"""
from engines.collect import demo

demo()
