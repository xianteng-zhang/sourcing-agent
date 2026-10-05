# 🛒 sourcing-agent — 跨境电商选品 + 上架 Agent

[![CI](https://github.com/xianteng-zhang/sourcing-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/xianteng-zhang/sourcing-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)

基于 LangGraph + 大语言模型的跨境电商选品 + 上架 Agent。从商品评论中自动挖掘**卖点、痛点、关键词和竞品差异**,并生成合规的多语言 Listing 上架文案。

## 功能一览

| 功能 | 输入 | 输出 |
|------|------|------|
| 🤖 **选品 Agent** | 竞品评论 CSV + 卖家产品规格 | LLM 自主决定调用 5 个工具中的哪些、什么顺序(实测支持并行调用);两类确定性闸门把关**违禁词**与**编造参数**,产出可上架 Listing |
| 📊 **评论挖掘** | 单个商品的评论 CSV | 结构化选品报告(官方星级分布 / 口碑 / 卖点 / 痛点 / 关键词 / 竞品线索 / 选品结论) |
| ⚔️ **竞品对比** | 多个竞品的评论 CSV | 对比报告(概览表 / 卖点痛点横向对比 / 差异化机会 / 选品建议) |
| 📈 **趋势分析** | 关键词 | Google 搜索热度走势 + 上升/下降判断 |
| 📚 **历史记录** | — | 每次分析自动保存,可回看 / 删除 |

## 技术栈

Python · Streamlit · LangChain · LangGraph · DeepSeek · pandas · SQLite · pytrends · JavaScript(浏览器扩展)

## 快速开始

```bash
# 1. 创建虚拟环境(Windows)
python -m venv .venv
.venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key:复制 .env.example 为 .env,填入 DeepSeek Key
copy .env.example .env

# 4. 启动
streamlit run app.py
```

浏览器会自动打开 http://localhost:8501

## 如何获取评论 CSV

**方法 1:自研评论抓取扩展(推荐,一键导出,支持亚马逊 + 速卖通)**

1. 在界面左侧边栏**下载扩展 zip** 并解压
2. Edge 打开 `edge://extensions/` → 开「开发人员模式」→「加载解压缩的扩展」→ 选解压出的文件夹
3. 亚马逊:打开评论页,往下滚动加载评论;速卖通:打开商品页即可
4. 点扩展图标 →「抓取当前页评论 → CSV」,自动下载

**方法 2:Instant Data Scraper 插件(通用备选)**

1. 浏览器(Edge/Chrome)安装 **Instant Data Scraper** 插件
2. 打开亚马逊商品的**评论页**(点「See all reviews / 查看所有评论」)
3. 往下滚动几屏,让评论加载出来
4. 点插件图标,弹出的预览**默认选中的往往不是评论**——点顶部的「**Try another table**」切换,直到预览变成「一条评论一行」
5. 确认后点「Download CSV」导出

**方法 3:手动复制到 Excel(简单可靠)**

1. 打开商品评论页,复制 20~30 条评论
2. Excel 里做两列:`rating`(评分 1~5)、`review`(评论内容)
3. 「文件 → 另存为」选 **CSV UTF-8**

**方法 4:先用示例数据体验**

- 不传文件,界面上勾选「使用示例数据」即可先看效果(示例在 `data/sample_reviews.csv`)

> 💡 工具**不挑平台**:亚马逊、速卖通、Shopee、TikTok Shop……任何平台导出的评论 CSV 都能分析。

## CSV 格式(智能识别)

**列名无所谓**,工具会自动识别。支持:

- 标准列名:`review` / `comment` / `内容` / `评论`(正文)、`rating` / `score` / `评分`(评分)
- 插件导出的网页类名:`a-size-base`、`cr-original-review-content`、`a-icon-alt` 等
- 识别逻辑:先按列名匹配 → 匹配不上就按**内容特征**推断(哪列文本最长 = 评论、哪列值像评分 = 评分)

没有评分列也能跑,只是少了「平均分 / 好评占比」统计。

> 自研扩展抓取的 CSV 第一行会带 `# 平均分=… 总评数=… 5星=…` 这类**官方星级分布元数据**,工具会自动识别并在报告里优先采信(基于全部评论,比样本更权威)。

## 工作原理

**评论挖掘**是典型的 Map-Reduce:

1. 读 CSV(自动探测 UTF-8 / GBK 编码)
2. 把评论分成小批(默认每批 25 条),每批让 LLM 提取「卖点 / 痛点 / 购买动机 / 竞品提及」
3. 汇总所有批次结果,再让 LLM 生成最终 Markdown 报告

**竞品对比**:

1. 用 `ThreadPoolExecutor` **多线程并行**分析每个竞品(LLM 调用是 IO 密集,并行提速)
2. 全部完成后,由一次汇总调用生成差异化对比报告

**选品 Agent**(LangGraph ReAct):

1. LLM 绑定 **5 个工具**:竞品评论挖掘 / 关键词挖掘 / Listing 生成 / 多语言翻译 / 合规审查
2. LLM **自主规划**调用哪些工具、什么顺序,并**流式执行**,每步实时展示。
   实测一次真实运行:第 1 轮**并行**调用 `analyze_reviews` + `extract_keywords`
   (5 个工具里只有这两个彼此无依赖),之后串行 `generate_listing` → `compliance_check`。
3. **两类确定性闸门**先于模型判定,命中即判不合格、**不调用模型**、LLM 无权推翻:
   - `compliance.py`:违禁词 / 促销语 / 绝对化用语 / 未授权认证
   - `spec_anchor.py`:**参数锚定** —— Listing 里的数值型参数必须能在卖家填写的规格里
     找到,查不到即为候选编造(规格写 320 克、文案写 320g 不算违规)
   两类都通过后,才交给 LLM 细查参数单位 / 标题格式 / 关键词堆砌。
   `generate_listing` 以其构成「生成 → 拦截 → 重写」闭环,最多 3 轮。
4. **卖家规格通过闭包注入工具,不经过模型中转** —— 否则模型漏抄或截断规格时不会报错,
   只会安静地产出一份全是【待填写】的 Listing。

## 测试

```bash
pip install -r requirements-dev.txt
pytest -q
```

47 个用例分两个文件,各钉一类**静默失效** —— 不报错、照常出结果、只是悄悄错了。

**`test_review_miner.py`(14 例):脏 CSV 的列识别与评分解析。**
这是整个工具唯一直接面对不可控外部数据的地方,上游插件换一版导出格式,识别逻辑就会失效。

- **纯数字评分列**:`_parse_rating` 原先只认 `5.0 out of 5 stars` / `4.0 颗星` 两种文本,
  而 `_detect_rating_column` 却把纯数字列也判定为评分列——于是评分列被「识别出来」却「解析成空」。
  自带示例数据正好走这条路径,概览卡片的平均分和好评占比全丢。
- **lightbox 干扰列**:只按「平均文本长度」挑评论列,会选中弹窗那种「行数少但单条极长」的列;
  `_detect_review_column` 的排序键必须是「非空行数优先、平均长度次之」。

**`test_compliance.py`(33 例):两类确定性闸门 + 重写闭环。**
闸门失效是「漏拦」(最危险),闭环失效是「空转」(最烧钱),两类都钉住。

- **误报边界**:`100% 棉` 是合法参数,`100% 有效` 才是违规;裸写 `CE` 会误命中 `CERTIFIED`,
  所以认证类规则用 ASCII 字母边界而非 `\b` —— 中文是 Unicode 词字符,`\b` 在「FDA认证」里根本不成立。
- **自检表引用违禁词**(真机联调踩到):模型会在 Listing 后附一张「合规自检对照」表,
  里面**引用**「最好 / 包邮 / 终身保修」来说明「未出现」。逐词扫描会把这些**引用**当成真实违规,
  **重写闭环于是永远无法收敛** —— 每轮白烧 3 倍 token,还始终报不合格。
  修复后同一用例从 **3 轮 15.6s 不通过** 降到 **1 轮 4.2s 通过**。
- **参数锚定的误报边界**:规格写「320 克」而文案写「320g」必须放行(所以判定比的是
  **数值**而不是「数值+单位」);`5 min` 不能被当成 `5m`、`8GB` 不能被当成 `8g`
  (单字母单位后面加了 ASCII 字母边界);全是【待填写】的文案是正确的,不能拦。
- **数值比对必须归一化**:规格写 `320`、文案写 `320.0g` 是同一件事。

## 项目结构

```
app.py                 # Streamlit 界面(五个标签页,含 Agent 流式展示)
config.py              # DeepSeek 配置
agent.py               # 选品+上架 Agent:LangGraph ReAct + 5 工具 + 流式执行
prompts.py             # 提示词(评论 / 汇总 / 竞品 / Listing / 合规 / 关键词 / 翻译)
review_miner.py        # 评论挖掘:CSV 解析 + 分批 LLM + 智能列识别 + 星级分布
listing_generator.py   # Listing 生成(含确定性重写闭环)/ 合规审查 / 关键词 / 翻译
compliance.py          # 确定性闸门之一:违禁词规则表 + 注释行剥离
spec_anchor.py         # 确定性闸门之二:参数锚定(数值必须来自卖家规格)
competitor_compare.py  # 竞品对比:多线程并行 + 汇总
trend_analyzer.py      # 趋势分析:pytrends
storage.py             # 历史记录:SQLite 单文件,分类存储(评论/竞品/Agent)
verify.py              # 命令行验证(不开界面也能测)
test_review_miner.py   # 单元测试:列识别 / 评分解析 / 分批抽样
test_compliance.py     # 单元测试:两类闸门 / 误报边界 / 重写闭环 / 步骤组装
extension/             # 浏览器扩展 zip(评论抓取 + 星级分布)
data/sample_reviews.csv # 示例评论数据
.streamlit/config.toml  # 界面主题配色
```

## 成本提示

- 评论分批喂给 LLM,数量越多 token 消耗越大。默认**最多分析 300 条**(超出均匀抽样)。
- 想省钱:`.env` 里把模型改成 `deepseek-flash`(更快更便宜)。

## 常见坑

- **趋势分析需要能访问 Google**:`pytrends` 访问 Google Trends,国内需要系统级代理;代理关了趋势会查不到,但评论挖掘和历史记录不受影响。
- **中文关键词趋势**:Google 上中文搜索量偏低,趋势图可能很平,英文关键词效果更好。
- **pip 安装慢**:清华镜像对部分包返回 403,可用阿里云镜像:`pip install -i https://mirrors.aliyun.com/pypi/simple/`。

## 界面主题

配色在 `.streamlit/config.toml` 里,默认「蓝色电商风」。改几行色值即可换风格:

```toml
[theme]
primaryColor = "#2563EB"   # 主色
backgroundColor = "#F8FAFC" # 页面背景
textColor = "#0F172A"      # 文字
```
