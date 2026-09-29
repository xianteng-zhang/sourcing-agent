# 🛒 product-research — 跨境电商选品调研工具

基于大语言模型的跨境电商选品分析工具。从商品评论中自动挖掘**卖点、痛点和竞品差异**,辅助选品决策。

## 功能一览

| 功能 | 输入 | 输出 |
|------|------|------|
| 📊 **评论挖掘** | 单个商品的评论 CSV | 结构化选品报告(口碑 / 卖点 / 痛点 / 改进机会 / 竞品线索 / 选品结论) |
| ⚔️ **竞品对比** | 多个竞品的评论 CSV | 对比报告(概览表 / 卖点痛点横向对比 / 差异化机会 / 选品建议) |
| 📈 **趋势分析** | 关键词 | Google 搜索热度走势 + 上升/下降判断 |
| 📚 **历史记录** | — | 每次分析自动保存,可回看 / 删除 |

## 技术栈

Python · Streamlit · LangChain · DeepSeek · pandas · SQLite · pytrends

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

**方法 1:Instant Data Scraper 插件(推荐,能自动抓)**

1. 浏览器(Edge/Chrome)安装 **Instant Data Scraper** 插件
2. 打开亚马逊商品的**评论页**(点「See all reviews / 查看所有评论」)
3. 往下滚动几屏,让评论加载出来
4. 点插件图标,弹出的预览**默认选中的往往不是评论**——点顶部的「**Try another table**」切换,直到预览变成「一条评论一行」
5. 确认后点「Download CSV」导出

**方法 2:手动复制到 Excel(简单可靠)**

1. 打开商品评论页,复制 20~30 条评论
2. Excel 里做两列:`rating`(评分 1~5)、`review`(评论内容)
3. 「文件 → 另存为」选 **CSV UTF-8**

**方法 3:先用示例数据体验**

- 不传文件,界面上勾选「使用示例数据」即可先看效果(示例在 `data/sample_reviews.csv`)

> 💡 工具**不挑平台**:亚马逊、速卖通、Shopee、TikTok Shop……任何平台导出的评论 CSV 都能分析。

## CSV 格式(智能识别)

**列名无所谓**,工具会自动识别。支持:

- 标准列名:`review` / `comment` / `内容` / `评论`(正文)、`rating` / `score` / `评分`(评分)
- 插件导出的网页类名:`a-size-base`、`cr-original-review-content`、`a-icon-alt` 等
- 识别逻辑:先按列名匹配 → 匹配不上就按**内容特征**推断(哪列文本最长 = 评论、哪列值像评分 = 评分)

没有评分列也能跑,只是少了「平均分 / 好评占比」统计。

## 工作原理

**评论挖掘**是典型的 Map-Reduce:

1. 读 CSV(自动探测 UTF-8 / GBK 编码)
2. 把评论分成小批(默认每批 25 条),每批让 LLM 提取「卖点 / 痛点 / 购买动机 / 竞品提及」
3. 汇总所有批次结果,再让 LLM 生成最终 Markdown 报告

**竞品对比**:

1. 用 `ThreadPoolExecutor` **多线程并行**分析每个竞品(LLM 调用是 IO 密集,并行提速)
2. 全部完成后,由一次汇总调用生成差异化对比报告

## 测试

```bash
pip install -r requirements-dev.txt
pytest -q
```

14 个用例集中在 `test_review_miner.py`,只钉一件事:**脏 CSV 的列识别与评分解析**。
这是整个工具唯一直接面对不可控外部数据的地方——上游插件换一版导出格式,识别逻辑会
**静默失效**:不报错,报告照常生成,只是平均分和好评占比悄悄没了。这种失败必须靠测试兜住。

其中两条是真实踩坑后的回归用例:

- **纯数字评分列**:`_parse_rating` 原先只认 `5.0 out of 5 stars` / `4.0 颗星` 两种文本,
  而 `_detect_rating_column` 却把纯数字列也判定为评分列——于是评分列被「识别出来」却「解析成空」。
  自带示例数据正好走这条路径,概览卡片的平均分和好评占比全丢。
  现已补上纯数字分支,并与识别规则保持一致。
- **lightbox 干扰列**:只按「平均文本长度」挑评论列,会选中弹窗那种「行数少但单条极长」的列;
  `_detect_review_column` 的排序键必须是「非空行数优先、平均长度次之」,并过滤掉
  URL / HTML / CSS 片段占多数的候选列。

## 项目结构

```
app.py                 # Streamlit 界面(四个标签页)
config.py              # DeepSeek 配置
prompts.py             # 提示词(评论分析 / 汇总 / 竞品对比)
review_miner.py        # 评论挖掘:CSV 解析 + 分批 LLM + 智能列识别
competitor_compare.py  # 竞品对比:多线程并行 + 汇总
trend_analyzer.py      # 趋势分析:pytrends
storage.py             # 历史记录:SQLite 单文件
verify.py              # 命令行验证(不开界面也能测)
test_review_miner.py   # 单元测试:列识别 / 评分解析 / 分批抽样
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
