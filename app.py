"""选品调研 Agent — Streamlit 界面。

用法:streamlit run app.py
"""
import os
import tempfile
from datetime import datetime

import streamlit as st

import config
import storage
from competitor_compare import run_compare
from review_miner import run_review_mining
from trend_analyzer import get_trends, trend_summary

st.set_page_config(page_title="跨境电商选品调研 Agent", page_icon="🛒", layout="wide")

# ---- 侧边栏 ----
with st.sidebar:
    st.markdown("## 🛒 选品调研 Agent")
    st.markdown("跨境电商评论分析 · 竞品对比 · 趋势研判")
    st.divider()
    st.markdown("**功能**")
    st.markdown("- 📊 评论挖掘\n- ⚔️ 竞品对比\n- 📈 趋势分析\n- 📚 历史记录")
    st.divider()
    st.caption("数据来自商品评论 CSV,支持中英文、任意导出格式。")

st.title("🛒 跨境电商选品调研 Agent")
st.caption("评论挖掘 + 竞品对比 + 趋势分析,帮你看清「什么好卖、值不值得做」")

tab_mine, tab_compare, tab_trend, tab_history = st.tabs(["📊 评论挖掘", "⚔️ 竞品对比", "📈 趋势分析", "📚 历史记录"])

HOW_TO_CSV = """
**方法 1:Instant Data Scraper 插件(推荐,能自动抓)**
1. 浏览器(Edge/Chrome)安装 **Instant Data Scraper** 插件
2. 打开亚马逊商品的**评论页**(点「See all reviews / 查看所有评论」)
3. 往下滚动几屏,让评论加载出来
4. 点插件图标,弹出的预览**默认选中的往往不是评论**——点顶部的「**Try another table**」切换,直到预览变成「一条评论一行」(能看到评论文本)
5. 确认后点「Download CSV」导出

**方法 2:手动复制到 Excel(简单可靠)**
1. 打开商品评论页,复制 20~30 条评论
2. Excel 里做两列:`rating`(评分 1~5)、`review`(评论内容)
3. 「文件 → 另存为」选 **CSV UTF-8**

**方法 3:先用示例数据体验**
- 不传文件,勾选「使用示例数据」即可先看效果

> 💡 本工具会自动识别列名(中英文、任何导出格式都行),不用担心 CSV 格式。
"""

# ---------------- 评论挖掘 ----------------
with tab_mine:
    st.header("评论挖掘:从用户评论里挖出卖点、痛点、机会")
    st.write("上传评论 CSV,AI 会自动分析并生成选品报告。没有数据也可以勾选示例数据先体验。")

    with st.expander("📖 如何获取商品的评论 CSV?"):
        st.markdown(HOW_TO_CSV)

    uploaded = st.file_uploader("上传评论 CSV 文件", type=["csv"])
    use_sample = st.checkbox("使用示例数据(不传文件也能跑)", value=False)
    batch_size = st.slider(
        "每批分析的评论条数", 10, 50, 25,
        help="评论多时分批分析,每批多少条喂给 AI。数值小更精细,但更慢、更耗 token。",
    )
    title = st.text_input("分析名称(可选,方便在历史里查找)", placeholder="例如:FOGFIRE 慢跑裤")

    if st.button("开始分析", type="primary"):
        if uploaded is None and not use_sample:
            st.warning("请先上传 CSV 文件,或勾选「使用示例数据」。")
        else:
            try:
                llm = config.get_llm()
                if uploaded is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                        tmp.write(uploaded.getbuffer())
                        csv_path = tmp.name
                else:
                    csv_path = "data/sample_reviews.csv"

                progress_bar = st.progress(0.0, text="准备中…")

                def on_progress(done: int, total: int):
                    progress_bar.progress(min(done / total, 1.0), text=f"已分析 {done}/{total} 条")

                result = run_review_mining(csv_path, llm, batch_size=batch_size, progress=on_progress)
                progress_bar.progress(1.0, text="分析完成")

                stats = result["stats"]
                st.subheader("📋 评论概览")
                c1, c2, c3 = st.columns(3)
                c1.metric("评论总数", stats.get("total_reviews", 0))
                c2.metric("平均评分", stats.get("avg_rating", "—"))
                sentiment = stats.get("sentiment", {})
                c3.metric("好评条数", sentiment.get("positive", "—"))
                if sentiment:
                    st.write(
                        f"好评 {sentiment.get('positive', 0)} · "
                        f"中评 {sentiment.get('neutral', 0)} · "
                        f"差评 {sentiment.get('negative', 0)}"
                    )

                st.subheader("📝 选品分析报告")
                st.markdown(result["report"])

                # 保存到历史记录
                storage.save_analysis(
                    title=title or f"分析 {datetime.now():%m-%d %H:%M}",
                    review_count=result["review_count"],
                    avg_rating=stats.get("avg_rating"),
                    stats=stats,
                    report=result["report"],
                )
                st.success("已保存到「历史记录」")
            except Exception as exc:
                st.error(f"出错了:{exc}")
                if "DEEPSEEK_API_KEY" in str(exc):
                    st.info("请复制 .env.example 为 .env,并填入你的 DeepSeek API Key。")

# ---------------- 竞品对比 ----------------
with tab_compare:
    st.header("竞品对比:同时分析多个竞品,找出差异化机会")
    st.write("上传多个竞品的评论 CSV(每个文件一个竞品,文件名会作为竞品名)。")

    with st.expander("📖 如何获取商品的评论 CSV?"):
        st.markdown(HOW_TO_CSV)

    uploaded_files = st.file_uploader(
        "上传多个竞品的评论 CSV", type=["csv"], accept_multiple_files=True
    )

    if uploaded_files and st.button("开始对比", type="primary"):
        try:
            llm = config.get_llm()

            tmpdir = tempfile.mkdtemp()
            products = []
            for f in uploaded_files:
                name = os.path.splitext(f.name)[0]
                path = os.path.join(tmpdir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                products.append((name, path))

            progress_bar = st.progress(0.0, text="准备中…")

            def on_progress(done: int, total: int):
                progress_bar.progress(done / total, text=f"已分析 {done}/{total} 个竞品")

            result = run_compare(products, llm, progress=on_progress)
            progress_bar.progress(1.0, text="对比完成")

            st.subheader("📋 各竞品概览")
            for name, info in result["per_product"].items():
                s = info["stats"]
                c1, c2 = st.columns(2)
                c1.metric(f"{name} · 评论数", s.get("total_reviews", 0))
                c2.metric(f"{name} · 平均分", s.get("avg_rating", "—"))

            st.subheader("📝 竞品对比报告")
            st.markdown(result["report"])
        except Exception as exc:
            st.error(f"出错了:{exc}")
            if "DEEPSEEK_API_KEY" in str(exc):
                st.info("请复制 .env.example 为 .env,并填入你的 DeepSeek API Key。")

# ---------------- 趋势分析 ----------------
with tab_trend:
    st.header("趋势分析:关键词在 Google 的搜索热度走势")
    st.caption("数据来自 Google Trends(免费)。英文关键词效果最好,中文搜索量可能偏低。")

    kw_text = st.text_input(
        "关键词(逗号分隔,最多 5 个)",
        value="wireless earbuds, noise cancelling headphones",
    )
    timeframe = st.selectbox("时间范围", ["today 3-m", "today 12-m", "today 5-y"], index=1)

    if st.button("查询趋势", type="primary"):
        keywords = [k.strip() for k in kw_text.split(",") if k.strip()]
        if not keywords:
            st.warning("请输入至少一个关键词。")
        else:
            try:
                with st.spinner("正在查询 Google Trends…"):
                    df = get_trends(keywords, timeframe)
                if df is not None and not df.empty:
                    st.subheader("📈 热度走势")
                    st.line_chart(df)
                    st.subheader("🧭 趋势判断")
                    st.markdown(trend_summary(df))
                else:
                    st.info("未获取到趋势数据,可能是该关键词搜索量过低。")
            except Exception as exc:
                st.error(f"查询失败:{exc}")
                st.info("pytrends 偶尔会被 Google 限流,稍后重试,或换成英文关键词。")

# ---------------- 历史记录 ----------------
with tab_history:
    st.header("历史记录")
    st.caption("每次分析都会自动保存到这里,方便回看和对比。")

    rows = storage.list_history()
    if not rows:
        st.info("还没有分析记录。去「评论挖掘」跑一次,结果会自动保存到这里。")
    else:
        for r in rows:
            label = f"{r['title']} ｜ {r['review_count']} 条评论"
            if r.get("avg_rating") is not None:
                label += f" ｜ 均分 {r['avg_rating']}"
            label += f" ｜ {r['created_at']}"
            with st.expander(label):
                full = storage.get_analysis(r["id"])
                if full and full.get("report"):
                    st.markdown(full["report"])
                if st.button("删除这条", key=f"del_{r['id']}"):
                    storage.delete_analysis(r["id"])
                    st.rerun()
