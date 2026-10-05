"""选品与上架 Agent — Streamlit 界面。

用法:streamlit run app.py
"""
import json
import os
import tempfile
from datetime import datetime

import streamlit as st

import config
import storage
from agent import run_agent_stream, write_outbox
from competitor_compare import run_compare
from review_miner import run_review_mining
from trend_analyzer import get_trends, trend_summary

import actions
import events

st.set_page_config(page_title="跨境电商选品与上架 Agent", page_icon="🛒", layout="wide")

# ---- 侧边栏 ----
with st.sidebar:
    st.markdown("## 🛒 选品与上架 Agent")
    st.markdown("选品 Agent · 评论分析 · 竞品对比 · 趋势研判")
    st.divider()
    st.markdown("**功能**")
    st.markdown("- 🤖 选品 Agent\n- 📊 评论挖掘\n- ⚔️ 竞品对比\n- 📈 趋势分析\n- 📚 历史记录")
    st.divider()
    st.markdown("**🔌 评论抓取扩展**")
    st.markdown("自研浏览器扩展,自动抓取亚马逊 / 速卖通评论导出 CSV。")
    zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extension", "amazon-review-scraper.zip")
    if os.path.exists(zip_path):
        with open(zip_path, "rb") as f:
            st.download_button(
                "⬇️ 下载扩展(zip)",
                f.read(),
                file_name="amazon-review-scraper.zip",
                mime="application/zip",
            )
    st.caption("用法:解压后,Edge 打开 edge://extensions/ → 开「开发人员模式」→「加载解压缩的扩展」→ 选解压出的文件夹。")

    st.divider()
    st.caption("数据来自商品评论 CSV,支持中英文、任意导出格式。")

    # 外部副作用动作的**唯一**放行入口 —— 模型没有别的路能把自己的提议批掉
    gate = actions.DEFAULT_GATE
    pending = gate.list_approvals(status="pending")
    approved = gate.list_approvals(status="approved")
    executed = {r["approval_id"] for r in gate.audit_log(limit=200)
                if r["decision"] == "allowed"}

    if pending or approved:
        st.divider()

    for rec in pending:
        st.markdown(f"**⏳ 待审批** `#{rec['id']}` · {rec['action']}")
        body = json.loads(rec["payload_json"]).get("listing_text", "")
        with st.expander("查看要发布的内容"):
            st.text(body[:500] + ("…" if len(body) > 500 else ""))
        c_ok, c_no = st.columns(2)
        if c_ok.button("✅ 批准", key=f"appr_{rec['id']}"):
            gate.decide(rec["id"], approved=True, by="user")
            st.rerun()
        if c_no.button("⛔ 拒绝", key=f"rej_{rec['id']}"):
            gate.decide(rec["id"], approved=False, by="user", reason="界面拒绝")
            st.rerun()

    for rec in approved:
        if rec["id"] in executed:
            continue  # 已经发布过的就不再给了
        st.markdown(f"**✅ 已批准** `#{rec['id']}` · {rec['action']}")
        if st.button("🚀 发布这条(执行已批准的内容)", key=f"pub_{rec['id']}"):
            payload = json.loads(rec["payload_json"])
            try:
                out = gate.execute(
                    "publish_listing", payload,
                    lambda: write_outbox(payload.get("listing_text", "")),
                    approval_id=rec["id"], actor="user",
                )
                st.success(f"已发布:{out['result']['path']}")
            except Exception as exc:  # noqa: BLE001 - 门禁拒绝时把原因显示出来
                st.error(f"发布失败:{exc}")

st.title("🛒 跨境电商选品与上架 Agent")
st.caption("选品 Agent + 评论挖掘 + 竞品对比 + 趋势分析,帮你看清「什么好卖、值不值得做」")

tab_mine, tab_agent, tab_compare, tab_trend, tab_history = st.tabs(["📊 评论挖掘", "🤖 选品 Agent", "⚔️ 竞品对比", "📈 趋势分析", "📚 历史记录"])

HOW_TO_CSV = """
**方法 1:自研评论抓取扩展(推荐,一键导出)**
1. 在左侧边栏**下载扩展 zip** 并解压
2. Edge 打开 `edge://extensions/` → 开「开发人员模式」→「加载解压缩的扩展」→ 选解压出的文件夹
3. 打开亚马逊评论页,往下滚动加载评论
4. 点扩展图标 →「抓取当前页评论 → CSV」,自动下载

**方法 2:Instant Data Scraper 插件(通用备选)**
1. 浏览器(Edge/Chrome)安装 **Instant Data Scraper** 插件
2. 打开亚马逊评论页,往下滚动加载评论
3. 点插件图标 → 弹出的预览默认不是评论,点「**Try another table**」切换到评论列表
4. 点「Download CSV」导出

**方法 3:手动复制到 Excel(简单可靠)**
1. 复制 20~30 条评论
2. Excel 两列:`rating`(评分 1~5)、`review`(评论内容)
3. 另存为 **CSV UTF-8**

**方法 4:先用示例数据体验**
- 不传文件,勾选「使用示例数据」即可

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
                    type="评论挖掘",
                )
                st.success("已保存到「历史记录」")
            except Exception as exc:
                st.error(f"出错了:{exc}")
                if "DEEPSEEK_API_KEY" in str(exc):
                    st.info("请复制 .env.example 为 .env,并填入你的 DeepSeek API Key。")

# ---------------- 选品 Agent ----------------
with tab_agent:
    st.header("选品 Agent:竞品评论 → 我的 Listing")
    st.write("上传**竞品**的评论 CSV,Agent 分析竞品的卖点/痛点/关键词,结合你填的产品规格,生成**你产品**的 Listing。")

    agent_csv = st.file_uploader("上传竞品评论 CSV", type=["csv"], key="agent_csv")

    st.markdown("**我的产品规格**(可选,硬参数,不填则对应处标【待填写】)")
    sc1, sc2 = st.columns(2)
    with sc1:
        spec_category = st.text_input("品类", placeholder="例如:男士慢跑裤 / 笔记本电脑")
        spec_material = st.text_input("材质", placeholder="例如:92%棉 8%氨纶")
    with sc2:
        spec_weight = st.text_input("重量", placeholder="例如:320g / 1.55千克")
        spec_cert = st.text_input("认证", placeholder="例如:FDA / 能源之星 / OEKO-TEX")
    spec_detail = st.text_area(
        "完整技术规格(可选,可粘贴详细参数,每行「参数名:参数值」)",
        placeholder="处理器:Intel N150(4核,3.6GHz)\n内存:8GB LPDDR5\n显示器:15.6英寸 FHD 1920x1080\n无线:Wi-Fi 6,蓝牙5.2\n端口:2x USB-A,1x USB-C,1x HDMI\n尺寸:359.2 x 234 x 17.9 毫米\n颜色:霜蓝色",
        height=160,
    )

    user_request = st.text_input(
        "你的需求(可选,留空则用默认流程)",
        placeholder="例如:重点分析竞品差评原因,并生成日语版 Listing",
    )

    if agent_csv and st.button("🚀 启动 Agent", type="primary"):
        try:
            llm = config.get_llm()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(agent_csv.getbuffer())
                csv_path = tmp.name

            # 实时展示工具调用过程
            process_box = st.empty()
            process_lines = []

            def on_step(tool, result_text):
                if result_text is None:
                    process_lines.append(f"🛠️ 正在调用 `{tool}` …")
                else:
                    process_lines.append(f"✅ `{tool}` 完成")
                process_box.markdown("\n\n".join(process_lines))

            spec_parts = []
            for label, val in [
                ("品类", spec_category), ("材质", spec_material),
                ("重量", spec_weight), ("认证", spec_cert),
            ]:
                if val and val.strip():
                    spec_parts.append(f"{label}:{val.strip()}")
            if spec_detail and spec_detail.strip():
                spec_parts.append("详细规格:\n" + spec_detail.strip())
            my_specs = "\n".join(spec_parts)

            # 规格通过闭包注入 Agent,不拼进对话 —— 避免模型漏抄/截断后静默降级
            result = run_agent_stream(csv_path, user_request, llm, on_step, my_specs)

            st.caption(
                f"运行 `{result['run_id']}` ｜ 事件已落库到 seq **{result['last_seq']}**"
                " —— 断线或刷新后，下面的回放只会补你没收到的那部分。"
            )

            # 完整过程(含每步详情,折叠)
            if result.get("steps"):
                with st.expander("🔍 查看完整工作过程(含每步参数与结果)"):
                    for i, step in enumerate(result["steps"], 1):
                        st.markdown(
                            f"**步骤 {i}:`{step['tool']}`** ｜ 动作分级 `{step.get('tier', '?')}`"
                        )
                        if step.get("args"):
                            st.caption(f"参数:{step['args']}")
                        if step.get("result"):
                            st.text(step["result"][:600] + ("…" if len(step["result"]) > 600 else ""))

            with st.expander("🧵 事件回放(Last-Event-ID:只补没收到的那部分)"):
                cursor = st.number_input(
                    "只显示 seq 大于", min_value=0, max_value=int(result["last_seq"]),
                    value=0, key="replay_cursor",
                )
                replayed = events.DEFAULT_LOG.events_since(result["run_id"], int(cursor))
                if replayed:
                    for ev in replayed:
                        st.text(f"#{ev['seq']:>3}  {ev['type']:<13} {ev['data']}")
                else:
                    st.caption("没有新事件(游标已经是最新的)。")

            st.subheader("📝 最终报告")
            st.markdown(result["answer"])

            # 保存到历史记录
            storage.save_analysis(
                title=user_request or "选品 Agent 默认流程",
                review_count=None,
                avg_rating=None,
                stats={},
                report=result["answer"],
                type="选品Agent",
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

            # 保存到历史记录
            names = " vs ".join(result["per_product"].keys())
            total = sum(info["stats"].get("total_reviews", 0) for info in result["per_product"].values())
            storage.save_analysis(
                title=f"竞品对比:{names}",
                review_count=total,
                avg_rating=None,
                stats={},
                report=result["report"],
                type="竞品对比",
            )
            st.success("已保存到「历史记录」")
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
        # 按类型筛选
        all_types = ["全部"] + sorted({r.get("type", "评论挖掘") for r in rows})
        selected = st.radio("按类型筛选", all_types, horizontal=True)
        filtered = rows if selected == "全部" else [r for r in rows if r.get("type") == selected]

        for r in filtered:
            label = f"[{r.get('type', '评论挖掘')}] {r['title']}"
            if r.get("review_count"):
                label += f" ｜ {r['review_count']} 条评论"
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
