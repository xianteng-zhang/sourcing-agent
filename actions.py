"""动作分级:只读 / 可补编辑 / 外部副作用 —— 外部副作用必须人工批准。

**为什么要有这一层**

Agent 手里的工具不是平等的:

    只读        查评论、算关键词、跑合规检查 —— 跑一百遍没人受伤
    可补编辑    生成 Listing、导出报告 —— 写错了删掉重来,代价只是重跑
    外部副作用  把 Listing 发到平台、对外发送 —— 面向真实世界、不可撤销、可能花钱

把三者混成一类交给 LLM 自主调用,等于让模型自己决定要不要动真格。这一层把分级
**声明化**(每个动作声明自己的 tier),再由代码**确定性**地执行门禁:

    只读        → 直接执行
    可补编辑    → 直接执行,但结果标记为草稿、记审计
    外部副作用  → **没有有效批准记录就不执行**(fail-closed)

**关键在最后一条:默认拒绝,而不是默认放行。** 模型说「用户已经同意了」不算数 ——
批准只能由人通过审批入口写进库。而且批准是**绑定到具体内容**的(payload 哈希),
所以「批准了这一版 Listing」不能拿去发改过的另一版。

已知边界:真实平台发布接口需要 SP-API 之类的凭据,当前 EXTERNAL 动作落到本地
outbox 并记账,门禁本身是完整的。另外,绕过 `execute()` 直接调用底层函数仍然可能 ——
门禁的效力来自于「所有对外动作都从这里走」这个约定。
"""
import hashlib
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

DB_PATH = Path(__file__).parent / "data" / "actions.db"

# 三级动作
READ = "read"          # 只读
DRAFT = "draft"        # 可补编辑(产出草稿/本地文件)
EXTERNAL = "external"  # 外部副作用(对外发送、发布)

TIERS = (READ, DRAFT, EXTERNAL)

# 动作 → 分级。声明式:分级是代码里写死的,不由模型决定。
ACTIONS: dict[str, str] = {
    "analyze_reviews": READ,
    "extract_keywords": READ,
    "compliance_check": READ,
    "generate_listing": DRAFT,
    "translate_listing": DRAFT,
    "export_listing": DRAFT,
    "publish_listing": EXTERNAL,
}

DEFAULT_APPROVAL_TTL = 3600  # 批准 1 小时内有效


class UnknownAction(Exception):
    """动作没有声明分级 —— 拒绝执行,而不是默认放行。"""


class ApprovalRequired(Exception):
    """外部副作用动作缺少有效批准。"""


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tier_of(action: str) -> str:
    """查一个动作的分级。没声明的动作直接拒绝(白名单,不是黑名单)。"""
    tier = ACTIONS.get(action)
    if tier is None:
        raise UnknownAction(
            f"动作 {action!r} 没有声明分级;请在 actions.ACTIONS 里明确它的 tier"
        )
    return tier


def payload_hash(payload: Any) -> str:
    """把动作参数规范化后取哈希 —— 批准绑定到内容,换一版就失效。"""
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


class ActionGate:
    """审批 + 审计的存储与门禁。一个实例对应一个库文件(测试可传临时路径)。"""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DB_PATH

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approvals (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    action        TEXT NOT NULL,
                    payload_hash  TEXT NOT NULL,
                    payload_json  TEXT NOT NULL,
                    status        TEXT NOT NULL DEFAULT 'pending',
                    requested_by  TEXT NOT NULL,
                    requested_at  TEXT NOT NULL,
                    expires_at    REAL NOT NULL,
                    decided_by    TEXT,
                    decided_at    TEXT,
                    reason        TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS action_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    action      TEXT NOT NULL,
                    tier        TEXT NOT NULL,
                    decision    TEXT NOT NULL,
                    actor       TEXT NOT NULL,
                    approval_id INTEGER,
                    detail      TEXT,
                    created_at  TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------ 审批

    def request_approval(self, action: str, payload: Any, requested_by: str = "agent",
                         ttl_seconds: float = DEFAULT_APPROVAL_TTL) -> dict:
        """登记一条待审批。外部副作用动作执行前必须先有它。"""
        tier_of(action)
        self.init_db()
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO approvals (action, payload_hash, payload_json, status, "
                "requested_by, requested_at, expires_at) VALUES (?, ?, ?, 'pending', ?, ?, ?)",
                (action, payload_hash(payload), json.dumps(payload, ensure_ascii=False, default=str),
                 requested_by, _now_iso(), time.time() + ttl_seconds),
            )
            conn.commit()
            return self.get_approval(int(cur.lastrowid))
        finally:
            conn.close()

    def decide(self, approval_id: int, approved: bool, by: str = "human",
               reason: str = "") -> dict:
        """人工裁决。只有这里能把状态写成 approved —— 模型没有别的入口。"""
        self.init_db()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE approvals SET status = ?, decided_by = ?, decided_at = ?, reason = ? "
                "WHERE id = ? AND status = 'pending'",
                ("approved" if approved else "rejected", by, _now_iso(), reason, approval_id),
            )
            conn.commit()
        finally:
            conn.close()
        return self.get_approval(approval_id)

    def get_approval(self, approval_id: int) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_approvals(self, status: str | None = None, limit: int = 50) -> list[dict]:
        conn = self._connect()
        try:
            sql = "SELECT * FROM approvals"
            args: tuple = ()
            if status:
                sql += " WHERE status = ?"
                args = (status,)
            sql += " ORDER BY id DESC LIMIT ?"
            return [dict(r) for r in conn.execute(sql, (*args, limit)).fetchall()]
        finally:
            conn.close()

    # ------------------------------------------------------------ 审计

    def log(self, action: str, tier: str, decision: str, actor: str,
            approval_id: int | None = None, detail: str | None = None) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO action_log (action, tier, decision, actor, approval_id, detail, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (action, tier, decision, actor, approval_id, detail, _now_iso()),
            )
            conn.commit()
        finally:
            conn.close()

    def audit_log(self, limit: int = 50) -> list[dict]:
        conn = self._connect()
        try:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM action_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()]
        finally:
            conn.close()

    # ------------------------------------------------------------ 门禁

    def check(self, action: str, payload: Any, approval_id: int | None,
              now: float | None = None) -> tuple[bool, str]:
        """判定一个外部副作用动作能否执行。返回 (放行?, 原因)。

        默认拒绝。只有「已批准 + 动作一致 + 内容一致 + 未过期」四条全中才放行。
        """
        now = time.time() if now is None else now
        if approval_id is None:
            return False, "外部副作用动作必须携带 approval_id,当前没有"
        rec = self.get_approval(approval_id)
        if rec is None:
            return False, f"批准记录 #{approval_id} 不存在"
        if rec["status"] != "approved":
            return False, f"批准记录 #{approval_id} 当前状态是 {rec['status']},未获批准"
        if rec["action"] != action:
            return False, f"批准记录 #{approval_id} 是给 {rec['action']!r} 的,不能用于 {action!r}"
        if rec["payload_hash"] != payload_hash(payload):
            return False, "批准记录对应的内容与本次要执行的不一致(可能被改动过)"
        if rec["expires_at"] <= now:
            return False, "批准记录已过期"
        return True, "已获人工批准"

    def execute(self, action: str, payload: Any, run: Callable[[], Any],
                approval_id: int | None = None, actor: str = "agent",
                now: float | None = None) -> dict:
        """在门禁下执行一个动作。外部副作用没批准就抛 ApprovalRequired。"""
        tier = tier_of(action)  # 未声明的动作在这里就被拦下
        self.init_db()

        if tier == EXTERNAL:
            allowed, why = self.check(action, payload, approval_id, now=now)
            if not allowed:
                self.log(action, tier, "blocked", actor, approval_id, why)
                raise ApprovalRequired(f"{action}:{why}")

        result = run()
        self.log(action, tier, "allowed", actor, approval_id, None)
        return {
            "ok": True,
            "action": action,
            "tier": tier,
            "result": result,
            "draft": tier == DRAFT,   # 可补编辑:结果只是草稿,可改可删
            "approved_by_human": tier == EXTERNAL,
        }


DEFAULT_GATE = ActionGate()
