"""动作分级与审批门禁的测试。

这个文件要钉住的核心只有一句话:**外部副作用动作默认拒绝**。
其余用例都在证明这个「拒绝」是有牙齿的 —— 不能靠模型自称已获同意绕过,
不能拿别的动作的批准来顶,不能拿改过的内容复用旧批准,过期也不算。
"""
import pytest

from actions import (
    ACTIONS,
    DRAFT,
    EXTERNAL,
    READ,
    ActionGate,
    ApprovalRequired,
    UnknownAction,
    payload_hash,
    tier_of,
)


@pytest.fixture()
def gate(tmp_path):
    return ActionGate(tmp_path / "actions.db")


def _ok():
    return {"published": True}


# ------------------------------------------------------------ 分级声明

def test_every_declared_action_has_a_known_tier():
    assert ACTIONS, "分级表不能是空的"
    for name, tier in ACTIONS.items():
        assert tier in (READ, DRAFT, EXTERNAL), f"{name} 的 tier {tier!r} 不是三级之一"


def test_the_three_tiers_are_all_used():
    """三级都要有真实动作,否则这个模式在这项目里是空谈。"""
    used = set(ACTIONS.values())
    assert used == {READ, DRAFT, EXTERNAL}


def test_unknown_action_is_refused():
    """白名单而不是黑名单:没声明的动作拒绝执行,不是默认放行。"""
    with pytest.raises(UnknownAction):
        tier_of("delete_everything")


def test_publish_listing_is_the_external_one():
    assert tier_of("publish_listing") == EXTERNAL
    assert tier_of("analyze_reviews") == READ
    assert tier_of("generate_listing") == DRAFT


# ------------------------------------------------- 只读 / 可补编辑:直接放行

def test_read_action_runs_without_approval(gate):
    out = gate.execute("analyze_reviews", {"csv": "a.csv"}, _ok)
    assert out["ok"] is True
    assert out["tier"] == READ
    assert out["result"] == {"published": True}


def test_draft_action_runs_but_is_marked_draft(gate):
    """可补编辑:能直接跑,但结果必须被标记成草稿 —— 它还不是对外的东西。"""
    out = gate.execute("generate_listing", {"insights": "x"}, _ok)
    assert out["draft"] is True
    assert out["approved_by_human"] is False


# ------------------------------------------- 外部副作用:默认拒绝

def test_external_action_blocked_without_approval(gate):
    """核心用例:没有批准就绝不能执行,而不是「先跑了再说」。"""
    calls = []

    with pytest.raises(ApprovalRequired):
        gate.execute("publish_listing", {"text": "listing"},
                     lambda: calls.append(1) or _ok())

    assert calls == [], "被拦截的动作绝不能被执行到"


def test_external_action_blocked_while_still_pending(gate):
    """登记了审批但人还没批 —— 依然拒绝。登记不等于同意。"""
    rec = gate.request_approval("publish_listing", {"text": "listing"})
    with pytest.raises(ApprovalRequired):
        gate.execute("publish_listing", {"text": "listing"}, _ok, approval_id=rec["id"])


def test_external_action_blocked_when_rejected(gate):
    rec = gate.request_approval("publish_listing", {"text": "listing"})
    gate.decide(rec["id"], approved=False, reason="文案还没定稿")
    with pytest.raises(ApprovalRequired):
        gate.execute("publish_listing", {"text": "listing"}, _ok, approval_id=rec["id"])


def test_external_action_runs_after_human_approval(gate):
    payload = {"text": "listing"}
    rec = gate.request_approval("publish_listing", payload, requested_by="agent")
    gate.decide(rec["id"], approved=True, by="zhang")

    out = gate.execute("publish_listing", payload, _ok, approval_id=rec["id"])

    assert out["ok"] is True
    assert out["approved_by_human"] is True
    assert out["draft"] is False


# ------------------------------------------------- 批准绑定到内容与动作

def test_approval_is_bound_to_payload(gate):
    """批准的是「这一版文案」。改过之后必须重新批 —— 否则门禁形同虚设。"""
    rec = gate.request_approval("publish_listing", {"text": "第一版"})
    gate.decide(rec["id"], approved=True)

    with pytest.raises(ApprovalRequired):
        gate.execute("publish_listing", {"text": "偷偷改成第二版"}, _ok,
                     approval_id=rec["id"])


def test_approval_is_bound_to_action(gate):
    rec = gate.request_approval("publish_listing", {"text": "x"})
    gate.decide(rec["id"], approved=True)

    with pytest.raises(UnknownAction):
        gate.execute("delete_listing", {"text": "x"}, _ok, approval_id=rec["id"])


def test_expired_approval_is_refused(gate):
    rec = gate.request_approval("publish_listing", {"text": "x"}, ttl_seconds=1)
    gate.decide(rec["id"], approved=True)

    with pytest.raises(ApprovalRequired):
        gate.execute("publish_listing", {"text": "x"}, _ok,
                     approval_id=rec["id"], now=rec["expires_at"] + 1)


def test_payload_hash_ignores_key_order():
    assert payload_hash({"a": 1, "b": 2}) == payload_hash({"b": 2, "a": 1})
    assert payload_hash({"a": 1}) != payload_hash({"a": 2})


# ------------------------------------------------------------ 审计

def test_blocked_attempt_is_audited(gate):
    with pytest.raises(ApprovalRequired):
        gate.execute("publish_listing", {"text": "x"}, _ok)

    log = gate.audit_log()
    assert log[0]["action"] == "publish_listing"
    assert log[0]["decision"] == "blocked"
    assert log[0]["tier"] == EXTERNAL
    assert "approval_id" in log[0]["detail"] or "批准" in log[0]["detail"]


def test_allowed_attempt_is_audited_with_approval_id(gate):
    rec = gate.request_approval("publish_listing", {"text": "x"})
    gate.decide(rec["id"], approved=True)
    gate.execute("publish_listing", {"text": "x"}, _ok, approval_id=rec["id"])

    log = gate.audit_log()
    assert log[0]["decision"] == "allowed"
    assert log[0]["approval_id"] == rec["id"]


def test_draft_and_read_attempts_are_audited_too(gate):
    gate.execute("analyze_reviews", {}, _ok)
    gate.execute("generate_listing", {}, _ok)
    decisions = {r["action"]: r["decision"] for r in gate.audit_log()}
    assert decisions == {"analyze_reviews": "allowed", "generate_listing": "allowed"}


def test_audit_records_rejection_reason_on_approval(gate):
    rec = gate.request_approval("publish_listing", {"text": "x"})
    gate.decide(rec["id"], approved=False, by="zhang", reason="价格还没核对")
    got = gate.get_approval(rec["id"])
    assert got["status"] == "rejected"
    assert got["reason"] == "价格还没核对"


def test_list_pending_approvals(gate):
    gate.request_approval("publish_listing", {"text": "a"})
    gate.request_approval("publish_listing", {"text": "b"})
    assert len(gate.list_approvals(status="pending")) == 2
