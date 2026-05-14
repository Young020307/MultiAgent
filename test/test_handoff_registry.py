# test/test_handoff_registry.py
import sys
sys.path.insert(0, "/home/neousys/桌面/MultiAgent")

from core.handoff_registry import register, unregister, get_all, get, update


def test_register_and_get():
    register("thread_1", {
        "conversation_id": "abc12345",
        "user_msg": "我要投诉",
        "escalate_reason": "user_requested",
        "sentiment": "critical",
    })
    info = get("thread_1")
    assert info is not None
    assert info["conversation_id"] == "abc12345"
    assert info["escalate_reason"] == "user_requested"
    assert "timestamp" in info


def test_get_all_returns_copy():
    register("thread_2", {
        "conversation_id": "def67890",
        "user_msg": "太差了",
        "escalate_reason": "negative_sentiment",
        "sentiment": "critical",
    })
    all_items = get_all()
    assert "thread_1" in all_items
    assert "thread_2" in all_items


def test_unregister():
    unregister("thread_1")
    assert get("thread_1") is None
    assert "thread_1" not in get_all()


def test_get_nonexistent():
    assert get("nonexistent") is None


def test_unregister_nonexistent_does_not_raise():
    unregister("nonexistent")  # should not raise


def test_register_timestamp_auto():
    register("thread_3", {"conversation_id": "test"})
    info = get("thread_3")
    assert "timestamp" in info
    assert info["timestamp"] != ""


def test_unregister_cleanup():
    register("thread_a", {"conversation_id": "a"})
    register("thread_b", {"conversation_id": "b"})
    unregister("thread_a")
    all_items = get_all()
    assert "thread_a" not in all_items
    assert "thread_b" in all_items


def test_update():
    register("thread_u", {"conversation_id": "u", "status": "pending"})
    update("thread_u", {"status": "active", "agent_name": "客服01"})
    info = get("thread_u")
    assert info["status"] == "active"
    assert info["agent_name"] == "客服01"
    assert info["conversation_id"] == "u"


def test_update_nonexistent_does_not_raise():
    update("nonexistent", {"status": "active"})  # should not raise
