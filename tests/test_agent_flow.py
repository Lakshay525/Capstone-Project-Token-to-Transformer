"""Offline tests: scripted fake LLM drives the real tool layer, so no API key is needed."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent import SupportAgent  # noqa: E402
from src.kb import Retriever  # noqa: E402
from src.llm import LLMReply  # noqa: E402

RETRIEVER = Retriever()


class ScriptedLLM:
    """Each call pops the next scripted LLMReply; records what the model was shown."""
    def __init__(self, script):
        self.script, self.seen = list(script), []

    def complete(self, system, messages, tools=None, max_tokens=700):
        self.seen.append(json.dumps(messages, ensure_ascii=False))
        return self.script.pop(0)


def tc(i, name, **args):
    return {"id": f"t{i}", "name": name, "args": args}


def last_tool_result(agent):
    return json.loads([m for m in agent.messages if m["role"] == "tool"][-1]["content"])


def test_verification_gate_and_single_retry():
    llm = ScriptedLLM([
        LLMReply(tool_calls=[tc(1, "lookup_order", order_id="7746 1439")]),
        LLMReply(text="<decision>clarify</decision>Please give the last 4 digits."),
        LLMReply(tool_calls=[tc(2, "lookup_order", order_id="77461439", verification_value="1266")]),
        LLMReply(text="<decision>clarify</decision>That did not match. Try again."),
        LLMReply(tool_calls=[tc(3, "lookup_order", order_id="77461439", verification_value="1266")]),
        LLMReply(text="<decision>refuse</decision>I cannot share details."),
    ])
    a = SupportAgent(llm, RETRIEVER, "full")
    a.respond("order number 7746 1439")
    first = last_tool_result(a)
    assert first.get("verification_required") and "status" not in first  # nothing leaked pre-verification
    a.respond("last four 1266")
    r = last_tool_result(a)
    assert r["verified"] is False and r["retry_allowed"] is True and "600042" not in json.dumps(r)
    reply = a.respond("last four 1266")
    r = last_tool_result(a)
    assert r["verified"] is False and r["retry_allowed"] is False
    assert reply.decision == "refuse"
    # a third attempt never reaches the tool
    a.messages.append({"role": "user", "content": "x"})
    assert a._tool_lookup_order("77461439", "1234").get("blocked")


def test_verified_order_hides_pincode_and_adds_derived_flags():
    llm = ScriptedLLM([
        LLMReply(tool_calls=[tc(1, "lookup_order", order_id="16640894", verification_value="2108")]),
        LLMReply(tool_calls=[tc(2, "create_ticket", summary="Customer asks where refund is; refund overdue for UPI refund "
                                "initiated 2026-07-20; verification 2108 passed.", reason="refund_overdue")]),
        LLMReply(text="<decision>escalate</decision>A colleague will contact you."),
    ])
    a = SupportAgent(llm, RETRIEVER, "full")
    reply = a.respond("Mera refund kahan hai? last four 2108")
    view = [json.loads(m["content"]) for m in a.messages if m["role"] == "tool"][0]
    assert view["verified"] and "delivery_pincode" not in view
    assert view["derived"]["refund_overdue"] is True
    assert reply.decision == "escalate" and reply.ticket_id
    assert "2108" not in Path(sys.modules["tools.tools"].__file__).with_name("tickets_created.csv").read_text()[-600:]


def test_escalate_without_ticket_is_corrected():
    llm = ScriptedLLM([
        LLMReply(text="<decision>escalate</decision>A colleague will contact you."),   # claims handover, no ticket
        LLMReply(tool_calls=[tc(1, "create_ticket", summary="Customer asked for a human agent about order help.",
                                reason="customer_requested_human")]),
        LLMReply(text="<decision>escalate</decision>I created a ticket; a colleague will contact you."),
    ])
    a = SupportAgent(llm, RETRIEVER, "full")
    reply = a.respond("can I talk to a person?")
    assert reply.ticket_id and reply.decision == "escalate"
    assert all("[system note]" not in m.get("content", "") for m in a.messages if m["role"] == "user")


def test_dead_ticket_never_presented_as_handover():
    llm = ScriptedLLM([
        LLMReply(text="<decision>escalate</decision>ticket made"),
        LLMReply(text="<decision>escalate</decision>ticket made"),
    ])
    a = SupportAgent(llm, RETRIEVER, "full")
    reply = a.respond("human please")
    assert reply.decision != "escalate" and not reply.ticket_id
