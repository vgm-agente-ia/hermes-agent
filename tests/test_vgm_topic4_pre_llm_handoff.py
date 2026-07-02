"""P3-1.4 regression tests for VGM topic-4 pre-LLM Kanban handoff.

The 2026-06-15 incident exposed two routing contracts that must be enforced
before the chachita LLM sees Gloria's topic-4 exegesis follow-up:

* ``ruta``/route-selection messages are chat responses, not automatic DOCX.
* The durable Kanban card must carry source-topic delivery metadata so the
  worker can deliver to the originating Telegram topic.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RUN_PY = REPO_ROOT / "gateway" / "run.py"
TOPIC4_MSG = {
    "platform": "telegram",
    "chat_id": "-1003826944617",
    "thread_id": 4,
    "message_id": "msg-20260615-1024",
    "text": "ruta 2 y 3",
    "profile": "vgm-chachita",
    "user_name": "Gloria",
}


def _chat_routing():
    return importlib.import_module("gateway.chat_routing")


def _source_for_function(source: str, function_name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{function_name} not found in gateway/run.py")


def test_route_only_topic4_message_routes_to_kanban_with_chat_response_mode():
    routing = _chat_routing()

    decision = routing.route_exegesis_topic_message(TOPIC4_MSG)

    assert decision["action"] == "route_to_vgm_exegesis"
    assert decision["delegate_target"] == "vgm-exegesis"
    assert decision["handoff_mode"] == "kanban"
    assert decision["delivery_mode"] == "chat_response"
    assert decision["build_docx"] is False
    assert decision["target_delivery"] == "telegram:-1003826944617:4"


def test_non_topic4_message_is_passthrough_and_not_routed_to_exegesis_worker():
    routing = _chat_routing()
    msg = dict(TOPIC4_MSG, thread_id=1096, text="estado de resultados en doc editable")

    decision = routing.route_exegesis_topic_message(msg)

    assert decision["action"] == "passthrough"
    assert decision["delegate_target"] is None
    assert decision["handoff_mode"] is None


def test_kanban_payload_includes_delivery_mode_and_source_topic_contract():
    routing = _chat_routing()

    payload = routing.build_exegesis_followup_kanban_task(
        TOPIC4_MSG,
        profile_config={"kanban": {"default_board": "vgm"}},
    )

    assert payload["assignee"] == "vgm-exegesis"
    assert payload["board"] == "vgm"
    assert payload["workspace_kind"] == "dir"
    assert payload["workspace_path"] == "/home/deployer/vgm-knowledge"
    assert payload["skills"] == ["exegesis-diaria"]
    assert payload["goal_mode"] is True
    assert payload["goal_max_turns"] == 12
    assert payload["max_runtime_seconds"] == 3600
    assert payload["idempotency_key"].startswith("vgm-exegesis-followup:-1003826944617:4:")

    body = payload["body"]
    required_lines = [
        "delivery_mode: chat_response",
        "build_docx: false",
        "source_platform: telegram",
        "source_chat_id: -1003826944617",
        "source_thread_id: 4",
        "target_delivery: telegram:-1003826944617:4",
        "requested_text: ruta 2 y 3",
    ]
    for line in required_lines:
        assert line in body

    assert "delegate_task" not in body
    assert "DOCX" not in body.split("## Delivery intent", 1)[0]


def test_explicit_docx_topic4_payload_flips_delivery_mode_but_keeps_source_topic():
    routing = _chat_routing()
    msg = dict(TOPIC4_MSG, message_id="msg-20260615-1032", text="genera un docX integrando todas las secciones")

    decision = routing.route_exegesis_topic_message(msg)
    payload = routing.build_exegesis_followup_kanban_task(
        msg,
        profile_config={"kanban": {"default_board": "vgm"}},
    )

    assert decision["delivery_mode"] == "docx_requested"
    assert decision["build_docx"] is True
    assert decision["target_delivery"] == "telegram:-1003826944617:4"
    assert "delivery_mode: docx_requested" in payload["body"]
    assert "build_docx: true" in payload["body"]
    assert "source_thread_id: 4" in payload["body"]
    assert "target_delivery: telegram:-1003826944617:4" in payload["body"]


def test_gateway_wires_topic4_handoff_before_agent_claim_and_llm_turn():
    source = RUN_PY.read_text(encoding="utf-8")

    assert "from gateway.chat_routing import" in source
    assert "route_exegesis_topic_message" in source
    assert "build_exegesis_followup_kanban_task" in source
    assert "_maybe_handle_exegesis_topic4_handoff" in source

    handle_source = _source_for_function(source, "_handle_message")
    handoff_idx = handle_source.find("_maybe_handle_exegesis_topic4_handoff")
    claim_idx = handle_source.find("_claim_active_session_slot")
    llm_idx = handle_source.find("await self._handle_message_with_agent")

    assert handoff_idx != -1, "_handle_message must call the deterministic topic-4 handoff"
    assert claim_idx != -1, "_handle_message must still claim sessions on normal path"
    assert llm_idx != -1, "_handle_message must still call the agent on normal path"
    assert handoff_idx < claim_idx < llm_idx, (
        "topic-4 handoff must run before active-session claim and before "
        "_handle_message_with_agent, so route-only follow-ups never reach the LLM"
    )


def test_gateway_handoff_helper_has_no_agent_or_delegate_dependency():
    from gateway.run import GatewayRunner

    helper = getattr(GatewayRunner, "_maybe_handle_exegesis_topic4_handoff", None)
    assert helper is not None
    helper_source = inspect.getsource(helper)

    forbidden = ["AIAgent", "delegate_task", "_handle_message_with_agent"]
    for token in forbidden:
        assert token not in helper_source

    assert "kanban_db.create_task" in helper_source or ".create_task(" in helper_source
