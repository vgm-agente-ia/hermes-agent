"""VGM exegesis topic-4 deterministic Kanban routing helpers.

This module is intentionally small and side-effect-free: it classifies the
incoming Telegram topic-4 message and builds a Kanban task payload, but it does
not write to SQLite or call any LLM/network/tool APIs. ``gateway/run.py`` owns
the durable create step at the pre-LLM ingress point.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any


_VGM_CHACHITA = "vgm-chachita"
_VGM_EXEGESIS = "vgm-exegesis"
_TELEGRAM = "telegram"
_CHACHITA_CHAT_ID = "-1003826944617"
_TOPIC4_THREAD_ID = "4"
_VGM_KNOWLEDGE = Path("/home/deployer/vgm-knowledge")
_VGM_TOOLS = _VGM_KNOWLEDGE / "tools"
_DELIVERY_INTENT_PATH = _VGM_TOOLS / "delivery_intent.py"
_TOPIC_RESOLVER_PATH = _VGM_TOOLS / "topic_resolver.py"
_ARTIFACT_ROOT = _VGM_KNOWLEDGE / "artifacts" / "exegesis"
_DEFAULT_BOARD = "vgm"

_FOLLOWUP_RE = re.compile(
    r"("
    r"\bruta(?:s)?\b"
    r"|\bla\s+(?:1|2|3|uno|dos|tres)\b"
    r"|\blas\s+tres\b"
    r"|\b(?:1|2|3)\b"
    r"|\bexploraci[oó]n\b"
    r"|\bexp[aá]nd(?:eme|ir|e|ela|elo|elas|elos)?\b"
    r"|\bopci[oó]n\b"
    r"|\bfrase\b"
    r"|\bhomil[ií]a\b"
    r"|\bradio\b"
    r"|\bredes\b"
    r"|\bm[ií]stica\b"
    r"|\bpastoral\b"
    r"|\bbienaventuranzas\b"
    r"|\b[aá]nimo\b"
    r"|\bcitas\b"
    r")",
    re.IGNORECASE,
)


@lru_cache(maxsize=None)
def _load_module_from_path(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _delivery_intent_module() -> ModuleType:
    return _load_module_from_path("vgm_delivery_intent", str(_DELIVERY_INTENT_PATH))


def _topic_resolver_module() -> ModuleType:
    return _load_module_from_path("vgm_topic_resolver", str(_TOPIC_RESOLVER_PATH))


def _platform_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower()


def _thread_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalized_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return unicodedata.normalize("NFC", text)


def _is_chachita_topic4(msg: dict[str, Any]) -> bool:
    if str(msg.get("profile") or "") != _VGM_CHACHITA:
        return False
    if _platform_value(msg.get("platform")) != _TELEGRAM:
        return False
    if str(msg.get("chat_id") or "") != _CHACHITA_CHAT_ID:
        return False
    return _thread_value(msg.get("thread_id")) == _TOPIC4_THREAD_ID


def _origin_target(msg: dict[str, Any]) -> str:
    return f"{_TELEGRAM}:{msg.get('chat_id')}:{msg.get('thread_id')}"


def _classify_delivery(text: str) -> dict[str, Any]:
    return dict(_delivery_intent_module().classify(text))


def _resolve_delivery_target(msg: dict[str, Any], text: str) -> dict[str, Any]:
    origin = _origin_target(msg)
    return dict(_topic_resolver_module().resolve_target(origin, text))


def _should_route_topic4_text(text: str, intent: dict[str, Any]) -> bool:
    if not text:
        return False
    if text.lstrip().startswith("/"):
        return False
    if intent.get("delivery_mode") == "docx_requested":
        return True
    return bool(_FOLLOWUP_RE.search(_normalized_text(text)))


def route_exegesis_topic_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Return the deterministic routing decision for a gateway message.

    Only ``vgm-chachita`` Telegram topic 4 is intercepted. Matching follow-up
    messages are routed to a durable Kanban handoff for ``vgm-exegesis``;
    everything else passes through to the normal gateway path.
    """
    text = str(msg.get("text") or "").strip()
    passthrough = {
        "action": "passthrough",
        "delegate_target": None,
        "handoff_mode": None,
        "reason": "no_routing_rule_matched",
        "delivery_mode": "chat_response",
        "build_docx": False,
        "target_delivery": None,
    }
    if not _is_chachita_topic4(msg):
        return dict(passthrough)

    intent = _classify_delivery(text)
    target = _resolve_delivery_target(msg, text)
    if not _should_route_topic4_text(text, intent):
        out = dict(passthrough)
        out["reason"] = "topic4_non_followup"
        out["delivery_mode"] = intent.get("delivery_mode", "chat_response")
        out["build_docx"] = bool(intent.get("build_docx", False))
        out["target_delivery"] = target.get("target")
        return out

    return {
        "action": "route_to_vgm_exegesis",
        "delegate_target": _VGM_EXEGESIS,
        "handoff_mode": "kanban",
        "reason": "exegesis_topic4_followup",
        "delivery_mode": intent.get("delivery_mode", "chat_response"),
        "build_docx": bool(intent.get("build_docx", False)),
        "target_delivery": target.get("target"),
        "intent": intent,
        "target": target,
    }


def _profile_default_board(profile_config: dict[str, Any] | None) -> str:
    if isinstance(profile_config, dict):
        kanban = profile_config.get("kanban")
        if isinstance(kanban, dict):
            board = str(kanban.get("default_board") or "").strip()
            if board:
                return board
    return _DEFAULT_BOARD


def _idempotency_key(msg: dict[str, Any]) -> str:
    seed = "|".join(
        [
            _platform_value(msg.get("platform")),
            str(msg.get("chat_id") or ""),
            _thread_value(msg.get("thread_id")),
            str(msg.get("message_id") or ""),
            str(msg.get("text") or ""),
        ]
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
    return f"vgm-exegesis-followup:{msg.get('chat_id')}:{msg.get('thread_id')}:{digest}"


def _bool_literal(value: Any) -> str:
    return "true" if bool(value) else "false"


def build_exegesis_followup_kanban_task(
    msg: dict[str, Any],
    *,
    profile_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the durable Kanban task payload for a routed topic-4 follow-up."""
    decision = route_exegesis_topic_message(msg)
    if decision.get("action") != "route_to_vgm_exegesis":
        raise ValueError(f"message is not routable to vgm-exegesis: {decision!r}")

    text = str(msg.get("text") or "").strip()
    target_delivery = str(decision.get("target_delivery") or _origin_target(msg))
    delivery_mode = str(decision.get("delivery_mode") or "chat_response")
    build_docx = bool(decision.get("build_docx"))
    board = _profile_default_board(profile_config)

    body = f"""# Gloria exegesis follow-up routed from Telegram topic 4

## Original request
{text}

## Routing metadata
source_platform: telegram
source_chat_id: {msg.get('chat_id')}
source_thread_id: {msg.get('thread_id')}
source_message_id: {msg.get('message_id') or ''}
target_delivery: {target_delivery}
requested_text: {text}
artifact_root: {_ARTIFACT_ROOT}

## Delivery intent
delivery_mode: {delivery_mode}
build_docx: {_bool_literal(build_docx)}
intent_reason: {decision.get('intent', {}).get('reason', '')}

## Worker contract
1. Read the latest relevant exegesis artifact from artifact_root before answering.
2. Produce the scoped Spanish response Gloria requested.
3. If delivery_mode is chat_response, answer in Telegram chat only; do not create a document artifact.
4. If delivery_mode is docx_requested, create the requested file artifact and deliver it with MEDIA syntax.
5. Deliver the final user-facing result to target_delivery exactly as written above.
6. Keep conversation.md/GitHub narrative in Markdown only; that is separate from Telegram delivery.
"""

    return {
        "title": "Gloria exegesis follow-up — topic 4",
        "body": body,
        "assignee": _VGM_EXEGESIS,
        "created_by": "gateway:vgm-chachita",
        "tenant": "vgm",
        "workspace_kind": "dir",
        "workspace_path": str(_VGM_KNOWLEDGE),
        "priority": 100,
        "skills": ["exegesis-diaria"],
        "idempotency_key": _idempotency_key(msg),
        "max_runtime_seconds": 3600,
        "goal_mode": True,
        "goal_max_turns": 12,
        "board": board,
    }


def build_exegesis_followup_kanban_pair(
    msg: dict[str, Any],
    *,
    profile_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build [BUILD, DELIVER] Kanban task payloads for the pair-routing path.

    When ``vgm.exegesis_routing`` is set to ``build_deliver_pair`` in the
    profile config, the gateway creates TWO tasks instead of one:

    1. **BUILD** — assigned to ``vgm-exegesis``, goal_mode=True, produces the
       exegesis content and saves the artifact. Does NOT deliver.
    2. **DELIVER** — assigned to ``vgm-chachita``, goal_mode=False, reads the
       BUILD task's artifact and delivers it to Gloria via the target channel.

    The DELIVER payload's ``parents`` is left absent; ``run.py`` sets it to
    ``[build_id]`` after the BUILD task is created.
    """
    decision = route_exegesis_topic_message(msg)
    if decision.get("action") != "route_to_vgm_exegesis":
        raise ValueError(f"message is not routable to vgm-exegesis: {decision!r}")

    text = str(msg.get("text") or "").strip()
    target_delivery = str(decision.get("target_delivery") or _origin_target(msg))
    delivery_mode = str(decision.get("delivery_mode") or "chat_response")
    build_docx = bool(decision.get("build_docx"))
    board = _profile_default_board(profile_config)
    build_key = _idempotency_key(msg)

    # ── BUILD payload ──────────────────────────────────────────────────
    build_body = f"""# Gloria exegesis follow-up — BUILD (topic 4)

## Original request
{text}

## Routing metadata
source_platform: telegram
source_chat_id: {msg.get('chat_id')}
source_thread_id: {msg.get('thread_id')}
source_message_id: {msg.get('message_id') or ''}
artifact_root: {_ARTIFACT_ROOT}

## Worker contract — BUILD ONLY
1. Read the latest relevant exegesis artifact from artifact_root before answering.
2. Produce the scoped Spanish response Gloria requested.
3. Save the final content as a Markdown artifact under artifact_root (YYYY-MM-DD/exegesis.md).
4. In your kanban_complete metadata, include `artifact_path` pointing to the saved file.
5. Do NOT deliver to Telegram. Do NOT use MEDIA syntax. Delivery is handled by a separate DELIVER task.
6. Keep conversation.md/GitHub narrative in Markdown only; that is separate from Telegram delivery.
"""

    build_payload = {
        "title": "Gloria exegesis follow-up — BUILD (topic 4)",
        "body": build_body,
        "assignee": _VGM_EXEGESIS,
        "created_by": "gateway:vgm-chachita",
        "tenant": "vgm",
        "workspace_kind": "dir",
        "workspace_path": str(_VGM_KNOWLEDGE),
        "priority": 100,
        "skills": ["exegesis-diaria"],
        "idempotency_key": build_key,
        "max_runtime_seconds": 3600,
        "goal_mode": True,
        "goal_max_turns": 12,
        "board": board,
    }

    # ── DELIVER payload ────────────────────────────────────────────────
    deliver_body = f"""# Gloria exegesis follow-up — DELIVER (topic 4)

## Delivery target
target_delivery: {target_delivery}
delivery_mode: {delivery_mode}
build_docx: {_bool_literal(build_docx)}

## BUILD task reference
The BUILD task is your parent task. Use kanban_show on your own task to find
the parent BUILD task_id, then kanban_show that BUILD task to read its
kanban_complete metadata for the `artifact_path` field.

## Worker contract — DELIVER ONLY
1. Find your parent BUILD task (it is listed in your task's parents field).
2. kanban_show the BUILD task and read its completion metadata for `artifact_path`.
3. Read the artifact file at that path.
4. Deliver the content to target_delivery exactly as specified above.
5. If delivery_mode is chat_response, send the text to the Telegram chat via send_message.
6. If delivery_mode is docx_requested, create the DOCX and deliver it with MEDIA syntax.
7. Do NOT produce new exegesis content. Do NOT load the exegesis-diaria skill.
"""

    deliver_payload = {
        "title": "Gloria exegesis follow-up — DELIVER (topic 4)",
        "body": deliver_body,
        "assignee": _VGM_CHACHITA,
        "created_by": "gateway:vgm-chachita",
        "tenant": "vgm",
        "workspace_kind": "dir",
        "workspace_path": str(_VGM_KNOWLEDGE),
        "priority": 100,
        "skills": [],
        "idempotency_key": f"{build_key}-deliver",
        "max_runtime_seconds": 300,
        "goal_mode": False,
        "board": board,
        # parents intentionally absent — run.py sets parents=[build_id] after BUILD creation
    }

    return [build_payload, deliver_payload]
