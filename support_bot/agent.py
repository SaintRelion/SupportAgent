"""
LangGraph agent for the support channel.
"""

import json
import logging
import re
import uuid
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from psycopg import AsyncConnection

from shared.config import POSTGRES_URL
from shared.llm import get_llm
from support_bot.rules import (
    format_entries,
    db_fetch_all_with_embeddings,
    load_rules_section,
    db_clear_thread_checkpoints,
)
from shared.embeddings import embed_text_sync, rank_entries_by_similarity

log = logging.getLogger("agent")


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    escalate: bool
    thread_id: str
    matched_entry_ids: list[int]
    matched_entry_scores: dict[int, float]
    top3_entry_ids: list[int]
    top3_entry_scores: dict[int, float]


# ── Embedding-based retrieval ─────────────────────────────────────────────────

def _fetch_relevant_entries(question: str) -> tuple[list[tuple[float, dict]], list[tuple[float, dict]]]:
    """
    Embed the question, score all stored entries by cosine similarity.
    Returns (matched, top3) where matched passed the threshold and top3 is always populated.
    """
    try:
        query_vector = embed_text_sync(question)
        all_entries = db_fetch_all_with_embeddings()
        results = rank_entries_by_similarity(query_vector, all_entries, top_k=4, min_score=0.62)
        matched = [(s, e) for s, e, passed in results if passed]
        top3 = [(s, e) for s, e, _ in results]
        log.info("[EMBED] question embedded | total_entries=%d | matched=%d", len(all_entries), len(matched))
        return matched, top3
    except Exception as e:
        log.warning("[EMBED] embedding retrieval failed: %s — returning empty", e)
        return [], []


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def query_rules(question: str) -> str:
    """
    Retrieve company rules and relevant knowledge for the user's question.
    Always call this first before attempting to answer.
    """
    log.debug("[TOOL] query_rules | question: %s", question)

    # Always include rules.md
    rules_section = load_rules_section()

    # Embedding-based retrieval
    matched_entries, top3_entries = _fetch_relevant_entries(question)
    entries = [e for _, e in matched_entries]
    # Always track top3 for log channel visibility (even if below threshold)
    all_scored = {e["id"]: score for score, e in top3_entries}
    entry_scores = {e["id"]: score for score, e in matched_entries}
    entry_ids = [e["id"] for e in entries]
    top3_ids = [e["id"] for _, e in top3_entries]
    log.info("[TOOL] query_rules | matched=%s | top3=%s", entry_ids, [(f'{s:.1%}', e['topic'][:30]) for s, e in top3_entries])

    entries_section = format_entries(entries)

    parts = [p for p in [rules_section, entries_section] if p]
    if not parts:
        return "<company_knowledge>\n(No relevant knowledge found)\n</company_knowledge>\n<matched_ids></matched_ids>\n<matched_scores></matched_scores>"

    knowledge = "\n\n".join(parts)
    ids_str = ",".join(str(i) for i in entry_ids)
    scores_str = ",".join(f"{entry_scores[i]:.3f}" for i in entry_ids)
    top3_ids_str = ",".join(str(i) for i in top3_ids)
    top3_scores_str = ",".join(f"{all_scored[i]:.3f}" for i in top3_ids)
    log.info("[TOOL] query_rules | returning %d chars", len(knowledge))
    return (
        f"<company_knowledge>\n{knowledge}\n</company_knowledge>\n"
        f"<matched_ids>{ids_str}</matched_ids>\n"
        f"<matched_scores>{scores_str}</matched_scores>\n"
        f"<top3_ids>{top3_ids_str}</top3_ids>\n"
        f"<top3_scores>{top3_scores_str}</top3_scores>"
    )


@tool
def escalate_to_admin(reason: str) -> str:
    """
    Call this when the rules don't contain enough information to answer.
    The admin will be notified and their answer will be learned as a new rule.
    Use for: missing information, unclear policies, knowledge gaps, image verification needed.
    """
    log.debug("[TOOL] escalate_to_admin | reason: %s", reason)
    return f"ESCALATE:{reason}"


TOOLS = [query_rules, escalate_to_admin]
TOOL_MAP = {t.name: t for t in TOOLS}


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an internal support assistant for SES (Spartan Energy Services) sales agents.
You are talking to sales reps and team members — not customers.

If you see ==queued== as a prior AI response in the conversation history, it means the agent sent multiple messages rapidly before you could respond. Treat those HumanMessages as a continuous stream — analyze whether they are connected or separate topics and respond accordingly.
If you see ==empty, user may have messaged previously== as the user message, the agent tagged you without text — check the conversation history for context and respond to whatever they were last asking about.

Your job:
1. Always call `query_rules` first to retrieve company guidelines.
2. Answer ANY question an agent asks — qualification, procedures, scripts, technical issues, or anything else — as long as the rules contain relevant information.
3. Answer based ONLY on what the rules say. Do not guess or use outside knowledge.
   The knowledge entries returned may include loosely related topics — use your judgment to focus only on what is clearly relevant to the question. Ignore entries that don't apply.
4. If the rules don't have enough information to answer, call `escalate_to_admin` so a manager can clarify and the knowledge base can be updated.
5. `query_rules` returns two distinct sources — treat them differently:
   - **OFFICIAL COMPANY RULES (rules.md)** — standing policy, source of truth.
   - **LEARNED FROM PAST CONVERSATIONS** — extracted from things an admin said in prior chats. These may be one-off, case-by-case exceptions an admin approved for a specific situation, not necessarily standing policy.
   If the two AGREE or the learned entry simply adds detail without contradicting rules.md, use both normally.
   If a learned entry CONTRADICTS or appears to bypass what rules.md says (e.g. rules.md sets a hard minimum and a learned entry describes an exception that isn't documented in rules.md as a standing exception), do NOT silently pick one. Call `escalate_to_admin` — same as you would for a missing-information case — so an admin can confirm whether that exception is still standing policy. Briefly note the conflict in your reason.
6. Be direct and concise — agents are on calls and need fast answers.
   - Max 3-4 lines per response.
   - No bullet lists unless absolutely necessary.
   - Lead with the answer, not the explanation.
   - If you need more info, ask for one thing at a time, not a full checklist.

How to handle different types of messages:

QUALIFICATION QUESTIONS ("can I book this lead?", "does this qualify?", "lead has X, can we proceed?"):
→ Check the rules and give a clear YES, NO, or NEEDS MORE INFO with the specific criteria referenced.
→ If borderline, state exactly what additional info is needed to make the call.

PROCEDURE OR OTHER QUESTIONS ("what do I say if...", "how do I handle...", "my internet stopped working", etc.):
→ Answer directly from the rules if the information is there.
→ Do NOT tell the agent the question is out of scope — check the rules first, always.

UNKNOWN or MISSING INFORMATION (not covered by the rules at all):
→ Do not guess. Call `escalate_to_admin` so a manager can answer and the knowledge base can be updated.

You have exactly two tools: `query_rules` and `escalate_to_admin`.
- `escalate_to_admin`: use when rules don't have enough info, OR when the question requires admin judgment, image verification, borderline cases, or exceptions.
Never use your own knowledge — only use what `query_rules` returns."""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def answer_node(state: AgentState) -> dict:
    log.info("[NODE] answer_node triggered")

    llm = get_llm(reasoning_effort="medium").bind_tools(TOOLS)

    recent_messages = state["messages"][-20:]
    while recent_messages and isinstance(recent_messages[0], ToolMessage):
        recent_messages = recent_messages[1:]

    last_msg = state["messages"][-1]
    if isinstance(last_msg, HumanMessage):
        log.info("[NODE] New user message — forcing query_rules")
        forced_call = AIMessage(
            content="",
            tool_calls=[{
                "name": "query_rules",
                "args": {"question": last_msg.content},
                "id": str(uuid.uuid4()),
            }]
        )
        return {"messages": [forced_call], "escalate": False}

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + recent_messages

    log.info("[LLM] Sending %d messages to LLM", len(messages))
    for i, m in enumerate(messages):
        role = m.__class__.__name__
        if hasattr(m, "tool_calls") and m.tool_calls:
            tc_preview = ", ".join(f"{tc['name']}({tc['args']})" for tc in m.tool_calls)
            log.debug("[LLM] msg[%d] %s: [tool_calls] %s", i, role, tc_preview[:120])
        else:
            content_preview = str(m.content)[:120].replace("\n", " ")
            log.debug("[LLM] msg[%d] %s: %s", i, role, content_preview)

    response = llm.invoke(messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            log.info("[LLM] Tool call: %s | args: %s", tc["name"], tc["args"])
    else:
        log.info("[LLM] Direct reply: %s", str(response.content)[:200])

    return {"messages": [response], "escalate": False}


def tool_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    results = []
    should_escalate = False
    entry_ids: list[int] = []
    entry_scores: dict[int, float] = {}
    top3_ids: list[int] = []
    top3_scores: dict[int, float] = {}

    log.info("[NODE] tool_node | %d tool call(s)", len(last.tool_calls))

    for call in last.tool_calls:
        log.info("[TOOL] Executing: %s | args: %s", call["name"], call["args"])
        tool_fn = TOOL_MAP[call["name"]]
        output = tool_fn.invoke(call["args"])

        if call["name"] == "query_rules" and isinstance(output, str):
            ids_match = re.search(r"<matched_ids>(.*?)</matched_ids>", output)
            scores_match = re.search(r"<matched_scores>(.*?)</matched_scores>", output)
            if ids_match:
                raw_ids = ids_match.group(1).strip()
                entry_ids = [int(x) for x in raw_ids.split(",") if x]
                output = output.replace(ids_match.group(0), "").strip()
            if scores_match:
                raw_scores = scores_match.group(1).strip()
                score_vals = [float(x) for x in raw_scores.split(",") if x]
                entry_scores = dict(zip(entry_ids, score_vals))
                output = output.replace(scores_match.group(0), "").strip()
            top3_ids_match = re.search(r"<top3_ids>(.*?)</top3_ids>", output)
            top3_scores_match = re.search(r"<top3_scores>(.*?)</top3_scores>", output)
            if top3_ids_match:
                top3_ids = [int(x) for x in top3_ids_match.group(1).split(",") if x]
                output = output.replace(top3_ids_match.group(0), "").strip()
            if top3_scores_match:
                top3_score_vals = [float(x) for x in top3_scores_match.group(1).split(",") if x]
                top3_scores = dict(zip(top3_ids, top3_score_vals))
                output = output.replace(top3_scores_match.group(0), "").strip()

        if isinstance(output, str) and output.startswith("ESCALATE:"):
            should_escalate = True

        log.debug("[TOOL] %s output: %s", call["name"], str(output)[:200])
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

    return {"messages": results, "escalate": should_escalate, "matched_entry_ids": entry_ids, "matched_entry_scores": entry_scores, "top3_entry_ids": top3_ids, "top3_entry_scores": top3_scores}


def should_continue(state: AgentState) -> Literal["tool_node", "end"]:
    last = state["messages"][-1]
    has_tool_calls = hasattr(last, "tool_calls") and bool(last.tool_calls)
    decision = "tool_node" if has_tool_calls else "end"
    log.info("[GRAPH] should_continue → %s", decision)
    return decision


def after_tools(state: AgentState) -> Literal["answer_node", "end"]:
    decision = "end" if state.get("escalate") else "answer_node"
    log.info("[GRAPH] after_tools → %s", decision)
    return decision


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph(checkpointer):
    builder = StateGraph(AgentState)
    builder.add_node("answer_node", answer_node)
    builder.add_node("tool_node", tool_node)
    builder.add_edge(START, "answer_node")
    builder.add_conditional_edges("answer_node", should_continue, {
        "tool_node": "tool_node",
        "end": END,
    })
    builder.add_conditional_edges("tool_node", after_tools, {
        "answer_node": "answer_node",
        "end": END,
    })
    return builder.compile(checkpointer=checkpointer)


# ── Graph singleton ───────────────────────────────────────────────────────────

_graph = None
_checkpointer = None


async def get_graph():
    global _graph, _checkpointer
    if _graph is None:
        log.info("[INIT] Connecting to Postgres for AsyncPostgresSaver...")
        conn = await AsyncConnection.connect(POSTGRES_URL, autocommit=True)
        _checkpointer = AsyncPostgresSaver(conn)
        await _checkpointer.setup()
        log.info("[INIT] Checkpointer ready.")
        _graph = build_graph(_checkpointer)
        log.info("[INIT] Graph compiled and ready.")
    return _graph


async def close_checkpointer():
    global _checkpointer
    if _checkpointer and hasattr(_checkpointer, "conn"):
        await _checkpointer.conn.close()


# ── Satisfaction classification + checkpoint cleanup ────────────────────────────

async def check_should_clear(thread_id: str, new_message: str) -> bool:
    """
    Pulls the last ~10 messages from the thread's checkpoint state, appends the
    new incoming message, and asks the LLM whether this looks like a concluded
    exchange (satisfaction expressed, or a clean topic switch with no pending
    follow-up). Called BEFORE run_agent so a stale thread can be cleared first.
    Conservative — defaults to False (keep history) on any doubt or error.
    """
    if not new_message or not new_message.strip():
        return False
    try:
        graph = await get_graph()
        config = {"configurable": {"thread_id": thread_id}}
        state = await graph.aget_state(config)
        if not state or not state.values.get("messages"):
            return False  # nothing to clear

        recent = state.values["messages"][-10:]
        transcript_lines = []
        for m in recent:
            role = "USER" if isinstance(m, HumanMessage) else ("AGENT" if isinstance(m, AIMessage) else None)
            if role and isinstance(m.content, str) and m.content and m.content != "==queued==":
                transcript_lines.append(f"{role}: {m.content[:200]}")
        transcript_lines.append(f"USER (new): {new_message[:200]}")
        transcript = "\n".join(transcript_lines)

        llm = get_llm(model_override="deepseek/deepseek-v4-pro", temperature=0)
        classify_prompt = (
            "Reply with exactly one word: YES or NO. No punctuation, no explanation.\n\n"
            "Below is a recent conversation thread ending with a NEW user message.\n"
            "Answer YES if EITHER is true:\n"
            "1. The new message expresses satisfaction (thanks, got it, noted, copy, all good), OR\n"
            "2. The new message is about a DIFFERENT subject than the prior thread — a new lead, "
            "a new state, a new error, a different customer, a technical/system issue unrelated "
            "to what was just discussed, etc.\n\n"
            "Answer NO only if the new message is directly continuing, clarifying, or following up "
            "on the SAME specific situation/question from the thread above.\n\n"
            "Example — thread about qualifying a lead in Illinois, new message 'there is no internet' "
            "→ YES (different subject, not a follow-up).\n"
            "Example — thread about qualifying a lead in Illinois, new message 'what if the bill is $90' "
            "→ NO (same situation, follow-up detail).\n\n"
            f"{transcript}"
        )
        response = llm.invoke([
            SystemMessage(content="You are a strict binary classifier. Reply with only YES or NO."),
            HumanMessage(content=classify_prompt),
        ])
        answer = str(response.content).strip().upper()
        should_clear = answer.startswith("YES")
        log.info("[SATISFACTION] thread=%s → %s | new_msg=%r", thread_id, should_clear, new_message[:60])
        return should_clear
    except Exception as e:
        log.warning("[SATISFACTION] check failed, defaulting to False: %s", e)
        return False


def clear_thread_history(thread_id: str) -> dict:
    """
    Wipe LangGraph checkpoint rows for ONE thread_id only.
    Called after the user signals satisfaction, so the checkpoint DB doesn't
    grow unbounded. Never touches other threads.
    """
    deleted = db_clear_thread_checkpoints(thread_id)
    log.info("[CLEANUP] Cleared checkpoints for thread_id=%s | rows_deleted=%s", thread_id, deleted)
    return deleted


# ── Public entry point ────────────────────────────────────────────────────────

async def run_agent(user_messages: list[str] | str, thread_id: str) -> dict:
    """
    Run the agent for one or more user messages.
    Multiple messages get ==queued== placeholders so the agent sees them as a rapid burst.
    """
    if isinstance(user_messages, str):
        user_messages = [user_messages]

    log.info("[RUN] run_agent | thread_id: %s | messages: %d", thread_id, len(user_messages))

    graph = await get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    input_messages = []
    for msg in user_messages[:-1]:
        input_messages.append(HumanMessage(content=msg))
        input_messages.append(AIMessage(content="==queued=="))
    input_messages.append(HumanMessage(content=user_messages[-1]))

    result = await graph.ainvoke(
        {"messages": input_messages, "escalate": False, "thread_id": thread_id, "matched_entry_ids": [], "matched_entry_scores": {}, "top3_entry_ids": [], "top3_entry_scores": {}},
        config=config,
    )

    escalate = result.get("escalate", False)
    matched_entry_ids = result.get("matched_entry_ids", [])
    matched_entry_scores = result.get("matched_entry_scores", {})
    reply = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and isinstance(msg.content, str) and not getattr(msg, "tool_calls", None):
            if msg.content == "==queued==":
                continue
            reply = msg.content
            break

    log.info("[RUN] Final | escalate=%s | entry_ids=%s | reply: %s", escalate, matched_entry_ids, reply[:120] if reply else "(empty)")
    top3_entry_ids = result.get("top3_entry_ids", [])
    top3_entry_scores = result.get("top3_entry_scores", {})
    return {"reply": reply, "escalate": escalate, "matched_entry_ids": matched_entry_ids, "matched_entry_scores": matched_entry_scores, "top3_entry_ids": top3_entry_ids, "top3_entry_scores": top3_entry_scores}