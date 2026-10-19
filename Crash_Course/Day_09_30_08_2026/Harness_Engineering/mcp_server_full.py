"""
mcp_server_full.py -- exposes the SAME tools from rag_engine.py over MCP,
so Claude Desktop / MCP Inspector / any other MCP client can use them.

Note what's NOT duplicated here versus streamlit_full_harness.py:
semantic chunking, re-ranking, and the 3 tool implementations all live
in rag_engine.py and are imported, not rewritten. Only the "how do
outside callers reach these tools" part differs between the two files.

This server does not run its own LLM or its own agent loop -- an MCP
server never does. Whatever client connects (Claude Desktop, etc.)
brings its own model and its own harness (its own FEEDFORWARD GUIDE via
system prompt, its own TOOL LOOP). This file only provides the tools
themselves, plus a best-effort FEEDBACK SENSOR and STATE & MEMORY layer
around each tool call, since an MCP server has no visibility into
whatever the client's LLM ultimately says.

RUNNING FOR MANUAL TESTING:
    mcp dev mcp_server_full.py

CONNECTING TO CLAUDE DESKTOP: same steps as mcp_policy_server.py,
just point "args" at this file's absolute path instead.
"""

from mcp.server import MCPServer

from rag_engine import (
    build_vector_db,
    build_reranker,
    make_tools,
    init_memory_db,
    save_turn,
)

# ============================================================
# 4. STATE & MEMORY -- log every tool call this server handles,
# independent of whichever client (Claude Desktop, Inspector, etc.)
# is calling it. session_id is fixed here since an MCP server doesn't
# have its own concept of a "browser session" the way Streamlit does.
# ============================================================

init_memory_db()
MCP_SESSION_ID = "mcp-server"

print("Loading semantic FAISS index...")
vector_db, _ = build_vector_db()

print("Loading re-ranking model...")
reranker = build_reranker()

print("Building tools...")
search_policy, raise_hr_ticket, check_public_holiday = make_tools(vector_db, reranker)


# ============================================================
# 1. TOOL LOOPS -- exposed over MCP instead of a local agent loop
# ============================================================

mcp = MCPServer("company-policy-full")


@mcp.tool()
def search_policy_tool(query: str) -> str:
    """Search the company policy document and return the most relevant
    passages for the given question (semantic chunking + re-ranking)."""
    result = search_policy.invoke({"query": query})
    save_turn(MCP_SESSION_ID, "tool:search_policy", f"query={query} -> {result[:200]}")
    return result


@mcp.tool()
def raise_hr_ticket_tool(employee_id: str, request_type: str, details: str) -> str:
    """Raise an HR ticket/request on behalf of an employee. Only call this
    after the user has explicitly confirmed they want the action taken."""
    result = raise_hr_ticket.invoke({
        "employee_id": employee_id, "request_type": request_type, "details": details
    })
    save_turn(MCP_SESSION_ID, "tool:raise_hr_ticket", result)
    return result


@mcp.tool()
def check_public_holiday_tool(date: str, country_code: str = "IN") -> str:
    """Check whether a given date is a public holiday in a specific country."""
    result = check_public_holiday.invoke({"date": date, "country_code": country_code})
    save_turn(MCP_SESSION_ID, "tool:check_public_holiday", result)
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
