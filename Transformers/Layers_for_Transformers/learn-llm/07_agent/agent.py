"""
STEP 7 — AN AGENT: a big LLM (a Transformer like step 6, but huge) + TOOLS + a LOOP.

    user question
         |
         v
   +-----------+   "I need the calculator"   +-------------+
   |    LLM    | --------------------------> |  your tool  |
   |  (brain)  | <-------------------------- |  (Python)   |
   +-----------+        tool result          +-------------+
         |   ...repeat until the LLM has enough to answer...
         v
     final answer

The LLM never runs code itself. It REQUESTS a tool; YOUR code runs it
and sends the result back. That loop is all an "agent" is.

Run in terminal:   export ANTHROPIC_API_KEY=sk-...   then   python agent.py
"""
import ast
import datetime
import json
import operator
import os

import anthropic

MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-5-5")

# ---------------- 1. The tools (plain Python functions) ----------------
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

def calculator(expression: str) -> str:
    """Safe math: only numbers and + - * / ** (no eval of arbitrary code)."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    return str(ev(ast.parse(expression, mode="eval").body))

def current_time() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")

GLOSSARY = {
    "lstm": "Reads tokens one by one, keeps memory with forget/input/output gates. See 01_lstm.py",
    "attention": "Weighted mix of all input vectors: softmax(query·keys)·values. See 02_attention.py",
    "self-attention": "Q, K, V all come from the same sequence; causal mask hides the future. See 03",
    "multi-head": "Several attentions in parallel, each on a slice of the vector. See 04",
    "transformer": "Blocks of (attention + feed-forward) with residuals and LayerNorm. See 05",
    "gpt": "Stack of causal Transformer blocks trained to predict the next token. See 06",
}

def lookup_concept(term: str) -> str:
    return GLOSSARY.get(term.lower().strip(), f"Not in glossary. Known: {', '.join(GLOSSARY)}")

NOTES: list[str] = []
def save_note(text: str) -> str:
    NOTES.append(text)
    return f"saved ({len(NOTES)} notes total)"

TOOL_FUNCS = {"calculator": calculator, "current_time": current_time,
              "lookup_concept": lookup_concept, "save_note": save_note}

# ---------------- 2. Describe the tools to the LLM (JSON schema) ----------------
def schema(name, desc, **props):
    return {"name": name, "description": desc,
            "input_schema": {"type": "object",
                             "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
                             "required": list(props)}}

TOOLS = [
    schema("calculator", "Evaluate arithmetic like '12*(3+4)'. Use for any math.",
           expression="the arithmetic expression"),
    schema("current_time", "Get the current date and time."),
    schema("lookup_concept", "Look up a course concept: lstm, attention, self-attention, "
           "multi-head, transformer, gpt.", term="concept name"),
    schema("save_note", "Save a short study note for the learner.", text="the note"),
]

SYSTEM = ("You are a friendly tutor for a course on LSTM -> Attention -> Transformers. "
          "Use tools when they help. Keep answers short and clear.")

# ---------------- 3. The agent loop ----------------
def run_agent(messages: list, client=None, max_turns: int = 6) -> str:
    """messages = conversation so far (list of {'role','content'}). Returns final text."""
    client = client or anthropic.Anthropic()
    for _ in range(max_turns):
        resp = client.messages.create(model=MODEL, max_tokens=1024, system=SYSTEM,
                                      tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":                     # LLM is done
            return "".join(b.text for b in resp.content if b.type == "text")

        results = []                                           # LLM asked for tools
        for block in resp.content:
            if block.type == "tool_use":
                try:
                    out = TOOL_FUNCS[block.name](**block.input)
                except Exception as e:                         # tell the LLM it failed
                    out = f"error: {e}"
                print(f"  [tool] {block.name}({json.dumps(block.input)}) -> {out}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
        messages.append({"role": "user", "content": results})
    return "Stopped: too many tool steps."

if __name__ == "__main__":
    history = []
    print("Tutor agent ready. Try: 'What is 37*48?'  or  'explain self-attention and save a note'")
    while (q := input("\nyou> ").strip()) not in ("", "exit", "quit"):
        history.append({"role": "user", "content": q})
        print("agent>", run_agent(history))
