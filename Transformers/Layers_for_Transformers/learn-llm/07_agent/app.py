"""
DEPLOY the agent as a web service (FastAPI).

  POST /chat   {"session_id": "abc", "message": "what is 2**10?"}  -> {"reply": "..."}
  GET  /       a tiny chat page to try it in the browser
  GET  /health for load balancers / uptime checks

Run locally:   uvicorn app:app --reload --port 8000   -> open http://localhost:8000
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent import run_agent

app = FastAPI(title="Tutor Agent")
SESSIONS: dict[str, list] = {}       # in-memory chat history (use Redis/DB in production)

class ChatIn(BaseModel):
    session_id: str = "default"
    message: str

@app.post("/chat")
def chat(body: ChatIn):
    history = SESSIONS.setdefault(body.session_id, [])
    history.append({"role": "user", "content": body.message})
    return {"reply": run_agent(history)}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def page():
    return """<!doctype html><meta name=viewport content="width=device-width">
<title>Tutor Agent</title>
<body style="font-family:sans-serif;max-width:640px;margin:2em auto;padding:0 1em">
<h2>Tutor Agent</h2><div id=log></div>
<form id=f><input id=m style="width:80%" placeholder="Ask about attention..."><button>Send</button></form>
<script>
const sid = Math.random().toString(36).slice(2);
f.onsubmit = async e => {
  e.preventDefault(); const q = m.value; m.value = "";
  log.innerHTML += `<p><b>You:</b> ${q}</p>`;
  const r = await fetch("/chat", {method:"POST", headers:{"Content-Type":"application/json"},
                                   body: JSON.stringify({session_id: sid, message: q})});
  log.innerHTML += `<p><b>Agent:</b> ${(await r.json()).reply}</p>`;
};
</script>"""
