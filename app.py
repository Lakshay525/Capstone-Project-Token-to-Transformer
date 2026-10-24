"""Local web demo: talk to the agent from your browser.

    python app.py            # then open http://localhost:8000

The page records the microphone, converts it to 16 kHz WAV in the browser, and posts it here.
The server runs Gnani STT -> agent -> Gnani TTS and returns the transcript, the reply, the sources
and the spoken reply. A text box is included for debugging (it skips STT but still speaks the reply).
"""
import base64
import json
import sys
import tempfile
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.agent import SupportAgent  # noqa: E402
from src.kb import Retriever  # noqa: E402
from src.llm import LLM, LLMError  # noqa: E402
from src.voice import Voice, VoiceError  # noqa: E402

RETRIEVER, LLM_CLIENT = Retriever(), LLM()
try:
    VOICE = Voice()
except VoiceError as exc:
    VOICE = None
    print(f"[warn] voice disabled: {exc}")
SESSIONS: dict = {}

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>ZentroMart Voice Support</title>
<style>
:root{--bg:#fafafa;--fg:#1c1c1c;--card:#fff;--muted:#6b6b6b;--accent:#0b6e4f;--line:#e2e2e2}
@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--card:#222;--muted:#9a9a9a;--accent:#4cc38a;--line:#333}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:640px;margin:0 auto;padding:16px}
h1{font-size:20px}#log{display:flex;flex-direction:column;gap:8px;margin:16px 0;min-height:200px}
.m{padding:10px 12px;border-radius:12px;max-width:85%;background:var(--card);border:1px solid var(--line)}
.u{align-self:flex-end;border-color:var(--accent)}.src{font-size:12px;color:var(--muted)}
button,input{font:inherit;padding:10px 14px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
#mic{background:var(--accent);color:#fff;border:0;width:100%;padding:16px;font-size:18px}
#mic.rec{background:#b3261e}form{display:flex;gap:8px;margin-top:12px}input{flex:1}
</style>
<main><h1>ZentroMart support (voice)</h1>
<div id=log></div>
<button id=mic>Hold to talk</button>
<form id=f><input id=t placeholder="or type a message" autocomplete=off><button>Send</button></form>
<p class=src>Say or type your order ID, then the last 4 digits of your registered phone or your delivery pincode.</p></main>
<script>
const sid=crypto.randomUUID(),log=document.getElementById('log'),mic=document.getElementById('mic');
function add(t,c,s){const d=document.createElement('div');d.className='m '+c;d.textContent=t;
 if(s){const p=document.createElement('div');p.className='src';p.textContent=s;d.appendChild(p)}log.appendChild(d);d.scrollIntoView()}
function show(r){if(r.error){add(r.error,'a');return}
 if(r.transcript!==undefined)add(r.transcript,'u');
 add(r.reply,'a',[r.decision,(r.chunks||[]).join(', ')].filter(Boolean).join(' | '));
 if(r.audio){new Audio('data:audio/wav;base64,'+r.audio).play()}}
async function post(url,body,type){const r=await fetch(url,{method:'POST',headers:{'X-Session':sid,'Content-Type':type},body});show(await r.json())}
document.getElementById('f').onsubmit=e=>{e.preventDefault();const t=document.getElementById('t');if(!t.value)return;
 add(t.value,'u');post('/api/text',t.value,'text/plain');t.value=''};
function wav(buf,rate){const n=buf.length,o=new DataView(new ArrayBuffer(44+n*2));
 const w=(i,s)=>[...s].forEach((c,k)=>o.setUint8(i+k,c.charCodeAt(0)));
 w(0,'RIFF');o.setUint32(4,36+n*2,true);w(8,'WAVEfmt ');o.setUint32(16,16,true);o.setUint16(20,1,true);o.setUint16(22,1,true);
 o.setUint32(24,rate,true);o.setUint32(28,rate*2,true);o.setUint16(32,2,true);o.setUint16(34,16,true);w(36,'data');o.setUint32(40,n*2,true);
 for(let i=0;i<n;i++){const s=Math.max(-1,Math.min(1,buf[i]));o.setInt16(44+i*2,s<0?s*0x8000:s*0x7fff,true)}return o.buffer}
let rec,chunks=[];
async function start(){const s=await navigator.mediaDevices.getUserMedia({audio:true});rec=new MediaRecorder(s);chunks=[];
 rec.ondataavailable=e=>chunks.push(e.data);
 rec.onstop=async()=>{s.getTracks().forEach(t=>t.stop());const ab=await new Blob(chunks).arrayBuffer();
  const ctx=new AudioContext({sampleRate:16000});const dec=await ctx.decodeAudioData(ab);
  post('/api/audio',wav(dec.getChannelData(0),16000),'audio/wav')};rec.start();mic.classList.add('rec');mic.textContent='Listening... release to send'}
function stop(){if(rec&&rec.state==='recording'){rec.stop();mic.classList.remove('rec');mic.textContent='Hold to talk'}}
mic.onpointerdown=start;mic.onpointerup=stop;mic.onpointerleave=stop;
</script>"""


def agent_for(sid: str) -> SupportAgent:
    if sid not in SESSIONS:
        SESSIONS[sid] = SupportAgent(LLM_CLIENT, RETRIEVER, "full", channel="voice")
    return SESSIONS[sid]


def turn(sid: str, transcript: str) -> dict:
    out = {"transcript": transcript}
    if not transcript.strip():
        reply, decision, chunks = Voice.recovery_text("en-IN"), "clarify", []
    else:
        try:
            r = agent_for(sid).respond(transcript)
            reply, decision, chunks = r.text, r.decision, r.chunks
        except LLMError as exc:
            reply, decision, chunks = "Sorry, I'm having trouble right now. Please try again.", "answer", []
            out["error_detail"] = str(exc)
    out.update(reply=reply, decision=decision, chunks=chunks)
    if VOICE:
        with tempfile.TemporaryDirectory() as d:
            sp = VOICE.speak(reply, Path(d) / "reply.wav")
            if sp["audio"]:
                out["audio"] = base64.b64encode(Path(sp["audio"]).read_bytes()).decode()
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8" if "text" in ctype or "json" in ctype else ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, PAGE, "text/html") if self.path == "/" else self._send(404, "{}")

    def do_POST(self):
        sid = self.headers.get("X-Session") or str(uuid.uuid4())
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            if self.path == "/api/text":
                res = turn(sid, body.decode("utf-8"))
            elif self.path == "/api/audio":
                if not VOICE:
                    return self._send(200, json.dumps({"error": "Voice is off: set GNANI_API_KEY in .env"}))
                with tempfile.TemporaryDirectory() as d:
                    p = Path(d) / "in.wav"
                    p.write_bytes(body)
                    heard = VOICE.listen(p)
                res = turn(sid, heard["transcript"])
            else:
                return self._send(404, "{}")
            self._send(200, json.dumps(res, ensure_ascii=False))
        except Exception as exc:
            self._send(200, json.dumps({"error": f"{type(exc).__name__}: {exc}"}))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = 8000
    print(f"ZentroMart voice agent on http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
