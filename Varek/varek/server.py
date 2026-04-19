import os
import re
import httpx
import json
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

# Use absolute paths relative to this file so the server can be started from any directory
BASE_DIR = Path(__file__).parent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "varek-gemma3")

SUMMARIZE_AFTER_PAIRS = 10
MAX_MESSAGES = 30

PORTRAIT_META = {
    "orla":           {"file": "orla.png",           "type": "bust",     "title": "Orla",                  "subtitle": "Fish Vendor, Ash Market"},
    "thessaly":       {"file": "thessaly.png",       "type": "bust",     "title": "Thessaly Vorne",        "subtitle": "Master Assessor"},
    "maret":          {"file": "maret.png",          "type": "bust",     "title": "Maret",                 "subtitle": "Tavern Keeper"},
    "ossian":         {"file": "ossian.png",         "type": "bust",     "title": "Ossian Frett",          "subtitle": "Tax Clerk, Dead Mail Division"},
    "vicar":          {"file": "vicar.png",          "type": "bust",     "title": "The Vicar-Who-Remains", "subtitle": "Priest of the Unnamed"},
    "drev":           {"file": "drev.png",           "type": "bust",     "title": "Constable Drev",        "subtitle": "City Guard"},
    "cartographer":   {"file": "cartographer.png",   "type": "bust",     "title": "The Cartographer",      "subtitle": "No Known Name"},
    "delea_street":   {"file": "delea_street.png",   "type": "fullbody", "title": "Delea",                 "subtitle": "Street Child"},
    "delea_familiar": {"file": "delea_familiar.png", "type": "fullbody", "title": "Delea",                 "subtitle": "Your Familiar"},
    "olver_street":   {"file": "olver_street.png",   "type": "fullbody", "title": "Olver",                 "subtitle": "Street Child"},
    "olver_familiar": {"file": "olver_familiar.png", "type": "fullbody", "title": "Olver",                 "subtitle": "Your Familiar"},
}

# ===== KEYWORD DETECTION =====

# Character detection — order matters, most specific first
CHARACTER_KEYWORDS = [
    ("thessaly",     [r"\bthessaly\b", r"\bvorne\b", r"\bmaster assessor\b"]),
    ("maret",        [r"\bmaret\b", r"\bthe warm floor\b", r"\bstitch lane\b"]),
    ("ossian",       [r"\bossian\b", r"\bfrett\b", r"\bdead mail\b"]),
    ("vicar",        [r"\bvicar[\s-]who[\s-]remains\b", r"\bthe vicar\b"]),
    ("drev",         [r"\bconstable drev\b", r"\bdrev\b"]),
    ("cartographer", [r"\bthe cartographer\b", r"\bcartographer\b"]),
    ("orla",         [r"\borla\b"]),
    ("delea_street", [r"\bdelea\b", r"\bcordelea\b"]),
    ("olver_street", [r"\bolver\b"]),
]

# Location detection — scan narrative for district/place references
LOCATION_KEYWORDS = [
    ("The Ash Market",        [r"\bash market\b"]),
    ("The Pale Quarter",      [r"\bpale quarter\b", r"\bthe temple\b"]),
    ("The Sump",              [r"\bthe sump\b"]),
    ("The Magistrate's Hill", [r"\bmagistrate'?s hill\b"]),
    ("The Old Bone",          [r"\bold bone\b"]),
    ("The Warm Floor",        [r"\bwarm floor\b", r"\bstitch lane\b"]),
    ("Dead Mail Division",    [r"\bdead mail\b"]),
]

# Time progression cycle
TIME_CYCLE = ["Dawn", "Morning", "Midday", "Afternoon", "Dusk", "Evening", "Night", "Deep Night"]


def detect_characters_from_text(text: str) -> list[str]:
    """Scan narrative for all character names mentioned. Returns list ordered by
    last-paragraph priority — characters in the final paragraph come first."""
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\n', text) if p.strip()]
    last_para = paragraphs[-1].lower() if paragraphs else ""
    full_lower = text.lower()

    found_last = []   # Characters in last paragraph (priority)
    found_rest = []   # Characters elsewhere

    for char_id, patterns in CHARACTER_KEYWORDS:
        in_last = any(re.search(p, last_para) for p in patterns)
        in_full = any(re.search(p, full_lower) for p in patterns)
        if in_last:
            found_last.append(char_id)
        elif in_full:
            found_rest.append(char_id)

    return found_last + found_rest


def detect_location_from_text(text: str) -> str | None:
    if not text:
        return None
    lower = text.lower()
    for location_name, patterns in LOCATION_KEYWORDS:
        for pattern in patterns:
            if re.search(pattern, lower):
                return location_name
    return None


def advance_time(current_time: str, turn_number: int = 0) -> str:
    """Advance time of day as a server-side fallback. Moves forward 1 step every
    3 turns so the day doesn't race to Deep Night within the first few exchanges.
    Turn 0 is the opening scene — time doesn't advance until turn 3.
    The model's explicit time output always takes precedence over this fallback."""
    if current_time not in TIME_CYCLE:
        return current_time
    if turn_number == 0 or turn_number % 3 != 0:
        return current_time  # Only advance at turns 3, 6, 9, ...
    idx = TIME_CYCLE.index(current_time)
    new_idx = min(idx + 1, len(TIME_CYCLE) - 1)
    return TIME_CYCLE[new_idx]


def normalize_time(raw_time: str) -> str | None:
    """Try to match a model's time string to our canonical list."""
    if not raw_time:
        return None
    lower = raw_time.strip().lower()
    for t in TIME_CYCLE:
        if t.lower() == lower:
            return t
    # Fuzzy matching for common model variations
    fuzzy = {
        "mid-day": "Midday", "mid day": "Midday", "noon": "Midday",
        "late night": "Deep Night", "midnight": "Deep Night",
        "early morning": "Dawn", "sunrise": "Dawn", "sunset": "Dusk",
        "twilight": "Dusk", "late afternoon": "Dusk",
    }
    return fuzzy.get(lower, None)


# ===== PROMPTS =====

SUMMARIZE_PROMPT = """Summarize this story so far. Keep it simple and clear. Write in past tense. Include: what happened, who the player met, where they went, what they learned, what is currently going on. Maximum 300 words. Start with PREVIOUSLY:"""


def build_system_prompt(player: dict) -> str:
    name = player.get("name", "Unknown")
    age = player.get("age", "Unknown")
    gender = player.get("gender", "Unknown")
    appearance = player.get("appearance", "Unremarkable looking")
    background = player.get("background", "Stranger to the city")

    return f"""The player character for this session:
Name: {name}
Age: {age}
Gender: {gender}
Appearance: {appearance}
Background: {background}

Use their name sometimes. Mention their appearance when it matters. Their background affects how NPCs treat them.

FAMILIAR RULES (Delea and Olver):
- They are fragments of the dead god in child shapes. Only one can bond with the player.
- The bond only happens after many sessions of genuine care. At least 5-6 meaningful kind interactions.
- The child chooses to bond. The player does not get to decide.
- If a bond happens, write on a new line: [BOND: delea] or [BOND: olver]
- If the player is cruel to a bonded familiar, write: [ABANDONED: delea] or [ABANDONED: olver]
- Do not trigger bonds easily. The player must earn it over a long time."""


def clean_text(text: str) -> str:
    text = text.replace('\u2014', '--').replace('\u2013', '-')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2026', '...')
    return text


def extract_narrative(content: str) -> str:
    """Pull just the narrative text from an assistant message, stripping JSON wrapper if present."""
    try:
        parsed = json.loads(content)
        return parsed.get("narrative", content)
    except Exception:
        return content


# ===== SUMMARIZATION =====

async def summarize_messages(messages: list, model: str) -> str:
    transcript = ""
    for msg in messages:
        role = "PLAYER" if msg["role"] == "user" else "NARRATOR"
        content = extract_narrative(msg["content"]) if msg["role"] == "assistant" else msg["content"]
        transcript += f"{role}: {content}\n\n"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": f"{SUMMARIZE_PROMPT}\n\n{transcript}"}],
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 500},
                },
            )
        if resp.status_code == 200:
            return resp.json().get("message", {}).get("content", "").strip()
    except Exception:
        pass
    return ""


async def prepare_messages(messages: list, model: str) -> list:
    """Prepare conversation history for the model. Strips JSON metadata from assistant
    messages so the model only sees narrative text, not structured data."""
    cleaned = []
    for msg in messages:
        if msg["role"] == "assistant":
            cleaned.append({"role": "assistant", "content": extract_narrative(msg["content"])})
        else:
            cleaned.append(msg)

    if len(cleaned) <= SUMMARIZE_AFTER_PAIRS * 2:
        return cleaned

    split_point = len(cleaned) - (SUMMARIZE_AFTER_PAIRS * 2)
    old_messages = cleaned[:split_point]
    recent_messages = cleaned[split_point:]
    summary = await summarize_messages(old_messages, model)
    if summary:
        summary_msg = {"role": "user", "content": f"[{summary}]\n\nContinue the story from here."}
        summary_response = {"role": "assistant", "content": "The story continues..."}
        return [summary_msg, summary_response] + recent_messages
    return cleaned[-MAX_MESSAGES:]


# ===== RESPONSE PARSING =====

def _parse_response(raw: str, current_location: str = "Varek", current_time: str = "Dawn",
                     current_role: str = "Stranger", turn_number: int = 0) -> dict:
    """Parse narrative from Ollama. Tries JSON first, falls back to plain text.
    Uses server-side detection as fallback for location/character/time."""
    text = raw.strip()

    # Try JSON first
    try:
        clean = text.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > 0:
            parsed = json.loads(clean[start:end])
            if "narrative" in parsed:
                for key in ["narrative", "location", "time", "role", "ooc"]:
                    if isinstance(parsed.get(key), str):
                        parsed[key] = clean_text(parsed[key])
                if not parsed.get("characters"):
                    parsed["characters"] = detect_characters_from_text(parsed.get("narrative", ""))
                # Backward compat: also set singular character as first match
                if parsed.get("characters"):
                    parsed["character"] = parsed["characters"][0]
                elif parsed.get("character"):
                    parsed["characters"] = [parsed["character"]]
                else:
                    parsed["characters"] = []
                    parsed["character"] = None
                if not parsed.get("location") or parsed["location"] in ("Varek", "Unknown", "null"):
                    parsed["location"] = detect_location_from_text(parsed.get("narrative", "")) or current_location
                norm_time = normalize_time(parsed.get("time", ""))
                if norm_time:
                    parsed["time"] = norm_time
                else:
                    # Model gave an unrecognised time — advance as fallback
                    parsed["time"] = advance_time(current_time, turn_number)
                return parsed
    except Exception:
        pass

    # Plain text parse
    location = current_location
    time_of_day = current_time
    role = current_role
    familiar_bond = None
    familiar_abandoned = None

    # Extract [LOCATION: ... | TIME: ... | ROLE: ...]
    status_match = re.search(
        r'\[LOCATION:\s*(.+?)\s*\|\s*TIME:\s*(.+?)\s*\|\s*ROLE:\s*(.+?)\s*\]',
        text, re.IGNORECASE
    )
    if status_match:
        location = status_match.group(1).strip()
        norm = normalize_time(status_match.group(2).strip())
        if norm:
            time_of_day = norm
        role = status_match.group(3).strip()
        text = text[:status_match.start()] + text[status_match.end():]

    # Extract [BOND: ...]
    bond_match = re.search(r'\[BOND:\s*(delea|olver)\s*\]', text, re.IGNORECASE)
    if bond_match:
        familiar_bond = bond_match.group(1).lower()
        text = text[:bond_match.start()] + text[bond_match.end():]

    # Extract [ABANDONED: ...]
    abandon_match = re.search(r'\[ABANDONED:\s*(delea|olver)\s*\]', text, re.IGNORECASE)
    if abandon_match:
        familiar_abandoned = abandon_match.group(1).lower()
        text = text[:abandon_match.start()] + text[abandon_match.end():]

    # Clean leftover bracketed meta
    text = re.sub(r'\[(?:LOCATION|TIME|ROLE|STATUS|CHARACTER|SETTING|SCENE)[^\]]*\]', '', text, flags=re.IGNORECASE)

    narrative = clean_text(text.strip())

    # Server-side fallback detection
    characters = detect_characters_from_text(narrative)
    detected_location = detect_location_from_text(narrative)
    if detected_location:
        location = detected_location

    # If model gave us a status line we use that time. Otherwise advance as fallback.
    if not status_match:
        time_of_day = advance_time(current_time, turn_number)

    return {
        "narrative": narrative,
        "location": location,
        "time": time_of_day,
        "role": role,
        "character": characters[0] if characters else None,
        "characters": characters,
        "familiar_bond": familiar_bond,
        "familiar_abandoned": familiar_abandoned,
        "ooc": None,
    }


# ===== STATIC FILES =====

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_game():
    with open(BASE_DIR / "static" / "index.html", "r") as f:
        return f.read()


# ===== NARRATION ENDPOINT =====

@app.post("/api/narrate")
async def narrate(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", None) or OLLAMA_MODEL
    player = body.get("player", {})
    current_location = body.get("currentLocation", "Varek")
    current_time = body.get("currentTime", "Dawn")
    current_role = body.get("currentRole", "Stranger")
    turn_number = body.get("turnNumber", 0)

    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    system_prompt = build_system_prompt(player)
    prepared = await prepare_messages(messages, model)

    # Build flat prompt — only narrative text, no JSON metadata
    history_text = ""
    for msg in prepared:
        role_label = "Player" if msg["role"] == "user" else "Narrator"
        history_text += f"{role_label}: {msg['content']}\n\n"
    full_prompt = f"{system_prompt}\n\n{history_text}Narrator:"

    chat_messages = [{"role": "user", "content": full_prompt}]

    async def event_stream():
        accumulated = ""
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={"model": model, "messages": chat_messages, "stream": True,
                          "options": {"temperature": 0.95, "num_predict": 2000}},
                ) as response:
                    if response.status_code != 200:
                        body_bytes = await response.aread()
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Ollama error {response.status_code}'})}\n\n"
                        return
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            accumulated += token
                            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                        if chunk.get("done"):
                            break
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        parsed = _parse_response(accumulated, current_location, current_time, current_role, turn_number)
        yield f"data: {json.dumps({'type': 'done', **parsed})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ===== PORTRAIT ENDPOINTS =====

@app.get("/api/portrait/{character_id}")
async def get_portrait(character_id: str):
    if character_id not in PORTRAIT_META:
        raise HTTPException(status_code=404, detail="Portrait not found")
    portrait_path = BASE_DIR / "static" / "portraits" / PORTRAIT_META[character_id]["file"]
    if not portrait_path.exists():
        raise HTTPException(status_code=404, detail="Portrait file not found")
    return FileResponse(portrait_path, media_type="image/png")


@app.get("/api/portraits")
async def list_portraits():
    return JSONResponse(content=PORTRAIT_META)


@app.get("/health")
async def health():
    return {"status": "ok", "ollama_url": OLLAMA_BASE_URL, "ollama_model": OLLAMA_MODEL}
