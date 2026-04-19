# VAREK — The Corpse City

A dark fantasy narrative game powered by a local Ollama model.

---

## Setup

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Create the Ollama model
```
ollama create varek-gemma3 -f Modelfile_varek_gemma3
```

### 3. Configure environment
Copy `.env.example` to `.env`:
```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=varek-gemma3
```

### 4. Add portraits
Drop portrait images into `static/portraits/`:
```
varek/static/portraits/
  orla.png
  thessaly.png
  maret.png
  ossian.png
  vicar.png
  drev.png
  cartographer.png
  delea_street.png
  delea_familiar.png
  olver_street.png
  olver_familiar.png
```
The game works without portraits — missing files are silently ignored.

### 5. Run
```
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000`

---

## Features
- Character creation — name, age, gender, appearance, background
- Named NPC cast with portraits detected by keyword scan on narrative text
- Familiar mechanic — meet Delea or Olver early, earn their bond over time
- Free-text player input — no preset choices, type whatever you want
- Automatic conversation summarization via the local model when history grows long
- Save / Load — downloads game state as JSON, reload anytime

---

## File structure
```
varek/
├── server.py
├── Modelfile_varek_gemma3
├── requirements.txt
├── .env.example
├── .env
└── static/
    ├── index.html
    └── portraits/
        └── *.png
```
