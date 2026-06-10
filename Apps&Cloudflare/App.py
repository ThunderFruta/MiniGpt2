from flask import Flask, request, jsonify, render_template
import os
import sys
import traceback
import subprocess
import shutil
import importlib
import importlib.util
import json
import time
import uuid
import threading
from threading import Lock

#cloudflared tunnel --url http://localhost:5000

# ----------------------------------------
# Make stdout unbuffered (important!)
# ----------------------------------------
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ----------------------------------------
# Flask app
# ----------------------------------------
app = Flask(__name__)

# ----------------------------------------
# Globals / Locks
# ----------------------------------------
MODEL_CACHE = {}
MODEL_LOADING_PERSONA = None
CACHE_LOCK = Lock()

CLOUDFLARED_PROC = None
CLOUDFLARED_URL = None
CLOUDFLARED_LOCK = Lock()
MODEL_LOADING = False

LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "1").lower() not in ("0", "false", "no")

# ----------------------------------------
# Helpers
# ----------------------------------------

def _prediction_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "Prediction")


def log_prediction(entry: dict):
    if not LOG_PREDICTIONS:
        return
    try:
        entry = dict(entry)
        entry.setdefault("id", uuid.uuid4().hex)
        entry.setdefault("ts", time.time())
        os.makedirs(_prediction_dir(), exist_ok=True)
        with open(os.path.join(_prediction_dir(), "predictions.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def log_load_event(entry: dict):
    if not LOG_PREDICTIONS:
        return
    try:
        entry = dict(entry)
        entry.setdefault("id", uuid.uuid4().hex)
        entry.setdefault("ts", time.time())
        os.makedirs(_prediction_dir(), exist_ok=True)
        with open(os.path.join(_prediction_dir(), "load_events.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _require_api_key(req):
    api_key = os.getenv("API_KEY")
    if not api_key:
        return True, None

    auth = req.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and auth.split(None, 1)[1] == api_key:
        return True, None

    if req.headers.get("X-API-Key") == api_key:
        return True, None

    try:
        j = req.get_json(silent=True) or {}
        if j.get("api_key") == api_key:
            return True, None
    except Exception:
        pass

    return False, (jsonify({"error": "Unauthorized"}), 401)

# ----------------------------------------
# Cloudflare Tunnel
# ----------------------------------------

def start_cloudflared_tunnel(target=None):
    global CLOUDFLARED_PROC, CLOUDFLARED_URL

    with CLOUDFLARED_LOCK:
        if CLOUDFLARED_PROC and CLOUDFLARED_PROC.poll() is None:
            return {"started": False, "reason": "already_running", "url": CLOUDFLARED_URL}

        cf = shutil.which("cloudflared")
        if not cf:
            return {"started": False, "reason": "cloudflared_not_found"}

        if not target:
            target = f"http://localhost:{os.getenv('FLASK_PORT', '5000')}"

        proc = subprocess.Popen(
            [cf, "tunnel", "--url", target],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        CLOUDFLARED_PROC = proc
        CLOUDFLARED_URL = None

        def reader():
            global CLOUDFLARED_URL
            for line in proc.stdout:
                print("[cloudflared]", line.rstrip())
                if "https://" in line and "trycloudflare.com" in line:
                    for token in line.split():
                        if token.startswith("https://") and "trycloudflare.com" in token:
                            CLOUDFLARED_URL = token

        threading.Thread(target=reader, daemon=True).start()

        return {"started": True, "url": CLOUDFLARED_URL}


def stop_cloudflared_tunnel():
    global CLOUDFLARED_PROC, CLOUDFLARED_URL
    with CLOUDFLARED_LOCK:
        if not CLOUDFLARED_PROC:
            return {"stopped": False}
        proc = CLOUDFLARED_PROC
        CLOUDFLARED_PROC = None
        proc.terminate()
        CLOUDFLARED_URL = None
        return {"stopped": True}


def cloudflared_status():
    with CLOUDFLARED_LOCK:
        return {
            "running": CLOUDFLARED_PROC is not None and CLOUDFLARED_PROC.poll() is None,
            "url": CLOUDFLARED_URL,
        }
    
def _load_predict_module():
    try:
        return importlib.import_module("MiniGpt2.Predict")
    except Exception:
        spec = importlib.util.spec_from_file_location(
            "predict", os.path.join(os.path.dirname(__file__), "Predict.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def generate_reply(model, tokenizer, prompt, max_new_tokens=128):
    predict_module = _load_predict_module()

    # build prompt (single-turn)
    prompt_text = predict_module.build_prompt(
        history=[("User", prompt)],
        neutral=True,
    )

    output = predict_module.generate(
        model,
        tokenizer,
        prompt_text,
        max_new_tokens=max_new_tokens,
    )
    return predict_module.postprocess(output)

# ----------------------------------------
# Model Loader (minimal, safe)
# ----------------------------------------

def get_model(persona="duk"):
    with CACHE_LOCK:
        if persona in MODEL_CACHE:
            return MODEL_CACHE[persona]

    predict_module = _load_predict_module()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- Correct absolute paths ----
    base_path = os.path.join(
        script_dir,
        "LoraAdapters",
        "merged_stage2"
    )

    adapter_path = os.path.join(
        script_dir,
        "LoraAdapters",
        "stage3-lora"
    )

    # ---- Hard validation (fail loud, not silent) ----
    if not os.path.isdir(base_path):
        raise RuntimeError(f"Base model not found: {base_path}")

    if not os.path.isdir(adapter_path):
        raise RuntimeError(f"LoRA adapter not found: {adapter_path}")

    print("[INFO] Base model path:", base_path)
    print("[INFO] Stage-3 LoRA path:", adapter_path)

    model, tokenizer = predict_module.load_model(
        base_path,
        adapter_path
    )

    with CACHE_LOCK:
        MODEL_CACHE[persona] = (model, tokenizer)

    print("[INFO] Model fully loaded (merged_stage2 + stage3 LoRA)")
    return model, tokenizer


def start_background_loader(persona="duk"):
    global MODEL_LOADING, MODEL_LOADING_PERSONA

    with CACHE_LOCK:
        if persona in MODEL_CACHE:
            return False
        if MODEL_LOADING:
            return False
        MODEL_LOADING = True
        MODEL_LOADING_PERSONA = persona

    def loader():
        global MODEL_LOADING, MODEL_LOADING_PERSONA
        try:
            get_model(persona)
            log_load_event({"event": "background_load_success", "persona": persona})
            print("[INFO] Model load complete")
        except Exception as e:
            log_load_event({"event": "background_load_error", "error": str(e)})
            print("[ERROR] Model load failed:", e)
        finally:
            with CACHE_LOCK:
                MODEL_LOADING = False
                MODEL_LOADING_PERSONA = None

    threading.Thread(target=loader, daemon=True).start()
    return True


# ----------------------------------------
# Routes
# ----------------------------------------

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/predict", methods=["POST"])
def predict():
    ok, resp = _require_api_key(request)
    if not ok:
        return resp

    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "missing prompt"}), 400

    persona = data.get("persona", "duk")
    max_new_tokens = int(data.get("max_new_tokens", 128))

    # --------------------------------------------------
    # 1) Check if model is already loaded
    # --------------------------------------------------
    with CACHE_LOCK:
        cached: tuple | None = MODEL_CACHE.get(persona)
        loading = (MODEL_LOADING_PERSONA == persona)

    if cached is None:
        if not loading:
            print(f"[INFO] Auto-loading model for persona={persona}")
            start_background_loader(persona)

        return jsonify({
            "error": "model loading",
            "status": "loading",
            "persona": persona
        }), 503
    else:
        model, tokenizer = cached
    


    model, tokenizer = cached

    # --------------------------------------------------
    # 3) Run generation
    # --------------------------------------------------
    start_ts = time.time()
    try:
        reply = generate_reply(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens
        )
    except Exception as e:
        tb = traceback.format_exc()
        log_prediction({
            "prompt": prompt,
            "error": str(e),
            "traceback": tb
        })
        return jsonify({"error": str(e)}), 500

    duration = time.time() - start_ts

    # --------------------------------------------------
    # 4) Log + return
    # --------------------------------------------------
    log_prediction({
        "prompt": prompt,
        "reply": reply,
        "duration": duration,
        "persona": persona,
    })

    return jsonify({
        "reply": reply,
        "duration": duration,
        "persona": persona
    })

@app.route("/model_status")
def model_status():
    persona = request.args.get("persona", "duk")
    with CACHE_LOCK:
        if persona in MODEL_CACHE:
            return jsonify({"status": "ready"})
        if MODEL_LOADING:
            return jsonify({"status": "loading"})
    return jsonify({"status": "not_loaded"})



@app.route("/load_model", methods=["POST"])
def load_model():
    ok, resp = _require_api_key(request)
    if not ok:
        return resp

    with CACHE_LOCK:
        if MODEL_CACHE:
            return jsonify({"status": "ready"})

    started = start_background_loader()
    return jsonify({
        "status": "loading",
        "started": started
    })



@app.route("/tunnel/start", methods=["POST"])
def tunnel_start():
    ok, resp = _require_api_key(request)
    if not ok:
        return resp
    return jsonify(start_cloudflared_tunnel())


@app.route("/tunnel/status")
def tunnel_status():
    ok, resp = _require_api_key(request)
    if not ok:
        return resp
    return jsonify(cloudflared_status())


@app.route("/tunnel/stop", methods=["POST"])
def tunnel_stop():
    ok, resp = _require_api_key(request)
    if not ok:
        return resp
    return jsonify(stop_cloudflared_tunnel())

# ----------------------------------------
# Main
# ----------------------------------------

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")

    print(f"[INFO] Flask starting on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
