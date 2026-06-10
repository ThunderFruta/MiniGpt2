from flask import Flask, request, jsonify, render_template
import os
import traceback
import subprocess
import shutil
import importlib
import importlib.util
import json
import time
import uuid
import threading

app = Flask(__name__)

# Lazy model loader: uses `MiniGpt2/Predict.py` helpers if `USE_MODEL` env var is truthy.
MODEL = None
TOKENIZER = None
MODEL_LOADING = False
# Per-persona cached models
MODEL_CACHE = {}
# Which persona is currently loading (background)
MODEL_LOADING_PERSONA = None
# Lock for cache updates
from threading import Lock
CACHE_LOCK = Lock()

# Prediction logging configuration: set LOG_PREDICTIONS=0 to disable
LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "1").lower() not in ("0", "false", "no")


def _prediction_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(script_dir, "Prediction")
    return pred_dir


def log_prediction(entry: dict):
    """Append a JSON line describing a prediction to `Prediction/predictions.jsonl`.

    Adds timestamp and uuid if missing. Respects `LOG_PREDICTIONS` flag.
    """
    if not LOG_PREDICTIONS:
        return
    try:
        d = dict(entry)
        d.setdefault("id", uuid.uuid4().hex)
        d.setdefault("ts", time.time())
        pred_dir = _prediction_dir()
        os.makedirs(pred_dir, exist_ok=True)
        path = os.path.join(pred_dir, "predictions.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    except Exception:
        pass


def log_load_event(entry: dict):
    """Append a JSON line describing a model load event to `Prediction/load_events.jsonl`.
    Adds timestamp and uuid if missing. Respects `LOG_PREDICTIONS` flag.
    """
    if not LOG_PREDICTIONS:
        return
    try:
        d = dict(entry)
        d.setdefault("id", uuid.uuid4().hex)
        d.setdefault("ts", time.time())
        pred_dir = _prediction_dir()
        os.makedirs(pred_dir, exist_ok=True)
        path = os.path.join(pred_dir, "load_events.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    except Exception:
        pass


def get_model(persona: str = "duk"):
    """Return cached (model, tokenizer) for the given persona, loading it if needed.

    Supported personas: 'duk' (default, merged + adapter) and 'base' (base instruct model, no LoRA).
    """
    global MODEL_CACHE
    with CACHE_LOCK:
        if persona in MODEL_CACHE:
            return MODEL_CACHE[persona]

    predict_module = None
    try:
        predict_module = importlib.import_module("MiniGpt2.Predict")
    except Exception:
        predict_path = os.path.join(os.path.dirname(__file__), "Predict.py")
        spec = importlib.util.spec_from_file_location("minigpt2_predict", predict_path)
        predict_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predict_module)

    # read quantization/offload envs
    no_8bit = os.getenv("NO_8BIT", "").lower() in ("1", "true", "yes")
    cpu_offload = os.getenv("USE_CPU_OFFLOAD", "0").lower() in ("1", "true", "yes")
    use_4bit = os.getenv("USE_4BIT", "1").lower() in ("1", "true", "yes")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Normalize persona: disable the explicit `base` persona and map it to `duk`.
    if isinstance(persona, str) and persona.lower() == "base":
        print("[INFO] 'base' persona disabled; using 'duk' persona instead.")
        persona = "duk"

    # default persona: merged + adapter (prefer merged_stage2 + stage3-lora)
    base = os.getenv("BASE_MODEL", "merged_stage2")
    adapter = os.getenv("ADAPTER", "stage3-lora")

    base_candidates = [
        base,
        os.path.join(script_dir, base),
        os.path.join(script_dir, "LoraAdapters", base),
    ]
    base_path = None
    for cand in base_candidates:
        if cand and os.path.exists(cand):
            base_path = os.path.abspath(cand)
            break
    if base_path is None:
        base_path = base

    adapter_candidates = [
        adapter,
        os.path.join(script_dir, adapter),
        os.path.join(script_dir, "LoraAdapters", adapter),
    ]
    adapter_path = None
    for cand in adapter_candidates:
        if cand and os.path.exists(cand):
            adapter_path = os.path.abspath(cand)
            break

    if adapter_path is None:
        # No separate adapter folder found — this may be a merged base checkpoint.
        # Proceed to load the base model alone (treat as merged checkpoint).
        print(f"[WARN] No LoRA adapter found in candidates {adapter_candidates}; proceeding without adapter (assuming merged base).")
        adapter_path = None

    print(f"[INFO] Resolved base_path={base_path}, adapter_path={adapter_path} for persona={persona}")
    model, tokenizer = predict_module.load_model(base_path, adapter_path, use_8bit=not no_8bit, use_4bit=use_4bit, cpu_offload=cpu_offload)
    with CACHE_LOCK:
        MODEL_CACHE[persona] = (model, tokenizer)
    return model, tokenizer


def _background_load(force_cpu_offload: bool = False, persona: str = "duk"):
    """Background thread target to load the model. Sets MODEL and TOKENIZER globals.

    If force_cpu_offload is True, sets USE_CPU_OFFLOAD=1 for this load.
    """
    global MODEL_LOADING, MODEL, TOKENIZER
    try:
        global MODEL_LOADING_PERSONA
        MODEL_LOADING_PERSONA = persona
        if force_cpu_offload:
            os.environ["USE_CPU_OFFLOAD"] = "1"
        # call get_model(persona) which will populate MODEL_CACHE
        get_model(persona)
        try:
            log_load_event({"event": "background_load_success", "force_cpu_offload": force_cpu_offload})
        except Exception:
            pass
    except Exception as e:
        try:
            log_load_event({"event": "background_load_error", "error": str(e)})
        except Exception:
            pass
    finally:
        MODEL_LOADING_PERSONA = None


def start_background_loader(force_cpu_offload: bool = False, persona: str = "duk"):
    """Start a non-blocking background thread to load the model.

    Returns True if a loader was started, False if one is already running or model already loaded.
    """
    global MODEL_LOADING_PERSONA
    with CACHE_LOCK:
        if persona in MODEL_CACHE:
            return False
        if MODEL_LOADING_PERSONA is not None:
            return False
        MODEL_LOADING_PERSONA = persona
    t = threading.Thread(target=_background_load, args=(force_cpu_offload, persona), daemon=True)
    t.start()
    return True


@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    """Trigger loading the model (synchronous). Returns status and message.

    This is safer than forcing the frontend to know env vars; frontend can call
    `/load_model` and wait for completion before enabling model usage.
    """
    ok, resp = _require_api_key(request)
    if not ok:
        return resp

    # If caller requested to wait, perform synchronous load; otherwise start background loader
    body = request.get_json(silent=True) or {}
    wait = body.get("wait", False)
    force_cpu_offload = body.get("cpu_offload", False)
    persona = body.get("persona", "duk")

    # If the requested persona is already loaded, return ready
    with CACHE_LOCK:
        if persona in MODEL_CACHE:
            return jsonify({"status": "ready", "message": f"Model for persona {persona} already loaded"})
        if MODEL_LOADING_PERSONA is not None:
            return jsonify({"status": "loading", "message": f"Model loading in progress for persona {MODEL_LOADING_PERSONA}"})

    if wait:
        try:
            # Load the requested persona synchronously
            model, tokenizer = get_model(persona)
            try:
                log_load_event({
                    "event": "load_model",
                    "persona": persona,
                    "client_ip": request.remote_addr,
                    "user_agent": request.headers.get("User-Agent"),
                })
            except Exception:
                pass
            return jsonify({"status": "ready", "message": f"Model for persona {persona} loaded successfully"})
        except Exception as e:
            try:
                log_load_event({
                    "event": "load_model_error",
                    "persona": persona,
                    "client_ip": request.remote_addr,
                    "user_agent": request.headers.get("User-Agent"),
                    "error": str(e)
                })
            except Exception:
                pass
            tb = traceback.format_exc()
            return jsonify({"status": "error", "message": str(e), "traceback": tb}), 500
    else:
        started = start_background_loader(force_cpu_offload=force_cpu_offload)
        if started:
            return jsonify({"status": "loading", "message": "Background model load started"})
        else:
            return jsonify({"status": "loading", "message": "Model already loading or loaded"})


def _require_api_key(req):
    """If `API_KEY` env var is set, require requests to present that key.

    Accepts either `Authorization: Bearer <key>` or `X-API-Key: <key>` header.
    Returns (True, None) when OK, or (False, response_tuple) when not authorized.
    """
    api_key = os.getenv("API_KEY")
    if not api_key:
        return True, None

    # Check Authorization header
    auth = req.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth.split(None, 1)[1].strip()
        if token == api_key:
            return True, None

    # Check X-API-Key
    xkey = req.headers.get("X-API-Key")
    if xkey and xkey == api_key:
        return True, None

    # Also allow key in JSON body (not recommended but helpful for curl)
    try:
        j = req.get_json(silent=True) or {}
        if isinstance(j, dict) and j.get("api_key") == api_key:
            return True, None
    except Exception:
        pass

    return False, (jsonify({"error": "Unauthorized"}), 401)


@app.route("/", methods=["GET"]) 
def index():
    return render_template("chat.html")


@app.route("/predict", methods=["POST"]) 
def predict():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "missing 'prompt' in JSON body"}), 400
    # enforce API key if configured
    ok, resp = _require_api_key(request)
    if not ok:
        return resp
    # Decide whether to use the model for this request.
    # Request-level `use_model` (boolean) takes precedence; otherwise use the env var `USE_MODEL`.
    req_use = data.get("use_model", None)
    if isinstance(req_use, bool):
        use_model = req_use
    else:
        use_model = os.getenv("USE_MODEL", "0").lower() in ("1", "true", "yes")

    if use_model:
        try:
            predict_module = None
            try:
                predict_module = importlib.import_module("MiniGpt2.Predict")
            except Exception:
                predict_path = os.path.join(os.path.dirname(__file__), "Predict.py")
                spec = importlib.util.spec_from_file_location("minigpt2_predict", predict_path)
                predict_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(predict_module)

            # If persona requests the base instruct model, prefer a dedicated base (no LoRA)
            persona = data.get("persona", "duk")

            # Prefer cached model if already loaded for this persona — don't reload.
            with CACHE_LOCK:
                cached = MODEL_CACHE.get(persona)
                any_cached = bool(MODEL_CACHE)
            if cached is not None:
                model, tokenizer = cached
            else:
                # If we have any cached model for another persona, prefer to use it
                if any_cached:
                    with CACHE_LOCK:
                        model, tokenizer = next(iter(MODEL_CACHE.values()))
                else:
                    # No model loaded at all — do NOT attempt a synchronous load here.
                    # Start a background loader for the requested persona and ask client to retry.
                    try:
                        started = start_background_loader(force_cpu_offload=os.getenv("USE_CPU_OFFLOAD", "0").lower() in ("1","true","yes"), persona=persona)
                        log_load_event({"event": "predict_triggered_background_load", "persona": persona, "started": started, "client_ip": request.remote_addr})
                    except Exception:
                        pass
                    return jsonify({"error": "Model not loaded yet. Background load started. Retry after a minute.", "persona": persona}), 503
                if isinstance(persona, str) and persona.lower() == "base":
                    # Determine base instruct model path or HF id
                    base_instruct = os.getenv("BASE_INSTRUCT", "meta-llama/Meta-Llama-3-8B-Instruct")
                    # Resolve local candidates similar to get_model
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    base_candidates = [
                        base_instruct,
                        os.path.join(script_dir, base_instruct),
                        os.path.join(script_dir, "LoraAdapters", base_instruct),
                    ]
                    base_path = None
                    for cand in base_candidates:
                        if cand and os.path.exists(cand):
                            base_path = os.path.abspath(cand)
                            break
                    if base_path is None:
                        base_path = base_instruct

                    no_8bit = os.getenv("NO_8BIT", "").lower() in ("1", "true", "yes")
                    cpu_offload = os.getenv("USE_CPU_OFFLOAD", "0").lower() in ("1", "true", "yes")
                    # Default to 4-bit quantization unless explicitly disabled
                    use_4bit = os.getenv("USE_4BIT", "1").lower() in ("1", "true", "yes")

                    # Load the base instruct model directly (no adapter)
                    try:
                        model, tokenizer = predict_module.load_model(base_path, adapter_path=None, use_8bit=not no_8bit, use_4bit=use_4bit, cpu_offload=cpu_offload)
                    except Exception as e:
                        # Loading failed; fallback to any already-loaded model instead of reloading
                        try:
                            log_load_event({"event": "load_error_base_fallback", "error": str(e), "persona": persona})
                        except Exception:
                            pass
                        with CACHE_LOCK:
                            if MODEL_CACHE:
                                model, tokenizer = next(iter(MODEL_CACHE.values()))
                            else:
                                raise
                else:
                    try:
                        # Do not perform blocking loads here; only use cached models. If none, we would have returned above.
                        model, tokenizer = get_model(persona) if False else (model, tokenizer)
                    except Exception as e:
                        # On failure, prefer to use any cached model rather than attempting another load.
                        try:
                            log_load_event({"event": "load_error_fallback", "error": str(e), "persona": persona})
                        except Exception:
                            pass
                        with CACHE_LOCK:
                            if MODEL_CACHE:
                                model, tokenizer = next(iter(MODEL_CACHE.values()))
                            else:
                                raise

            # Accept optional history: list of pairs or objects. If none provided, use single user prompt.
            history_in = data.get("history")
            if history_in and isinstance(history_in, list):
                # normalize to list of (role, text)
                history = []
                for item in history_in:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        history.append((str(item[0]), str(item[1])))
                    elif isinstance(item, dict) and "role" in item and "text" in item:
                        history.append((str(item["role"]), str(item["text"])))
                    else:
                        # fallback: append as user text
                        history.append(("User", str(item)))
                    # ensure final user prompt appended if prompt provided
                    if prompt:
                        history.append(("User", prompt))
            else:
                history = [("User", prompt)]

            # persona selection affects prompt style: 'base' -> neutral instruct-style, otherwise Duk Bot style
            persona = data.get("persona", "duk")
            neutral_flag = True if (isinstance(persona, str) and persona.lower() == "base") else data.get("neutral", False)
            prompt_text = predict_module.build_prompt(history, neutral=neutral_flag)

            start_ts = time.time()
            # Try generate; if it fails with an offload/dispatch error, start a background reload with offload enabled and return an informative error.
            try:
                out = predict_module.generate(
                    model,
                    tokenizer,
                    prompt_text,
                    max_new_tokens=data.get("max_new_tokens", 64),
                )
            except Exception as gen_exc:
                msg = str(gen_exc)
                is_offload_error = (
                    "Some modules are dispatched on the CPU or the disk" in msg
                    or "load_in_8bit_fp32_cpu_offload" in msg
                    or "dispatch" in msg
                )
                if is_offload_error:
                    # Start a background reload with CPU offload enabled and inform the client to retry later.
                    try:
                        print("[WARN] Detected offload/dispatch error during generate; scheduling background reload with CPU offload")
                        log_load_event({
                            "event": "offload_retry_scheduled",
                            "error": msg,
                            "client_ip": request.remote_addr,
                        })
                    except Exception:
                        pass

                    # Clear current model and start background loader with offload
                    try:
                        global MODEL, TOKENIZER
                        MODEL = None
                        TOKENIZER = None
                        start_background_loader(force_cpu_offload=True)
                    except Exception:
                        pass

                    return jsonify({
                        "error": "Model needs CPU offload and is being reloaded in background. Please retry in a minute.",
                        "detail": msg
                    }), 503
                else:
                    # not an offload-related error; propagate
                    raise
            out = predict_module.postprocess(out)
            duration = time.time() - start_ts
            # Log prediction
            try:
                safe_headers = {k: v for k, v in request.headers.items()}
                safe_headers.pop('Authorization', None)
                safe_headers.pop('X-API-Key', None)
                log_prediction({
                    "prompt": prompt,
                    "history_len": len(history) if 'history' in locals() else None,
                    "use_model": True,
                    "reply": out,
                    "duration": duration,
                    "client_ip": request.remote_addr,
                    "user_agent": request.headers.get("User-Agent"),
                    "headers": safe_headers,
                })
            except Exception:
                pass
            return jsonify({"reply": out})
        except Exception as e:
            tb = traceback.format_exc()
            try:
                safe_headers = {k: v for k, v in request.headers.items()}
                safe_headers.pop('Authorization', None)
                safe_headers.pop('X-API-Key', None)
                log_prediction({
                    "prompt": prompt,
                    "history": data.get("history"),
                    "use_model": True,
                    "error": str(e),
                    "traceback": tb,
                    "client_ip": request.remote_addr,
                    "user_agent": request.headers.get("User-Agent"),
                    "headers": safe_headers,
                })
            except Exception:
                pass
            return jsonify({"error": str(e), "traceback": tb}), 500

    # Default: echo-style lightweight reply (fast, no model load)
    reply = f"Echo: {prompt}"
    try:
        safe_headers = {k: v for k, v in request.headers.items()}
        safe_headers.pop('Authorization', None)
        safe_headers.pop('X-API-Key', None)
        log_prediction({
            "prompt": prompt,
            "use_model": False,
            "reply": reply,
            "client_ip": request.remote_addr,
            "user_agent": request.headers.get("User-Agent"),
            "headers": safe_headers,
        })
    except Exception:
        pass
    return jsonify({"reply": reply})


@app.route('/model_status', methods=['GET'])
def model_status():
    """Return whether a persona-specific model is loaded or loading.

    Query param: `persona` (default: 'duk')
    """
    ok, resp = _require_api_key(request)
    if not ok:
        return resp

    persona = request.args.get('persona', 'duk')
    with CACHE_LOCK:
        if persona in MODEL_CACHE:
            return jsonify({"status": "ready", "persona": persona})
        if MODEL_LOADING_PERSONA is not None:
            return jsonify({"status": "loading", "persona": MODEL_LOADING_PERSONA})

    # Fallback: report whether any local model files look present and whether USE_MODEL env is set
    use_model_env = os.getenv("USE_MODEL", "0").lower() in ("1", "true", "yes")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_candidate = os.getenv("ADAPTER", "stage2-lora")
    candidates = [
        os.path.join(script_dir, "LoraAdapters", "merged_stage1"),
        os.path.join(script_dir, "LoraAdapters", adapter_candidate),
    ]
    model_files_found = any(os.path.exists(c) for c in candidates)
    return jsonify({"status": "not_loaded", "persona": persona, "use_model_env": use_model_env, "model_files_found": model_files_found})


@app.route('/debug_devices', methods=['GET'])
def debug_devices():
    """Return device placement info for cached models.

    Query param: `persona` (optional). If provided, returns info for that persona only.
    """
    ok, resp = _require_api_key(request)
    if not ok:
        return resp

    persona = request.args.get('persona', None)
    results = {}
    with CACHE_LOCK:
        items = [(p, MODEL_CACHE[p]) for p in MODEL_CACHE.keys() if (persona is None or p == persona)]
    for p, (m, tok) in items:
        info = {}
        # try to get HF device_map if available
        info["hf_device_map"] = getattr(m, "hf_device_map", None) or getattr(m, "device_map", None)
        # collect a small sample of parameter device placements
        try:
            devs = {}
            for i, (n, param) in enumerate(m.named_parameters()):
                d = str(param.device)
                devs[d] = devs.get(d, 0) + 1
                if i >= 200:
                    break
            info["param_device_sample_counts"] = devs
        except Exception as e:
            info["param_device_sample_error"] = str(e)
        # cuda info
        try:
            import torch as _t
            info["cuda_available"] = _t.cuda.is_available()
            if info["cuda_available"]:
                info["cuda_memory_allocated"] = _t.cuda.memory_allocated()
                info["cuda_memory_reserved"] = _t.cuda.memory_reserved()
        except Exception:
            pass
        results[p] = info

    if persona and persona not in results:
        return jsonify({"error": "persona not found or not loaded", "persona": persona}), 404
    return jsonify(results)


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") in ("1", "true", "yes")
    enable_ssl = os.getenv("ENABLE_SSL", "0").lower() in ("1", "true", "yes")

    def ensure_cert(cert_path: str, key_path: str, common_name: str = "localhost"):
        if os.path.exists(cert_path) and os.path.exists(key_path):
            return
        if shutil.which("openssl") is None:
            raise RuntimeError("OpenSSL not found; cannot generate self-signed cert.\n" \
                               "Install openssl or provide SSL_CERT/SSL_KEY paths.")
        print(f"[INFO] Generating self-signed cert {cert_path} / {key_path} (CN={common_name})")
        cmd = [
            "openssl",
            "req",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            key_path,
            "-x509",
            "-days",
            "365",
            "-out",
            cert_path,
            "-subj",
            f"/CN={common_name}",
        ]
        subprocess.check_call(cmd)

    ssl_context = None
    if enable_ssl:
        cert_file = os.getenv("SSL_CERT", "cert.pem")
        key_file = os.getenv("SSL_KEY", "key.pem")
        try:
            ensure_cert(cert_file, key_file, common_name=host)
        except Exception as e:
            print(f"[ERROR] Failed to ensure cert: {e}")
            raise
        ssl_context = (cert_file, key_file)

    print(f"Starting Flask on {host}:{port} (debug={debug}, ssl={enable_ssl})")
    app.run(host=host, port=port, debug=debug, ssl_context=ssl_context)
