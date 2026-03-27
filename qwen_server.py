import base64
import io
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Optional

import torch
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from openai import OpenAI
from PIL import Image
from PIL import UnidentifiedImageError
from transformers import AutoProcessor
from transformers import AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration

from rag_retriever import AncientArchitectureRAG


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer env %s=%r, fallback to %s", name, value, default)
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float env %s=%r, fallback to %s", name, value, default)
        return default


def _get_torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    mapping = {
        "auto": torch.float16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        logger.warning("Unknown TORCH_DTYPE=%r, fallback to bfloat16", name)
        return torch.bfloat16
    return mapping[normalized]


@dataclass
class ServerConfig:
    model_path: str = os.getenv("MODEL_PATH", "/home/yy/models/qwen2.5-vl-72b")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _get_env_int("PORT", 11435)
    use_remote_api: bool = _get_env_bool("USE_REMOTE_API", bool(os.getenv("SILICONFLOW_API_KEY")))
    siliconflow_api_key: str = os.getenv("SILICONFLOW_API_KEY", "")
    siliconflow_base_url: str = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    siliconflow_model: str = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    torch_dtype: torch.dtype = _get_torch_dtype(os.getenv("TORCH_DTYPE", "bfloat16"))
    max_new_tokens: int = _get_env_int("MAX_NEW_TOKENS", 512)
    temperature: float = _get_env_float("TEMPERATURE", 0.7)
    top_p: float = _get_env_float("TOP_P", 0.9)
    default_system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        "你是一位专业的中国古建筑 AI 助手。请优先结合可用资料作答，表达清晰，"
        "如果信息不足请明确说明不确定。",
    )
    enable_rag: bool = _get_env_bool("ENABLE_RAG", True)


config = ServerConfig()

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
CORS(app)

model = None
tokenizer = None
processor = None
rag: Optional[AncientArchitectureRAG] = None
image_matcher = None
image_matcher_error: Optional[str] = None
remote_client: Optional[OpenAI] = None


def _remote_api_enabled() -> bool:
    return config.use_remote_api and bool(config.siliconflow_api_key)


def _service_mode() -> str:
    if _remote_api_enabled():
        return "remote_api"
    if model is not None:
        return "local_model"
    return "degraded"


def resolve_input_device() -> str:
    if _remote_api_enabled():
        return "remote-api"

    if model is None:
        return "cpu"

    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        pass

    model_device = getattr(model, "device", None)
    if model_device is not None:
        return str(model_device)

    return "cpu"


def decode_image(image_base64: str) -> Image.Image:
    payload = image_base64.split(",", 1)[-1] if "," in image_base64 else image_base64
    image_bytes = base64.b64decode(payload)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _parse_top_k(value: Any, default: int = 3) -> int:
    if value in {None, ""}:
        return default
    try:
        return min(max(int(value), 1), 10)
    except (TypeError, ValueError):
        return default


def _matcher_project_root() -> Path:
    return (Path(__file__).resolve().parents[1] / "compterdesign").resolve()


def _matcher_render_root() -> Path:
    return _matcher_project_root() / "outputs" / "renders"


def get_remote_client() -> OpenAI:
    global remote_client

    if not _remote_api_enabled():
        raise RuntimeError("Remote API is not enabled. Please set SILICONFLOW_API_KEY.")

    if remote_client is None:
        remote_client = OpenAI(
            api_key=config.siliconflow_api_key,
            base_url=config.siliconflow_base_url,
        )

    return remote_client


def _build_remote_messages(question: str, context: str, image_base64: Optional[str]) -> list[dict[str, Any]]:
    if context:
        user_text = f"参考资料：\n{context}\n\n问题：{question}"
    else:
        user_text = question

    if not image_base64:
        return [
            {"role": "system", "content": config.default_system_prompt},
            {"role": "user", "content": user_text},
        ]

    return [
        {"role": "system", "content": config.default_system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_base64}},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def _extract_remote_answer_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")).strip())
                continue

            item_type = getattr(item, "type", None)
            if item_type == "text":
                parts.append(str(getattr(item, "text", "")).strip())

        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def generate_remote_answer(question: str, context: str, image_base64: Optional[str]) -> str:
    client = get_remote_client()
    response = client.chat.completions.create(
        model=config.siliconflow_model,
        messages=_build_remote_messages(question=question, context=context, image_base64=image_base64),
        max_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )

    if not response.choices:
        raise RuntimeError("Remote API returned no choices.")

    answer = _extract_remote_answer_content(response.choices[0].message.content)
    if not answer:
        raise RuntimeError("Remote API returned an empty answer.")
    return answer


def get_image_matcher():
    global image_matcher, image_matcher_error

    if image_matcher is not None:
        return image_matcher

    if image_matcher_error is not None:
        raise RuntimeError(image_matcher_error)

    project_root = _matcher_project_root()
    if not project_root.exists():
        image_matcher_error = f"Matcher project not found: {project_root}"
        raise RuntimeError(image_matcher_error)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from matcher_service import SingleImageMatcher
        from matcher_service import SingleImageMatcherConfig

        image_matcher = SingleImageMatcher(
            SingleImageMatcherConfig(
                project_root=project_root,
                render_root=project_root / "outputs" / "renders",
                query_root=project_root / "outputs" / "agent_queries",
            )
        )
        return image_matcher
    except Exception as exc:
        image_matcher_error = f"Failed to initialize image matcher: {exc}"
        logger.exception("Image matcher initialization failed")
        raise RuntimeError(image_matcher_error) from exc


def _format_match_answer(match_result: dict[str, Any]) -> str:
    matches = match_result.get("matches", [])
    if not matches:
        return "No matching model was found."

    best_match = matches[0]
    lines = [
        "I matched the uploaded image against the pre-rendered 3D model library.",
        f"Best match: model {best_match['model_name']} (score {best_match['score']:.4f}).",
        "Top candidates:",
    ]

    for index, item in enumerate(matches, start=1):
        view = item.get("best_view", {})
        lines.append(
            f"{index}. model {item['model_name']} | score {item['score']:.4f} | "
            f"azimuth {view.get('azimuth')} | elevation {view.get('elevation')}"
        )

    return "\n".join(lines)


def _run_match_pipeline(question: str, image_base64: str, top_k: int) -> dict[str, Any]:
    image = decode_image(image_base64)
    matcher = get_image_matcher()

    upload_root = Path(__file__).resolve().parent / "tmp_uploads"
    upload_root.mkdir(parents=True, exist_ok=True)

    query_id = f"agent_{uuid.uuid4().hex[:12]}"
    upload_path = upload_root / f"{query_id}.png"
    image.save(upload_path)

    try:
        match_result = matcher.match_image(
            image_path=str(upload_path),
            top_k=top_k,
            query_id=query_id,
        )
    finally:
        if upload_path.exists():
            upload_path.unlink()

    return {
        "question": question,
        "answer": _format_match_answer(match_result),
        "used_knowledge": False,
        "has_image": True,
        "used_matcher": True,
        "match_result": match_result,
    }


def build_messages(question: str, context: str, image: Optional[Image.Image]) -> list[dict[str, Any]]:
    if context:
        user_text = f"参考资料：\n{context}\n\n问题：{question}"
    else:
        user_text = question

    if image is None:
        return [
            {"role": "system", "content": config.default_system_prompt},
            {"role": "user", "content": user_text},
        ]

    return [
        {"role": "system", "content": config.default_system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def generate_answer(messages: list[dict[str, Any]], image: Optional[Image.Image]) -> str:
    input_device = resolve_input_device()

    if image is None:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(input_device)
    else:
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(input_device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.temperature > 0,
            top_p=config.top_p,
        )

    input_length = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    generated_ids = outputs[:, input_length:] if input_length else outputs
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded[0].strip()


def _load_rag_if_enabled() -> Optional[AncientArchitectureRAG]:
    if not config.enable_rag:
        return None

    try:
        loaded_rag = AncientArchitectureRAG()
        logger.info("RAG retriever loaded successfully")
        return loaded_rag
    except Exception:
        logger.exception("RAG retriever failed to load; continuing without RAG")
        return None


def load_model() -> bool:
    global model, tokenizer, processor, rag

    rag = _load_rag_if_enabled()

    if _remote_api_enabled():
        logger.info(
            "Remote API mode enabled. Using SiliconFlow model %s via %s",
            config.siliconflow_model,
            config.siliconflow_base_url,
        )
        model = None
        tokenizer = None
        processor = None
        return True

    logger.info("Loading local model from: %s", config.model_path)

    if not os.path.exists(config.model_path):
        logger.error("Local model path does not exist: %s", config.model_path)
        return False

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(config.model_path, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_path,
            device_map="auto",
            torch_dtype=config.torch_dtype,
            trust_remote_code=True,
        )
        logger.info("Local model loaded successfully")
        return True
    except Exception:
        logger.exception("Local model failed to load")
        return False


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/ai/health", methods=["GET"])
def health():
    matcher_assets_ready = _matcher_render_root().exists()
    service_ready = _remote_api_enabled() or model is not None
    return jsonify(
        {
            "status": "up" if service_ready else "degraded",
            "service": "Chinese Ancient Architecture AI Assistant",
            "service_mode": _service_mode(),
            "model_loaded": model is not None or _remote_api_enabled(),
            "remote_api_enabled": _remote_api_enabled(),
            "remote_model": config.siliconflow_model if _remote_api_enabled() else None,
            "rag_loaded": rag is not None,
            "model_path": config.model_path,
            "device": resolve_input_device(),
            "matcher_render_root": str(_matcher_render_root()),
            "matcher_assets_ready": matcher_assets_ready,
            "matcher_error": image_matcher_error,
        }
    )


@app.route("/api/agent/match", methods=["POST"])
def agent_match():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    image_base64 = data.get("image")
    top_k = _parse_top_k(data.get("top_k"), default=3)

    if not image_base64:
        return jsonify({"error": "image is required"}), 400

    try:
        response_body = _run_match_pipeline(question=question, image_base64=image_base64, top_k=top_k)
        return app.response_class(
            response=json.dumps(response_body, ensure_ascii=False),
            mimetype="application/json",
        )
    except (ValueError, UnidentifiedImageError, OSError):
        logger.exception("Image decode failed for matcher route")
        return jsonify({"error": "image is not a valid base64 image"}), 400
    except RuntimeError as exc:
        logger.exception("Matcher route is unavailable")
        return jsonify({"error": str(exc)}), 503
    except Exception:
        logger.exception("Matcher route failed")
        return jsonify({"error": "internal server error"}), 500


@app.route("/api/ai/chat", methods=["POST"])
def chat():
    if not _remote_api_enabled() and (model is None or tokenizer is None or processor is None):
        return jsonify({"error": "model is not ready"}), 503

    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    image_base64 = data.get("image")
    has_image = bool(image_base64)

    if not question:
        return jsonify({"error": "question cannot be empty"}), 400

    logger.info("Received question: %s", question)
    logger.info("Has image: %s", has_image)

    try:
        image = decode_image(image_base64) if has_image else None
    except (ValueError, UnidentifiedImageError, OSError):
        logger.exception("Image decode failed")
        return jsonify({"error": "image is not a valid base64 image"}), 400

    try:
        context = rag.get_context(question) if rag else ""
        if _remote_api_enabled():
            answer = generate_remote_answer(
                question=question,
                context=context,
                image_base64=image_base64 if has_image else None,
            )
        else:
            messages = build_messages(question=question, context=context, image=image)
            answer = generate_answer(messages=messages, image=image)

        logger.info("Answer generated successfully, length=%s", len(answer))

        response_body = {
            "question": question,
            "answer": answer,
            "used_knowledge": bool(context),
            "has_image": has_image,
            "service_mode": _service_mode(),
        }
        return app.response_class(
            response=json.dumps(response_body, ensure_ascii=False),
            mimetype="application/json",
        )
    except Exception:
        logger.exception("Chat request failed")
        return jsonify({"error": "internal server error"}), 500


if __name__ == "__main__":
    if load_model():
        logger.info("Service started successfully on %s:%s", config.host, config.port)
        app.run(host=config.host, port=config.port, threaded=True)
    raise SystemExit("Service failed to start")
