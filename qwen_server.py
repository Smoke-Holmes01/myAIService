import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from rag_retriever import AncientArchitectureRAG


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("环境变量 %s=%r 不是合法整数，回退为 %s", name, value, default)
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("环境变量 %s=%r 不是合法浮点数，回退为 %s", name, value, default)
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
        logger.warning("未知 TORCH_DTYPE=%r，回退为 bfloat16", name)
        return torch.bfloat16
    return mapping[normalized]


@dataclass
class ServerConfig:
    model_path: str = os.getenv("MODEL_PATH", "/home/yy/models/qwen2.5-vl-72b")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _get_env_int("PORT", 11435)
    torch_dtype: torch.dtype = _get_torch_dtype(os.getenv("TORCH_DTYPE", "bfloat16"))
    max_new_tokens: int = _get_env_int("MAX_NEW_TOKENS", 512)
    temperature: float = _get_env_float("TEMPERATURE", 0.7)
    top_p: float = _get_env_float("TOP_P", 0.9)
    default_system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        "你是一位专业的中国古建筑 AI 助手。请优先结合检索到的资料回答，"
        "回答时保持准确、清晰；如果资料不足，请明确说明不确定。",
    )
    enable_rag: bool = os.getenv("ENABLE_RAG", "true").lower() not in {"0", "false", "no"}


config = ServerConfig()

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
CORS(app)

model = None
tokenizer = None
processor = None
rag: Optional[AncientArchitectureRAG] = None


def resolve_input_device() -> torch.device:
    if model is None:
        return torch.device("cpu")

    try:
        return next(model.parameters()).device
    except StopIteration:
        pass

    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)

    return torch.device("cpu")


def decode_image(image_base64: str) -> Image.Image:
    payload = image_base64.split(",", 1)[-1] if "," in image_base64 else image_base64
    image_bytes = base64.b64decode(payload)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def build_messages(question: str, context: str, image: Optional[Image.Image]) -> list[dict]:
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


def generate_answer(messages: list[dict], image: Optional[Image.Image]) -> str:
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


def load_model() -> bool:
    global model, tokenizer, processor, rag
    logger.info("正在加载模型: %s", config.model_path)

    if not os.path.exists(config.model_path):
        logger.error("模型路径不存在: %s", config.model_path)
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
        logger.info("模型加载成功")
    except Exception:
        logger.exception("模型加载失败")
        return False

    rag = None
    if config.enable_rag:
        try:
            rag = AncientArchitectureRAG()
            logger.info("RAG 检索器加载成功")
        except Exception:
            logger.exception("RAG 检索器加载失败，服务将以纯模型模式运行")

    return True


@app.route("/api/ai/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "up" if model is not None else "degraded",
            "service": "Chinese Ancient Architecture AI Assistant",
            "model_loaded": model is not None,
            "rag_loaded": rag is not None,
            "model_path": config.model_path,
            "device": str(resolve_input_device()),
        }
    )


@app.route("/api/ai/chat", methods=["POST"])
def chat():
    if model is None or tokenizer is None or processor is None:
        return jsonify({"error": "模型尚未加载完成"}), 503

    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    image_base64 = data.get("image")
    has_image = bool(image_base64)

    if not question:
        return jsonify({"error": "question 不能为空"}), 400

    logger.info("收到问题: %s", question)
    logger.info("是否包含图片: %s", has_image)

    try:
        image = decode_image(image_base64) if has_image else None
    except (ValueError, UnidentifiedImageError, OSError):
        logger.exception("图片解析失败")
        return jsonify({"error": "image 不是合法的 base64 图片"}), 400

    try:
        context = rag.get_context(question) if rag else ""
        messages = build_messages(question=question, context=context, image=image)
        answer = generate_answer(messages=messages, image=image)
        logger.info("回答生成成功，长度=%s", len(answer))

        response_body = {
            "question": question,
            "answer": answer,
            "used_knowledge": bool(context),
            "has_image": has_image,
        }
        return app.response_class(
            response=json.dumps(response_body, ensure_ascii=False),
            mimetype="application/json",
        )
    except Exception:
        logger.exception("问答处理失败")
        return jsonify({"error": "服务内部错误"}), 500


if __name__ == "__main__":
    if load_model():
        logger.info("服务启动成功，监听 %s:%s", config.host, config.port)
        app.run(host=config.host, port=config.port, threaded=True)
    else:
        raise SystemExit("模型加载失败，服务未启动")
