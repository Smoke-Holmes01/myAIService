import logging
import os
import sys

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


MODEL_PATH = os.getenv("MODEL_PATH", "/home/yy/models/qwen2.5-vl-72b")
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "bfloat16").lower()


def resolve_dtype() -> torch.dtype:
    if TORCH_DTYPE in {"float16", "fp16"}:
        return torch.float16
    if TORCH_DTYPE in {"float32", "fp32"}:
        return torch.float32
    return torch.bfloat16


if __name__ == "__main__":
    logger.info("开始检查模型目录: %s", MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        logger.error("模型路径不存在")
        sys.exit(1)

    logger.info("加载 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    logger.info("tokenizer 加载成功，词表大小=%s", len(tokenizer))

    logger.info("加载 processor")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    logger.info("processor 加载成功，类型=%s", type(processor).__name__)

    logger.info("加载模型，dtype=%s", resolve_dtype())
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=resolve_dtype(),
        trust_remote_code=True,
    )
    first_device = next(model.parameters()).device
    logger.info("模型加载成功，输入设备=%s", first_device)
