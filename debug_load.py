import sys
print("1. 开始导入")
sys.stdout.flush()

import torch
print("2. torch 导入完成")
sys.stdout.flush()

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
print("3. transformers 导入完成")
sys.stdout.flush()

import os
print("4. os 导入完成")
sys.stdout.flush()

MODEL_PATH = "/home/yy/models/qwen2.5-vl-72b"
print(f"5. 模型路径: {MODEL_PATH}")
sys.stdout.flush()

print("6. 检查路径是否存在")
if not os.path.exists(MODEL_PATH):
    print(f"❌ 路径不存在: {MODEL_PATH}")
    sys.exit(1)
print("7. 路径存在")
sys.stdout.flush()

print("8. 开始加载 tokenizer...")
sys.stdout.flush()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("9. tokenizer 加载成功")
sys.stdout.flush()

print("10. 开始加载 processor...")
sys.stdout.flush()
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("11. processor 加载成功")
sys.stdout.flush()

print("12. 开始加载模型（这一步最慢，需要 5-10 分钟）...")
sys.stdout.flush()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
print("13. ✅ 模型加载成功！")
sys.stdout.flush()

print("14. 完成！")
