import torch
import json
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import os
import logging
import sys

print("=" * 50)
print("1. 开始导入...")
sys.stdout.flush()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("2. 导入完成，开始配置...")
sys.stdout.flush()

MODEL_PATH = "/home/yy/models/qwen2.5-vl-72b"
HOST = "0.0.0.0"
PORT = 11435

print(f"3. 模型路径: {MODEL_PATH}")
print(f"4. 路径存在: {os.path.exists(MODEL_PATH)}")
sys.stdout.flush()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

print("5. Flask 配置完成")
sys.stdout.flush()

model = None
tokenizer = None
processor = None
rag = None

print("6. 全局变量初始化完成，开始加载模型...")
sys.stdout.flush()

def load_model():
    global model, tokenizer, processor
    logger.info(f"正在加载模型: {MODEL_PATH}")
    print("7. 进入 load_model 函数")
    sys.stdout.flush()
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"路径不存在: {MODEL_PATH}")
        return False
    
    try:
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
        
        print("12. 开始加载模型（这一步最慢，可能需要几分钟）...")
        sys.stdout.flush()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print(f"13. 模型加载成功！设备: {model.device}")
        sys.stdout.flush()
        
        return True
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        print(f"错误: {e}")
        sys.stdout.flush()
        return False

@app.route('/api/ai/health', methods=['GET'])
def health():
    return jsonify({
        "status": "up",
        "service": "Qwen2.5-VL Service",
        "model": "qwen2.5-vl:72b",
        "device": str(model.device) if model else "not loaded"
    })

@app.route('/api/ai/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        messages = [
            {"role": "system", "content": "你是一位专业的中国古建筑专家。"},
            {"role": "user", "content": question}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        return app.response_class(
            response=json.dumps({
                "question": question,
                "answer": response
            }, ensure_ascii=False),
            mimetype='application/json'
        )
        
    except Exception as e:
        logger.error(f"错误: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("14. 进入主函数")
    sys.stdout.flush()
    if load_model():
        print(f"15. 服务启动成功！监听 {HOST}:{PORT}")
        sys.stdout.flush()
        app.run(host=HOST, port=PORT, threaded=True)
    else:
        print("16. 模型加载失败")
        sys.stdout.flush()
