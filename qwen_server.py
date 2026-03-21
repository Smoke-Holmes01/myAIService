import torch
import json
import base64
import io
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from rag_retriever import AncientArchitectureRAG
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "/home/yy/models/qwen2.5-vl-72b"
HOST = "0.0.0.0"
PORT = 11435

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

# 全局变量声明
model = None
tokenizer = None
processor = None
rag = None

def load_model():
    global model, tokenizer, processor, rag
    logger.info(f"正在加载模型: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"路径不存在: {MODEL_PATH}")
        return False
    
    try:
        # 加载主模型
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        logger.info(f"✅ 模型加载成功！设备: {model.device}")
        
        # ===== 在这里加载RAG检索器 =====
        try:
            from rag_retriever import AncientArchitectureRAG
            rag = AncientArchitectureRAG()
            logger.info("✅ RAG检索器加载成功")
        except Exception as e:
            logger.warning(f"⚠️ RAG检索器加载失败: {e}")
            rag = None
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
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
    global rag
    try:
        data = request.get_json()
        question = data.get('question', '')
        image_base64 = data.get('image', None)
        
        logger.info(f"收到问题: {question}")
        logger.info(f"是否有图片: {bool(image_base64)}")
        
        # 从知识库检索
        context = ""
        if rag:
            context = rag.get_context(question)
        
        # 构建系统提示
        if context:
            system_prompt = "你是一位专业的中国古建筑专家。请基于以下参考资料回答问题。"
            user_content = f"参考资料：\n{context}\n\n问题：{question}"
        else:
            system_prompt = "你是一位专业的中国古建筑专家。"
            user_content = question
        
        # ===== 处理输入（区分有无图片）=====
        if image_base64:
            logger.info("📷 处理图片输入...")
            
            # 解码图片
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
            logger.info(f"   图片尺寸: {image.size}")
            
            # 多模态消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_content}
                ]}
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            ).to(model.device)
            
        else:
            # 纯文本
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        # 解码回答
        if image_base64:
            response = processor.decode(outputs[0], skip_special_tokens=True)
        else:
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        logger.info(f"✅ 生成回答: {response[:100]}...")
        
        return app.response_class(
            response=json.dumps({
                "question": question,
                "answer": response,
                "used_knowledge": bool(context),
                "has_image": bool(image_base64)
            }, ensure_ascii=False),
            mimetype='application/json'
        )
        
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
