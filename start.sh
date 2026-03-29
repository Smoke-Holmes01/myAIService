#!/bin/bash
export LOCAL_MODEL_PATH=/home/yy/models/Qwen3.5-9B
export USE_REMOTE_API=true
export SILICONFLOW_API_KEY=sk-lpnkyqmqhjsfotmuwamqicastbpdnnrtqqoitnnyfyxtjqwj
export SILICONFLOW_MODEL=Qwen/Qwen3.5-397B-A17B
export ENABLE_RAG=false
export HOST=0.0.0.0
export PORT=11435


cd /home/yy/Agent/my-ai-service
python qwen_server.py
