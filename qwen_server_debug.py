from qwen_server import app, config, load_model, logger


if __name__ == "__main__":
    if load_model():
        logger.info("调试服务启动成功，监听 %s:%s", config.host, config.port)
        app.run(host=config.host, port=config.port, threaded=True, debug=True)
    else:
        raise SystemExit("模型加载失败，调试服务未启动")
