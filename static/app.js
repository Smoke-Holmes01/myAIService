const elements = {
  form: document.getElementById("chat-form"),
  messages: document.getElementById("messages"),
  question: document.getElementById("question-input"),
  submit: document.getElementById("submit-button"),
  imageInput: document.getElementById("image-input"),
  previewCard: document.getElementById("image-preview-card"),
  previewImage: document.getElementById("image-preview"),
  previewName: document.getElementById("image-name"),
  removeImage: document.getElementById("remove-image"),
  clearChat: document.getElementById("clear-chat"),
  helperText: document.getElementById("helper-text"),
  statusDot: document.getElementById("status-dot"),
  statusText: document.getElementById("status-text"),
  statusMeta: document.getElementById("status-meta"),
  template: document.getElementById("message-template"),
};

let selectedImageBase64 = "";
let selectedImageName = "";

function scrollMessagesToBottom() {
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function createMessage(role, content, options = {}) {
  const node = elements.template.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  if (options.loading) {
    node.classList.add("loading");
  }

  const badge = node.querySelector(".message-badge");
  const body = node.querySelector(".message-body");
  badge.textContent = role === "assistant" ? "AI" : "我";
  body.textContent = content;

  if (options.imageUrl) {
    const image = document.createElement("img");
    image.src = options.imageUrl;
    image.alt = "用户上传图片";
    image.style.width = "100%";
    image.style.maxWidth = "280px";
    image.style.display = "block";
    image.style.borderRadius = "16px";
    image.style.marginBottom = "14px";
    image.style.border = "1px solid rgba(54, 36, 23, 0.12)";
    body.prepend(image);
  }

  elements.messages.appendChild(node);
  scrollMessagesToBottom();
  return node;
}

function setPendingState(pending) {
  elements.submit.disabled = pending;
  elements.question.disabled = pending;
  elements.imageInput.disabled = pending;
  elements.helperText.textContent = pending
    ? "模型正在生成回答，请保持页面开启并耐心等待。"
    : "模型回复可能需要几十秒，请耐心等待。";
}

function clearImageSelection() {
  selectedImageBase64 = "";
  selectedImageName = "";
  elements.imageInput.value = "";
  elements.previewCard.classList.add("hidden");
  elements.previewImage.removeAttribute("src");
  elements.previewName.textContent = "未选择文件";
}

function shouldUseMatcher(question, hasImage) {
  if (!hasImage) {
    return false;
  }

  const matcherKeywords = [
    "match",
    "matcher",
    "3d",
    "obj",
    "mesh",
    "model",
    "匹配",
    "模型",
    "三维",
    "对应",
  ];
  const normalizedQuestion = String(question || "").toLowerCase();
  return matcherKeywords.some((keyword) => normalizedQuestion.includes(keyword));
}

async function updateHealthStatus() {
  try {
    const response = await fetch("/api/ai/health");
    const data = await response.json();
    const isOnline = response.ok && data.model_loaded;
    const status = isOnline ? "服务在线，可开始演示" : "服务未就绪";

    elements.statusDot.classList.remove("offline", "online");
    elements.statusDot.classList.add(isOnline ? "online" : "offline");
    elements.statusText.textContent = status;
    elements.statusMeta.textContent = `设备：${data.device || "未知"} ｜ RAG：${data.rag_loaded ? "已启用" : "未启用"} ｜ 模型路径：${data.model_path || "未知"}`;
  } catch (error) {
    elements.statusDot.classList.remove("online");
    elements.statusDot.classList.add("offline");
    elements.statusText.textContent = "无法连接 AI 服务";
    elements.statusMeta.textContent = "请确认 qwen_server.py 已启动，并检查服务端口是否可访问。";
  }
}

elements.imageInput.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) {
    clearImageSelection();
    return;
  }

  selectedImageName = file.name;
  const reader = new FileReader();
  reader.onload = () => {
    selectedImageBase64 = String(reader.result || "");
    elements.previewImage.src = selectedImageBase64;
    elements.previewName.textContent = selectedImageName;
    elements.previewCard.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
});

elements.removeImage.addEventListener("click", () => {
  clearImageSelection();
});

elements.clearChat.addEventListener("click", () => {
  elements.messages.innerHTML = "";
  createMessage(
    "assistant",
    "对话已清空。你可以继续提问，也可以重新上传一张古建筑图片。"
  );
});

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = elements.question.value.trim();
  if (!question) {
    elements.question.focus();
    return;
  }

  const userText = selectedImageName
    ? `${question}\n\n[已上传图片：${selectedImageName}]`
    : question;

  createMessage("user", userText, {
    imageUrl: selectedImageBase64 || null,
  });

  const loadingMessage = createMessage("assistant", "正在分析，请稍候...", {
    loading: true,
  });

  setPendingState(true);

  try {
    const useMatcher = shouldUseMatcher(question, Boolean(selectedImageBase64));
    const endpoint = useMatcher ? "/api/agent/match" : "/api/ai/chat";

    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        image: selectedImageBase64 || undefined,
        use_matcher: useMatcher,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "请求失败");
    }

    let answer = data.used_knowledge
      ? `${data.answer}\n\n[本次回答已结合知识库检索结果]`
      : `${data.answer}\n\n[本次回答基于模型直接生成]`;

    if (data.used_matcher) {
      answer = `${data.answer}\n\n[This result came from the compterdesign matcher.]`;
    }

    loadingMessage.querySelector(".message-body").textContent = answer;
    loadingMessage.classList.remove("loading");
  } catch (error) {
    loadingMessage.querySelector(".message-body").textContent = `调用失败：${error.message}`;
    loadingMessage.classList.remove("loading");
  } finally {
    setPendingState(false);
    elements.question.value = "";
    clearImageSelection();
    scrollMessagesToBottom();
  }
});

updateHealthStatus();
window.setInterval(updateHealthStatus, 30000);
