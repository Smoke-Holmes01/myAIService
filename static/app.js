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
  providerSwitch: document.getElementById("provider-switch"),
  providerRadios: document.querySelectorAll('input[name="provider"]'),
  template: document.getElementById("message-template"),
};

if (typeof marked !== 'undefined') {
  marked.setOptions({
    breaks: true,
    gfm: true
  });
}

function preprocessMarkdown(text) {
  if (!text) return text;
  let processed = text;
  // Separate run-on lists that AI might output on a single line
  processed = processed.replace(/([：:。！？\?])\s*\*(?=[^\s\*])/g, '$1\n\n* ');
  // Ensure bullets have a trailing space so marked.js recognizes them
  processed = processed.replace(/(^|\n)\s*\*(?=[^\s\*])/g, '$1* ');
  return processed;
}

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function normalizeModelMarkdown(text) {
  if (!text) return "";

  let normalized = preprocessMarkdown(String(text).replace(/\r\n?/g, "\n"));
  normalized = normalized.replace(/\*{3,}/g, "**");
  normalized = normalized.replace(/(^|\n)(#{1,6})(?=\S)/g, "$1$2 ");
  normalized = normalized.replace(/([。！？；：:])\s*(#{1,6})(?=\S)/g, "$1\n\n$2 ");
  normalized = normalized.replace(/([。！？；：:])\s*(\d+\.)\s*(?=\S)/g, "$1\n\n$2 ");
  normalized = normalized.replace(/([。！？；：:])\s*((?:[-*])\s*)(?=\S)/g, "$1\n\n$2");
  normalized = normalized.replace(/([。！？；：:])\s*(\*\*[^*\n]+\*\*)/g, "$1\n\n$2");
  normalized = normalized.replace(/(^|\n)\s*([*-])(?=\S)/g, "$1$2 ");
  normalized = normalized.replace(/\n{3,}/g, "\n\n");
  return normalized;
}

function markdownToPlainText(text) {
  if (!text) return "";

  return String(text)
    .replace(/\r\n?/g, "\n")
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/(\*\*|__)(.*?)\1/g, "$2")
    .replace(/(\*|_)(.*?)\1/g, "$2")
    .replace(/`{1,3}([^`]*)`{1,3}/g, "$1")
    .replace(/^\s*>\s?/gm, "")
    .replace(/^\s*[-*]\s+/gm, "")
    .replace(/^\s*\d+\.\s+/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function renderMarkdown(text) {
  const normalized = normalizeModelMarkdown(text);

  if (typeof marked === "undefined") {
    return escapeHtml(markdownToPlainText(normalized)).replace(/\n/g, "<br>");
  }

  return marked.parse(normalized);
}

let selectedImageBase64 = "";
let selectedImageName = "";
let activeImageBase64 = "";
let activeImageName = "";
let lastToolMode = "";

const mouthImages = {
  A: "/static/mouth_shapes/mouth_A.png",
  B: "/static/mouth_shapes/mouth_B.png",
  C: "/static/mouth_shapes/mouth_C.png",
  D: "/static/mouth_shapes/mouth_D.png",
  E: "/static/mouth_shapes/mouth_E.png",
  G: "/static/mouth_shapes/mouth_G.png",
  H: "/static/mouth_shapes/mouth_H.png",
};

const defaultMouthShape = "A";

let currentAudio = null;
let isSpeaking = false;
let lipSyncInterval = null;
let simpleMouthInterval = null;
let currentShapeIndex = 0;
let mouthShapesData = [];
let autoPlay = false;

const digitalHuman = document.getElementById("digital-human");
const speakingBubble = document.getElementById("speaking-bubble");
const bubbleContent = document.querySelector(".bubble-content");

function scrollMessagesToBottom() {
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function ensureProviderSwitch() {
  if (!elements.providerSwitch) {
    const inputStack = document.querySelector(".input-stack");
    const composerActions = inputStack ? inputStack.querySelector(".composer-actions") : null;
    if (inputStack && composerActions) {
      const switchNode = document.createElement("div");
      switchNode.className = "provider-switch";
      switchNode.id = "provider-switch";
      switchNode.innerHTML = `
        <span class="provider-label">推理来源</span>
        <label class="provider-option" data-provider="remote_api">
          <input type="radio" name="provider" value="remote_api" checked>
          <span>API</span>
        </label>
        <label class="provider-option" data-provider="local_model">
          <input type="radio" name="provider" value="local_model">
          <span>本地模型</span>
        </label>
      `;
      inputStack.insertBefore(switchNode, composerActions);
      elements.providerSwitch = switchNode;
      elements.providerRadios = switchNode.querySelectorAll('input[name="provider"]');
    }
  }

  if (!document.getElementById("provider-switch-style")) {
    const style = document.createElement("style");
    style.id = "provider-switch-style";
    style.textContent = `
      .provider-switch {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
      }
      .provider-label {
        font-size: 14px;
        color: var(--muted);
      }
      .provider-option {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: rgba(255, 248, 235, 0.88);
        cursor: pointer;
        transition: border-color 0.18s ease, background 0.18s ease, color 0.18s ease;
      }
      .provider-option input {
        margin: 0;
      }
      .provider-option.is-active {
        border-color: rgba(159, 47, 31, 0.28);
        background: rgba(159, 47, 31, 0.1);
        color: var(--accent);
      }
      .provider-option.is-disabled {
        opacity: 0.55;
        cursor: not-allowed;
      }
    `;
    document.head.appendChild(style);
  }
}

function getSelectedProvider() {
  const checked = Array.from(elements.providerRadios || []).find((radio) => radio.checked);
  return checked ? checked.value : "remote_api";
}

function setSelectedProvider(provider) {
  Array.from(elements.providerRadios || []).forEach((radio) => {
    radio.checked = radio.value === provider;
  });
  refreshProviderSwitch();
}

function refreshProviderSwitch() {
  if (!elements.providerSwitch) {
    return;
  }

  elements.providerSwitch.querySelectorAll(".provider-option").forEach((option) => {
    const radio = option.querySelector('input[name="provider"]');
    const isActive = radio && radio.checked;
    const isDisabled = radio && radio.disabled;
    option.classList.toggle("is-active", Boolean(isActive));
    option.classList.toggle("is-disabled", Boolean(isDisabled));
  });
}

function updateProviderAvailability(data) {
  const providers = data && data.providers ? data.providers : {};
  const remoteEnabled = Boolean(providers.remote_api && providers.remote_api.enabled);
  const localEnabled = Boolean(providers.local_model && providers.local_model.enabled);

  Array.from(elements.providerRadios || []).forEach((radio) => {
    if (radio.value === "remote_api") {
      radio.disabled = !remoteEnabled;
    }
    if (radio.value === "local_model") {
      radio.disabled = !localEnabled;
    }
  });

  const selectedProvider = getSelectedProvider();
  if ((selectedProvider === "remote_api" && !remoteEnabled) || (selectedProvider === "local_model" && !localEnabled)) {
    const fallbackProvider = data && data.default_provider && data.default_provider !== "degraded"
      ? data.default_provider
      : (remoteEnabled ? "remote_api" : "local_model");
    if (fallbackProvider) {
      setSelectedProvider(fallbackProvider);
    }
  }

  refreshProviderSwitch();
}

function attachPlayButton(body, text) {
  if (!body || !text || body.querySelector(".play-btn")) {
    return;
  }

  const playBtn = document.createElement("button");
  playBtn.className = "play-btn";
  playBtn.textContent = "播放";
  playBtn.style.marginLeft = "12px";
  playBtn.style.padding = "4px 12px";
  playBtn.style.borderRadius = "20px";
  playBtn.style.border = "none";
  playBtn.style.background = "#f0f0f0";
  playBtn.style.cursor = "pointer";
  playBtn.style.fontSize = "12px";
  playBtn.onclick = () => speakText(markdownToPlainText(text));
  body.appendChild(playBtn);
}

function finalizeAssistantMessage(messageNode, text, options = {}) {
  if (!messageNode) {
    return;
  }

  const body = messageNode.querySelector(".message-body");
  body.innerHTML = renderMarkdown(text);
  attachPlayButton(body, text);
  messageNode.classList.remove("loading");

  if (options.autoPlay && text) {
    speakText(markdownToPlainText(text));
  }

  scrollMessagesToBottom();
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
  body.innerHTML = renderMarkdown(content);

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

  if (role === "assistant" && content && !options.loading && options.enablePlayback !== false) {
    attachPlayButton(body, content);
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

function clearConversationImageContext() {
  activeImageBase64 = "";
  activeImageName = "";
  lastToolMode = "";
}

function getEffectiveImageState() {
  if (selectedImageBase64) {
    return {
      imageBase64: selectedImageBase64,
      imageName: selectedImageName,
      usingConversationImage: false,
    };
  }

  if (activeImageBase64) {
    return {
      imageBase64: activeImageBase64,
      imageName: activeImageName,
      usingConversationImage: true,
    };
  }

  return {
    imageBase64: "",
    imageName: "",
    usingConversationImage: false,
  };
}

function shouldUseMatcher(question, options = {}) {
  const {
    hasImage = false,
  } = options;

  if (!hasImage) {
    return false;
  }

  const explicitMatcherKeywords = [
    "match",
    "matcher",
    "3d",
    "obj",
    "mesh",
    "model",
    "which model",
    "similar model",
    "匹配",
    "模型",
    "三维",
    "对应",
  ];
  const normalizedQuestion = String(question || "").toLowerCase();
  return explicitMatcherKeywords.some((keyword) => normalizedQuestion.includes(keyword));
}

async function updateHealthStatus() {
  try {
    const response = await fetch("/api/ai/health");
    const data = await response.json();
    const isOnline = response.ok && data.status === "up";
    const status = isOnline ? "服务在线，可开始演示" : "服务未就绪";
    const modeMeta = data.remote_api_enabled
      ? `模式: Remote API | 模型: ${data.remote_model || "未知"}`
      : `模式: Local Model | 路径: ${data.model_path || "未知"}`;

    elements.statusDot.classList.remove("offline", "online");
    elements.statusDot.classList.add(isOnline ? "online" : "offline");
    elements.statusText.textContent = status;
    elements.statusMeta.textContent = `设备: ${data.device || "未知"} | RAG: ${data.rag_loaded ? "已启用" : "未启用"} | ${modeMeta}`;
  } catch (error) {
    elements.statusDot.classList.remove("online");
    elements.statusDot.classList.add("offline");
    elements.statusText.textContent = "无法连接 AI 服务";
    elements.statusMeta.textContent = "请确认 qwen_server.py 已启动，并检查服务端口是否可访问。";
  }
}

async function sendStreamingChat({ question, imageBase64, loadingMessage, provider }) {
  const response = await fetch("/api/ai/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      provider,
      question,
      image: imageBase64 || undefined,
    }),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || "请求失败");
  }

  if (!response.body) {
    throw new Error("流式响应不可用");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  const body = loadingMessage.querySelector(".message-body");

  let buffer = "";
  let answer = "";
  let usedKnowledge = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }

      const event = JSON.parse(trimmed);
      if (event.type === "start") {
        usedKnowledge = Boolean(event.used_knowledge);
        continue;
      }

      if (event.type === "delta") {
        answer += event.delta || "";
        body.innerHTML = renderMarkdown(answer || "正在分析，请稍候...");
        loadingMessage.classList.remove("loading");
        scrollMessagesToBottom();
        continue;
      }

      if (event.type === "done") {
        answer = event.answer || answer;
        const finalText = usedKnowledge
          ? `${answer}\n\n[本次回答已结合知识库检索结果]`
          : `${answer}\n\n[本次回答由远程模型流式生成]`;
        finalizeAssistantMessage(loadingMessage, finalText, { autoPlay });
        continue;
      }

      if (event.type === "error") {
        throw new Error(event.error || "流式请求失败");
      }
    }
  }
}

async function sendStandardChat({ question, imageBase64, loadingMessage, provider }) {
  const response = await fetch("/api/ai/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      provider,
      question,
      image: imageBase64 || undefined,
    }),
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || "请求失败");
  }

  const providerNote = provider === "local_model"
    ? "[本次回答由本地模型生成]"
    : "[本次回答由 API 生成]";
  const finalText = data.used_knowledge
    ? `${data.answer}\n\n[本次回答已结合知识库检索结果]\n${providerNote}`
    : `${data.answer}\n\n${providerNote}`;

  finalizeAssistantMessage(loadingMessage, finalText, { autoPlay });
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
  clearConversationImageContext();
  clearImageSelection();
  createMessage(
    "assistant",
    "对话已清空。你可以继续提问，也可以重新上传一张古建筑图片。",
  );
});

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = elements.question.value.trim();
  if (!question) {
    elements.question.focus();
    return;
  }

  const effectiveImageState = getEffectiveImageState();
  const hasImage = Boolean(effectiveImageState.imageBase64);
  const selectedProvider = getSelectedProvider();
  const userText = selectedImageName
    ? `${question}\n\n[已上传图片：${selectedImageName}]`
    : effectiveImageState.usingConversationImage
      ? `${question}\n\n[沿用上一张图片：${effectiveImageState.imageName || "未命名图片"}]`
      : question;

  createMessage("user", userText, {
    imageUrl: selectedImageBase64 || null,
  });

  const loadingMessage = createMessage("assistant", "正在分析，请稍候...", {
    loading: true,
    enablePlayback: false,
  });

  setPendingState(true);

  try {
    if (selectedImageBase64) {
      activeImageBase64 = selectedImageBase64;
      activeImageName = selectedImageName;
    }

    const useMatcher = shouldUseMatcher(question, {
      hasImage,
      usingConversationImage: effectiveImageState.usingConversationImage,
      lastToolMode,
    });

    if (useMatcher) {
      const response = await fetch("/api/agent/match", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          image: effectiveImageState.imageBase64 || undefined,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "请求失败");
      }

      finalizeAssistantMessage(
        loadingMessage,
        `${data.answer}\n\n[本次结果由本地 3D 模型匹配工具生成]`,
        { autoPlay },
      );
      lastToolMode = "matcher";
    } else {
      if (selectedProvider === "remote_api") {
        await sendStreamingChat({
          question,
          imageBase64: effectiveImageState.imageBase64,
          loadingMessage,
          provider: selectedProvider,
        });
      } else {
        await sendStandardChat({
          question,
          imageBase64: effectiveImageState.imageBase64,
          loadingMessage,
          provider: selectedProvider,
        });
      }
      lastToolMode = "chat";
    }
  } catch (error) {
    finalizeAssistantMessage(loadingMessage, `调用失败：${error.message}`, { autoPlay: false });
  } finally {
    setPendingState(false);
    elements.question.value = "";
    clearImageSelection();
    scrollMessagesToBottom();
  }
});

function initMouthImage() {
  let mouthImg = document.getElementById("mouth-img");
  if (!mouthImg) {
    mouthImg = document.createElement("img");
    mouthImg.id = "mouth-img";
    mouthImg.style.position = "absolute";
    mouthImg.style.bottom = "20px";
    mouthImg.style.left = "50%";
    mouthImg.style.transform = "translateX(-50%)";
    mouthImg.style.width = "50px";
    mouthImg.style.height = "auto";
    mouthImg.style.transition = "all 0.05s linear";
    mouthImg.style.zIndex = "10";
    mouthImg.style.pointerEvents = "none";

    const avatarWrapper = document.querySelector(".avatar-wrapper");
    if (avatarWrapper) {
      avatarWrapper.style.position = "relative";
      avatarWrapper.appendChild(mouthImg);
    }
  }
  return mouthImg;
}

function updateMouthShape(shapeCode) {
  const mouthImg = document.getElementById("mouth-img");
  if (!mouthImg) {
    return;
  }

  const imagePath = mouthImages[shapeCode] || mouthImages[defaultMouthShape];
  mouthImg.src = imagePath;
}

function startLipSyncAnimation(mouthShapes) {
  stopLipSyncAnimation();

  mouthShapesData = mouthShapes;
  currentShapeIndex = 0;

  if (!mouthShapesData || mouthShapesData.length === 0) {
    startSimpleMouthAnimation();
    return;
  }

  initMouthImage();

  lipSyncInterval = setInterval(() => {
    if (!currentAudio || currentAudio.paused || currentAudio.ended) {
      return;
    }

    const currentTime = currentAudio.currentTime;

    while (
      currentShapeIndex < mouthShapesData.length &&
      currentTime >= mouthShapesData[currentShapeIndex].end
    ) {
      currentShapeIndex += 1;
    }

    if (
      currentShapeIndex < mouthShapesData.length &&
      currentTime >= mouthShapesData[currentShapeIndex].start
    ) {
      updateMouthShape(mouthShapesData[currentShapeIndex].shape);
    }
  }, 50);
}

function startSimpleMouthAnimation() {
  stopLipSyncAnimation();

  initMouthImage();
  let frameIndex = 0;
  const shapes = ["A", "B", "C", "D", "C", "B"];

  simpleMouthInterval = setInterval(() => {
    if (!currentAudio || currentAudio.paused || currentAudio.ended) {
      return;
    }
    frameIndex = (frameIndex + 1) % shapes.length;
    updateMouthShape(shapes[frameIndex]);
  }, 150);
}

function stopLipSyncAnimation() {
  if (lipSyncInterval) {
    clearInterval(lipSyncInterval);
    lipSyncInterval = null;
  }
  if (simpleMouthInterval) {
    clearInterval(simpleMouthInterval);
    simpleMouthInterval = null;
  }
}

function resetMouthShape() {
  const mouthImg = document.getElementById("mouth-img");
  if (mouthImg) {
    updateMouthShape("A");
  }
}

function updateSpeakingBubble(text) {
  if (bubbleContent) {
    bubbleContent.textContent = text.substring(0, 100) + (text.length > 100 ? "..." : "");
  }
  if (speakingBubble) {
    speakingBubble.style.opacity = "1";
  }
}

function clearSpeakingBubble() {
  setTimeout(() => {
    if (digitalHuman && !digitalHuman.classList.contains("speaking")) {
      if (speakingBubble) {
        speakingBubble.style.opacity = "0";
      }
      if (bubbleContent) {
        setTimeout(() => {
          bubbleContent.textContent = "";
        }, 300);
      }
    }
  }, 500);
}

function activateDigitalHuman(isActive) {
  if (!digitalHuman) {
    return;
  }

  if (isActive) {
    digitalHuman.classList.add("speaking");
  } else {
    digitalHuman.classList.remove("speaking");
    clearSpeakingBubble();
  }
}

function stopCurrentPlayback() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
    currentAudio = null;
  }

  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }

  activateDigitalHuman(false);
  stopLipSyncAnimation();
  resetMouthShape();
  isSpeaking = false;
}

async function speakText(text) {
  if (!text || text.trim() === "") {
    return;
  }

  stopCurrentPlayback();

  try {
    activateDigitalHuman(true);
    updateSpeakingBubble(text);

    let mouthShapes = [];
    try {
      const lipResponse = await fetch("/api/ai/lipsync", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (lipResponse.ok) {
        const lipData = await lipResponse.json();
        mouthShapes = lipData.mouth_shapes || [];
      }
    } catch (error) {
      console.warn("Lip sync generation failed, falling back to simple mouth animation.", error);
    }

    const audioResponse = await fetch("/api/ai/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!audioResponse.ok) {
      throw new Error("TTS request failed");
    }

    const audioBlob = await audioResponse.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    currentAudio = new Audio(audioUrl);

    if (mouthShapes.length > 0) {
      startLipSyncAnimation(mouthShapes);
    } else {
      startSimpleMouthAnimation();
    }

    currentAudio.onended = () => {
      activateDigitalHuman(false);
      stopLipSyncAnimation();
      resetMouthShape();
      URL.revokeObjectURL(audioUrl);
      currentAudio = null;
      isSpeaking = false;
    };

    currentAudio.onerror = () => {
      activateDigitalHuman(false);
      stopLipSyncAnimation();
      resetMouthShape();
      URL.revokeObjectURL(audioUrl);
      currentAudio = null;
      isSpeaking = false;
    };

    await currentAudio.play();
    isSpeaking = true;
  } catch (error) {
    console.error("Voice playback failed:", error);
    activateDigitalHuman(false);
    stopLipSyncAnimation();
    resetMouthShape();

    if ("speechSynthesis" in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "zh-CN";
      utterance.onend = () => {
        activateDigitalHuman(false);
        stopLipSyncAnimation();
        resetMouthShape();
      };
      window.speechSynthesis.speak(utterance);
      startSimpleMouthAnimation();
    }
  }
}

function toggleAutoPlay() {
  autoPlay = !autoPlay;
  const voiceToggle = document.getElementById("voice-toggle");
  if (!voiceToggle) {
    return;
  }

  if (autoPlay) {
    voiceToggle.classList.add("auto-play");
    voiceToggle.textContent = "自动播放 (开)";
  } else {
    voiceToggle.classList.remove("auto-play");
    voiceToggle.textContent = "自动播放 (关)";
  }
}

if (digitalHuman) {
  digitalHuman.addEventListener("click", stopCurrentPlayback);
}

const voiceToggle = document.getElementById("voice-toggle");
if (voiceToggle) {
  voiceToggle.addEventListener("click", toggleAutoPlay);
  if (autoPlay) {
    voiceToggle.classList.add("auto-play");
    voiceToggle.textContent = "自动播放 (开)";
  }
}

async function legacyUpdateHealthStatus() {
  try {
    const response = await fetch("/api/ai/health");
    const data = await response.json();
    const isOnline = response.ok && data.status === "up";
    const status = isOnline ? "服务在线，可开始演示" : "服务未就绪";

    updateProviderAvailability(data);

    const remoteMeta = data.remote_api_enabled
      ? `API: 已就绪 (${data.remote_model || "未命名模型"})`
      : "API: 未就绪";
    const localMeta = data.local_model_loaded
      ? `本地: 已就绪 (${data.local_model_path || "未配置路径"})`
      : "本地: 未就绪";

    elements.statusDot.classList.remove("offline", "online");
    elements.statusDot.classList.add(isOnline ? "online" : "offline");
    elements.statusText.textContent = status;
    elements.statusMeta.textContent = `设备: ${data.device || "未知"} | RAG: ${data.rag_loaded ? "已启用" : "未启用"} | ${remoteMeta} | ${localMeta}`;
  } catch (error) {
    elements.statusDot.classList.remove("online");
    elements.statusDot.classList.add("offline");
    elements.statusText.textContent = "无法连接 AI 服务";
    elements.statusMeta.textContent = "请确认 qwen_server.py 已启动，并检查服务端口是否可访问。";
  }
}

ensureProviderSwitch();
Array.from(elements.providerRadios || []).forEach((radio) => {
  radio.addEventListener("change", refreshProviderSwitch);
});
refreshProviderSwitch();

updateHealthStatus();
window.setInterval(updateHealthStatus, 30000);
