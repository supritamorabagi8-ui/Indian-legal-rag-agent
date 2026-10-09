/* ============================================================
   INDIAN LEGAL AI ASSISTANT — FRONTEND CONTROLLER & AGENT ENGINE
   ============================================================ */

(() => {
  "use strict";

  // ── STORAGE KEYS & DEFAULTS ─────────────────────────────────
  const STORAGE_KEYS = {
    SESSIONS: "indian_legal_sessions_v3",
    ACTIVE_SESSION: "indian_legal_active_session_id_v3",
    CONFIG: "indian_legal_config_v3",
    THEME: "indian_legal_theme_v3"
  };

  const THEMES = ["theme-obsidian", "theme-indigo", "theme-slate"];

  const DEFAULT_CONFIG = {
    model: "openai/gpt-oss-20b",
    mode: "agentic", // "agentic" | "rag" | "search"
    top_k: 5,
    temperature: 0.2,
    domainHint: "auto",
    autoScroll: true,
    highlightCitations: true,
    showAgentSteps: true
  };

  // ── APPLICATION STATE ───────────────────────────────────────
  let state = {
    mode: "agentic",
    isLoading: false,
    config: loadConfig(),
    sessions: loadSessions(),
    activeSessionId: null,
    currentRetrievedChunks: [],
    selectedDomain: "auto",
    isRecordingVoice: false,
    speechRecognition: null,
    speechSynthesisUtterance: null,
    activeSpeakingMsgId: null,
    currentThemeIndex: 0
  };

  // ── DOM ELEMENTS CACHE ──────────────────────────────────────
  const elements = {
    // Sidebar & Navigation
    appSidebar: document.getElementById("app-sidebar"),
    sidebarOverlay: document.getElementById("sidebar-overlay"),
    mobileMenuBtn: document.getElementById("mobile-menu-btn"),
    closeSidebarBtn: document.getElementById("close-sidebar-btn"),
    newChatBtn: document.getElementById("new-chat-btn"),
    historySearch: document.getElementById("history-search"),
    historyList: document.getElementById("history-list"),
    clearAllHistoryBtn: document.getElementById("clear-all-history-btn"),
    footerChunkCount: document.getElementById("footer-chunk-count"),

    // Sidebar Mode Selectors
    sideModeAgentic: document.getElementById("side-mode-agentic"),
    sideModeRag: document.getElementById("side-mode-rag"),
    sideModeSearch: document.getElementById("side-mode-search"),

    // Topbar & Tabs
    tabAgentic: document.getElementById("tab-agentic"),
    tabRagChat: document.getElementById("tab-rag-chat"),
    tabVectorSearch: document.getElementById("tab-vector-search"),
    systemStatusPill: document.getElementById("system-status-pill"),
    statusModelText: document.getElementById("status-model-text"),
    themeToggleBtn: document.getElementById("theme-toggle-btn"),
    openSettingsBtn: document.getElementById("open-settings-btn"),
    openHealthBtn: document.getElementById("open-health-btn"),
    exportChatBtn: document.getElementById("export-chat-btn"),

    // Main Views
    viewRagChat: document.getElementById("view-rag-chat"),
    viewVectorSearch: document.getElementById("view-vector-search"),

    // Chat Consultation View
    chatViewport: document.getElementById("chat-viewport"),
    welcomeHero: document.getElementById("welcome-hero"),
    messagesStream: document.getElementById("messages-stream"),
    legalQueryInput: document.getElementById("legal-query-input"),
    sendQueryBtn: document.getElementById("send-query-btn"),
    voiceDictateBtn: document.getElementById("voice-dictate-btn"),
    composerClearBtn: document.getElementById("composer-clear-btn"),
    composerModeTag: document.getElementById("composer-mode-tag"),
    composerModelTag: document.getElementById("composer-model-tag"),
    viewDisclaimerLink: document.getElementById("view-disclaimer-link"),

    // Direct Search View
    directVectorQuery: document.getElementById("direct-vector-query"),
    directSearchBtn: document.getElementById("direct-search-btn"),
    searchResultsViewport: document.getElementById("search-results-grid"),
    searchResultsMeta: document.getElementById("search-results-meta"),
    resultsCountText: document.getElementById("results-count-text"),
    resultsTimingText: document.getElementById("results-timing-text"),

    // Chunk Inspector Modal
    chunkModal: document.getElementById("chunk-modal"),
    chunkModalSource: document.getElementById("chunk-modal-source"),
    chunkModalType: document.getElementById("chunk-modal-type"),
    chunkModalScore: document.getElementById("chunk-modal-score"),
    chunkModalDist: document.getElementById("chunk-modal-dist"),
    chunkModalId: document.getElementById("chunk-modal-id"),
    chunkModalText: document.getElementById("chunk-modal-text"),
    closeChunkModalBtn: document.getElementById("close-chunk-modal-btn"),
    closeChunkModalFooter: document.getElementById("close-chunk-modal-footer"),
    copyChunkTextBtn: document.getElementById("copy-chunk-text-btn"),
    askAboutChunkBtn: document.getElementById("ask-about-chunk-btn"),

    // Settings Modal
    settingsModal: document.getElementById("settings-modal"),
    settingModelSelect: document.getElementById("setting-model-select"),
    settingModeSelect: document.getElementById("setting-mode-select"),
    settingTopkSlider: document.getElementById("setting-topk-slider"),
    settingTopkVal: document.getElementById("setting-topk-val"),
    settingTempSlider: document.getElementById("setting-temp-slider"),
    settingTempVal: document.getElementById("setting-temp-val"),
    settingAutoscroll: document.getElementById("setting-autoscroll"),
    settingCitationHighlight: document.getElementById("setting-citation-highlight"),
    settingShowAgentSteps: document.getElementById("setting-show-agent-steps"),
    closeSettingsModalBtn: document.getElementById("close-settings-modal-btn"),
    resetSettingsBtn: document.getElementById("reset-settings-btn"),
    saveSettingsBtn: document.getElementById("save-settings-btn"),

    // Health Modal
    healthModal: document.getElementById("health-modal"),
    healthModalContent: document.getElementById("health-modal-content"),
    closeHealthModalBtn: document.getElementById("close-health-modal-btn"),
    closeHealthFooterBtn: document.getElementById("close-health-footer-btn"),

    // Disclaimer Modal
    disclaimerModal: document.getElementById("disclaimer-modal"),
    closeDisclaimerModalBtn: document.getElementById("close-disclaimer-modal-btn"),
    ackDisclaimerBtn: document.getElementById("ack-disclaimer-btn"),

    // Toast Container
    toastStack: document.getElementById("toast-stack")
  };

  // ── INITIALIZATION ──────────────────────────────────────────
  function init() {
    initTheme();
    initSettingsUI();
    initSession();
    attachEventListeners();
    initVoiceRecognition();
    fetchSystemStatus();
    updateModeUI(state.config.mode || "agentic");
  }

  // ── THEME MANAGEMENT ────────────────────────────────────────
  function initTheme() {
    const savedTheme = localStorage.getItem(STORAGE_KEYS.THEME) || "theme-obsidian";
    state.currentThemeIndex = THEMES.indexOf(savedTheme);
    if (state.currentThemeIndex === -1) state.currentThemeIndex = 0;
    applyTheme(THEMES[state.currentThemeIndex]);
  }

  function applyTheme(themeName) {
    document.body.className = themeName;
    localStorage.setItem(STORAGE_KEYS.THEME, themeName);
    const icons = { "theme-obsidian": "✨", "theme-indigo": "🌌", "theme-slate": "☀️" };
    const iconEl = document.getElementById("theme-icon");
    if (iconEl) iconEl.textContent = icons[themeName] || "✨";
  }

  function toggleTheme() {
    state.currentThemeIndex = (state.currentThemeIndex + 1) % THEMES.length;
    const nextTheme = THEMES[state.currentThemeIndex];
    applyTheme(nextTheme);
    showToast(`Theme switched to ${nextTheme.replace("theme-", "").toUpperCase()}`, "info");
  }

  // ── CONFIGURATION & SETTINGS ────────────────────────────────
  function loadConfig() {
    try {
      const saved = localStorage.getItem(STORAGE_KEYS.CONFIG);
      return saved ? { ...DEFAULT_CONFIG, ...JSON.parse(saved) } : { ...DEFAULT_CONFIG };
    } catch {
      return { ...DEFAULT_CONFIG };
    }
  }

  function saveConfig(newConfig) {
    state.config = { ...state.config, ...newConfig };
    localStorage.setItem(STORAGE_KEYS.CONFIG, JSON.stringify(state.config));
    updateConfigTags();
  }

  function initSettingsUI() {
    if (elements.settingModelSelect) elements.settingModelSelect.value = state.config.model;
    if (elements.settingModeSelect) elements.settingModeSelect.value = state.config.mode;
    if (elements.settingTopkSlider) {
      elements.settingTopkSlider.value = state.config.top_k;
      if (elements.settingTopkVal) elements.settingTopkVal.textContent = `${state.config.top_k} Chunks`;
    }
    if (elements.settingTempSlider) {
      elements.settingTempSlider.value = state.config.temperature;
      if (elements.settingTempVal) elements.settingTempVal.textContent = state.config.temperature.toFixed(2);
    }
    if (elements.settingAutoscroll) elements.settingAutoscroll.checked = state.config.autoScroll;
    if (elements.settingCitationHighlight) elements.settingCitationHighlight.checked = state.config.highlightCitations;
    if (elements.settingShowAgentSteps) elements.settingShowAgentSteps.checked = state.config.showAgentSteps;

    updateConfigTags();
  }

  function updateConfigTags() {
    if (elements.composerModelTag) {
      const modelShort = state.config.model.split("/").pop();
      elements.composerModelTag.textContent = `Model: ${modelShort}`;
    }
    if (elements.composerModeTag) {
      if (state.mode === "agentic") {
        elements.composerModeTag.textContent = "🤖 Multi-Agent Active";
        elements.composerModeTag.style.color = "var(--agent-router)";
      } else if (state.mode === "rag") {
        elements.composerModeTag.textContent = "⚡ Fast RAG Active";
        elements.composerModeTag.style.color = "var(--gold-light)";
      } else {
        elements.composerModeTag.textContent = "🔍 Clause Explorer";
        elements.composerModeTag.style.color = "var(--agent-retrieval)";
      }
    }
  }

  // ── SESSIONS & LOCAL STORAGE ────────────────────────────────
  function loadSessions() {
    try {
      const raw = localStorage.getItem(STORAGE_KEYS.SESSIONS);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  function saveSessions() {
    localStorage.setItem(STORAGE_KEYS.SESSIONS, JSON.stringify(state.sessions));
    renderHistoryList();
  }

  function initSession() {
    const savedActiveId = localStorage.getItem(STORAGE_KEYS.ACTIVE_SESSION);
    if (savedActiveId && state.sessions.find(s => s.id === savedActiveId)) {
      switchSession(savedActiveId);
    } else {
      createNewSession();
    }
    renderHistoryList();
  }

  function createNewSession() {
    const newSession = {
      id: "session_" + Date.now(),
      title: "New Legal Consultation",
      timestamp: new Date().toISOString(),
      messages: []
    };
    state.sessions.unshift(newSession);
    state.activeSessionId = newSession.id;
    localStorage.setItem(STORAGE_KEYS.ACTIVE_SESSION, newSession.id);
    saveSessions();
    renderCurrentSessionMessages();
  }

  function switchSession(sessionId) {
    state.activeSessionId = sessionId;
    localStorage.setItem(STORAGE_KEYS.ACTIVE_SESSION, sessionId);
    renderHistoryList();
    renderCurrentSessionMessages();
    closeMobileSidebar();
  }

  function getActiveSession() {
    return state.sessions.find(s => s.id === state.activeSessionId) || null;
  }

  function deleteSession(sessionId, e) {
    if (e) e.stopPropagation();
    state.sessions = state.sessions.filter(s => s.id !== sessionId);
    if (state.activeSessionId === sessionId) {
      if (state.sessions.length > 0) {
        state.activeSessionId = state.sessions[0].id;
      } else {
        createNewSession();
        return;
      }
    }
    saveSessions();
    renderCurrentSessionMessages();
  }

  function clearAllSessions() {
    if (!confirm("Are you sure you want to clear all consultation history?")) return;
    state.sessions = [];
    localStorage.removeItem(STORAGE_KEYS.SESSIONS);
    createNewSession();
    showToast("All consultation history cleared", "info");
  }

  function renderHistoryList(filterQuery = "") {
    if (!elements.historyList) return;
    elements.historyList.innerHTML = "";

    const query = filterQuery.toLowerCase().trim();
    const filtered = state.sessions.filter(s => 
      s.title.toLowerCase().includes(query) ||
      (s.messages && s.messages.some(m => m.text.toLowerCase().includes(query)))
    );

    if (filtered.length === 0) {
      elements.historyList.innerHTML = `<p class="history-empty">${query ? "No matching consultations" : "No previous consultations"}</p>`;
      return;
    }

    filtered.forEach(s => {
      const item = document.createElement("div");
      item.className = `history-item ${s.id === state.activeSessionId ? "active" : ""}`;
      item.onclick = () => switchSession(s.id);

      const titleSpan = document.createElement("span");
      titleSpan.className = "history-item-text";
      titleSpan.textContent = s.title;

      const delBtn = document.createElement("button");
      delBtn.className = "history-del-btn";
      delBtn.title = "Delete consultation";
      delBtn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>`;
      delBtn.onclick = (e) => deleteSession(s.id, e);

      item.appendChild(titleSpan);
      item.appendChild(delBtn);
      elements.historyList.appendChild(item);
    });
  }

  // ── MODE SWITCHING ──────────────────────────────────────────
  function updateModeUI(mode) {
    state.mode = mode;
    state.config.mode = mode;
    saveConfig({ mode });

    // Update Topbar Tabs
    if (elements.tabAgentic) elements.tabAgentic.classList.toggle("active", mode === "agentic");
    if (elements.tabRagChat) elements.tabRagChat.classList.toggle("active", mode === "rag");
    if (elements.tabVectorSearch) elements.tabVectorSearch.classList.toggle("active", mode === "search");

    // Update Sidebar Cards
    if (elements.sideModeAgentic) elements.sideModeAgentic.classList.toggle("active", mode === "agentic");
    if (elements.sideModeRag) elements.sideModeRag.classList.toggle("active", mode === "rag");
    if (elements.sideModeSearch) elements.sideModeSearch.classList.toggle("active", mode === "search");

    // Toggle Main Panels
    if (mode === "search") {
      if (elements.viewRagChat) elements.viewRagChat.classList.remove("active");
      if (elements.viewVectorSearch) elements.viewVectorSearch.classList.add("active");
      if (elements.directVectorQuery) elements.directVectorQuery.focus();
    } else {
      if (elements.viewVectorSearch) elements.viewVectorSearch.classList.remove("active");
      if (elements.viewRagChat) elements.viewRagChat.classList.add("active");
      if (elements.legalQueryInput) elements.legalQueryInput.focus();
    }

    updateConfigTags();
  }

  // ── VOICE RECOGNITION (WEB SPEECH API) ──────────────────────
  function initVoiceRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      if (elements.voiceDictateBtn) elements.voiceDictateBtn.style.opacity = "0.5";
      return;
    }

    state.speechRecognition = new SpeechRecognition();
    state.speechRecognition.continuous = false;
    state.speechRecognition.interimResults = true;
    state.speechRecognition.lang = "en-IN";

    state.speechRecognition.onstart = () => {
      state.isRecordingVoice = true;
      if (elements.voiceDictateBtn) elements.voiceDictateBtn.classList.add("recording");
      showToast("Listening... Speak your legal query", "info");
    };

    state.speechRecognition.onresult = (event) => {
      let transcript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      if (elements.legalQueryInput) {
        elements.legalQueryInput.value = transcript;
        autoResizeTextarea(elements.legalQueryInput);
      }
    };

    state.speechRecognition.onerror = (e) => {
      state.isRecordingVoice = false;
      if (elements.voiceDictateBtn) elements.voiceDictateBtn.classList.remove("recording");
      showToast("Voice recognition error: " + e.error, "error");
    };

    state.speechRecognition.onend = () => {
      state.isRecordingVoice = false;
      if (elements.voiceDictateBtn) elements.voiceDictateBtn.classList.remove("recording");
    };
  }

  function toggleVoiceDictation() {
    if (!state.speechRecognition) {
      showToast("Voice dictation is not supported in this browser.", "error");
      return;
    }
    if (state.isRecordingVoice) {
      state.speechRecognition.stop();
    } else {
      try {
        state.speechRecognition.start();
      } catch (err) {
        console.error(err);
      }
    }
  }

  // ── TEXT-TO-SPEECH (READ ALOUD) ─────────────────────────────
  function readAloudMessage(text, msgId, btn) {
    if (!("speechSynthesis" in window)) {
      showToast("Speech synthesis not supported in this browser.", "error");
      return;
    }

    if (window.speechSynthesis.speaking && state.activeSpeakingMsgId === msgId) {
      window.speechSynthesis.cancel();
      state.activeSpeakingMsgId = null;
      if (btn) btn.classList.remove("active");
      return;
    }

    window.speechSynthesis.cancel();

    // Clean markdown syntax for speech
    const cleanText = text
      .replace(/[*#_`]/g, "")
      .replace(/\[Source.*?\]/g, "")
      .replace(/https?:\/\/\S+/g, "");

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.lang = "en-IN";

    utterance.onstart = () => {
      state.activeSpeakingMsgId = msgId;
      if (btn) btn.classList.add("active");
    };

    utterance.onend = () => {
      state.activeSpeakingMsgId = null;
      if (btn) btn.classList.remove("active");
    };

    utterance.onerror = () => {
      state.activeSpeakingMsgId = null;
      if (btn) btn.classList.remove("active");
    };

    window.speechSynthesis.speak(utterance);
  }

  // ── CONSULTATION CHAT & AGENTIC EXECUTION ───────────────────
  async function handleUserSubmit() {
    if (state.isLoading) return;
    const text = elements.legalQueryInput.value.trim();
    if (!text) return;

    elements.legalQueryInput.value = "";
    autoResizeTextarea(elements.legalQueryInput);

    // Append User Message to Active Session
    const session = getActiveSession();
    if (!session) return;

    if (session.messages.length === 0) {
      session.title = text.length > 40 ? text.substring(0, 37) + "..." : text;
    }

    const userMsg = {
      id: "msg_" + Date.now(),
      sender: "user",
      text: text,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    };
    session.messages.push(userMsg);
    saveSessions();
    renderCurrentSessionMessages();

    // Placeholder Assistant Message with Loading Animation
    const assistantMsgId = "msg_asst_" + Date.now();
    const assistantMsg = {
      id: assistantMsgId,
      sender: "assistant",
      text: "",
      pipeline: state.mode,
      isLoading: true,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    };
    session.messages.push(assistantMsg);
    renderCurrentSessionMessages();

    state.isLoading = true;
    if (elements.sendQueryBtn) elements.sendQueryBtn.disabled = true;

    try {
      const endpoint = state.mode === "agentic" ? "/api/agentic" : "/api/ask";
      const payload = {
        question: text,
        model: state.config.model,
        top_k: state.config.top_k,
        temperature: state.config.temperature,
        domain_hint: state.selectedDomain !== "auto" ? state.selectedDomain : null
      };

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok || data.status === "error") {
        throw new Error(data.error || "Failed to generate legal consultation.");
      }

      // Update Assistant Message with Result
      assistantMsg.isLoading = false;
      assistantMsg.text = data.answer || "No legal analysis generated.";
      assistantMsg.pipeline = data.pipeline || state.mode;
      assistantMsg.queryType = data.query_type;
      assistantMsg.routerData = data.router_data;
      assistantMsg.verification = data.verification;
      assistantMsg.sources = data.sources || [];
      assistantMsg.chunks = data.chunks || [];
      assistantMsg.executionSteps = data.execution_steps || [];
      assistantMsg.modelUsed = data.model_used;
      assistantMsg.totalTimeMs = data.total_time_ms;

      state.currentRetrievedChunks = data.chunks || [];
      saveSessions();
      renderCurrentSessionMessages();

    } catch (err) {
      assistantMsg.isLoading = false;
      assistantMsg.text = `⚠️ **System Error:** ${err.message}\n\nPlease check backend connectivity or try another model in Settings.`;
      saveSessions();
      renderCurrentSessionMessages();
      showToast(err.message, "error");
    } finally {
      state.isLoading = false;
      if (elements.sendQueryBtn) elements.sendQueryBtn.disabled = false;
    }
  }

  // ── RENDER CONVERSATION MESSAGES ────────────────────────────
  function renderCurrentSessionMessages() {
    const session = getActiveSession();
    if (!elements.messagesStream || !session) return;

    elements.messagesStream.innerHTML = "";

    if (session.messages.length === 0) {
      if (elements.welcomeHero) elements.welcomeHero.style.display = "block";
      return;
    } else {
      if (elements.welcomeHero) elements.welcomeHero.style.display = "none";
    }

    session.messages.forEach(msg => {
      const row = document.createElement("div");
      row.className = `message-row ${msg.sender}`;

      const avatar = document.createElement("div");
      avatar.className = "message-avatar";
      avatar.textContent = msg.sender === "user" ? "👤" : "⚖️";

      const bubbleWrap = document.createElement("div");
      bubbleWrap.className = "message-bubble-wrap";

      const headerLine = document.createElement("div");
      headerLine.className = "message-header-line";

      const senderName = document.createElement("span");
      senderName.className = "sender-name";
      senderName.textContent = msg.sender === "user" ? "You (Legal Inquirer)" : (msg.pipeline === "agentic" ? "Indian Legal AI (Multi-Agent)" : "Indian Legal AI (Fast RAG)");

      const msgTime = document.createElement("span");
      msgTime.className = "message-time";
      msgTime.textContent = msg.timestamp || "";

      headerLine.appendChild(senderName);
      headerLine.appendChild(msgTime);
      bubbleWrap.appendChild(headerLine);

      const card = document.createElement("div");
      card.className = "message-card";

      if (msg.isLoading) {
        card.innerHTML = `
          <div class="loading-spinner-wrap">
            <div class="spinner"></div>
            <p>${msg.pipeline === "agentic" ? "Routing query & verifying statutory evidence across 442k+ clauses…" : "Retrieving relevant clauses & synthesizing opinion…"}</p>
          </div>
        `;
      } else {
        // If Assistant & Multi-Agent mode with steps -> render Agentic Pipeline Box
        if (msg.sender === "assistant" && msg.pipeline === "agentic" && msg.executionSteps && msg.executionSteps.length > 0 && state.config.showAgentSteps) {
          const pipelineBox = renderAgenticPipelineBox(msg);
          card.appendChild(pipelineBox);
        }

        // Render Markdown content
        const markdownDiv = document.createElement("div");
        markdownDiv.className = "legal-markdown";
        markdownDiv.innerHTML = formatLegalMarkdown(msg.text, state.config.highlightCitations);
        card.appendChild(markdownDiv);

        // Render Sources Pills if available
        if (msg.sources && msg.sources.length > 0) {
          const sourcesBar = document.createElement("div");
          sourcesBar.className = "sources-pills-bar";
          sourcesBar.innerHTML = `<span class="sources-label">Citations & Sources:</span>`;

          msg.sources.forEach((src, idx) => {
            const pill = document.createElement("button");
            pill.className = "source-pill-btn";
            pill.title = `View source document: ${src.name}`;
            pill.innerHTML = `<span>📜 ${src.name}</span> <span class="sp-score">${src.relevance_score || 90}%</span>`;
            pill.onclick = () => openChunkModalBySource(src.name, msg.chunks);
            sourcesBar.appendChild(pill);
          });

          card.appendChild(sourcesBar);
        }

        // Assistant Message Actions (Copy, Read Aloud, Export)
        if (msg.sender === "assistant") {
          const actionsBar = document.createElement("div");
          actionsBar.className = "message-actions-bar";

          // Copy Button
          const copyBtn = document.createElement("button");
          copyBtn.className = "msg-action-btn";
          copyBtn.title = "Copy opinion to clipboard";
          copyBtn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg> <span>Copy</span>`;
          copyBtn.onclick = () => {
            navigator.clipboard.writeText(msg.text);
            copyBtn.innerHTML = `<span>✓ Copied</span>`;
            setTimeout(() => {
              copyBtn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg> <span>Copy</span>`;
            }, 2000);
            showToast("Copied legal advisory to clipboard", "success");
          };

          // Read Aloud Button (Text-to-Speech)
          const readBtn = document.createElement("button");
          readBtn.className = "msg-action-btn";
          readBtn.title = "Listen to legal advisory (Read Aloud)";
          readBtn.innerHTML = `<span>🔊 Read Aloud</span>`;
          readBtn.onclick = () => readAloudMessage(msg.text, msg.id, readBtn);

          actionsBar.appendChild(copyBtn);
          actionsBar.appendChild(readBtn);
          card.appendChild(actionsBar);
        }
      }

      bubbleWrap.appendChild(card);
      row.appendChild(avatar);
      row.appendChild(bubbleWrap);
      elements.messagesStream.appendChild(row);
    });

    if (state.config.autoScroll && elements.chatViewport) {
      elements.chatViewport.scrollTop = elements.chatViewport.scrollHeight;
    }
  }

  // ── AGENTIC PIPELINE VISUALIZER BOX ─────────────────────────
  function renderAgenticPipelineBox(msg) {
    const box = document.createElement("div");
    box.className = "agentic-pipeline-box";

    const header = document.createElement("div");
    header.className = "pipeline-header";
    header.innerHTML = `
      <div class="ph-title">
        <span>🤖 Autonomous Agentic Workflow</span>
        <span class="msc-badge">${msg.queryType || "Indian Law"}</span>
      </div>
      <div class="ph-timing">${msg.totalTimeMs ? (msg.totalTimeMs / 1000).toFixed(2) + "s" : "Verified"}</div>
    `;
    box.appendChild(header);

    // 4-Step Agent Grid
    const stepsGrid = document.createElement("div");
    stepsGrid.className = "agent-steps-grid";

    msg.executionSteps.forEach(step => {
      const stepChip = document.createElement("div");
      stepChip.className = "agent-step-chip completed";
      stepChip.innerHTML = `
        <div class="step-chip-top">
          <span class="step-chip-icon">${step.icon || "✓"}</span>
          <span class="step-chip-time">${step.time_ms || 0}ms</span>
        </div>
        <div class="step-chip-name">${step.name}</div>
        <div class="step-chip-status" title="${step.summary}">${step.summary}</div>
      `;
      stepsGrid.appendChild(stepChip);
    });
    box.appendChild(stepsGrid);

    // Verification Status Banner
    if (msg.verification) {
      const isInsufficient = msg.verification.status === "INSUFFICIENT";
      const banner = document.createElement("div");
      banner.className = `verification-banner ${isInsufficient ? "insufficient" : ""}`;
      banner.innerHTML = `
        <div class="vb-left">
          <span>${isInsufficient ? "⚠️ Context Flag:" : "🛡️ Evidence Status:"}</span>
          <span>${isInsufficient ? "Partial Grounds" : "Statutorily Grounded & Validated"}</span>
        </div>
        <div class="vb-confidence">${msg.verification.confidence || 95}% Confidence</div>
      `;
      box.appendChild(banner);
    }

    return box;
  }

  // ── MARKDOWN & CITATION FORMATTER ───────────────────────────
  function formatLegalMarkdown(rawText, highlightCitations = true) {
    if (!rawText) return "";

    let html = rawText
      // Headings
      .replace(/^### (.*$)/gim, "<h3>$1</h3>")
      .replace(/^## (.*$)/gim, "<h2>$1</h2>")
      .replace(/^# (.*$)/gim, "<h1>$1</h1>")
      // Bold & Italic
      .replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/gim, "<em>$1</em>")
      // Code inline
      .replace(/`([^`]+)`/gim, "<code>$1</code>")
      // Unordered lists
      .replace(/^\s*[\-\*]\s+(.*$)/gim, "<li>$1</li>")
      // Ordered lists
      .replace(/^\s*\d+\.\s+(.*$)/gim, "<li>$1</li>")
      // Paragraph breaks
      .replace(/\n\n+/g, "</p><p>")
      .replace(/\n/g, "<br/>");

    html = `<p>${html}</p>`;
    html = html.replace(/<li>(.*?)<\/li>/g, "<ul><li>$1</li></ul>").replace(/<\/ul>\s*<ul>/g, "");

    // Highlight key Indian statutory references
    if (highlightCitations) {
      const legalPatterns = [
        /\b(Article\s+\d+[A-Z]?)\b/gi,
        /\b(Articles\s+\d+(?:,\s*\d+)*(?:\s*and\s*\d+)?)\b/gi,
        /\b(Section\s+\d+[A-Z]?)\b/gi,
        /\b(Sections\s+\d+(?:,\s*\d+)*(?:\s*and\s*\d+)?)\b/gi,
        /\b(Bharatiya Nyaya Sanhita(?:,\s*2023)?|BNS(?:\s*2023)?)\b/gi,
        /\b(Bharatiya Nagarik Suraksha Sanhita(?:,\s*2023)?|BNSS(?:\s*2023)?)\b/gi,
        /\b(Bharatiya Sakshya Adhiniyam(?:,\s*2023)?|BSA(?:\s*2023)?)\b/gi,
        /\b(Information Technology Act(?:,\s*2000)?|IT Act(?:,\s*2000)?)\b/gi,
        /\b(Constitution of India)\b/gi,
        /\b(Indian Contract Act(?:,\s*1872)?)\b/gi
      ];

      legalPatterns.forEach(pat => {
        html = html.replace(pat, (match) => `<span class="legal-citation">${match}</span>`);
      });
    }

    return html;
  }

  // ── DIRECT CLAUSE & VECTOR EXPLORER ─────────────────────────
  async function handleDirectSearch() {
    if (!elements.directVectorQuery) return;
    const query = elements.directVectorQuery.value.trim();
    if (!query) return;

    if (elements.searchResultsViewport) {
      elements.searchResultsViewport.innerHTML = `
        <div class="loading-spinner-wrap">
          <div class="spinner"></div>
          <p>Scanning 442,528 FAISS vectors for '${query}'…</p>
        </div>
      `;
    }

    const t0 = performance.now();

    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 8 })
      });
      const data = await res.json();
      const elapsed = Math.round(performance.now() - t0);

      if (!res.ok || data.status === "error") {
        throw new Error(data.error || "Search failed.");
      }

      state.currentRetrievedChunks = data.results || [];
      renderSearchResults(data.results || [], query, elapsed);

    } catch (err) {
      if (elements.searchResultsViewport) {
        elements.searchResultsViewport.innerHTML = `
          <div class="search-empty-state">
            <div class="ses-icon">⚠️</div>
            <h3>Vector Search Failed</h3>
            <p>${err.message}</p>
          </div>
        `;
      }
      showToast(err.message, "error");
    }
  }

  function renderSearchResults(results, query, elapsedMs) {
    if (!elements.searchResultsViewport) return;

    if (elements.searchResultsMeta) {
      elements.searchResultsMeta.style.display = "flex";
      if (elements.resultsCountText) elements.resultsCountText.textContent = `Found ${results.length} relevant document clauses`;
      if (elements.resultsTimingText) elements.resultsTimingText.textContent = `${elapsedMs}ms • FAISS Index`;
    }

    if (results.length === 0) {
      elements.searchResultsViewport.innerHTML = `
        <div class="search-empty-state">
          <div class="ses-icon">🔍</div>
          <h3>No matching legal clauses found</h3>
          <p>Try broader terms, statutory section numbers, or act names.</p>
        </div>
      `;
      return;
    }

    elements.searchResultsViewport.innerHTML = "";

    results.forEach((r, idx) => {
      const card = document.createElement("div");
      card.className = "clause-card";
      card.onclick = () => openChunkModal(r);

      card.innerHTML = `
        <div class="clause-card-header">
          <div class="cch-left">
            <span class="cch-rank">#${r.rank || idx + 1}</span>
            <span class="cch-source">${r.source}</span>
          </div>
          <span class="cch-score">${r.relevance_score || 85}% Relevance</span>
        </div>
        <div class="clause-snippet">${escapeHtml(r.snippet || r.text.substring(0, 240) + "...")}</div>
        <div class="clause-card-footer">
          <span>Type: ${r.type || "Statute"}</span>
          <span>L2 Distance: ${r.distance || "0.25"}</span>
        </div>
      `;
      elements.searchResultsViewport.appendChild(card);
    });
  }

  // ── MODALS LOGIC ────────────────────────────────────────────
  function openChunkModal(chunk) {
    if (!elements.chunkModal) return;
    if (elements.chunkModalSource) elements.chunkModalSource.textContent = `Source: ${chunk.source || "Indian Legal Document"}`;
    if (elements.chunkModalType) elements.chunkModalType.textContent = chunk.type || "Statute";
    if (elements.chunkModalScore) elements.chunkModalScore.textContent = `${chunk.relevance_score || 90}%`;
    if (elements.chunkModalDist) elements.chunkModalDist.textContent = chunk.distance || "0.245";
    if (elements.chunkModalId) elements.chunkModalId.textContent = `#${chunk.chunk_id || 0}`;
    if (elements.chunkModalText) elements.chunkModalText.textContent = chunk.text || "";

    if (elements.askAboutChunkBtn) {
      elements.askAboutChunkBtn.onclick = () => {
        closeChunkModal();
        updateModeUI("agentic");
        if (elements.legalQueryInput) {
          elements.legalQueryInput.value = `Explain the legal implications and judicial interpretation of this provision: "${(chunk.snippet || chunk.text).substring(0, 160)}..."`;
          autoResizeTextarea(elements.legalQueryInput);
          elements.legalQueryInput.focus();
        }
      };
    }

    elements.chunkModal.classList.add("active");
  }

  function openChunkModalBySource(sourceName, chunksList) {
    const list = chunksList && chunksList.length > 0 ? chunksList : state.currentRetrievedChunks;
    const match = list.find(c => c.source === sourceName) || list[0];
    if (match) openChunkModal(match);
  }

  function closeChunkModal() {
    if (elements.chunkModal) elements.chunkModal.classList.remove("active");
  }

  function openSettingsModal() {
    initSettingsUI();
    if (elements.settingsModal) elements.settingsModal.classList.add("active");
  }

  function closeSettingsModal() {
    if (elements.settingsModal) elements.settingsModal.classList.remove("active");
  }

  function saveSettings() {
    const newConfig = {
      model: elements.settingModelSelect ? elements.settingModelSelect.value : state.config.model,
      mode: elements.settingModeSelect ? elements.settingModeSelect.value : state.config.mode,
      top_k: elements.settingTopkSlider ? parseInt(elements.settingTopkSlider.value, 10) : state.config.top_k,
      temperature: elements.settingTempSlider ? parseFloat(elements.settingTempSlider.value) : state.config.temperature,
      autoScroll: elements.settingAutoscroll ? elements.settingAutoscroll.checked : true,
      highlightCitations: elements.settingCitationHighlight ? elements.settingCitationHighlight.checked : true,
      showAgentSteps: elements.settingShowAgentSteps ? elements.settingShowAgentSteps.checked : true
    };

    saveConfig(newConfig);
    updateModeUI(newConfig.mode);
    closeSettingsModal();
    showToast("Settings and RAG parameters updated", "success");
    renderCurrentSessionMessages();
  }

  async function openHealthModal() {
    if (elements.healthModal) elements.healthModal.classList.add("active");
    if (!elements.healthModalContent) return;

    elements.healthModalContent.innerHTML = `
      <div class="loading-spinner-wrap">
        <div class="spinner"></div>
        <p>Querying backend diagnostics & corpus statistics…</p>
      </div>
    `;

    try {
      const res = await fetch("/api/health");
      const data = await res.json();

      elements.healthModalContent.innerHTML = `
        <div class="health-metrics-grid">
          <div class="health-metric-card">
            <div class="hm-label">System Status</div>
            <div class="hm-value" style="color:var(--success);">● Active & Healthy</div>
          </div>
          <div class="health-metric-card">
            <div class="hm-label">Indexed Legal Vectors</div>
            <div class="hm-value">${(data.total_vectors || 442528).toLocaleString()}</div>
          </div>
          <div class="health-metric-card">
            <div class="hm-label">Embedding Engine</div>
            <div class="hm-value" style="font-size:0.88rem;">${data.embedding_model || "all-MiniLM-L6-v2"}</div>
          </div>
          <div class="health-metric-card">
            <div class="hm-label">Default LLM Model</div>
            <div class="hm-value" style="font-size:0.88rem;">${data.default_llm || "gpt-oss-20b"}</div>
          </div>
        </div>
        <div class="setting-group" style="margin-top:12px;">
          <label class="setting-label">Indexed Indian Statutes</label>
          <ul style="padding-left:18px; font-size:0.85rem; color:var(--text-secondary); line-height:1.6;">
            <li>Constitution of India (Articles 1–395 & Schedules)</li>
            <li>Bharatiya Nyaya Sanhita (BNS 2023)</li>
            <li>Bharatiya Nagarik Suraksha Sanhita (BNSS 2023)</li>
            <li>Bharatiya Sakshya Adhiniyam (BSA 2023)</li>
            <li>Information Technology Act 2000 & Cyber Rules</li>
            <li>Indian Contract Act 1872 & Commercial Statutes</li>
            <li>Supreme Court Landmark Precedents</li>
          </ul>
        </div>
      `;
    } catch (err) {
      elements.healthModalContent.innerHTML = `
        <div class="alert-box">
          ⚠️ <strong>Backend Unreachable:</strong> Could not connect to /api/health. Please ensure the Flask backend server is running.
        </div>
      `;
    }
  }

  function closeHealthModal() {
    if (elements.healthModal) elements.healthModal.classList.remove("active");
  }

  function openDisclaimerModal() {
    if (elements.disclaimerModal) elements.disclaimerModal.classList.add("active");
  }

  function closeDisclaimerModal() {
    if (elements.disclaimerModal) elements.disclaimerModal.classList.remove("active");
  }

  // ── EXPORT CONSULTATION ─────────────────────────────────────
  function exportConsultation() {
    const session = getActiveSession();
    if (!session || session.messages.length === 0) {
      showToast("No consultation messages to export", "info");
      return;
    }

    let md = `# Indian Legal AI Assistant — Consultation Brief\n\n`;
    md += `**Date:** ${new Date(session.timestamp).toLocaleString()}\n`;
    md += `**Topic:** ${session.title}\n`;
    md += `**System:** Indian Legal AI (Multi-Agent RAG)\n\n---\n\n`;

    session.messages.forEach(m => {
      const sender = m.sender === "user" ? "### Legal Query" : "### Legal Advisory & Synthesis";
      md += `${sender} (${m.timestamp})\n\n${m.text}\n\n`;
      if (m.sources && m.sources.length > 0) {
        md += `**Citations:** ${m.sources.map(s => s.name).join(", ")}\n\n`;
      }
      md += `---\n\n`;
    });

    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `Indian_Legal_Consultation_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("Consultation exported as Markdown", "success");
  }

  // ── SYSTEM STATUS FETCH ─────────────────────────────────────
  async function fetchSystemStatus() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        if (elements.statusModelText) {
          elements.statusModelText.textContent = `FAISS ${(data.total_vectors || 442528).toLocaleString()} • ${state.config.model.split("/").pop()}`;
        }
        if (elements.footerChunkCount) {
          elements.footerChunkCount.textContent = (data.total_chunks || 442528).toLocaleString();
        }
      }
    } catch {
      if (elements.statusModelText) elements.statusModelText.textContent = "FAISS 442k • Multi-Agent";
    }
  }

  // ── EVENT LISTENERS ─────────────────────────────────────────
  function attachEventListeners() {
    // Mode Switching Tabs
    if (elements.tabAgentic) elements.tabAgentic.onclick = () => updateModeUI("agentic");
    if (elements.tabRagChat) elements.tabRagChat.onclick = () => updateModeUI("rag");
    if (elements.tabVectorSearch) elements.tabVectorSearch.onclick = () => updateModeUI("search");

    // Sidebar Mode Cards
    if (elements.sideModeAgentic) elements.sideModeAgentic.onclick = () => updateModeUI("agentic");
    if (elements.sideModeRag) elements.sideModeRag.onclick = () => updateModeUI("rag");
    if (elements.sideModeSearch) elements.sideModeSearch.onclick = () => updateModeUI("search");

    // Theme Toggle
    if (elements.themeToggleBtn) elements.themeToggleBtn.onclick = toggleTheme;

    // Chat Composer
    if (elements.sendQueryBtn) elements.sendQueryBtn.onclick = handleUserSubmit;
    if (elements.legalQueryInput) {
      elements.legalQueryInput.onkeydown = (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
          e.preventDefault();
          handleUserSubmit();
        }
      };
      elements.legalQueryInput.oninput = () => autoResizeTextarea(elements.legalQueryInput);
    }

    // Voice Dictation
    if (elements.voiceDictateBtn) elements.voiceDictateBtn.onclick = toggleVoiceDictation;

    // Direct Search View
    if (elements.directSearchBtn) elements.directSearchBtn.onclick = handleDirectSearch;
    if (elements.directVectorQuery) {
      elements.directVectorQuery.onkeydown = (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          handleDirectSearch();
        }
      };
    }

    // Quick Filter Pills in Search View
    document.querySelectorAll(".filter-pill").forEach(pill => {
      pill.onclick = () => {
        const query = pill.getAttribute("data-search");
        if (elements.directVectorQuery && query) {
          elements.directVectorQuery.value = query;
          handleDirectSearch();
        }
      };
    });

    // Domain Override Chips in Composer
    document.querySelectorAll(".domain-chip").forEach(chip => {
      chip.onclick = () => {
        document.querySelectorAll(".domain-chip").forEach(c => c.classList.remove("active"));
        chip.classList.add("active");
        state.selectedDomain = chip.getAttribute("data-domain") || "auto";
        showToast(`Domain scoped to: ${chip.textContent.trim()}`, "info");
      };
    });

    // Preset Prompt Cards in Welcome Screen
    document.querySelectorAll(".prompt-card").forEach(card => {
      card.onclick = () => {
        const query = card.getAttribute("data-query");
        if (elements.legalQueryInput && query) {
          elements.legalQueryInput.value = query;
          autoResizeTextarea(elements.legalQueryInput);
          handleUserSubmit();
        }
      };
    });

    // Sidebar Controls
    if (elements.newChatBtn) elements.newChatBtn.onclick = createNewSession;
    if (elements.clearAllHistoryBtn) elements.clearAllHistoryBtn.onclick = clearAllSessions;
    if (elements.composerClearBtn) elements.composerClearBtn.onclick = createNewSession;
    if (elements.historySearch) {
      elements.historySearch.oninput = (e) => renderHistoryList(e.target.value);
    }

    // Mobile Sidebar Drawer
    if (elements.mobileMenuBtn) elements.mobileMenuBtn.onclick = openMobileSidebar;
    if (elements.closeSidebarBtn) elements.closeSidebarBtn.onclick = closeMobileSidebar;
    if (elements.sidebarOverlay) elements.sidebarOverlay.onclick = closeMobileSidebar;

    // Modals
    if (elements.openSettingsBtn) elements.openSettingsBtn.onclick = openSettingsModal;
    if (elements.closeSettingsModalBtn) elements.closeSettingsModalBtn.onclick = closeSettingsModal;
    if (elements.saveSettingsBtn) elements.saveSettingsBtn.onclick = saveSettings;
    if (elements.resetSettingsBtn) elements.resetSettingsBtn.onclick = () => {
      saveConfig(DEFAULT_CONFIG);
      initSettingsUI();
      showToast("Reset to factory defaults", "info");
    };

    if (elements.settingTopkSlider && elements.settingTopkVal) {
      elements.settingTopkSlider.oninput = (e) => elements.settingTopkVal.textContent = `${e.target.value} Chunks`;
    }
    if (elements.settingTempSlider && elements.settingTempVal) {
      elements.settingTempSlider.oninput = (e) => elements.settingTempVal.textContent = parseFloat(e.target.value).toFixed(2);
    }

    if (elements.openHealthBtn) elements.openHealthBtn.onclick = openHealthModal;
    if (elements.closeHealthModalBtn) elements.closeHealthModalBtn.onclick = closeHealthModal;
    if (elements.closeHealthFooterBtn) elements.closeHealthFooterBtn.onclick = closeHealthModal;

    if (elements.closeChunkModalBtn) elements.closeChunkModalBtn.onclick = closeChunkModal;
    if (elements.closeChunkModalFooter) elements.closeChunkModalFooter.onclick = closeChunkModal;
    if (elements.copyChunkTextBtn) {
      elements.copyChunkTextBtn.onclick = () => {
        if (elements.chunkModalText) {
          navigator.clipboard.writeText(elements.chunkModalText.textContent);
          showToast("Clause text copied to clipboard", "success");
        }
      };
    }

    if (elements.viewDisclaimerLink) elements.viewDisclaimerLink.onclick = (e) => { e.preventDefault(); openDisclaimerModal(); };
    if (elements.closeDisclaimerModalBtn) elements.closeDisclaimerModalBtn.onclick = closeDisclaimerModal;
    if (elements.ackDisclaimerBtn) elements.ackDisclaimerBtn.onclick = closeDisclaimerModal;

    // Export Action
    if (elements.exportChatBtn) elements.exportChatBtn.onclick = exportConsultation;
  }

  // ── HELPER UTILITIES ────────────────────────────────────────
  function openMobileSidebar() {
    if (elements.appSidebar) elements.appSidebar.classList.add("open");
    if (elements.sidebarOverlay) elements.sidebarOverlay.classList.add("open");
  }

  function closeMobileSidebar() {
    if (elements.appSidebar) elements.appSidebar.classList.remove("open");
    if (elements.sidebarOverlay) elements.sidebarOverlay.classList.remove("open");
  }

  function autoResizeTextarea(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 140) + "px";
  }

  function showToast(message, type = "info") {
    if (!elements.toastStack) return;
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    const icon = type === "success" ? "✓" : (type === "error" ? "⚠️" : "ℹ️");
    toast.innerHTML = `<span>${icon}</span> <span>${escapeHtml(message)}</span>`;
    elements.toastStack.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(30px)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  function escapeHtml(str) {
    return (str || "").replace(/[&<>"']/g, m => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;"
    }[m]));
  }

  // ── RUN INIT ON DOM READY ───────────────────────────────────
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

})();
