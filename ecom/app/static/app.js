/**
 * ecom support — Frontend Application State & Client Logic
 */

const STATE = {
  token: localStorage.getItem("ecom_token") || null,
  currentUser: null,
  activeSessionId: null,
  isTraceOpen: false,
};

// ─── DOM References ──────────────────────────────────────────────────────────
const dom = {
  app: document.getElementById("app"),
  userName: document.getElementById("userName"),
  userRole: document.getElementById("userRole"),
  userBadge: document.getElementById("userBadge"),
  loginModalBtn: document.getElementById("loginModalBtn"),
  logoutBtn: document.getElementById("logoutBtn"),
  themeToggleBtn: document.getElementById("themeToggleBtn"),
  traceToggleBtn: document.getElementById("traceToggleBtn"),
  newChatBtn: document.getElementById("newChatBtn"),
  sessionsList: document.getElementById("sessionsList"),
  adminPanel: document.getElementById("adminPanel"),
  reembedBtn: document.getElementById("reembedBtn"),
  pendingRefundsBadge: document.getElementById("pendingRefundsBadge"),
  pendingRefundsList: document.getElementById("pendingRefundsList"),
  chatMessages: document.getElementById("chatMessages"),
  chatForm: document.getElementById("chatForm"),
  messageInput: document.getElementById("messageInput"),
  sendBtn: document.getElementById("sendBtn"),
  traceDrawer: document.getElementById("traceDrawer"),
  closeTraceBtn: document.getElementById("closeTraceBtn"),
  metricLatency: document.getElementById("metricLatency"),
  metricCost: document.getElementById("metricCost"),
  metricPromptTokens: document.getElementById("metricPromptTokens"),
  metricCompTokens: document.getElementById("metricCompTokens"),
  traceId: document.getElementById("traceId"),
  traceUserDetails: document.getElementById("traceUserDetails"),
  traceStatus: document.getElementById("traceStatus"),
  stepsList: document.getElementById("stepsList"),

  // Authentication Modal Elements
  authModal: document.getElementById("authModal"),
  closeAuthModalBtn: document.getElementById("closeAuthModalBtn"),
  tabSignIn: document.getElementById("tabSignIn"),
  tabSignUp: document.getElementById("tabSignUp"),
  signInForm: document.getElementById("signInForm"),
  signUpForm: document.getElementById("signUpForm"),
  loginIdentifier: document.getElementById("loginIdentifier"),
  loginPassword: document.getElementById("loginPassword"),
  signInSubmitBtn: document.getElementById("signInSubmitBtn"),
  loginError: document.getElementById("loginError"),
  goToSignUpLink: document.getElementById("goToSignUpLink"),
  signupName: document.getElementById("signupName"),
  signupEmail: document.getElementById("signupEmail"),
  signupPassword: document.getElementById("signupPassword"),
  signupRole: document.getElementById("signupRole"),
  signUpSubmitBtn: document.getElementById("signUpSubmitBtn"),
  signUpError: document.getElementById("signUpError"),
  goToSignInLink: document.getElementById("goToSignInLink"),
};

// ─── API Client Helper ────────────────────────────────────────────────────────
async function api(endpoint, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };
  if (STATE.token) {
    headers["Authorization"] = `Bearer ${STATE.token}`;
  }

  const res = await fetch(endpoint, { ...options, headers });
  if (res.status === 401) {
    console.warn("Session unauthorized. Clearing token.");
    localStorage.removeItem("ecom_token");
    STATE.token = null;
    STATE.currentUser = null;
    updateUserUI();
    openAuthModal("signin");
    showAuthError("login", "Your session has expired. Please sign in again.");
    return null;
  }
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

// ─── Authentication Handlers ──────────────────────────────────────────────────
async function loadCurrentUser() {
  if (!STATE.token) {
    updateUserUI();
    openAuthModal("signin");
    return;
  }
  try {
    const user = await api("/api/auth/me");
    if (!user) {
      updateUserUI();
      openAuthModal("signin");
      return;
    }
    STATE.currentUser = user;
    updateUserUI();
    await loadSessions();
    if (STATE.currentUser.role === "admin") {
      await loadPendingRefunds();
    }
  } catch (err) {
    console.warn("Could not load current user:", err);
    STATE.token = null;
    STATE.currentUser = null;
    localStorage.removeItem("ecom_token");
    updateUserUI();
    openAuthModal("signin");
  }
}

function updateUserUI() {
  const isLoggedIn = !!STATE.currentUser;

  if (isLoggedIn) {
    if (dom.userBadge) {
      dom.userBadge.classList.remove("hidden");
      dom.userBadge.style.display = "flex";
    }
    if (dom.userName) dom.userName.textContent = STATE.currentUser.name;
    if (dom.userRole) {
      dom.userRole.textContent = STATE.currentUser.role;
      dom.userRole.className = `role-pill ${STATE.currentUser.role}`;
    }
    if (dom.loginModalBtn) {
      dom.loginModalBtn.classList.add("hidden");
      dom.loginModalBtn.style.display = "none";
    }
    if (dom.logoutBtn) {
      dom.logoutBtn.classList.remove("hidden");
      dom.logoutBtn.style.display = "inline-flex";
    }
  } else {
    if (dom.userBadge) {
      dom.userBadge.classList.add("hidden");
      dom.userBadge.style.display = "none";
    }
    if (dom.loginModalBtn) {
      dom.loginModalBtn.classList.remove("hidden");
      dom.loginModalBtn.style.display = "inline-flex";
    }
    if (dom.logoutBtn) {
      dom.logoutBtn.classList.add("hidden");
      dom.logoutBtn.style.display = "none";
    }
  }

  const isAdmin = isLoggedIn && STATE.currentUser.role === "admin";

  // Feature: Remove Admin Panel and Trace Inspector completely from Customer view
  if (isAdmin) {
    if (dom.adminPanel) {
      dom.adminPanel.classList.remove("hidden");
      dom.adminPanel.style.display = "block";
    }
    if (dom.traceToggleBtn) {
      dom.traceToggleBtn.classList.remove("hidden");
      dom.traceToggleBtn.style.display = "inline-flex";
    }
  } else {
    if (dom.adminPanel) {
      dom.adminPanel.classList.add("hidden");
      dom.adminPanel.style.display = "none";
    }
    if (dom.traceToggleBtn) {
      dom.traceToggleBtn.classList.add("hidden");
      dom.traceToggleBtn.style.display = "none";
    }
    if (STATE.isTraceOpen) {
      toggleTraceDrawer();
    }
  }
}

// ─── Auth Modal Management ────────────────────────────────────────────────────
function openAuthModal(tab = "signin") {
  if (dom.authModal) {
    dom.authModal.classList.remove("hidden");
    dom.authModal.style.display = "flex";
  }
  switchAuthTab(tab);
}

function closeAuthModal() {
  if (dom.authModal) {
    dom.authModal.classList.add("hidden");
    dom.authModal.style.display = "none";
  }
  clearAuthErrors();
}

function switchAuthTab(tab) {
  clearAuthErrors();
  if (tab === "signin") {
    if (dom.tabSignIn) dom.tabSignIn.classList.add("active");
    if (dom.tabSignUp) dom.tabSignUp.classList.remove("active");
    if (dom.signInForm) {
      dom.signInForm.classList.remove("hidden");
      dom.signInForm.style.display = "block";
    }
    if (dom.signUpForm) {
      dom.signUpForm.classList.add("hidden");
      dom.signUpForm.style.display = "none";
    }
    if (dom.loginIdentifier) setTimeout(() => dom.loginIdentifier.focus(), 80);
  } else {
    if (dom.tabSignUp) dom.tabSignUp.classList.add("active");
    if (dom.tabSignIn) dom.tabSignIn.classList.remove("active");
    if (dom.signUpForm) {
      dom.signUpForm.classList.remove("hidden");
      dom.signUpForm.style.display = "block";
    }
    if (dom.signInForm) {
      dom.signInForm.classList.add("hidden");
      dom.signInForm.style.display = "none";
    }
    if (dom.signupName) setTimeout(() => dom.signupName.focus(), 80);
  }
}

function clearAuthErrors() {
  if (dom.loginError) {
    dom.loginError.textContent = "";
    dom.loginError.classList.add("hidden");
    dom.loginError.style.display = "none";
  }
  if (dom.signUpError) {
    dom.signUpError.textContent = "";
    dom.signUpError.classList.add("hidden");
    dom.signUpError.style.display = "none";
  }
}

function showAuthError(form, msg) {
  const el = form === "login" ? dom.loginError : dom.signUpError;
  if (el) {
    el.textContent = msg;
    el.classList.remove("hidden");
    el.style.display = "block";
  }
}

async function handleSignInSubmit(e) {
  if (e) e.preventDefault();
  clearAuthErrors();

  const identifier = dom.loginIdentifier.value.trim();
  const password = dom.loginPassword.value;

  if (!identifier || !password) {
    showAuthError("login", "Please enter both User ID/Email and password.");
    return;
  }

  dom.signInSubmitBtn.disabled = true;
  dom.signInSubmitBtn.textContent = "Signing In...";

  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identifier, password }),
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      showAuthError("login", data.detail || "Invalid User ID/Email or password.");
      return;
    }

    STATE.token = data.access_token;
    localStorage.setItem("ecom_token", STATE.token);
    STATE.currentUser = data.user;

    closeAuthModal();
    updateUserUI();
    await loadSessions();
    if (STATE.currentUser.role === "admin") {
      await loadPendingRefunds();
    }
    startNewChat();
  } catch (err) {
    showAuthError("login", "Authentication error: " + err.message);
  } finally {
    dom.signInSubmitBtn.disabled = false;
    dom.signInSubmitBtn.textContent = "Sign In";
  }
}

async function handleSignUpSubmit(e) {
  if (e) e.preventDefault();
  clearAuthErrors();

  const name = dom.signupName.value.trim();
  const email = dom.signupEmail.value.trim();
  const password = dom.signupPassword.value;
  const role = dom.signupRole.value;

  if (!name || !email || !password) {
    showAuthError("signup", "All fields are required.");
    return;
  }
  if (!email.includes("@")) {
    showAuthError("signup", "Please enter a valid email address.");
    return;
  }
  if (password.length < 6) {
    showAuthError("signup", "Password must be at least 6 characters.");
    return;
  }

  dom.signUpSubmitBtn.disabled = true;
  dom.signUpSubmitBtn.textContent = "Creating Account...";

  try {
    const res = await fetch("/api/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password, role }),
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      showAuthError("signup", data.detail || "Registration failed.");
      return;
    }

    STATE.token = data.access_token;
    localStorage.setItem("ecom_token", STATE.token);
    STATE.currentUser = data.user;

    closeAuthModal();
    updateUserUI();
    await loadSessions();
    if (STATE.currentUser.role === "admin") {
      await loadPendingRefunds();
    }
    startNewChat();
  } catch (err) {
    showAuthError("signup", "Registration error: " + err.message);
  } finally {
    dom.signUpSubmitBtn.disabled = false;
    dom.signUpSubmitBtn.textContent = "Create Account";
  }
}

function handleLogout() {
  localStorage.removeItem("ecom_token");
  STATE.token = null;
  STATE.currentUser = null;
  STATE.activeSessionId = null;

  if (dom.sessionsList) {
    dom.sessionsList.innerHTML = '<div class="empty-state-sm">Sign in to view conversations</div>';
  }

  startNewChat();
  updateUserUI();
  openAuthModal("signin");
}

// ─── Sessions & Chat History ──────────────────────────────────────────────────
async function loadSessions() {
  try {
    const sessions = await api("/api/chat/sessions");
    dom.sessionsList.innerHTML = "";
    if (!sessions || sessions.length === 0) {
      dom.sessionsList.innerHTML = `<div class="empty-state-sm">No conversations yet</div>`;
      return;
    }

    sessions.forEach((s) => {
      const item = document.createElement("div");
      item.className = `session-item ${s.id === STATE.activeSessionId ? "active" : ""}`;
      item.textContent = s.title || "Untitled Session";
      item.onclick = () => selectSession(s.id);
      dom.sessionsList.appendChild(item);
    });
  } catch (err) {
    console.error("Failed to load sessions:", err);
  }
}

async function selectSession(sessionId) {
  STATE.activeSessionId = sessionId;
  document.querySelectorAll(".session-item").forEach((el) => el.classList.remove("active"));
  await loadSessions();

  try {
    const messages = await api(`/api/chat/sessions/${sessionId}/messages`);
    dom.chatMessages.innerHTML = "";
    if (messages && messages.length > 0) {
      messages.forEach((m) => renderMessage(m.role, m.content, m.tool_calls));
      scrollChatToBottom();
    }
  } catch (err) {
    console.error("Failed to load messages:", err);
  }
}

function startNewChat() {
  STATE.activeSessionId = null;
  document.querySelectorAll(".session-item").forEach((el) => el.classList.remove("active"));
  dom.chatMessages.innerHTML = `
    <div class="welcome-hero">
      <div class="hero-icon">🛍️</div>
      <h2>How can I help you today?</h2>
      <p>I can track your shipments, check catalog inventory, evaluate return windows, and process refunds according to store policies.</p>
      
      <div class="suggestions-grid">
        <button class="suggestion-chip" data-prompt="Where is my order ORD-2024-001 and when was it delivered?">
          📦 Track order ORD-2024-001
        </button>
        <button class="suggestion-chip" data-prompt="Can I return the Wireless Bluetooth Headphones from order ORD-2024-001?">
          🔄 Check return eligibility for ORD-2024-001
        </button>
        <button class="suggestion-chip" data-prompt="Do you have any 4K webcams in stock, and how long is their warranty?">
          🔍 Check 4K webcam stock & warranty
        </button>
        <button class="suggestion-chip" data-prompt="What happens if a package shows delivered but I haven't received it?">
          📋 What is your lost package policy?
        </button>
      </div>
    </div>
  `;
  attachSuggestionListeners();
}

// ─── Message Rendering & Text Formatting ──────────────────────────────────────
function formatMessageContent(rawText) {
  let text = rawText || "";

  // Highlight Citations [Source: ...]
  text = text.replace(/\[Source:\s*([^\]]+)\]/gi, (match, src) => {
    return `<span class="citation-badge">📄 ${src.trim()}</span>`;
  });

  // Code blocks
  text = text.replace(/```([\s\S]*?)```/g, '<pre class="step-details-pre"><code>$1</code></pre>');

  // Inline backticks
  text = text.replace(/`([^`]+)`/g, '<code class="font-mono">$1</code>');

  // Bold text
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  // Newlines
  text = text.replace(/\n/g, "<br>");

  return text;
}

function renderMessage(role, content, toolCalls = null) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  let toolsHtml = "";
  if (toolCalls && toolCalls.length > 0) {
    toolsHtml = `<div class="tool-pill-container">`;
    toolCalls.forEach((tc) => {
      const toolName = tc.tool || (tc.function && tc.function.name) || "tool";
      toolsHtml += `<span class="tool-pill" title="Click to view tool details">🔧 ${toolName}</span>`;
    });
    toolsHtml += `</div>`;
  }

  row.innerHTML = `
    ${toolsHtml}
    <div class="message-bubble">
      ${formatMessageContent(content)}
    </div>
  `;

  dom.chatMessages.appendChild(row);
}

function showThinkingIndicator() {
  const row = document.createElement("div");
  row.id = "thinkingIndicator";
  row.className = "message-row assistant";
  row.innerHTML = `
    <div class="thinking-row">
      <div class="spinner"></div>
      <span>AI reasoning & executing tools...</span>
    </div>
  `;
  dom.chatMessages.appendChild(row);
  scrollChatToBottom();
}

function removeThinkingIndicator() {
  const el = document.getElementById("thinkingIndicator");
  if (el) el.remove();
}

function scrollChatToBottom() {
  dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
}

// ─── Sending Messages ─────────────────────────────────────────────────────────
async function handleSendMessage(e) {
  if (e) e.preventDefault();
  if (!STATE.token || !STATE.currentUser) {
    openAuthModal("signin");
    showAuthError("login", "Please sign in or create an account to start chatting.");
    return;
  }
  const msg = dom.messageInput.value.trim();
  if (!msg) return;

  dom.messageInput.value = "";
  dom.messageInput.style.height = "auto";

  const hero = dom.chatMessages.querySelector(".welcome-hero");
  if (hero) hero.remove();

  renderMessage("user", msg);
  showThinkingIndicator();

  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        message: msg,
        session_id: STATE.activeSessionId,
      }),
    });

    removeThinkingIndicator();

    if (data) {
      STATE.activeSessionId = data.session_id;
      renderMessage("assistant", data.response, data.tool_calls);
      scrollChatToBottom();
      await loadSessions();

      if (data.trace) {
        updateTraceInspector(data.trace);
      }
    }
  } catch (err) {
    removeThinkingIndicator();
    renderMessage("assistant", `⚠️ Error: ${err.message}`);
    scrollChatToBottom();
  }
}

// ─── Trace Inspector Drawer ───────────────────────────────────────────────────
function updateTraceInspector(trace) {
  dom.traceId.textContent = trace.trace_id || "-";
  dom.metricLatency.textContent = `${trace.total_latency_ms || 0} ms`;
  dom.metricCost.textContent = `$${(trace.estimated_cost_usd || 0).toFixed(6)}`;
  dom.metricPromptTokens.textContent = trace.prompt_tokens || 0;
  dom.metricCompTokens.textContent = trace.completion_tokens || 0;

  // Feature: Display User Details in Traces
  if (dom.traceUserDetails) {
    const u = trace.user_details;
    if (u && typeof u === "object") {
      dom.traceUserDetails.textContent = `${u.name || ""} (${u.email || ""}) [${u.role || ""}]`;
    } else if (typeof u === "string") {
      try {
        const parsed = JSON.parse(u);
        dom.traceUserDetails.textContent = `${parsed.name} (${parsed.email}) [${parsed.role}]`;
      } catch {
        dom.traceUserDetails.textContent = u;
      }
    } else {
      dom.traceUserDetails.textContent = "-";
    }
  }

  dom.traceStatus.textContent = trace.status || "success";
  dom.traceStatus.className = `status-badge ${trace.status === "success" ? "status-success" : ""}`;

  dom.stepsList.innerHTML = "";
  if (!trace.steps || trace.steps.length === 0) {
    dom.stepsList.innerHTML = `<div class="empty-state-sm">Zero tool calls (Direct response)</div>`;
    return;
  }

  trace.steps.forEach((step) => {
    const card = document.createElement("div");
    card.className = "step-card";
    const detailsStr = JSON.stringify(step.details || {}, null, 2);
    card.innerHTML = `
      <div class="step-header">
        <span>Step ${step.step_number}: ${step.name}</span>
        <span class="step-duration">${step.duration_ms} ms</span>
      </div>
      <pre class="step-details-pre"><code>${detailsStr}</code></pre>
    `;
    dom.stepsList.appendChild(card);
  });
}

function toggleTraceDrawer() {
  STATE.isTraceOpen = !STATE.isTraceOpen;
  dom.traceDrawer.classList.toggle("open", STATE.isTraceOpen);
}

// ─── Admin Pending Refunds ───────────────────────────────────────────────────
async function loadPendingRefunds() {
  try {
    const data = await api("/api/admin/refunds/pending");
    if (!data) return;

    dom.pendingRefundsBadge.textContent = data.pending_count || 0;
    dom.pendingRefundsList.innerHTML = "";

    if (!data.refunds || data.refunds.length === 0) {
      dom.pendingRefundsList.innerHTML = `<div class="empty-state-sm">No refunds awaiting review</div>`;
      return;
    }

    data.refunds.forEach((r) => {
      const item = document.createElement("div");
      item.className = "pending-refund-item";
      item.innerHTML = `
        <div style="display:flex; justify-content:space-between;">
          <span class="font-mono"><strong>${r.refund_id}</strong> (${r.order_id})</span>
          <span class="pending-refund-amount">$${r.amount.toFixed(2)}</span>
        </div>
        <div style="color:var(--text-secondary); margin-top:2px;">${r.customer_email}</div>
        <button class="btn approve-btn" data-id="${r.refund_id}">Approve Refund</button>
      `;

      item.querySelector(".approve-btn").onclick = () => approveRefund(r.refund_id);
      dom.pendingRefundsList.appendChild(item);
    });
  } catch (err) {
    console.error("Failed to load pending refunds:", err);
  }
}

async function approveRefund(refundId) {
  try {
    const data = await api(`/api/admin/refunds/${refundId}/approve`, { method: "POST" });
    if (data) {
      alert(`Refund ${refundId} approved successfully.`);
      await loadPendingRefunds();
    }
  } catch (err) {
    alert(`Failed to approve refund: ${err.message}`);
  }
}

// ─── Dynamic Re-embedding ────────────────────────────────────────────────────
async function handleReembed() {
  if (!confirm("Re-chunk and compute fresh embeddings for all documents in data/policies/?")) return;
  dom.reembedBtn.disabled = true;
  dom.reembedBtn.innerHTML = `<span>⏳</span> Embedding...`;
  try {
    const data = await api("/api/admin/reembed", { method: "POST" });
    if (data) {
      alert(data.message || `Successfully re-embedded ${data.chunks_count} policy sections!`);
    }
  } catch (err) {
    alert(`Re-embedding failed: ${err.message}`);
  } finally {
    dom.reembedBtn.disabled = false;
    dom.reembedBtn.innerHTML = `<span>🔄</span> Re-embed Policies`;
  }
}

// ─── Suggestion Chips ─────────────────────────────────────────────────────────
function attachSuggestionListeners() {
  document.querySelectorAll(".suggestion-chip").forEach((chip) => {
    chip.onclick = () => {
      if (!STATE.token || !STATE.currentUser) {
        openAuthModal("signin");
        showAuthError("login", "Please sign in or create an account to start chatting.");
        return;
      }
      const prompt = chip.getAttribute("data-prompt");
      if (prompt) {
        dom.messageInput.value = prompt;
        handleSendMessage();
      }
    };
  });
}

// ─── Event Listeners ──────────────────────────────────────────────────────────
function initEvents() {
  dom.chatForm.onsubmit = handleSendMessage;

  dom.messageInput.onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  dom.newChatBtn.onclick = startNewChat;
  dom.reembedBtn.onclick = handleReembed;
  dom.traceToggleBtn.onclick = toggleTraceDrawer;
  dom.closeTraceBtn.onclick = toggleTraceDrawer;

  // Authentication UI events
  if (dom.loginModalBtn) {
    dom.loginModalBtn.onclick = () => openAuthModal("signin");
  }
  if (dom.logoutBtn) {
    dom.logoutBtn.onclick = handleLogout;
  }
  if (dom.closeAuthModalBtn) {
    dom.closeAuthModalBtn.onclick = closeAuthModal;
  }
  if (dom.tabSignIn) {
    dom.tabSignIn.onclick = () => switchAuthTab("signin");
  }
  if (dom.tabSignUp) {
    dom.tabSignUp.onclick = () => switchAuthTab("signup");
  }
  if (dom.goToSignUpLink) {
    dom.goToSignUpLink.onclick = () => switchAuthTab("signup");
  }
  if (dom.goToSignInLink) {
    dom.goToSignInLink.onclick = () => switchAuthTab("signin");
  }
  if (dom.signInForm) {
    dom.signInForm.onsubmit = handleSignInSubmit;
  }
  if (dom.signUpForm) {
    dom.signUpForm.onsubmit = handleSignUpSubmit;
  }

  dom.themeToggleBtn.onclick = () => {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", nextTheme);
    dom.themeToggleBtn.textContent = nextTheme === "dark" ? "🌙" : "☀️";
  };

  attachSuggestionListeners();
}

// ─── Bootstrap ────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", async () => {
  initEvents();
  await loadCurrentUser();
});
