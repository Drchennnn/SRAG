"use strict";

const API = "/api/workspace";
const byId = (id) => document.getElementById(id);
const ui = Object.fromEntries([
  "sessions", "new-session", "catalog", "import-form", "import-course", "import-files",
  "replace-label", "replace-document", "import-button", "import-results", "workspace-status",
  "scope-summary", "clear-file", "location", "location-kind", "location-title",
  "preview-location", "parent-location", "previous-location", "next-location", "messages",
  "turn-status", "turn-actions", "composer", "question", "composer-status", "send", "anchor-bar",
  "anchor-label", "clear-anchor", "reader", "reader-title", "reader-body",
  "reader-location", "close-reader",
].map((id) => [id, byId(id)]));

const state = {
  sessions: [],
  session: null,
  courses: [],
  nodes: new Map(),
  documents: new Map(),
  expanded: new Set(),
  epoch: 0,
  snapshotSequence: 0,
  navigating: false,
  loading: false,
  uploading: false,
  anchor: null,
  outbox: new Map(),
  turnOptions: new Map(),
  threadKey: "",
  actionKey: "",
  pollTimer: null,
};

const kindText = {
  course: "课程", document: "资料", chapter: "章节", section: "章节", page: "页",
  page_object: "页内对象", cell: "Cell", notebook_output: "保存输出",
};
const readText = {
  not_attempted: "未读取", succeeded: "已读取", partial: "部分读取", failed: "读取失败",
};
const statusText = {
  running: "正在处理", completed: "已完成", clarification: "需要澄清",
  failed: "处理失败", cancelled: "已取消",
};
const errors = {
  not_accessible: "当前入口不可访问，请检查会话和课程范围。",
  scope_changed: "本会话的范围已更改，请核对范围后重试。",
  stale_reference: "资料版本已更改，请从目录重新选择。",
  original_unavailable: "这个版本的原件已不可用。",
  file_too_large: "文件超出导入大小限制。",
  invalid_filename: "请选择文件名有效的 PDF、PPTX 或 Notebook。",
  empty_file: "当前文件为空。",
  idempotency_conflict: "这次操作的提交内容已发生变化，请重新发送。",
  invalid_request: "请检查所选范围和输入。",
  interrupted: "运行已中断。",
  turn_busy: "本会话还有一条问题正在处理。",
};

function node(tag, className, label) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (label != null) element.textContent = label;
  return element;
}

function button(label, onClick, className = "quiet") {
  const element = node("button", className, label);
  element.type = "button";
  element.addEventListener("click", onClick);
  return element;
}

function setStatus(element, text) {
  element.textContent = text ?? "";
  if (element === ui["workspace-status"]) element.hidden = !text;
}

function describeError(error) {
  return errors[error?.code] ?? error?.message ?? "工作区暂时不可用，请重试。";
}

function currentPath() {
  if (!state.session) throw new Error("请先建立会话。");
  return sessionPath(state.session.session_id);
}

function sessionPath(id) {
  return `${API}/sessions/${encodeURIComponent(id)}`;
}

async function api(path, {method = "GET", body, binary = false} = {}) {
  const headers = {Accept: "application/json"};
  if (body !== undefined) {
    headers["Content-Type"] = binary ? "application/octet-stream" : "application/json";
  }
  let response;
  try {
    response = await fetch(path, {
      method, headers, credentials: "same-origin", redirect: "error", cache: "no-store",
      body: body === undefined || binary ? body : JSON.stringify(body),
    });
  } catch {
    throw new Error("无法连接工作区，请重试。");
  }
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error("工作区未返回可用数据，请重试。");
  }
  if (!response.ok) {
    const detail = payload?.detail;
    const code = typeof detail === "string" ? detail : (detail?.code ?? detail?.error);
    throw Object.assign(new Error(errors[code] ?? `请求未完成（${response.status}）。`), {
      code, status: response.status,
    });
  }
  return payload;
}

function mediaURL(value, sessionId, view = "inline") {
  if (typeof value !== "string" || !value.startsWith("/")) return null;
  try {
    const url = new URL(value, window.location.origin);
    const prefix = `${sessionPath(sessionId)}/`;
    const relative = url.pathname.slice(prefix.length);
    const correctPath = /^(messages\/[^/]+\/media\/[^/]+|catalog\/[^/]+\/media)$/.test(relative);
    if (url.origin !== location.origin || !url.pathname.startsWith(prefix) ||
        !correctPath || url.username || url.password || url.hash) return null;
    url.search = relative.startsWith("messages/") ? new URLSearchParams({view}).toString() : "";
    return url.pathname + url.search;
  } catch {
    return null;
  }
}

function mode() {
  return ui.composer.elements.kind.value;
}

function pickMode(value) {
  for (const input of ui.composer.querySelectorAll('input[name="kind"]')) {
    input.checked = input.value === value;
  }
}

function anchor(value = null, label = "") {
  state.anchor = value;
  ui["anchor-bar"].hidden = !value;
  ui["anchor-label"].textContent = label;
}

function useAnchor(value, label, question = "") {
  pickMode("ask");
  anchor(value, label);
  if (question && !ui.question.value.trim()) ui.question.value = question;
  ui.question.focus();
  renderControls();
}

function captureSession() {
  return {id: state.session?.session_id, epoch: state.epoch};
}

function isCurrent(capture) {
  return state.epoch === capture.epoch && state.session?.session_id === capture.id;
}

function loadSavedSession() {
  try { return localStorage.getItem("srag-workspace-session"); }
  catch { return null; }
}

function setSavedSession(value) {
  try { localStorage.setItem("srag-workspace-session", value); }
  catch { /* The authenticated session list can still restore the latest session. */ }
}

function indexCatalog(courses) {
  state.courses = courses;
  state.nodes.clear();
  state.documents.clear();
  function visit(item, documentEntry, parent) {
    const entry = {...item, doc: documentEntry, parent};
    state.nodes.set(entry.node_id, entry);
    for (const child of item.children ?? []) visit(child, documentEntry, entry);
  }
  for (const course of courses) {
    for (const item of course.documents ?? []) {
      const entry = {...item, course, kind: "document"};
      state.documents.set(entry.document_id, entry);
      state.nodes.set(entry.node_id, entry);
      for (const child of item.nodes ?? []) visit(child, entry, entry);
    }
  }
}

function revealFocus(item) {
  if (!item) return;
  const documentEntry = item.doc ?? item;
  state.expanded.add(documentEntry.node_id);
  state.expanded.add(documentEntry.course?.course_id);
  for (let parent = item.parent; parent; parent = parent.parent) {
    state.expanded.add(parent.node_id);
  }
  state.expanded.add(item.node_id);
}

function setSession(payload) {
  if (payload.session_id === state.session?.session_id &&
      payload.scope_revision < state.session.scope_revision) return;
  state.session = payload;
  for (const turn of payload.turns ?? []) {
    if (Array.isArray(turn.options)) state.turnOptions.set(turn.turn_id, turn.options);
    if (turn.status !== "running") {
      const local = state.outbox.get(payload.session_id);
      if (local?.payload.client_turn_id === turn.client_turn_id) {
        state.outbox.delete(payload.session_id);
      }
    }
  }
  render();
}

async function loadSession(id) {
  const epoch = ++state.epoch;
  state.session = null;
  state.loading = true;
  state.threadKey = "";
  state.expanded.clear();
  indexCatalog([]);
  anchor();
  ui.question.value = "";
  if (ui.reader.open) ui.reader.close();
  clearTimeout(state.pollTimer);
  setStatus(ui["workspace-status"], "");
  render();
  try {
    const path = sessionPath(id);
    const [session, catalog] = await Promise.all([api(path), api(`${path}/catalog`)]);
    if (state.epoch !== epoch) return;
    state.loading = false;
    indexCatalog(catalog.courses ?? []);
    for (const course of state.courses) {
      if (session.scope.course_ids.includes(course.course_id)) state.expanded.add(course.course_id);
    }
    for (const id of session.scope.document_ids) revealFocus(state.documents.get(id));
    revealFocus(state.nodes.get(session.focus?.node_id));
    setSavedSession(id);
    setSession(session);
  } catch (error) {
    if (state.epoch === epoch) setStatus(ui["workspace-status"], describeError(error));
  } finally {
    if (state.epoch === epoch) {
      state.loading = false;
      renderControls();
      renderCatalog();
    }
  }
}

async function reloadCatalog(capture) {
  const output = await api(`${sessionPath(capture.id)}/catalog`);
  if (!isCurrent(capture)) return;
  indexCatalog(output.courses ?? []);
  render();
}

async function reloadSession(capture) {
  const sequence = ++state.snapshotSequence;
  const output = await api(sessionPath(capture.id));
  if (isCurrent(capture) && sequence === state.snapshotSequence) setSession(output);
}

async function createSession() {
  ui["new-session"].disabled = true;
  try {
    const created = await api(`${API}/sessions`, {method: "POST", body: {}});
    try {
      const listing = await api(`${API}/sessions`);
      state.sessions = listing.sessions ?? [];
    } catch {
      state.sessions.push(created);
    }
    await loadSession(created.session_id);
  } catch (error) {
    setStatus(ui["workspace-status"], describeError(error));
  } finally {
    ui["new-session"].disabled = false;
  }
}

async function bootstrap() {
  try {
    const listing = await api(`${API}/sessions`);
    state.sessions = listing.sessions ?? [];
    const saved = loadSavedSession();
    const initial = state.sessions.find((item) => item.session_id === saved) ?? state.sessions[0];
    if (initial) await loadSession(initial.session_id);
    else await createSession();
  } catch (error) {
    if (error.status === 404) await createSession();
    else setStatus(ui["workspace-status"], describeError(error));
  }
}

function renderSessions() {
  const options = state.sessions.map((item) => {
    const option = node("option", "", item.title || "课程新会话");
    option.value = item.session_id;
    return option;
  });
  ui.sessions.replaceChildren(...options);
  if (state.session) ui.sessions.value = state.session.session_id;
  ui.sessions.disabled = state.loading || !options.length;
}

function row(label, kind, click, {selected, open, version} = {}) {
  const element = button("", click, "catalog-row");
  const indicator = node("span", "catalog-kind", open === undefined ? "·" : (open ? "▾" : "▸"));
  indicator.setAttribute("aria-hidden", "true");
  element.append(indicator, node("span", "catalog-label", label));
  element.title = kindText[kind] ?? kind;
  if (version != null) element.append(node("span", "catalog-version", `v${version}`));
  if (open !== undefined) element.setAttribute("aria-expanded", String(open));
  if (selected) element.setAttribute("aria-current", "location");
  return element;
}

function renderCatalog() {
  ui.catalog.setAttribute("aria-busy", String(state.loading));
  if (!state.courses.length) {
    const note = state.loading ? "正在加载资料库…" : "当前没有已授权课程。";
    ui.catalog.replaceChildren(node("p", "library-status", note));
    return;
  }
  const list = node("ul", "catalog-list");
  const focus = state.session?.focus?.node_id;
  const scope = state.session?.scope;
  if (state.courses.length > 1) {
    const all = node("li");
    all.append(row("全部课程", "course", () => void withNavigation(async () => {
      const ids = state.courses.map((course) => course.course_id);
      for (const id of ids) state.expanded.add(id);
      await updateScope(ids, []);
    }), {selected: scope?.course_ids.length === state.courses.length && !scope.document_ids.length}));
    list.append(all);
  }
  function renderNode(entry, items = entry.children ?? []) {
    const li = node("li");
    const isOpen = state.expanded.has(entry.node_id);
    const docSelected = entry.kind === "document" && scope?.document_ids.includes(entry.document_id);
    const selected = focus === entry.node_id || (!focus && docSelected);
    const nextOpen = items.length ? (selected ? !isOpen : true) : false;
    const itemButton = row(entry.label, entry.kind, () => {
      if (items.length) {
        if (nextOpen) state.expanded.add(entry.node_id);
        else state.expanded.delete(entry.node_id);
      }
      void chooseNode(entry, !selected);
    }, {selected, open: items.length ? isOpen : undefined, version: entry.kind === "document" ? entry.version : null});
    itemButton.dataset.nodeId = entry.node_id;
    li.append(itemButton);
    if (isOpen && items.length) {
      const sublist = node("ul", "catalog-list");
      for (const item of items) sublist.append(renderNode(state.nodes.get(item.node_id)));
      li.append(sublist);
    }
    return li;
  }
  for (const course of state.courses) {
    const li = node("li");
    const isOpen = state.expanded.has(course.course_id);
    const selected = scope?.course_ids.length === 1 && scope.course_ids[0] === course.course_id;
    li.append(row(course.label || course.course_id, "course", () => {
      if (isOpen && selected) state.expanded.delete(course.course_id);
      else state.expanded.add(course.course_id);
      if (!selected || scope?.document_ids.length) void chooseCourse(course.course_id);
      else renderCatalog();
    }, {selected: selected && !scope.document_ids.length, open: isOpen}));
    if (isOpen) {
      const sublist = node("ul", "catalog-list");
      for (const documentEntry of course.documents ?? []) {
        sublist.append(renderNode(state.documents.get(documentEntry.document_id), documentEntry.nodes ?? []));
      }
      if (!sublist.children.length) sublist.append(node("li", "library-status", "暂无资料。从导入开始。"));
      li.append(sublist);
    }
    list.append(li);
  }
  ui.catalog.replaceChildren(list);
}

async function updateScope(courseIds, documentIds) {
  const capture = captureSession();
  ++state.snapshotSequence;
  const payload = await api(`${currentPath()}/scope`, {
    method: "PATCH",
    body: {course_ids: courseIds, document_ids: documentIds, scope_revision: state.session.scope_revision},
  });
  if (isCurrent(capture)) {
    anchor();
    setSession(payload);
  }
}

async function withNavigation(action) {
  if (!state.session || state.navigating) return;
  state.navigating = true;
  setStatus(ui["workspace-status"], "");
  renderControls();
  try { await action(); }
  catch (error) {
    setStatus(ui["workspace-status"], describeError(error));
    if (error.code === "scope_changed" && state.session) {
      try { await reloadSession(captureSession()); } catch { /* Keep the visible error. */ }
    }
  } finally {
    state.navigating = false;
    renderControls();
  }
}

async function chooseCourse(id) {
  await withNavigation(async () => {
    await updateScope([id], []);
    render();
  });
}

async function chooseNode(entry, setFocus = true) {
  if (!setFocus) {
    renderCatalog();
    return;
  }
  await withNavigation(async () => {
    const capture = captureSession();
    const documentEntry = entry.doc ?? entry;
    const scope = state.session.scope;
    const courseId = documentEntry.course.course_id;
    if (scope.course_ids.length !== 1 || scope.course_ids[0] !== courseId ||
        scope.document_ids.length !== 1 || scope.document_ids[0] !== documentEntry.document_id) {
      await updateScope([courseId], [documentEntry.document_id]);
    }
    if (!isCurrent(capture)) return;
    ++state.snapshotSequence;
    const payload = await api(`${currentPath()}/focus`, {
      method: "PUT",
      body: {node_id: entry.node_id, scope_revision: state.session.scope_revision},
    });
    if (!isCurrent(capture)) return;
    revealFocus(entry.parent);
    anchor();
    setSession(payload);
  });
}

function renderLocation() {
  const session = state.session;
  const courseCount = session?.scope?.course_ids.length ?? 0;
  const documentIds = session?.scope?.document_ids ?? [];
  const labels = documentIds.map((id) => state.documents.get(id)?.file_name).filter(Boolean);
  const courseLabels = (session?.scope?.course_ids ?? []).map((id) => (
    state.courses.find((course) => course.course_id === id)?.label || id
  ));
  ui["scope-summary"].textContent = !session ? "选择或创建会话后开始。" :
    !courseCount ? "本会话暂无可检索范围。" :
    labels.length ? `当前范围 · ${labels.join("、")}` :
    `当前范围 · ${courseLabels.join("、")}的全部资料`;
  ui["clear-file"].hidden = !documentIds.length;
  const entry = state.nodes.get(session?.focus?.node_id);
  ui.location.hidden = !entry;
  if (!entry) return;
  const documentEntry = entry.doc ?? entry;
  ui["location-kind"].textContent = `${kindText[entry.kind] ?? entry.kind} · v${documentEntry.version}`;
  ui["location-title"].textContent = entry.label;
  ui["preview-location"].hidden = !entry.has_media && !entry.text;
  ui["parent-location"].disabled = !entry.parent;
  ui["previous-location"].disabled = !state.nodes.has(entry.previous_id);
  ui["next-location"].disabled = !state.nodes.has(entry.next_id);
}

function sourceLocation(block, evidence) {
  const key = block.display_source ?? evidence?.revision?.source;
  const locator = evidence?.locator ?? block.locator;
  const where = locator?.page != null ? `第 ${locator.page} 页` :
    Number.isInteger(locator?.cell_index) ? `Cell ${locator.cell_index + 1}` : key?.source_id;
  return [evidence?.file_name, key?.version != null ? `v${key.version}` : null, where]
    .filter(Boolean).join(" · ");
}

function renderSourceCard(block, message, position) {
  const evidence = message.sources?.[block.evidence_id];
  const title = block.caption || sourceLocation(block, evidence) || "原件图像";
  const details = sourceLocation(block, evidence);
  const figure = node("figure", "source-card");
  figure.dataset.displayId = block.display_id;
  figure.dataset.imageOrdinal = String(position);
  const picture = node("div", "source-card-picture");
  const notice = node("span", "", "正在加载图片…");
  notice.setAttribute("role", "status");
  picture.append(notice);
  const caption = node("figcaption");
  caption.append(node("p", "source-caption", `图 ${position} · ${title}`));
  const status = readText[block.read_status ?? evidence?.read_status] ?? "未读取";
  caption.append(node("p", "source-details", [details, status, block.fallback_reason].filter(Boolean).join(" · ")));
  const actions = node("div", "source-actions");
  const sessionId = state.session.session_id;
  const inline = mediaURL(block.media_url, sessionId);
  const isNotebookOutput = block.rendition_kind === "notebook_output";
  const ask = button("问这张图", () => useAnchor({
    type: "display", message_id: message.message_id, display_id: block.display_id,
  }, `引用图 ${position} · ${title}`));
  const controls = [ask, button(isNotebookOutput ? "查看原件" : "查看图片",
    () => openReader({url: inline, title, details}))];
  if (!isNotebookOutput) {
    const page = mediaURL(block.media_url, sessionId, "page");
    controls.push(button("回原件", () => openReader({
      url: page, title, details, rect: block.rect_visible_norm,
    })));
  }
  for (const control of controls) control.disabled = true;
  actions.append(...controls);
  caption.append(actions);
  if (inline) {
    const image = node("img");
    image.alt = title;
    image.hidden = true;
    image.addEventListener("load", () => {
      notice.hidden = true;
      image.hidden = false;
      for (const control of controls) control.disabled = false;
    });
    image.addEventListener("error", () => {
      picture.dataset.state = "failed";
      notice.textContent = "图片加载失败，原件可能已不可用。";
      notice.hidden = false;
      image.hidden = true;
      for (const control of controls) control.disabled = true;
    });
    picture.append(image);
    image.src = inline;
  } else {
    picture.dataset.state = "failed";
    notice.textContent = "原件不可访问。";
  }
  figure.append(picture, caption);
  return figure;
}

function renderMessages() {
  const messages = state.session?.messages ?? [];
  const key = JSON.stringify([state.session?.session_id, messages]);
  if (key === state.threadKey) return;
  state.threadKey = key;
  if (!messages.length) {
    ui.messages.replaceChildren(node("p", "empty-thread", state.session ?
      "从资料库选择位置，搜索原文或开始提问。" : "在这里继续你的课程复习。"));
    return;
  }
  ui.messages.replaceChildren(...messages.map((message) => {
    const isUser = message.role === "user";
    const article = node("article", isUser ? "message message-user" : "message");
    article.dataset.messageId = message.message_id;
    article.append(node("h3", "", isUser ? "你" : "课程助手"));
    const content = node("div", "message-content");
    const order = message.visible_display_ids ?? [];
    for (const block of message.blocks ?? []) {
      if (block.type === "source_image") {
        const position = order.indexOf(block.display_id);
        if (position >= 0) content.append(renderSourceCard(block, message, position + 1));
        else content.append(node("p", "notice-block", "原件当前不可展示。"));
      } else {
        const text = block.text ?? block.message ?? (block.type === "notice" ? "本条内容不可用。" : "");
        content.append(node("p", block.type === "text" ? "text-block" : "notice-block", text));
        const references = (block.citations ?? []).map((id) => {
          const evidence = message.sources?.[typeof id === "string" ? id : id.evidence_id];
          return evidence ? sourceLocation({}, evidence) : "";
        }).filter(Boolean);
        if (references.length) content.append(node("p", "citations", `依据 · ${references.join(" / ")}`));
      }
    }
    article.append(content);
    return article;
  }));
}

function isRunning() {
  const session = state.session;
  return session && (state.outbox.get(session.session_id)?.inFlight ||
    (session.turns ?? []).some((turn) => turn.status === "running"));
}

function renderTurns() {
  ui["turn-status"].replaceChildren();
  const turns = state.session?.turns ?? [];
  const local = state.session && state.outbox.get(state.session.session_id);
  const latest = turns.at(-1);
  let turn = turns.findLast((item) => item.status === "running");
  if (!turn && local?.inFlight) turn = {...local.payload, status: "running"};
  turn ??= latest;
  const action = turn?.status === "running" ? "cancel" :
    turn?.status === "failed" && turn.retryable ? "retry" : null;
  const actionKey = action ? JSON.stringify([state.session.session_id, turn.client_turn_id, action]) : "";
  if (actionKey !== state.actionKey) {
    state.actionKey = actionKey;
    ui["turn-actions"].replaceChildren();
    if (action) {
      const label = action === "cancel" ? "取消本轮" : "重试本轮";
      ui["turn-actions"].append(button(label, () => void turnAction(turn, action)));
    }
  }
  if (!turn || turn.status === "completed") return;
  const card = node("section", "turn-state");
  card.append(node("p", "", describeTurn(turn)));
  if (turn.status === "clarification") {
    const options = turn.options ?? state.turnOptions.get(turn.turn_id) ?? [];
    const choices = node("div", "clarification-actions");
    for (const option of options) {
      if (!option.option_id) continue;
      choices.append(button(option.label, () => useAnchor({
        type: "clarification", turn_id: turn.turn_id, option_id: option.option_id,
      }, option.label, turn.text)));
    }
    if (choices.children.length) card.append(choices);
  }
  ui["turn-status"].append(card);
}

function describeTurn(turn) {
  const code = typeof turn.error === "string" ? turn.error : turn.error?.code;
  return [statusText[turn.status], errors[code]].filter(Boolean).join(" · ");
}

function renderControls() {
  const session = state.session;
  ui["clear-file"].disabled = state.navigating;
  ui.question.disabled = !session;
  const local = session && state.outbox.get(session.session_id);
  const canSummary = session?.scope?.document_ids.length === 1;
  const localRetry = local?.indeterminate;
  const canSend = mode() === "summary" ? canSummary : Boolean(ui.question.value.trim());
  ui.send.disabled = !session || state.navigating || isRunning() || (!canSend && !localRetry);
  ui.send.textContent = localRetry ? "重发本轮" : "发送";
  setStatus(ui["composer-status"], localRetry ? "提交结果尚未确认，可使用相同请求重发。" :
    isRunning() ? "正在处理，可取消本轮。" :
    mode() === "summary" && !canSummary ? "请选择一份资料后总结。" : "");
  const imports = state.courses.map((course) => {
    const option = node("option", "", course.label || course.course_id);
    option.value = course.course_id;
    return option;
  });
  const currentCourse = ui["import-course"].value;
  ui["import-course"].replaceChildren(...imports);
  const defaultCourse = session?.scope?.course_ids[0];
  const proposed = [currentCourse, defaultCourse].find((id) => state.courses.some((course) => course.course_id === id));
  if (proposed) ui["import-course"].value = proposed;
  ui["import-course"].disabled = state.uploading || !imports.length;
  ui["import-files"].disabled = state.uploading || !imports.length;
  ui["import-button"].disabled = state.uploading || !imports.length || !session;
  const selectedDocument = state.documents.get(session?.scope?.document_ids[0]);
  ui["replace-label"].hidden = !canSummary || selectedDocument?.course.course_id !== ui["import-course"].value;
  if (ui["replace-label"].hidden) ui["replace-document"].checked = false;
  ui["replace-document"].disabled = state.uploading;
}

function render() {
  renderSessions();
  renderCatalog();
  renderLocation();
  renderMessages();
  renderTurns();
  renderControls();
  schedulePolling();
}

function schedulePolling() {
  clearTimeout(state.pollTimer);
  if (!isRunning()) return;
  const capture = captureSession();
  state.pollTimer = setTimeout(async () => {
    if (!isCurrent(capture)) return;
    try { await reloadSession(capture); }
    catch { if (isCurrent(capture)) schedulePolling(); }
  }, 900);
}

async function submitTurn(event) {
  event.preventDefault();
  if (ui.send.disabled) return;
  const captured = captureSession();
  let incoming = state.outbox.get(captured.id);
  if (!incoming?.indeterminate) {
    const kind = mode();
    const text = ui.question.value.trim() || (kind === "summary" ? "总结当前资料" : "");
    if (!text) return;
    const payload = {client_turn_id: crypto.randomUUID(), kind, text, scope_revision: state.session.scope_revision};
    if (state.anchor) payload.anchor = {...state.anchor};
    incoming = {payload, inFlight: false, indeterminate: false};
    state.outbox.set(captured.id, incoming);
    ui.question.value = "";
    anchor();
  }
  incoming.inFlight = true;
  incoming.indeterminate = false;
  setStatus(ui["workspace-status"], "");
  render();
  try {
    const response = await api(`${sessionPath(captured.id)}/turns`, {
      method: "POST", body: incoming.payload,
    });
    if (Array.isArray(response.options) && response.turn) {
      state.turnOptions.set(response.turn.turn_id, response.options);
    }
    if (state.outbox.get(captured.id) === incoming) state.outbox.delete(captured.id);
    if (isCurrent(captured)) await reloadSession(captured);
  } catch (error) {
    if (state.outbox.get(captured.id) === incoming) {
      if (!error.status) {
        incoming.indeterminate = true;
      } else {
        state.outbox.delete(captured.id);
        if (isCurrent(captured) && !ui.question.value.trim()) ui.question.value = incoming.payload.text;
      }
      if (isCurrent(captured)) setStatus(ui["workspace-status"], describeError(error));
    }
  } finally {
    incoming.inFlight = false;
    if (isCurrent(captured)) render();
  }
}

async function turnAction(turn, action) {
  const capture = captureSession();
  const path = `${sessionPath(capture.id)}/turns/${encodeURIComponent(turn.client_turn_id)}`;
  const retry = action === "retry" ? {payload: turn, inFlight: true, indeterminate: false} : null;
  try {
    if (retry) {
      state.outbox.set(capture.id, retry);
      render();
    }
    const result = await api(`${path}/${action}`, {method: "POST"});
    if (Array.isArray(result.options) && result.turn) {
      state.turnOptions.set(result.turn.turn_id, result.options);
    }
    if (state.outbox.get(capture.id) === retry) state.outbox.delete(capture.id);
    if (isCurrent(capture)) await reloadSession(capture);
  } catch (error) {
    if (state.outbox.get(capture.id) === retry) state.outbox.delete(capture.id);
    if (isCurrent(capture)) setStatus(ui["workspace-status"], describeError(error));
  } finally {
    if (isCurrent(capture)) render();
  }
}

async function importFiles(event) {
  event.preventDefault();
  if (!state.session || state.uploading) return;
  const capture = captureSession();
  const files = Array.from(ui["import-files"].files);
  if (!files.length) return;
  const courseId = ui["import-course"].value;
  const replaceId = ui["replace-document"].checked ? state.session.scope.document_ids[0] : null;
  if (replaceId && files.length !== 1) {
    setStatus(ui["workspace-status"], "更新版本时请选择一个文件。");
    return;
  }
  state.uploading = true;
  ui["import-results"].replaceChildren();
  renderControls();
  try {
    for (const file of files) {
      const entry = node("li", "", `${file.name} · 导入中`);
      ui["import-results"].append(entry);
      const query = new URLSearchParams({course_id: courseId, filename: file.name});
      if (replaceId) query.set("document_id", replaceId);
      try {
        const payload = await api(`${sessionPath(capture.id)}/import?${query}`, {
          method: "POST", body: file, binary: true,
        });
        if (!isCurrent(capture)) break;
        const status = {
          published: "导入完成", indexed: "导入完成", reused: "复用已导入原件", failed: "导入失败",
        }[payload.status] ?? "导入完成";
        const version = payload.version == null ? "" : ` · v${payload.version}`;
        entry.textContent = `${file.name} · ${status}${version}`;
        if (payload.status === "failed") entry.dataset.error = "true";
      } catch (error) {
        entry.dataset.error = "true";
        entry.textContent = `${file.name} · ${describeError(error)}`;
      }
    }
    if (isCurrent(capture)) {
      state.expanded.add(courseId);
      await reloadCatalog(capture);
      await reloadSession(capture);
      ui["import-files"].value = "";
      ui["replace-document"].checked = false;
    }
  } catch (error) {
    if (isCurrent(capture)) setStatus(ui["workspace-status"], describeError(error));
  } finally {
    state.uploading = false;
    renderControls();
  }
}

function validRect(rect) {
  return Array.isArray(rect) && rect.length === 4 &&
    rect.every((value) => Number.isFinite(value) && value >= 0 && value <= 1) &&
    rect[0] < rect[2] && rect[1] < rect[3];
}

function openReader({url, title, details, rect, fallback}) {
  ui["reader-title"].textContent = title;
  ui["reader-location"].textContent = details ?? "";
  ui["reader-body"].replaceChildren();
  if (!url) {
    ui["reader-body"].append(node("p", "text-block", fallback || "没有可供预览的图片。"));
  } else {
    const controls = node("div", "reader-controls");
    const picture = node("div", "reader-picture");
    const indicator = node("p", "", "正在加载原件…");
    indicator.setAttribute("role", "status");
    const image = node("img");
    image.alt = title;
    picture.hidden = true;
    const zoom = button("放大", () => {
      zoom.textContent = picture.classList.toggle("is-zoomed") ? "适应窗口" : "放大";
    });
    zoom.disabled = true;
    controls.append(zoom);
    picture.append(image);
    if (validRect(rect)) {
      const [left, top, right, bottom] = rect;
      const highlight = node("span", "reader-highlight");
      highlight.setAttribute("aria-label", "引用区域");
      highlight.style.left = `${left * 100}%`;
      highlight.style.top = `${top * 100}%`;
      highlight.style.width = `${(right - left) * 100}%`;
      highlight.style.height = `${(bottom - top) * 100}%`;
      picture.append(highlight);
    }
    image.addEventListener("load", () => {
      indicator.hidden = true;
      picture.hidden = false;
      zoom.disabled = false;
    });
    image.addEventListener("error", () => {
      picture.hidden = true;
      indicator.hidden = false;
      indicator.textContent = fallback || "原件不可用或访问权限已失效。";
      zoom.disabled = true;
    });
    ui["reader-body"].append(controls, indicator, picture);
    image.src = url;
  }
  if (!ui.reader.open) ui.reader.showModal();
}

function previewFocus() {
  const entry = state.nodes.get(state.session?.focus?.node_id);
  if (!entry) return;
  const path = `${currentPath()}/catalog/${encodeURIComponent(entry.node_id)}/media`;
  openReader({
    url: entry.has_media ? mediaURL(path, state.session.session_id) : null,
    title: entry.label,
    details: `${kindText[entry.kind] ?? entry.kind} · v${entry.version}`,
    fallback: entry.text,
  });
}

ui["new-session"].addEventListener("click", () => void createSession());
ui.sessions.addEventListener("change", () => void loadSession(ui.sessions.value));
ui["import-course"].addEventListener("change", renderControls);
ui["import-form"].addEventListener("submit", importFiles);
ui["clear-file"].addEventListener("click", () => {
  void withNavigation(() => updateScope(state.session.scope.course_ids, []));
});
ui["preview-location"].addEventListener("click", previewFocus);
for (const [control, getId] of [
  ["parent-location", (item) => item.parent?.node_id],
  ["previous-location", (item) => item.previous_id],
  ["next-location", (item) => item.next_id],
]) {
  ui[control].addEventListener("click", () => {
    const current = state.nodes.get(state.session?.focus?.node_id);
    const target = current && state.nodes.get(getId(current));
    if (target) void chooseNode(target);
  });
}
ui["clear-anchor"].addEventListener("click", () => { anchor(); renderControls(); });
ui.composer.addEventListener("submit", submitTurn);
ui.composer.addEventListener("input", () => {
  if (mode() !== "ask") anchor();
  renderControls();
});
ui["close-reader"].addEventListener("click", () => ui.reader.close());
ui.reader.addEventListener("close", () => ui["reader-body"].replaceChildren());

void bootstrap();
