"use strict";

const $ = (id) => document.getElementById(id);
const state = { user: null, info: null, mode: "login", previewUrl: null, requestId: 0 };

// ---------- helpers ----------
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

async function api(path, { body, json } = {}) {
  const opts = { method: body === undefined && json === undefined ? "GET" : "POST", headers: {}, credentials: "same-origin" };
  if (opts.method === "POST") opts.headers["X-Requested-With"] = "fetch";
  if (json !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(json); }
  if (body !== undefined) opts.body = body;
  const res = await fetch(path, opts);
  let data = {};
  try { data = await res.json(); } catch { /* empty */ }
  if (!res.ok) { const e = new Error(data.error || "Something went wrong. Please try again."); e.status = res.status; throw e; }
  return data;
}

let toastTimer;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 3200);
}

const pct = (p) => (p * 100).toFixed(p >= 0.995 || p < 0.001 ? 0 : 1) + "%";
const className = (code) => state.info?.classes[code]?.name ?? code;
const swatch = (model) => { const s = el("span", { class: "swatch" }); s.style.background = `var(--m-${model})`; return s; };

// ---------- tabs ----------
function showTab(name) {
  for (const a of document.querySelectorAll("nav.tabs a")) a.classList.toggle("active", a.dataset.tab === name);
  for (const id of ["try", "compare", "method", "privacy"]) $(id).hidden = id !== name;
}
for (const a of document.querySelectorAll("nav.tabs a")) {
  a.addEventListener("click", (e) => { e.preventDefault(); showTab(a.dataset.tab); history.replaceState(null, "", "#" + a.dataset.tab); });
}
if (["#compare", "#method", "#privacy"].includes(location.hash)) showTab(location.hash.slice(1));

// ---------- account ----------
function setUser(user) {
  state.user = user;
  $("auth").hidden = !!user;
  $("app").hidden = !user;
  $("who").hidden = !user;
  $("signout").hidden = !user;
  $("signin-top").hidden = !!user;
  $("delete-section").hidden = !user;
  if (user) {
    $("username").textContent = user;
    $("avatar").textContent = user[0];
  }
  clearPhoto();
}

function setMode(mode) {
  state.mode = mode;
  const signup = mode === "signup";
  $("seg-login").setAttribute("aria-selected", String(!signup));
  $("seg-signup").setAttribute("aria-selected", String(signup));
  $("confirm-field").hidden = !signup;
  $("user-hint").hidden = !signup;
  $("pass-hint").hidden = !signup;
  $("f-pass").autocomplete = signup ? "new-password" : "current-password";
  $("auth-submit").textContent = signup ? "Create account" : "Sign in";
  $("auth-error").hidden = true;
}
$("seg-login").onclick = () => setMode("login");
$("seg-signup").onclick = () => setMode("signup");
$("signin-top").onclick = () => { showTab("try"); setMode("login"); $("f-user").focus(); };

$("toggle-pw").onclick = () => {
  const show = $("f-pass").type === "password";
  for (const id of ["f-pass", "f-pass2"]) $(id).type = show ? "text" : "password";
  $("toggle-pw").textContent = show ? "Hide" : "Show";
  $("toggle-pw").setAttribute("aria-label", show ? "Hide password" : "Show password");
};

function authError(msg) { $("auth-error").textContent = msg; $("auth-error").hidden = false; }

$("auth-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = $("f-user").value.trim(), password = $("f-pass").value;
  if (state.mode === "signup") {
    if (!/^[A-Za-z0-9_.-]{3,32}$/.test(username)) return authError("Username must be 3 to 32 characters: letters, numbers, dot, dash or underscore.");
    if (password.length < 8) return authError("Password must be at least 8 characters.");
    if (password !== $("f-pass2").value) return authError("The two passwords don't match.");
  } else if (!username || !password) {
    return authError("Enter your username and password.");
  }
  const btn = $("auth-submit");
  btn.disabled = true;
  try {
    const data = await api(state.mode === "signup" ? "/api/signup" : "/api/login", { json: { username, password } });
    $("auth-form").reset();
    setUser(data.user);
    toast(state.mode === "signup" ? `Account created. Welcome, ${data.user}.` : `Signed in as ${data.user}.`);
  } catch (err) {
    authError(err.message);
  } finally {
    btn.disabled = false;
  }
});

$("signout").onclick = async () => {
  try { await api("/api/logout", { json: {} }); } catch { /* signing out locally anyway */ }
  setUser(null); setMode("login"); showTab("try");
  toast("Signed out.");
};

// Delete account
$("delete-open").onclick = () => { $("delete-error").hidden = true; $("delete-form").reset(); $("delete-dialog").showModal(); };
$("delete-cancel").onclick = () => $("delete-dialog").close();
$("delete-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("delete-confirm").disabled = true;
  try {
    await api("/api/delete-account", { json: { password: $("d-pass").value } });
    $("delete-dialog").close();
    setUser(null); setMode("signup"); showTab("try");
    toast("Your account has been deleted.");
  } catch (err) {
    $("delete-error").textContent = err.message; $("delete-error").hidden = false;
  } finally {
    $("delete-confirm").disabled = false;
  }
});

// ---------- photo upload ----------
const zone = $("zone"), fileInput = $("file");
$("choose").onclick = () => fileInput.click();
zone.onclick = () => fileInput.click();
zone.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } };
fileInput.onchange = () => { if (fileInput.files[0]) analyse(fileInput.files[0], null); fileInput.value = ""; };
zone.ondragover = (e) => { e.preventDefault(); zone.classList.add("over"); };
zone.ondragleave = () => zone.classList.remove("over");
zone.ondrop = (e) => { e.preventDefault(); zone.classList.remove("over"); if (e.dataTransfer.files[0]) analyse(e.dataTransfer.files[0], null); };
document.addEventListener("paste", (e) => {
  if (!state.user || $("app").hidden) return;
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) analyse(item.getAsFile(), null);
});
$("clear").onclick = () => { clearPhoto(); toast("Photo removed."); };

// Placeholder cards with the same shape as real results: "waiting" before a photo, "loading" while the models run
function placeholders(loading) {
  if (!state.info) return;
  $("models").replaceChildren(...state.info.models.map((m) => el("div", { class: "mcard placeholder" + (loading ? " loading" : "") },
    el("div", { class: "mhead" }, el("span", { class: "mname" }, swatch(m.id), m.name)),
    el("p", { class: "top", text: loading ? "Analysing" : "Waiting for a photo" }),
    el("div", { class: "bars" }, ...[72, 38, 24, 14].map((w) => {
      const s = el("span", { class: "skel" });
      s.style.width = w + "%";
      return s;
    })))));
}

function clearPhoto() {
  state.requestId++;  // ignore any result still on its way
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  $("preview").removeAttribute("src"); $("preview").hidden = true;
  $("hint").hidden = false; $("busy").hidden = true; $("clear").hidden = true;
  $("truth").textContent = ""; $("error").hidden = true;
  $("verdict").hidden = true;
  placeholders(false);
}

async function analyse(blob, truth) {
  if (!blob.type.startsWith("image/")) { showError("Please choose an image file (JPG or PNG)."); return; }
  if (blob.size > 20 * 1024 * 1024) { showError("That image is over 20 MB. Please choose a smaller one."); return; }
  const id = ++state.requestId;
  $("error").hidden = true;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(blob);
  $("preview").src = state.previewUrl; $("preview").hidden = false; $("hint").hidden = true;
  $("busy").hidden = false; $("clear").hidden = false;
  $("verdict").hidden = true;
  placeholders(true);
  $("truth").textContent = "";
  if (truth) $("truth").append("True diagnosis: ", el("b", { text: className(truth) }));
  try {
    const data = await api("/api/predict", { body: blob });
    if (id !== state.requestId) return;
    render(data.results, truth);
  } catch (err) {
    if (id !== state.requestId) return;
    if (err.status === 401) { setUser(null); toast("Your session ended. Please sign in again."); return; }
    placeholders(false);
    showError(err.message);
  } finally {
    if (id === state.requestId) $("busy").hidden = true;
  }
}

function showError(msg) { $("error").textContent = msg; $("error").hidden = false; }

function render(results, truth) {
  // Majority vote; ties go to the class with the higher total probability across models
  const votes = {}, mass = {};
  for (const r of results) {
    votes[r.top] = (votes[r.top] || 0) + 1;
    for (const p of r.probs) mass[p.code] = (mass[p.code] || 0) + p.p;
  }
  const [winner, n] = Object.entries(votes).sort((a, b) => b[1] - a[1] || mass[b[0]] - mass[a[0]])[0];
  const info = state.info.classes[winner];

  let sub = n === results.length ? "All four models agree." : `${n} of ${results.length} models agree.`;
  if (truth) sub += truth === winner ? " This matches the true diagnosis." : ` The true diagnosis is ${className(truth)}.`;
  const v = $("verdict");
  v.replaceChildren(
    el("div", { class: "label", text: "Most models say" }),
    el("div", { class: "big" }, info.name, el("span", { class: "tag " + (info.serious ? "warn" : "ok"), text: info.serious ? "Can be serious" : "Usually harmless" })),
    el("div", { class: "sub", text: sub }),
    el("p", { class: "about", text: info.about }),
  );
  v.hidden = false;

  $("models").replaceChildren(...results.map((r, i) => {
    const bars = r.probs.map((p) => {
      const fill = el("span", { class: "fill" });
      fill.style.width = (p.p * 100).toFixed(1) + "%";
      if (p.code === r.top) fill.style.background = `var(--m-${r.model})`;
      return el("div", { class: "bar" },
        el("span", { class: "lab", title: className(p.code), text: className(p.code) }),
        el("span", {}, fill),
        el("span", { class: "val", text: pct(p.p) }));
    });
    const card = el("div", { class: "mcard enter" },
      el("div", { class: "mhead" }, el("span", { class: "mname" }, swatch(r.model), r.name), el("span", { class: "ms", text: `${r.ms} ms` })),
      el("p", { class: "top" }, className(r.top), el("span", { text: pct(r.probs[0].p) })),
      el("div", { class: "bars" }, ...bars));
    card.style.setProperty("--i", i);
    return card;
  }));
}

// ---------- comparison table ----------
const COLS = [
  ["test_macro_f1", "Macro F1", "max"], ["test_balanced_acc", "Balanced acc.", "max"], ["test_accuracy", "Accuracy", "max"],
  ["test_melanoma_recall", "Melanoma recall", "max"], ["ext_macro_f1", "External macro F1", "max"],
  ["file_mb", "File size (MB)", "min"], ["cpu_latency_ms", "CPU ms / photo", "min"], ["params_millions", "Params (M)", "min"],
];

function renderComparison(info) {
  const val = (m, k) => (Array.isArray(m[k]) ? m[k][0] : m[k]);
  const best = {};
  for (const [k, , dir] of COLS) {
    const vals = info.models.map((m) => val(m, k));
    best[k] = dir === "max" ? Math.max(...vals) : Math.min(...vals);
  }
  const head = el("tr", {}, el("th", { text: "Model" }), ...COLS.map(([, label]) => el("th", { text: label })));
  const rows = info.models.map((m) => el("tr", {}, el("td", {}, el("span", { class: "mname" }, swatch(m.id), m.name)), ...COLS.map(([k]) => {
    const v = m[k], td = el("td", { class: val(m, k) === best[k] ? "best" : "" });
    if (Array.isArray(v)) {
      td.append(v[0].toFixed(3));
      if (info.seeds > 1) td.append(el("span", { class: "sd", text: "±" + v[1].toFixed(3) }));
    } else td.append(String(v));
    return td;
  })));
  $("table").replaceChildren(el("thead", {}, head), el("tbody", {}, ...rows));
  if (info.seeds > 1) {
    $("legend-note").prepend(`Mean of ${info.seeds} training runs per model; the small grey number is the standard deviation. `);
    $("seeds-text").textContent = `Each model was trained ${info.seeds} times with different random seeds. The tables show the mean and spread, so one lucky run can't decide the winner.`;
  }

  const f1 = (m) => `${m.test_macro_f1[0].toFixed(3)}${info.seeds > 1 ? " ± " + m.test_macro_f1[1].toFixed(3) : ""}`;
  const byF1 = [...info.models].sort((a, b) => b.test_macro_f1[0] - a.test_macro_f1[0]);
  const [first, second] = byF1;
  // A gap smaller than the run-to-run spread is not a real difference
  const tied = info.seeds > 1 && first.test_macro_f1[0] - second.test_macro_f1[0] < Math.max(first.test_macro_f1[1], second.test_macro_f1[1]);
  // Fastest model that is still reasonably accurate (within 0.06 macro F1 of the best)
  const quick = [...info.models].filter((m) => first.test_macro_f1[0] - m.test_macro_f1[0] <= 0.06)
    .sort((a, b) => a.cpu_latency_ms - b.cpu_latency_ms)[0];
  const mel = [...info.models].sort((a, b) => b.test_melanoma_recall[0] - a.test_melanoma_recall[0])[0];
  const points = [
    tied ? `${first.name} (macro F1 ${f1(first)}) and ${second.name} (${f1(second)}) are effectively tied. The gap is smaller than the variation between training runs.`
         : `${first.name} leads with macro F1 ${f1(first)}, ahead of ${second.name} (${f1(second)}).`,
    `${mel.name} catches the most melanomas (recall ${mel.test_melanoma_recall[0].toFixed(3)}).`,
  ];
  if (quick && quick.id !== first.id) {
    points.push(`For a slow laptop or a phone, ${quick.name} is ${(first.cpu_latency_ms / quick.cpu_latency_ms).toFixed(1)}× faster than ${first.name} on a CPU ` +
      `and ${(first.file_mb / quick.file_mb).toFixed(1)}× smaller, for ${(first.test_macro_f1[0] - quick.test_macro_f1[0]).toFixed(3)} less macro F1.`);
  }
  $("winner").replaceChildren(el("h3", { text: tied ? `${first.name} and ${second.name} come out on top` : `${first.name} comes out on top` }),
    el("ul", {}, ...points.map((t) => el("li", { text: t }))));
  $("winner").hidden = false;

  const leak = info.leakage_demo;
  if (leak) {
    const bi = leak.split_by_image, bl = leak.split_by_lesion;
    $("leak-text").textContent = `We measured what goes wrong with MobileNetV3: split by photo, the test score looks better, but on the outside test set the model is no better. The "improvement" is memorisation.`;
    const stat = (big, small) => el("div", { class: "stat" }, el("b", { text: big }), el("span", { text: small }));
    $("leak-stats").replaceChildren(
      stat(`${Math.round(leak.test_images_with_lesion_seen_in_training * 100)}%`, "of test photos showed a lesion already seen in training"),
      stat(`+${(bi.own_test_set.macro_f1 - bl.own_test_set.macro_f1).toFixed(3)}`, `inflated test macro F1 (${bi.own_test_set.macro_f1.toFixed(3)} vs ${bl.own_test_set.macro_f1.toFixed(3)})`),
      stat(`${(bi.external_test_set.macro_f1 - bl.external_test_set.macro_f1).toFixed(3)}`, `change on the outside test set (${bi.external_test_set.macro_f1.toFixed(3)} vs ${bl.external_test_set.macro_f1.toFixed(3)})`));
  }
}

// ---------- examples ----------
function renderExamples(info) {
  // ?v= changes whenever export_models.py picks a different photo, so browsers never show a stale cached sample
  const src = (code) => `examples/${code}.jpg?v=${info.examples?.[code] ?? ""}`;
  for (const img of document.querySelectorAll(".mosaic img[data-code]")) img.src = src(img.dataset.code);
  const box = $("examples");
  for (const code of Object.keys(info.classes)) {
    const b = el("button", { class: "ex", type: "button", title: info.classes[code].name, "aria-label": `Try a sample ${info.classes[code].name} photo` },
      el("img", { src: src(code), alt: "" }));
    b.onclick = async () => analyse(await (await fetch(src(code))).blob(), code);
    box.append(b);
  }
}

// ---------- start ----------
(async () => {
  try {
    const [info, me] = await Promise.all([api("/api/info"), api("/api/me")]);
    state.info = info;
    renderComparison(info);
    renderExamples(info);
    setUser(me.user);
  } catch {
    $("auth").hidden = false;
    authError("Can't reach the app server. Make sure run.bat (or python app/server.py) is still running.");
  }
})();
