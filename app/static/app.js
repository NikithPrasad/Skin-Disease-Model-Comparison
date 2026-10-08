"use strict";

const $ = (id) => document.getElementById(id);
const state = { user: null, info: null, mode: "signup", previewUrl: null, requestId: 0 };
const MIN_ANALYSIS_MS = 1100; // keep the "looking at your image" steps on screen long enough to read
const CONSIDER = "Consider discussing this result with a qualified healthcare professional";
const SIMILAR = "Because some skin conditions can look similar, a qualified healthcare professional can look at it properly and tell you for sure.";

// ---------- helpers ----------
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null && c !== false && c !== "") node.append(c);
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
  toastTimer = setTimeout(() => { t.hidden = true; }, 3500);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const pct = (p) => (p * 100).toFixed(p >= 0.995 || p < 0.001 ? 0 : 1) + "%";
const whole = (p) => (p > 0 && p < 0.1 ? (p * 100).toFixed(1) : Math.round(p * 100)) + "%";  // small rates keep a decimal
const className = (code) => state.info?.classes[code]?.name ?? code;
const lower = (code) => className(code).toLowerCase();

// ---------- progress stepper ----------
function setStep(n) {
  for (const li of document.querySelectorAll("#stepper li")) {
    const s = Number(li.dataset.step);
    li.classList.toggle("done", s < n);
    li.classList.toggle("current", s === n);
    if (s === n) li.setAttribute("aria-current", "step"); else li.removeAttribute("aria-current");
  }
}

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
  document.querySelector('label[for="f-user"]').textContent = signup ? "Choose a username" : "Username";
  $("f-pass").autocomplete = signup ? "new-password" : "current-password";
  $("auth-submit").textContent = signup ? "Create account" : "Sign in";
  $("auth-error").hidden = true;
}
$("seg-login").onclick = () => setMode("login");
$("seg-signup").onclick = () => setMode("signup");
$("signin-top").onclick = () => { setMode("login"); $("check").scrollIntoView({ behavior: "smooth" }); $("f-user").focus({ preventScroll: true }); };
$("hero-cta").addEventListener("click", () => setTimeout(() => (state.user ? $("choose") : $("f-user")).focus({ preventScroll: true }), 400));

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
    if (!/^[A-Za-z0-9_.-]{3,32}$/.test(username)) return authError("Please choose a username of 3 to 32 letters or numbers (a dot, dash or underscore is fine too).");
    if (password.length < 8) return authError("Please choose a password of at least 8 characters.");
    if (password !== $("f-pass2").value) return authError("The two passwords don't match. Please type them again.");
  } else if (!username || !password) {
    return authError("Please enter your username and password.");
  }
  const btn = $("auth-submit");
  btn.disabled = true;
  try {
    const data = await api(state.mode === "signup" ? "/api/signup" : "/api/login", { json: { username, password } });
    $("auth-form").reset();
    setUser(data.user);
    toast(state.mode === "signup" ? `Welcome, ${data.user}. You're ready to check an image.` : `Welcome back, ${data.user}.`);
    $("choose").focus();
  } catch (err) {
    authError(err.message);
  } finally {
    btn.disabled = false;
  }
});

$("signout").onclick = async () => {
  try { await api("/api/logout", { json: {} }); } catch { /* signing out locally anyway */ }
  setUser(null); setMode("login");
  toast("You're signed out.");
};

$("delete-open").onclick = () => { $("delete-error").hidden = true; $("delete-form").reset(); $("delete-dialog").showModal(); };
$("delete-cancel").onclick = () => $("delete-dialog").close();
$("delete-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("delete-confirm").disabled = true;
  try {
    await api("/api/delete-account", { json: { password: $("d-pass").value } });
    $("delete-dialog").close();
    setUser(null); setMode("signup");
    toast("Your account has been deleted.");
  } catch (err) {
    $("delete-error").textContent = err.message; $("delete-error").hidden = false;
  } finally {
    $("delete-confirm").disabled = false;
  }
});

// ---------- upload ----------
const zone = $("zone"), fileInput = $("file");
$("choose").onclick = () => fileInput.click();
zone.onclick = () => fileInput.click();
zone.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } };
fileInput.onchange = () => { if (fileInput.files[0]) check(fileInput.files[0], null); fileInput.value = ""; };
zone.ondragover = (e) => { e.preventDefault(); zone.classList.add("over"); };
zone.ondragleave = () => zone.classList.remove("over");
zone.ondrop = (e) => { e.preventDefault(); zone.classList.remove("over"); if (e.dataTransfer.files[0]) check(e.dataTransfer.files[0], null); };
document.addEventListener("paste", (e) => {
  if (!state.user || $("app").hidden) return;
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) check(item.getAsFile(), null);
});
$("clear").onclick = () => { clearPhoto(); toast("Image removed."); };

function clearPhoto() {
  state.requestId++;  // ignore any result still on its way
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  $("preview").removeAttribute("src"); $("preview").hidden = true;
  $("hint").hidden = false; $("clear").hidden = true;
  $("truth").textContent = ""; $("error").hidden = true;
  $("placeholder").hidden = false; $("analysing").hidden = true; $("result").hidden = true;
  setStep(1);
}

function showError(msg) { $("error").textContent = msg; $("error").hidden = false; }

async function showProgress(id) {
  const items = [...document.querySelectorAll(".progress-list li")];
  items.forEach((li) => li.classList.remove("active", "done"));
  for (const li of items) {
    if (id !== state.requestId) return;
    li.classList.add("active");
    await sleep(MIN_ANALYSIS_MS / items.length);
    li.classList.replace("active", "done");
  }
}

async function check(blob, truth) {
  if (!blob.type.startsWith("image/")) { showError("Please choose an image file, such as a JPG or PNG photo."); return; }
  if (blob.size > 20 * 1024 * 1024) { showError("That image is larger than 20 MB. Please choose a smaller one."); return; }
  const id = ++state.requestId;
  $("error").hidden = true;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(blob);
  $("preview").src = state.previewUrl; $("preview").hidden = false; $("hint").hidden = true; $("clear").hidden = false;
  $("truth").textContent = truth ? `Example image. The confirmed answer is: ${className(truth)}.` : "";
  $("placeholder").hidden = true; $("result").hidden = true; $("analysing").hidden = false;
  setStep(2);
  try {
    const [data] = await Promise.all([api("/api/predict", { body: blob }), showProgress(id)]);
    if (id !== state.requestId) return;
    $("analysing").hidden = true;
    render(data.results, truth);
    setStep(4);
    $("result").hidden = false;
    $("result").focus({ preventScroll: true });
    $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    if (id !== state.requestId) return;
    if (err.status === 401) { setUser(null); toast("Your session ended. Please sign in again."); return; }
    $("analysing").hidden = true; $("placeholder").hidden = false; setStep(1);
    showError(err.message);
  }
}

function anotherImageButton() {
  const b = el("button", { class: "btn ghost big", type: "button", text: "Check another image" });
  b.onclick = () => { clearPhoto(); $("upload-card").scrollIntoView({ behavior: "smooth", block: "start" }); $("choose").focus({ preventScroll: true }); };
  return el("div", { class: "result-actions" }, b);
}


// ---------- result ----------
const swatch = (model) => { const s = el("span", { class: "swatch" }); s.style.background = `var(--m-${model})`; return s; };

function agreement(n, total) {
  const level = n === total ? ["All four models agree", 3] : n === 3 ? ["3 of 4 models agree", 2] : [`${n} of ${total} models agree`, 1];
  const meter = el("span", { class: "meter", "aria-hidden": "true" }, ...[1, 2, 3].map((i) => el("span", { class: i <= level[1] ? "on" : "" })));
  return el("span", { class: "chip neutral" }, meter, level[0]);
}

// Majority vote decides what it most closely matches; any model seeing melanoma, or the models
// disagreeing, makes the advice "worth discussing with a professional": missing a cancer is the costly mistake.
function render(results, truth) {
  const votes = {}, mass = {};
  for (const r of results) {
    votes[r.top] = (votes[r.top] || 0) + 1;
    for (const p of r.probs) mass[p.code] = (mass[p.code] || 0) + p.p;
  }
  const [winner, n] = Object.entries(votes).sort((a, b) => b[1] - a[1] || mass[b[0]] - mass[a[0]])[0];
  const m = state.info, info = m.classes[winner], care = info.care, total = results.length, melVotes = votes.mel || 0;
  let concern = care.level !== "selfcare", why = "", steps = care.steps, treatment = care.treatment;
  if (melVotes && winner !== "mel") {
    concern = true;
    why = `Most models say ${lower(winner)}, but ${melVotes} of ${total} ${melVotes === 1 ? "sees" : "see"} features of melanoma.`;
    steps = m.classes.mel.care.steps;
    treatment = m.classes.mel.care.treatment;
  } else if (!concern && n < 3) {
    concern = true;
    why = "The AI models don't agree about this image.";
  }

  const chip = concern ? el("span", { class: "chip info", text: "Worth discussing with a professional" })
    : el("span", { class: "chip ok", text: "Appears less concerning" });

  const bars = (r) => r.probs.map((p) => {
    const fill = el("span", { class: "fill" });
    fill.style.width = (p.p * 100).toFixed(1) + "%";
    if (p.code === r.top) fill.style.background = `var(--m-${r.model})`;
    return el("div", { class: "bar" }, el("span", { class: "lab", text: className(p.code) }),
      el("span", { class: "track" }, fill), el("span", { class: "val", text: pct(p.p) }));
  });

  $("result").replaceChildren(...[
    el("p", { class: "match-label", text: "Your image most closely matches" }),
    el("h3", { class: "match-name", text: info.name }),
    el("div", { class: "chips" }, chip, agreement(n, total)),
    truth ? el("p", { class: "truth", text: truth === winner ? "This matches the confirmed answer for this example." : `The confirmed answer for this example is ${className(truth)}.` }) : null,

    el("div", { class: "block" }, el("h3", { text: "What this means" }),
      el("p", { text: info.about }),
      el("p", { text: `It typically looks like ${care.looks}. These are general signs, not something the AI measured.` })),

    el("div", { class: "next " + (concern ? "info" : "ok") },
      el("h3", { text: concern ? CONSIDER : care.headline }),
      concern ? el("p", { text: [why, SIMILAR].filter(Boolean).join(" ") }) : null,
      el("p", {}, el("b", { text: "What you can do now" })),
      el("ol", { class: "steps" }, ...steps.map((s) => el("li", { text: s })))),

    el("div", { class: "block" }, el("h3", { text: "How it's usually managed" }), el("p", { text: treatment })),

    el("div", { class: "block" }, el("h3", { text: "When to talk to a doctor" }),
      el("p", { text: "Whatever the result, it's worth talking to a doctor if a skin spot:" }),
      el("ul", { class: "signs" }, ...m.urgent_signs.map((s) => el("li", { text: s.replace(/^It /, "") })))),

    el("details", { class: "fold" }, el("summary", { text: "See where each AI model looked" }),
      el("div", {},
        el("div", { class: "seen-grid" }, ...results.map((r) => el("figure", {},
          el("img", { src: r.attention, alt: `Your image as seen by ${r.name}: areas it paid less attention to are dimmed` }),
          el("figcaption", {}, swatch(r.model), `${r.name}: ${lower(r.top)}`)))),
        el("p", { class: "muted", text: "The brighter area is the part of your image that most influenced each model. A model looking away from the skin spot is less reliable." }))),

    el("details", { class: "fold" }, el("summary", { text: "See each model's answer" }),
      el("div", { class: "model-list" }, ...results.map((r) => el("div", {},
        el("h4", {}, el("span", { class: "mname" }, swatch(r.model), r.name), el("span", { class: "top", text: `${className(r.top)}, ${pct(r.probs[0].p)}` })),
        el("div", { class: "bars" }, ...bars(r)))))),

    el("p", { class: "disclaimer" }, el("b", { text: "This is an AI-based prediction, not a medical diagnosis. " }),
      "An AI prediction is only one piece of information. You're always welcome to ask a doctor about any skin concern."),
    anotherImageButton(),
  ].filter(Boolean));
}

// ---------- model comparison ----------
const COLS = [
  ["test_macro_f1", "Macro F1", "max"], ["test_accuracy", "Accuracy", "max"], ["test_melanoma_recall", "Melanomas found", "max"],
  ["ext_macro_f1", "Macro F1, test set 2", "max"], ["file_mb", "Size (MB)", "min"], ["cpu_latency_ms", "Time per image (ms)", "min"],
];

function renderComparison(info) {
  const val = (m, k) => (Array.isArray(m[k]) ? m[k][0] : m[k]);
  const best = {};
  for (const [k, , dir] of COLS) {
    const vals = info.models.map((m) => val(m, k));
    best[k] = dir === "max" ? Math.max(...vals) : Math.min(...vals);
  }
  const head = el("tr", {}, el("th", { text: "AI model" }), ...COLS.map(([, label]) => el("th", { text: label })));
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
    $("legend-note").prepend(`Average of ${info.seeds} training runs per model; the small grey number shows how much it varied. `);
    $("seeds-text").textContent = `Each model was trained ${info.seeds} times with different random starting points, and the table shows the average, so one lucky run can't decide the winner.`;
  }

  const byF1 = [...info.models].sort((a, b) => b.test_macro_f1[0] - a.test_macro_f1[0]);
  const [first, second] = byF1;
  // A gap smaller than the run-to-run spread is not a real difference
  const tied = info.seeds > 1 && first.test_macro_f1[0] - second.test_macro_f1[0] < Math.max(first.test_macro_f1[1], second.test_macro_f1[1]);
  // Fastest model that is still reasonably accurate (within 0.06 macro F1 of the best)
  const quick = [...info.models].filter((m) => first.test_macro_f1[0] - m.test_macro_f1[0] <= 0.06).sort((a, b) => a.cpu_latency_ms - b.cpu_latency_ms)[0];
  const mel = [...info.models].sort((a, b) => b.test_melanoma_recall[0] - a.test_melanoma_recall[0])[0];
  const scratch = info.models.find((x) => x.id === "cnn");
  const points = [
    tied ? `${first.name} and ${second.name} are effectively tied: the difference between them is smaller than the variation between training runs.`
         : `${first.name} scores highest, ahead of ${second.name}.`,
    `${mel.name} finds the most melanomas.`,
  ];
  if (quick && quick.id !== first.id) points.push(`${quick.name} is ${(first.cpu_latency_ms / quick.cpu_latency_ms).toFixed(1)} times faster and ${(first.file_mb / quick.file_mb).toFixed(1)} times smaller than ${first.name}, for a slightly lower score, so it suits phones and slow laptops.`);
  if (scratch) points.push(`The basic CNN, trained from scratch, scores much lower (${scratch.test_macro_f1[0].toFixed(2)}): starting from general image knowledge helps a lot.`);
  $("winner").replaceChildren(el("h3", { text: tied ? `${first.name} and ${second.name} come out on top` : `${first.name} comes out on top` }),
    el("ul", {}, ...points.map((t) => el("li", { text: t }))));
  $("winner").hidden = false;

  const leak = info.leakage_demo;
  if (leak) {
    const bi = leak.split_by_image, bl = leak.split_by_lesion;
    $("leak-text").textContent = "We measured what happens if you don't: the test score looks better, but on images from a different collection the AI is no better. It had simply memorised spots.";
    const stat = (big, small) => el("div", { class: "stat" }, el("b", { text: big }), el("span", { text: small }));
    $("leak-stats").replaceChildren(
      stat(`${Math.round(leak.test_images_with_lesion_seen_in_training * 100)}%`, "of test photos showed a spot already seen in training"),
      stat(`+${(bi.own_test_set.macro_f1 - bl.own_test_set.macro_f1).toFixed(3)}`, "falsely higher test score"),
      stat(`${(bi.external_test_set.macro_f1 - bl.external_test_set.macro_f1).toFixed(3)}`, "change on a different collection"));
  }
}

// ---------- examples (shown as names, not pictures) ----------
function renderExamples(info) {
  const src = (code) => `examples/${code}.jpg?v=${info.examples?.[code] ?? ""}`;
  for (const code of Object.keys(info.classes)) {
    const b = el("button", { type: "button", text: info.classes[code].name });
    b.onclick = async () => check(await (await fetch(src(code))).blob(), code);
    $("examples").append(b);
  }
}

// ---------- start ----------
(async () => {
  try {
    const [info, me] = await Promise.all([api("/api/info"), api("/api/me")]);
    state.info = info;
    renderComparison(info);
    renderExamples(info);
    setMode("signup");
    setUser(me.user);
  } catch {
    $("auth").hidden = false;
    authError("We can't reach the app right now. Please make sure run.bat (or python app/server.py) is still running.");
  }
})();
