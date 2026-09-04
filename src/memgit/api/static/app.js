import { layout, renderGraph } from "./graph.js";

// Mirrors memgit.core.diff.ChangeKind's string values and the same glyph
// map used by src/memgit/mcp/server.py's _DIFF_PREFIXES and cli.py's
// _print_diff_human -- kept here as a literal rather than a new endpoint,
// since eight fixed entries are simpler than a round trip to fetch them.
const DIFF_GLYPHS = {
  added: "+",
  removed: "-",
  reaffirmed: "~",
  contradicted: "!",
  value_added: ">",
  value_removed: "<",
  mixed: "*",
  unchanged: " ",
};

class ApiError extends Error {
  constructor(body, status) {
    super((body && (body.detail || body.error)) || `HTTP ${status}`);
    this.status = status;
    this.body = body || {};
  }
}

const API_KEY_STORAGE_KEY = "memgitApiKey";

// sessionStorage, not localStorage: the key should not outlive the tab, and
// never lands in a URL or a cookie either -- there is no CSRF story to
// write because there is nothing ambient for a third-party page to ride on.
function getStoredApiKey() {
  try {
    return sessionStorage.getItem(API_KEY_STORAGE_KEY);
  } catch (err) {
    return null;
  }
}

function setStoredApiKey(key) {
  try {
    sessionStorage.setItem(API_KEY_STORAGE_KEY, key);
  } catch (err) {
    // sessionStorage unavailable -- the key just won't persist between
    // requests in this tab; not fatal, the next 401 prompts again.
  }
}

async function fetchWithApiKey(path, init) {
  const headers = new Headers((init && init.headers) || {});
  const key = getStoredApiKey();
  if (key) headers.set("X-API-Key", key);
  const response = await fetch(path, { ...init, headers });
  let body = null;
  try {
    body = await response.json();
  } catch (err) {
    body = null;
  }
  return { response, body };
}

async function api(path, init) {
  let { response, body } = await fetchWithApiKey(path, init);
  if (response.status === 401) {
    // Auth is off (the loopback-dev default) whenever this branch never
    // runs -- the first request already succeeded above. Only a server
    // started with --api-key ever gets here.
    const key = window.prompt("memgit API key required:");
    if (key) {
      setStoredApiKey(key);
      ({ response, body } = await fetchWithApiKey(path, init));
    }
  }
  if (!response.ok) throw new ApiError(body, response.status);
  return body;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

const state = {
  repo: null,
  commits: [],
  layoutResult: null,
  before: null,
  after: null,
  includeUnchanged: false,
  useMergeBase: false,
  tab: "diff",
  diff: null,
  diffError: null,
  stateData: null,
  stateError: null,
  replay: {
    busy: false,
    result: null,
    error: null,
  },
};

async function loadDiff() {
  if (!state.after) {
    state.diff = null;
    state.diffError = null;
    return;
  }
  const params = new URLSearchParams({ after: state.after });
  if (state.before) params.set("before", state.before);
  if (state.includeUnchanged) params.set("include_unchanged", "true");
  if (state.useMergeBase) params.set("use_merge_base", "true");
  try {
    state.diff = await api(`/api/diff?${params}`);
    state.diffError = null;
  } catch (err) {
    state.diff = null;
    state.diffError = err.message;
  }
}

async function loadState() {
  const rev = state.after || "HEAD";
  try {
    state.stateData = await api(`/api/state?rev=${encodeURIComponent(rev)}`);
    state.stateError = null;
  } catch (err) {
    state.stateData = null;
    state.stateError = err.message;
  }
}

function fillReplayKey(subject, predicate, factHash) {
  document.getElementById("replay-subject").value = subject;
  document.getElementById("replay-predicate").value = predicate;
  document.getElementById("replay-fact-hash").value = factHash || "";
  document.getElementById("replay-rev").value = state.after || "HEAD";
}

async function selectNode(hash, shiftKey) {
  if (shiftKey) {
    state.before = hash === state.before ? null : hash;
  } else {
    state.after = hash;
    state.before = null;
  }
  await Promise.all([loadDiff(), state.tab === "state" ? loadState() : Promise.resolve()]);
  render();
}

function renderHeader() {
  const repo = state.repo;
  document.getElementById("field-root").textContent = repo ? repo.root : "";
  if (repo) {
    const branchLabel = repo.head.detached
      ? `detached @ ${(repo.head.commit || "").slice(0, 8)}`
      : repo.head.branch || "(unborn)";
    document.getElementById("field-branch").innerHTML =
      `branch: <strong>${escapeHtml(branchLabel)}</strong>`;
    document.getElementById("field-head").textContent = repo.head.commit
      ? `HEAD ${repo.head.commit.slice(0, 8)}`
      : "no commits yet";
  }
}

function renderGraphPane() {
  if (!state.layoutResult) return;
  const svg = document.getElementById("graph-svg");
  renderGraph(svg, state.layoutResult, {
    headHash: state.repo ? state.repo.head.commit : null,
    before: state.before,
    after: state.after,
    onSelectNode: (hash, shiftKey) => {
      selectNode(hash, shiftKey);
    },
  });
}

function renderDiffTab() {
  const container = document.getElementById("tab-content");
  const parts = [];

  parts.push(`
    <div class="stat-line">
      <label><input type="checkbox" id="opt-unchanged" ${state.includeUnchanged ? "checked" : ""}> include unchanged</label>
      &nbsp;
      <label><input type="checkbox" id="opt-merge-base" ${state.useMergeBase ? "checked" : ""}> use merge-base</label>
      &nbsp;
      ${state.before ? `before <code>${state.before.slice(0, 8)}</code> vs ` : "(implicit first parent) vs "}
      after <code>${state.after ? state.after.slice(0, 8) : "?"}</code>
    </div>
  `);

  if (state.diffError) {
    parts.push(`<div class="error-notice">${escapeHtml(state.diffError)}</div>`);
  } else if (state.diff) {
    const diff = state.diff.diff;
    const stat = diff.stat;
    parts.push(
      `<p class="stat-line">${stat.keys_changed} key(s) changed -- ` +
        `+${stat.facts_added} -${stat.facts_removed} ~${stat.facts_reaffirmed}. ` +
        `Cardinality map embedded in this result.</p>`
    );
    if (diff.violations.length > 0) {
      parts.push(
        `<div class="warning">${diff.violations.length} cardinality violation(s) -- see the CLI's --strict-cardinality for detail.</div>`
      );
    }
    parts.push("<table><thead><tr><th></th><th>subject</th><th>predicate</th><th>detail</th></tr></thead><tbody>");
    for (const kd of diff.keys) {
      const glyph = DIFF_GLYPHS[kd.kind] ?? "?";
      const details = kd.values
        .map((v) => {
          const conf =
            v.before && v.after && v.before.confidence !== v.after.confidence
              ? ` <span class="confidence-arrow">(${v.before.confidence.toFixed(2)} &rarr; ${v.after.confidence.toFixed(2)})</span>`
              : "";
          return `${escapeHtml(v.object)}${conf}`;
        })
        .join(", ");
      parts.push(`
        <tr class="fact-row" data-subject="${escapeHtml(kd.subject)}" data-predicate="${escapeHtml(kd.predicate)}">
          <td class="glyph ${kd.kind}">${glyph}</td>
          <td>${escapeHtml(kd.subject)}</td>
          <td>${escapeHtml(kd.predicate)}</td>
          <td>${details}</td>
        </tr>
      `);
    }
    parts.push("</tbody></table>");
  } else {
    parts.push("<p class=\"stat-line\">Select a commit in the graph to see its diff.</p>");
  }

  container.innerHTML = parts.join("");

  const unchangedBox = document.getElementById("opt-unchanged");
  if (unchangedBox) {
    unchangedBox.addEventListener("change", (event) => {
      state.includeUnchanged = event.target.checked;
      loadDiff().then(render);
    });
  }
  const mergeBaseBox = document.getElementById("opt-merge-base");
  if (mergeBaseBox) {
    mergeBaseBox.addEventListener("change", (event) => {
      state.useMergeBase = event.target.checked;
      loadDiff().then(render);
    });
  }
  container.querySelectorAll(".fact-row").forEach((row) => {
    row.addEventListener("click", () => {
      fillReplayKey(row.dataset.subject, row.dataset.predicate, "");
    });
  });
}

function renderStateTab() {
  const container = document.getElementById("tab-content");
  const parts = [];

  if (state.stateError) {
    parts.push(`<div class="error-notice">${escapeHtml(state.stateError)}</div>`);
  } else if (state.stateData) {
    const facts = state.stateData.state.facts;
    parts.push(`<p class="stat-line">${facts.length} fact(s) at ${(state.stateData.resolved || "").slice(0, 8)}</p>`);
    parts.push("<table><thead><tr><th>subject</th><th>predicate</th><th>object</th><th>confidence</th></tr></thead><tbody>");
    for (const fact of facts) {
      parts.push(`
        <tr class="fact-row" data-subject="${escapeHtml(fact.subject)}" data-predicate="${escapeHtml(fact.predicate)}">
          <td>${escapeHtml(fact.subject)}</td>
          <td>${escapeHtml(fact.predicate)}</td>
          <td>${escapeHtml(fact.object)}</td>
          <td>${fact.confidence.toFixed(2)}</td>
        </tr>
      `);
    }
    parts.push("</tbody></table>");
  } else {
    parts.push("<p class=\"stat-line\">Select a commit in the graph to see its memory state.</p>");
  }

  container.innerHTML = parts.join("");
  container.querySelectorAll(".fact-row").forEach((row) => {
    row.addEventListener("click", () => {
      fillReplayKey(row.dataset.subject, row.dataset.predicate, "");
    });
  });
}

function renderReplayResult() {
  const container = document.getElementById("replay-result");
  const submit = document.getElementById("replay-submit");
  submit.disabled = state.replay.busy;
  submit.textContent = state.replay.busy
    ? "replaying (two model calls, ~10-40s)..."
    : "Run replay comparison";

  if (state.replay.error) {
    container.innerHTML = `<div class="error-notice">${escapeHtml(state.replay.error)}</div>`;
    return;
  }
  const result = state.replay.result;
  if (!result) {
    container.innerHTML = "";
    return;
  }

  const badgeClass = result.changed ? "yes" : "no";
  const pinned = result.retrieval
    ? `<p class="stat-line">${result.retrieval.k} of ${result.retrieval.candidates} fact(s) pinned for both sides.</p>`
    : "";

  container.innerHTML = `
    <span class="changed-badge ${badgeClass}">changed: ${result.changed ? "yes" : "no"}</span>
    <div class="replay-columns">
      <div>
        <strong>baseline</strong>
        <pre>${escapeHtml(result.baseline.reply)}</pre>
      </div>
      <div>
        <strong>ablated</strong>
        <pre>${escapeHtml(result.ablated.reply)}</pre>
      </div>
    </div>
    ${pinned}
    <p class="caveat">Evidence, not proof: removing one fact can have downstream effects on others that reference it.</p>
  `;
}

function render() {
  renderHeader();
  renderGraphPane();
  document.getElementById("tab-diff").classList.toggle("active", state.tab === "diff");
  document.getElementById("tab-state").classList.toggle("active", state.tab === "state");
  if (state.tab === "diff") {
    renderDiffTab();
  } else {
    renderStateTab();
  }
  renderReplayResult();
}

function wireStaticControls() {
  document.getElementById("tab-diff").addEventListener("click", () => {
    state.tab = "diff";
    render();
  });
  document.getElementById("tab-state").addEventListener("click", async () => {
    state.tab = "state";
    if (!state.stateData) await loadState();
    render();
  });

  document.getElementById("replay-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.target;
    const body = {
      rev: form.rev.value || "HEAD",
      subject: form.subject.value,
      predicate: form.predicate.value,
      query: form.query.value,
    };
    if (form.fact_hash.value) body.fact_hash = form.fact_hash.value;

    state.replay.busy = true;
    state.replay.error = null;
    state.replay.result = null;
    render();
    try {
      state.replay.result = await api("/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (err) {
      state.replay.error = err.message;
    } finally {
      state.replay.busy = false;
      render();
    }
  });
}

async function boot() {
  wireStaticControls();
  try {
    const [repo, log] = await Promise.all([api("/api/repo"), api("/api/log?all=true&limit=200")]);
    state.repo = repo;
    state.commits = log.commits;
    state.layoutResult = layout(log.commits);
    if (repo.head.commit) {
      state.after = repo.head.commit;
      await loadDiff();
    }
  } catch (err) {
    document.getElementById("graph-pane").innerHTML =
      `<div class="error-notice">${escapeHtml(err.message)}</div>`;
    return;
  }
  render();
}

boot();
