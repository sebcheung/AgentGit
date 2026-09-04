// Commit-graph lane layout and SVG rendering -- no library.
//
// `layout` is a single forward pass over `commits` (newest-first, exactly
// the order /api/log returns), git-log-graph style: a commit reuses the
// lane a child already reserved for it (or takes the leftmost free lane, a
// tip), then reserves that same lane for its own first parent so the line
// continues downward; every extra parent (a merge) forks a new lane.

const SVG_NS = "http://www.w3.org/2000/svg";

export const PAD = 20;
export const LANE_W = 18;
export const ROW_H = 26;
export const NODE_R = 5;
export const LANE_COLORS = 6;

function firstFreeLane(lanes) {
  const index = lanes.indexOf(null);
  if (index !== -1) return index;
  lanes.push(null);
  return lanes.length - 1;
}

export function layout(commits) {
  const byHash = new Map(commits.map((c) => [c.hash, c]));
  const lanes = [];
  const nodes = [];
  const nodeByHash = new Map();

  commits.forEach((commit, row) => {
    let lane = lanes.indexOf(commit.hash);
    if (lane === -1) lane = firstFreeLane(lanes);

    // A converging branch: another lane was also waiting for this same
    // commit (two children sharing a parent). Free it now that the node
    // is about to be drawn from `lane` -- otherwise it sits reserved
    // forever, wasting a column no commit will ever reuse.
    for (let j = 0; j < lanes.length; j += 1) {
      if (j !== lane && lanes[j] === commit.hash) lanes[j] = null;
    }

    const node = { hash: commit.hash, lane, row, commit };
    nodes.push(node);
    nodeByHash.set(commit.hash, node);

    const parents = commit.parents.filter((p) => byHash.has(p));
    lanes[lane] = parents.length > 0 ? parents[0] : null;
    for (let i = 1; i < parents.length; i += 1) {
      lanes[firstFreeLane(lanes)] = parents[i];
    }
  });

  const edges = [];
  for (const node of nodes) {
    for (const parent of node.commit.parents) {
      const target = nodeByHash.get(parent);
      if (target) edges.push({ from: node, to: target });
    }
  }

  const laneCount = nodes.reduce((max, n) => Math.max(max, n.lane + 1), 1);
  return { nodes, edges, laneCount };
}

function edgePath(x1, y1, x2, y2) {
  if (x1 === x2) return `M ${x1} ${y1} V ${y2}`;
  const c1y = y1 + ROW_H / 2;
  const c2y = y2 - ROW_H / 2;
  return `M ${x1} ${y1} C ${x1} ${c1y}, ${x2} ${c2y}, ${x2} ${y2}`;
}

/**
 * Render `layoutResult` into `svg` (an existing <svg> element).
 *
 * `onSelectNode(hash, shiftKey)` fires on click; a plain click sets `after`,
 * shift-click sets `before` -- see app.js.
 */
export function renderGraph(svg, layoutResult, { headHash, before, after, onSelectNode } = {}) {
  const { nodes, edges, laneCount } = layoutResult;
  const textX = PAD + laneCount * LANE_W + 10;
  const width = textX + 420;
  const height = PAD * 2 + Math.max(nodes.length, 1) * ROW_H;

  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.textContent = "";

  const x = (lane) => PAD + lane * LANE_W;
  const y = (row) => PAD + row * ROW_H;

  for (const edge of edges) {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", edgePath(x(edge.from.lane), y(edge.from.row), x(edge.to.lane), y(edge.to.row)));
    path.setAttribute("class", `edge lane-${edge.from.lane % LANE_COLORS}`);
    svg.appendChild(path);
  }

  nodes.forEach((node) => {
    const cx = x(node.lane);
    const cy = y(node.row);

    const clipId = `memgit-graph-clip-${node.row}`;
    const clipPath = document.createElementNS(SVG_NS, "clipPath");
    clipPath.setAttribute("id", clipId);
    const clipRect = document.createElementNS(SVG_NS, "rect");
    clipRect.setAttribute("x", String(textX));
    clipRect.setAttribute("y", String(cy - ROW_H / 2));
    clipRect.setAttribute("width", "400");
    clipRect.setAttribute("height", String(ROW_H));
    clipPath.appendChild(clipRect);
    svg.appendChild(clipPath);

    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", String(cx));
    circle.setAttribute("cy", String(cy));
    circle.setAttribute("r", String(NODE_R));
    const classes = ["node", `lane-${node.lane % LANE_COLORS}`];
    if (node.commit.is_merge) classes.push("merge");
    if (node.commit.is_root) classes.push("root");
    if (node.hash === headHash) classes.push("head");
    if (node.hash === before || node.hash === after) classes.push("selected");
    circle.setAttribute("class", classes.join(" "));
    svg.appendChild(circle);

    const text = document.createElementNS(SVG_NS, "text");
    text.setAttribute("x", String(textX));
    text.setAttribute("y", String(cy + 4));
    text.setAttribute("class", "label");
    text.setAttribute("clip-path", `url(#${clipId})`);
    text.textContent = `${node.hash.slice(0, 8)}  ${node.commit.summary}`;
    svg.appendChild(text);

    const hit = document.createElementNS(SVG_NS, "rect");
    hit.setAttribute("x", "0");
    hit.setAttribute("y", String(cy - ROW_H / 2));
    hit.setAttribute("width", String(width));
    hit.setAttribute("height", String(ROW_H));
    hit.setAttribute("class", "hit");
    if (onSelectNode) {
      hit.addEventListener("click", (event) => onSelectNode(node.hash, event.shiftKey));
    }
    svg.appendChild(hit);
  });
}
