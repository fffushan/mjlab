#!/usr/bin/env node
// Build the D1 data-collection pipeline figure.
//
//   node docs/reports/build_pipeline_diagram.mjs
//   -> docs/reports/x2_diffusion_data_collection_pipeline.svg
//
// Every count comes from
// docs/reports/x2_diffusion_data_collection_review_20261029.md (revision 2) and
// from the launch driver
// logs/diagnostics/preview-flow-port-20261020/multimotion144-bulk/run-bulk.sh.
// Change a number here only when the note changes.

import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, "x2_diffusion_data_collection_pipeline.svg");

const MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace";
const SANS = "'Inter','Segoe UI',system-ui,-apple-system,'Helvetica Neue',Arial,sans-serif";

const C = {
  page: "#eef1f5",
  cardTop: "#ffffff",
  bg: "#f8fafb",
  rule: "#e3e7ec",
  ruleLit: "#b4bcc6",
  steelF: "#ced4db",
  steelT: "#e7ebef",
  steelS: "#a9b1bb",
  steelL: "#d9dee4",
  steelHi: "#eff2f6",
  steelTop: "#f5f7fa",
  pipe: "#b9c1cb",
  ink: "#333b45",
  text: "#0e1317",
  dim: "#454f5b",
  faint: "#616b76",
  src: "#737e89",
  mid: "#576069",
  amber: "#96600a",
  amberD: "#d9a441",
  amberF: "#fdf3dc",
  cyan: "#0d6b7a",
  cyanD: "#93c9d3",
  cyanF: "#e3f2f5",
  rust: "#a53a1c",
  rustD: "#e0b0a0",
  rustF: "#fceee9",
  well: "#eceff4",
  pit: "#f3f6f9",
  dot: "#aeb6c0",
  bar: "#c2c9d2",
  barA: "#d99a1f",
  barC: "#3d9fb0",
  barR: "#cf6a45",
  barG: "#a7b0ba",
};

const W = 1680;
const H = 1120;

const esc = (s) =>
  String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function T(x, y, s, o = {}) {
  const {
    size = 13,
    fill = C.text,
    anchor = "start",
    weight = 400,
    ls = 0,
    mono = false,
    cls = "",
  } = o;
  const a = [
    `x="${x}"`,
    `y="${y}"`,
    `font-size="${size}"`,
    `fill="${fill}"`,
    `text-anchor="${anchor}"`,
    `font-weight="${weight}"`,
    mono ? `font-family="${MONO}"` : "",
    ls ? `letter-spacing="${ls}"` : "",
    cls ? `class="${cls}"` : "",
  ]
    .filter(Boolean)
    .join(" ");
  return `<text ${a}>${esc(s)}</text>`;
}

const P = (d, fill, stroke = "none", sw = 0, extra = "") =>
  `<path d="${d}" fill="${fill}"${
    stroke !== "none" ? ` stroke="${stroke}" stroke-width="${sw}" stroke-linejoin="round"` : ""
  }${extra ? " " + extra : ""}/>`;

const R = (x, y, w, h, fill, o = {}) => {
  const { stroke = "none", sw = 0, r = 0, extra = "" } = o;
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}"${r ? ` rx="${r}"` : ""} fill="${fill}"${
    stroke !== "none" ? ` stroke="${stroke}" stroke-width="${sw}"` : ""
  }${extra ? " " + extra : ""}/>`;
};

const CIR = (cx, cy, r, fill, o = {}) => {
  const { stroke = "none", sw = 0, extra = "" } = o;
  return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}"${
    stroke !== "none" ? ` stroke="${stroke}" stroke-width="${sw}"` : ""
  }${extra ? " " + extra : ""}/>`;
};

const LINE = (x1, y1, x2, y2, stroke, sw = 1, extra = "") =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${sw}"${
    extra ? " " + extra : ""
  }/>`;

/** Shallow extruded solid: top plane, right plane, front face. */
function solid(x, y, w, h, o = {}) {
  const {
    dx = 6,
    dy = -6,
    front = C.steelF,
    top = C.steelT,
    side = C.steelS,
    stroke = C.ink,
    sw = 1,
  } = o;
  const x2 = x + w;
  const y2 = y + h;
  const X = x + dx;
  const Y = y + dy;
  const X2 = x2 + dx;
  const Y2 = y2 + dy;
  return (
    `<g stroke="${stroke}" stroke-width="${sw}" stroke-linejoin="round">` +
    P(`M${x},${y} L${X},${Y} L${X2},${Y} L${x2},${y} Z`, top) +
    P(`M${x2},${y} L${X2},${Y} L${X2},${Y2} L${x2},${y2} Z`, side) +
    R(x, y, w, h, front) +
    `</g>`
  );
}

/** Drum: side body with elliptical top and bottom. */
function drum(cx, cy, rx, ry, h, o = {}) {
  const { front = C.steelF, top = C.steelT, stroke = C.ink, sw = 1 } = o;
  return (
    `<g stroke="${stroke}" stroke-width="${sw}">` +
    R(cx - rx, cy, rx * 2, h, front) +
    `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" fill="${top}"/>` +
    `<ellipse cx="${cx}" cy="${cy + h}" rx="${rx}" ry="${ry}" fill="${front}"/>` +
    `</g>`
  );
}

/** Flow rail: scale ticks plus a drifting dashed line. No arrowheads. */
function rail(x0, x1, y, o = {}) {
  const { step = 16, tick = 5, tone = C.ruleLit } = o;
  let out = LINE(x0, y, x1, y, tone, 1.5);
  for (let x = x0; x <= x1; x += step) out += LINE(x, y - tick, x, y, C.faint, 1);
  out += LINE(x0, y + 8, x1, y + 8, C.rule, 4);
  out += `<line class="flow" x1="${x0}" y1="${y + 8}" x2="${x1}" y2="${y + 8}" stroke="${C.dim}" stroke-width="1.3" stroke-dasharray="2 20"/>`;
  out += P(`M${x1 - 12},${y + 2} L${x1},${y + 8} L${x1 - 12},${y + 14} Z`, C.dim);
  return out;
}

const stations = [];

function plate(x, y, w, name, value, src) {
  return (
    R(x, y, w, 54, "#ffffff", { stroke: C.rule, sw: 1, r: 3, extra: 'class="plate"' }) +
    R(x, y, 3, 54, C.ruleLit) +
    T(x + 12, y + 17, name, { size: 11.5, weight: 600, ls: 0.65 }) +
    T(x + 12, y + 33, value, { size: 11, fill: C.dim, mono: true }) +
    T(x + 12, y + 46, src, { size: 9.4, fill: C.src, mono: true })
  );
}

/** Station: content authored in local coords (0..w, 0..h), plate beneath. */
function station(id, x, y, w, h, name, value, src, drawing) {
  stations.push({ id });
  return (
    `<g class="st" data-i="${id}" tabindex="0" role="listitem" aria-label="${esc(name)}: ${esc(value)}">` +
    `<g transform="translate(${x},${y})">${drawing}</g>` +
    plate(x, y + h + 10, w, name, value, src) +
    `</g>`
  );
}

const chip = (x, y, w, label, value, tone) =>
  R(x, y, w, 44, "#ffffff", { stroke: C.rule, sw: 1, r: 3 }) +
  R(x, y, w, 3, tone) +
  T(x + 12, y + 17, label, { size: 9.2, fill: C.dim, ls: 1.1, weight: 600 }) +
  T(x + 12, y + 34, value, { size: 13.5, mono: true, weight: 500 });

// ================================================================ header ====

let head = "";
head += T(52, 58, "FIG 1", { size: 9.6, fill: C.faint, ls: 2.4, weight: 700 });
head += T(52, 92, "D1 data collection pipeline", { size: 27, weight: 600, ls: -0.3 });
head += T(
  52,
  116,
  "frozen VAE rollout, eight parallel workers, one store, one window index.  X2 tennis cohort, 144 clips.",
  { size: 12.5, fill: C.dim },
);
head += chip(1136, 42, 166, "PLANNED TRIALS", "37 152", C.barA);
head += chip(1316, 42, 166, "WINDOWS", "2 322 881", C.barA);
head += chip(1496, 42, 136, "TRAIN CACHE", "50.9 GB", C.barC);
head += LINE(52, 134, 1632, 134, C.rule, 1);

// ========================================================= band A: queue ====

const AY = 178; // drawing top
const AH = 92; // drawing height
let bandA = "";
bandA += T(52, 164, "A · SCHEDULE", { size: 9.6, fill: C.faint, ls: 2, weight: 700 });
bandA += T(196, 164, "one start frame becomes 86 tickets · one clip becomes 258", { size: 11, fill: C.faint });

// A1: rack of source documents
let a1 = `<g>`;
a1 += solid(8, 58, 116, 6, { dx: 4, dy: -4, front: C.steelHi, top: C.steelTop });
a1 += solid(14, 46, 108, 6, { dx: 4, dy: -4, front: C.steelL, top: C.steelHi });
a1 += solid(20, 34, 100, 6, { dx: 4, dy: -4, front: C.steelF, top: C.steelL });
a1 += solid(6, 64, 132, 22, { dx: 6, dy: -6 });
a1 += CIR(16, 75, 1.5, C.ink);
a1 += CIR(128, 75, 1.5, C.ink);
for (let i = 0; i < 3; i += 1) {
  a1 += LINE(24, 27 + i * 12, 116, 27 + i * 12, i === 2 ? C.amberD : C.faint, 1);
}
a1 += T(152, 22, "contract.yaml", { size: 10, fill: C.dim, mono: true });
a1 += T(152, 37, "125 kept", { size: 9.4, fill: C.amber, mono: true });
a1 += T(152, 51, "250 control", { size: 9.4, fill: C.faint, mono: true });
a1 += T(152, 69, "manifest 144", { size: 10, fill: C.dim, mono: true });
a1 += T(152, 83, "vae.pt  standing.onnx", { size: 9.4, fill: C.faint, mono: true });
a1 += `</g>`;
bandA += station("inputs", 60, AY, 240, AH, "INPUTS", "contract · manifest · 3 files", "run-bulk.sh:11-25", a1);

// A2: index plate with three detents
let a2 = `<g>`;
a2 += T(148, 10, "floor(f · (N−1))", { size: 9.4, fill: C.faint, mono: true, anchor: "end" });
a2 += solid(6, 16, 160, 10, { dx: 5, dy: -5, front: C.steelHi, top: C.steelTop });
a2 += R(14, 26, 144, 44, C.steelS, { stroke: C.ink, sw: 1 });
[0, 1, 2].forEach((i) => {
  const x = 26 + i * 44;
  a2 += R(x, 32, 28, 32, C.pit, { stroke: C.ruleLit, sw: 1 });
  a2 += R(x + 6, 38, 16, 20, i === 0 ? C.amberF : C.steelS, {
    stroke: i === 0 ? C.amberD : C.faint,
    sw: 1,
  });
});
a2 += solid(6, 82, 160, 10, { dx: 6, dy: -6 });
[0, 1, 2].forEach((i) => {
  const x = 26 + i * 44;
  a2 += T(x + 14, 74, ["0.1", "0.5", "0.8"][i], {
    size: 9.6,
    fill: i === 0 ? C.amber : C.text,
    mono: true,
    anchor: "middle",
    weight: i === 0 ? 600 : 400,
  });
});
a2 += `</g>`;
bandA += station("starts", 392, AY, 172, AH, "3 START FRAMES", "21 … 361 · 153 distinct", "runtime_x2.py:55 · 90-97", a2);

// A3: phase shutter
let a3 = `<g>`;
a3 += T(76, 10, "pair-local", { size: 9.4, fill: C.faint, anchor: "middle" });
a3 += solid(6, 16, 140, 10, { dx: 5, dy: -5, front: C.steelHi, top: C.steelTop });
a3 += R(14, 32, 60, 38, C.steelS, { stroke: C.ink, sw: 1 });
a3 += R(78, 32, 60, 38, C.steelS, { stroke: C.ink, sw: 1 });
a3 += R(14, 32, 60, 8, C.amber, {});
a3 += R(78, 32, 60, 8, C.rustD, {});
a3 += T(44, 58, "clean", { size: 9.6, fill: C.text, anchor: "middle" });
a3 += T(108, 58, "OU", { size: 9.6, fill: C.text, anchor: "middle" });
a3 += T(108, 68, "θ.8 σ.1", { size: 8.6, fill: C.faint, mono: true, anchor: "middle" });
a3 += solid(6, 70, 140, 22, { dx: 6, dy: -6 });
a3 += `</g>`;
bandA += station("phase", 600, AY, 152, AH, "CLEAN + OU", "×2 phases, paired", "collector.py:516-521", a3);

// A4: 43-position seed drum
let a4 = `<g>`;
a4 += drum(68, 20, 40, 9, 58, {});
a4 += `<g class="spin" style="transform-box:fill-box;transform-origin:50% 50%">`;
for (let i = 0; i < 43; i += 1) {
  const ang = (i / 43) * Math.PI * 2;
  a4 += LINE(
    68 + Math.cos(ang) * 30,
    49 + Math.sin(ang) * 26,
    68 + Math.cos(ang) * 39,
    49 + Math.sin(ang) * 33,
    i === 0 ? C.amber : C.faint,
    1.3,
  );
}
a4 += `</g>`;
a4 += T(68, 46, "43", { size: 14, mono: true, anchor: "middle", weight: 600 });
a4 += T(68, 58, "seeds", { size: 8.2, fill: C.faint, anchor: "middle", ls: 0.6 });
a4 += solid(18, 78, 100, 14, { dx: 5, dy: -5 });
a4 += `</g>`;
bandA += station("seeds", 788, AY, 148, AH, "43 SEEDS", "0 … 42", "run-bulk.sh:14", a4);

// A5: ticket stack
let a5 = `<g>`;
for (let i = 0; i < 11; i += 1) {
  a5 += solid(6 + i * 2.6, 16 + i * 1.4, 158, 62, {
    dx: 5,
    dy: -5,
    front: i % 2 ? C.steelS : C.steelF,
    top: i % 2 ? C.steelF : C.steelT,
    side: "#b7bfc9",
  });
}
a5 += R(6, 16, 158, 62, "none", { stroke: C.amberD, sw: 1 });
a5 += T(85, 44, "37 152", { size: 19, fill: C.amber, anchor: "middle", mono: true, weight: 600 });
a5 += T(85, 59, "18 576 groups × 2", { size: 8.8, fill: "#252d37", anchor: "middle", mono: true });
a5 += T(85, 72, "144 × 3 × 43 × 2", { size: 8.6, fill: "#39424d", anchor: "middle", mono: true });
a5 += solid(2, 84, 166, 8, { dx: 5, dy: -5 });
a5 += `</g>`;
bandA += station("trials", 972, AY, 176, AH, "TICKET STACK", "37 152 planned trials", "144 · 3 · 43 · 2", a5);

// pilot gate note in the empty slot of band A
bandA += R(1180, 190, 448, 66, C.pit, { stroke: C.rule, sw: 1, r: 3 });
bandA += T(1196, 210, "PILOT GATE RUNS FIRST", { size: 9.4, fill: C.dim, ls: 1.1, weight: 700 });
bandA += T(1196, 228, "4 seeds · 440 trials per worker", { size: 10, fill: C.faint, mono: true });
bandA += T(1196, 244, "gate: each worker 18 motions, no overlap, rows within ±10 %, abandoned pairs ≤ 2 %", {
  size: 9.6,
  fill: C.faint,
});

// flow between the queue stations: chevrons in the gaps, no rail
[346, 582, 770, 954].forEach((x) => {
  bandA += P(
    `M${x - 7},212 L${x + 2},221 L${x - 7},230 M${x + 3},212 L${x + 12},221 L${x + 3},230`,
    "none",
    C.ruleLit,
    1.6,
  );
});
// colour key
bandA += R(1180, 272, 448, 62, "none", {});
bandA += R(1180, 274, 12, 12, C.amberF, { stroke: C.amberD, sw: 1 });
bandA += T(1202, 284, "indexed: kept rows, windows, tokens", { size: 9.6, fill: C.dim });
bandA += R(1180, 294, 12, 12, C.cyanF, { stroke: C.cyanD, sw: 1 });
bandA += T(1202, 304, "offline stage, or the standing policy after the clip ends", { size: 9.6, fill: C.dim });
bandA += R(1180, 314, 12, 12, C.rustF, { stroke: C.rustD, sw: 1 });
bandA += T(1202, 324, "failed trial: every row leaves the window index", { size: 9.6, fill: C.dim });
bandA += LINE(52, 338, 1632, 338, C.rule, 1);

// ====================================================== band B: machines ====

let bandB = "";
bandB += T(52, 366, "B · COLLECTION", { size: 9.6, fill: C.faint, ls: 2, weight: 700 });
bandB += T(206, 366, "8 workers · 64 environments each · 18 clips each · 4700 trial cap", {
  size: 11,
  fill: C.faint,
});

// distribution manifold from the ticket stack to the lanes
bandB += P("M1100,332 L1100,348 L668,348 L668,386 L648,386", "none", C.steelS, 9);
bandB += P("M1100,332 L1100,348 L668,348 L668,386 L648,386", "none", C.steelF, 5);
bandB += R(88, 382, 560, 8, C.steelS, { stroke: C.rule, sw: 1, r: 4 });

let lanes = `<g class="st" data-i="lanes" tabindex="0" role="listitem" aria-label="eight collector workers">`;
for (let i = 0; i < 8; i += 1) {
  const x0 = 60 + i * 76;
  lanes += LINE(x0 + 30, 390, x0 + 30, 404, C.ruleLit, 1.5);
  lanes += solid(x0 + 4, 404, 52, 14, { dx: 5, dy: -5, front: C.steelHi, top: C.steelTop });
  lanes += solid(x0 + 8, 418, 44, 164, { dx: 6, dy: -6, front: "#d7dce3" });
  // 64 environments
  for (let r = 0; r < 8; r += 1) {
    for (let c = 0; c < 8; c += 1) {
      lanes += CIR(x0 + 15 + c * 5.1, 432 + r * 5.4, 1.55, r === 0 && c === 7 ? C.amber : C.dot);
    }
  }
  // seed strips
  for (let s = 0; s < 5; s += 1) lanes += R(x0 + 13, 486 + s * 6, 34, 2.4, C.bar);
  lanes += solid(x0 + 2, 582, 56, 16, { dx: 6, dy: -6 });
  lanes += LINE(x0 + 30, 598, x0 + 30, 630, C.ruleLit, 1.5);
  lanes += T(x0 + 30, 660, `GPU ${i}`, { size: 9.6, fill: C.dim, mono: true, anchor: "middle" });
  lanes += T(x0 + 30, 673, "18 clips", { size: 8.8, fill: C.faint, mono: true, anchor: "middle" });
}
lanes += R(88, 626, 560, 8, C.steelS, { stroke: C.rule, sw: 1, r: 4 });
// merge pipe: right of the lanes, then left along the band gap into MERGE
lanes += P("M660,632 L660,700 L162,700 L162,784", "none", C.steelS, 9);
lanes += P("M660,632 L660,700 L162,700 L162,784", "none", C.steelF, 5);
lanes += T(672, 690, "64 envs · MJLAB_D1_WORKERS=1 · retry once with a trial offset", {
  size: 9.2,
  fill: C.faint,
  mono: true,
});
lanes += T(672, 706, "one merge pipe carries all eight stores into the chain", {
  size: 9.2,
  fill: C.faint,
});
lanes += `</g>`;
bandB += lanes;

// ---- the trial inset
const ix = 690;
const iy = 372;
const iw = 928;
const ih = 300;
let ins = "";
ins += R(ix, iy, iw, ih, C.pit, { stroke: C.rule, sw: 1, r: 4 });
ins += T(ix + 18, iy + 26, "ONE TRIAL · 250 CONTROL STEPS", { size: 10.2, fill: C.dim, ls: 1.3, weight: 700 });
ins += T(ix + 18, iy + 42, "the standing policy takes over at the clip end, not at a fall", {
  size: 10,
  fill: C.faint,
});
ins += T(ix + 18, iy + 62, "example: this clip ends at step 190", { size: 9.4, fill: C.faint, mono: true });

const tx0 = ix + 30;
const barW = 620;
const tx1 = tx0 + barW;
const sp = barW / 250;
const bY = iy + 104;
const RETAIN = 125;
const CLIPEND = 190;

ins += R(tx0, bY, sp * RETAIN, 32, C.amberF, { stroke: C.amberD, sw: 1 });
ins += R(tx0 + sp * RETAIN, bY, sp * (CLIPEND - RETAIN), 32, C.well, { stroke: C.steelS, sw: 1 });
ins += R(tx0 + sp * CLIPEND, bY, sp * (250 - CLIPEND), 32, C.cyanF, { stroke: C.cyanD, sw: 1 });
ins += T(tx0 + (sp * RETAIN) / 2, bY + 21, "VAE · kept", { size: 10, fill: C.amber, anchor: "middle", weight: 600 });
ins += T(tx0 + sp * (RETAIN + 32), bY + 21, "not indexed", { size: 9.2, fill: C.dim, anchor: "middle" });
ins += T(tx0 + sp * (CLIPEND + 30), bY + 21, "standing ppo", { size: 9.2, fill: C.cyan, anchor: "middle" });

for (let s = 0; s <= 250; s += 25) {
  const x = tx0 + s * sp;
  ins += LINE(x, bY + 32, x, bY + 38, C.faint, 1);
  if (s % 50 === 0) {
    ins += T(x, bY + 50, String(s), {
      size: 8.6,
      fill: C.faint,
      mono: true,
      anchor: s === 0 ? "start" : s === 250 ? "end" : "middle",
    });
  }
}
ins += LINE(tx0 + RETAIN * sp, bY - 12, tx0 + RETAIN * sp, bY + 38, C.amber, 1.2);
ins += T(tx0 + RETAIN * sp + 6, bY - 16, "125 retained", { size: 9.2, fill: C.amber, mono: true });
ins += LINE(tx0 + CLIPEND * sp, bY - 12, tx0 + CLIPEND * sp, bY + 38, C.cyan, 1.2);
ins += T(tx0 + CLIPEND * sp + 6, bY - 16, "clip end → handoff", { size: 9.2, fill: C.cyan, mono: true });

// stored rows vs indexable rows
ins += R(tx0, bY + 66, barW, 11, C.well, { stroke: C.steelS, sw: 1 });
ins += T(tx0, bY + 92, "250 rows written to the store", { size: 9.6, fill: C.dim, mono: true });
ins += R(tx0, bY + 104, sp * RETAIN, 11, C.amberD, { stroke: C.amber, sw: 1 });
ins += T(tx0, bY + 130, "125 rows indexable  (segment_qualified)", { size: 9.6, fill: C.amber, mono: true });

// failure panel, below the bar and clear of it: any step can fail
const px = ix + iw - 276;
const py = iy + 168;
ins += R(px, py, 258, 120, C.rustF, { stroke: C.rustD, sw: 1, r: 3 });
ins += R(px, py, 3, 120, C.rust);
ins += T(px + 14, py + 20, "FAILURE REMOVES THE TRIAL", { size: 9.4, fill: C.rust, weight: 700, ls: 0.4 });
ins += T(px + 14, py + 38, "fall · reset · teleport ·", { size: 9.2, fill: C.dim, mono: true });
ins += T(px + 14, py + 51, "tracking rejection · non-finite", { size: 9.2, fill: C.dim, mono: true });
ins += T(px + 14, py + 69, "865 trials (2.33 %) no window", { size: 9.2, fill: C.rust, mono: true });
ins += T(px + 14, py + 82, "failed rows stay in the store", { size: 9, fill: C.faint, mono: true });
ins += T(px + 14, py + 95, "with segment_qualified = false", { size: 9, fill: C.faint, mono: true });
ins += T(px + 14, py + 111, "no rebuild can recover them", { size: 9, fill: C.faint, mono: true });

ins += LINE(ix + 18, iy + ih - 46, px - 22, iy + ih - 46, C.rule, 1);
ins += T(ix + 18, iy + ih - 26, "qualified = no failure code and steps == 250", { size: 9.6, fill: C.dim, mono: true });
ins += T(ix + 18, iy + ih - 10, "vae_steps can be shorter than 125 when the clip ends, or when the trial fails", {
  size: 9.6,
  fill: C.faint,
  mono: true,
});
bandB += ins;

// ========================================================= band C: chain ====

const CYT = 760;
const CHH = 92;
let bandC = "";
bandC += T(196, 742, "C · OFFLINE CHAIN", { size: 9.6, fill: C.faint, ls: 2, weight: 700 });
bandC += T(340, 742, "no robot in the loop: inspect, replay, build, statistics, cache", { size: 11, fill: C.faint });

const SX = [60, 322, 584, 846, 1108, 1368];
const SWD = 226;

// C1 merge
let c1 = `<g>`;
c1 += T(92, 14, "8 stores + retries", { size: 9, fill: C.faint, anchor: "end", mono: true });
c1 += P(`M24,30 L180,30 L158,54 L46,54 Z`, C.steelT, C.ink, 1);
for (let i = 0; i < 8; i += 1) {
  c1 += LINE(38 + i * 18, 30, 82 + i * 5, 54, C.ruleLit, 1);
}
c1 += solid(78, 54, 48, 24, { dx: 5, dy: -5, front: C.steelS });
c1 += P(`M84,78 L120,78 L132,88 L72,88 Z`, C.steelF, C.ink, 1);
for (let i = 0; i < 5; i += 1) {
  c1 += R(80 + i * 11, 78, 3, 10, i === 0 ? C.amber : C.steelL, { stroke: C.ink, sw: 0.5 });
}
c1 += solid(52, 88, 100, 8, { dx: 5, dy: -5 });
c1 += `</g>`;
bandC += station("merge", SX[0], CYT, SWD, CHH, "MERGE", "→ store/ · 2 241 shards", "diffusion_merge", c1);

// C2 inspect: dial gauge
let c2 = `<g>`;
c2 += CIR(113, 46, 40, C.steelF, { stroke: C.ink, sw: 1 });
c2 += CIR(113, 46, 33, C.pit, { stroke: C.ruleLit, sw: 1 });
for (let i = 0; i <= 12; i += 1) {
  const a = (-150 + i * 25) * (Math.PI / 180);
  c2 += LINE(
    113 + Math.cos(a) * 27,
    46 + Math.sin(a) * 27,
    113 + Math.cos(a) * 32,
    46 + Math.sin(a) * 32,
    i === 8 ? C.amber : C.faint,
    1.1,
  );
}
c2 += LINE(113, 46, 133, 30, C.amber, 1.8);
c2 += CIR(113, 46, 2.6, C.amber);
c2 += solid(103, 86, 20, 6, { dx: 4, dy: -4, front: C.steelL });
c2 += `</g>`;
bandC += station("inspect", SX[1], CYT, SWD, CHH, "INSPECT", "schema + evidence", "inspect.json", c2);

// C3 replay: two reels
let c3 = `<g>`;
c3 += CIR(64, 44, 26, C.steelF, { stroke: C.ink, sw: 1 });
c3 += CIR(162, 44, 26, C.steelF, { stroke: C.ink, sw: 1 });
c3 += CIR(64, 44, 9, C.pit, { stroke: C.ruleLit, sw: 1 });
c3 += CIR(162, 44, 9, C.pit, { stroke: C.ruleLit, sw: 1 });
[0, 60, 120, 180, 240, 300].forEach((d) => {
  const a = (d * Math.PI) / 180;
  c3 += LINE(
    64 + Math.cos(a) * 12,
    44 + Math.sin(a) * 12,
    64 + Math.cos(a) * 24,
    44 + Math.sin(a) * 24,
    C.ruleLit,
    1.1,
  );
});

c3 += R(92, 38, 42, 11, C.amberF, { stroke: C.amberD, sw: 1 });
c3 += LINE(90, 44, 64, 44, C.amberD, 1.4);
c3 += LINE(136, 44, 162, 44, C.amberD, 1.4);
c3 += solid(42, 78, 148, 8, { dx: 5, dy: -5 });
c3 += T(113, 20, "re-run", { size: 9, fill: C.faint, anchor: "middle", mono: true });
c3 += `</g>`;
bandC += station("replay", SX[2], CYT, SWD, CHH, "REPLAY", "every action re-run", "replay.json", c3);

// C4 window index: 41-tooth comb
let c4 = `<g>`;
c4 += T(113, 10, "41 rows · stride 1", { size: 9, fill: C.faint, anchor: "middle", mono: true });
c4 += solid(10, 20, 206, 10, { dx: 5, dy: -5, front: C.steelHi, top: C.steelTop });
c4 += `<g class="comb" style="transform-box:fill-box;transform-origin:50% 0">`;
for (let i = 0; i < 41; i += 1) {
  c4 += R(16 + i * 4.66, 30, 2.6, 32, i < 7 ? "#b9c1cb" : i === 7 ? C.amber : C.steelHi, {});
}
c4 += `</g>`;
c4 += R(10, 62, 206, 7, C.steelF, { stroke: C.ink, sw: 1 });
c4 += P(`M14,69 L30,86 L196,86 L212,69 Z`, C.amberF, C.amberD, 1);
for (let i = 0; i < 9; i += 1) {
  c4 += R(34 + i * 18, 74, 12, 8, C.amberD, { stroke: C.amber, sw: 0.7 });
}
c4 += `</g>`;
bandC += station("index", SX[3], CYT, SWD, CHH, "WINDOW INDEX", "2 322 881 windows", "build · dataset.py:334", c4);

// C5 statistics: a balance, loaded on the train side only
let c5 = `<g>`;
c5 += T(34, 16, "train", { size: 8.6, fill: C.amber, anchor: "middle", mono: true });
c5 += T(192, 16, "hold", { size: 8.6, fill: C.cyan, anchor: "middle", mono: true });
c5 += R(24, 21, 178, 4, C.steelHi, { stroke: C.ink, sw: 1 });
c5 += LINE(113, 25, 113, 78, C.steelL, 3);
c5 += R(97, 78, 32, 6, C.steelF, { stroke: C.ink, sw: 1 });
c5 += LINE(34, 25, 34, 42, C.ruleLit, 1.2);
c5 += LINE(192, 25, 192, 42, C.ruleLit, 1.2);
c5 += R(20, 42, 28, 6, C.steelF, { stroke: C.ink, sw: 1 });
c5 += R(178, 42, 28, 6, C.steelF, { stroke: C.ink, sw: 1 });
c5 += R(23, 35, 22, 7, C.amberF, { stroke: C.amber, sw: 1 });
c5 += R(26, 28, 16, 7, C.amberF, { stroke: C.amberD, sw: 1 });
c5 += `</g>`;
bandC += station("stats", SX[4], CYT, SWD, CHH, "STATISTICS", "0.8 / 0.1 / 0.1 by group", "stats.json", c5);

// C6 token cache: press, ram, token on an anvil, tick
let c6 = `<g>`;
c6 += solid(62, 12, 52, 22, { dx: 5, dy: -5, front: C.steelHi, top: C.steelTop });
c6 += R(83, 34, 10, 8, C.steelS, { stroke: C.ink, sw: 1 });
c6 += R(66, 42, 68, 32, C.cyanF, { stroke: C.cyanD, sw: 1 });
c6 += T(100, 58, "41×167", { size: 9.4, fill: C.cyan, anchor: "middle", mono: true });
c6 += T(100, 70, "f32", { size: 8.6, fill: C.cyan, anchor: "middle", mono: true });
c6 += solid(48, 76, 112, 10, { dx: 5, dy: -5, front: C.steelF });
c6 += CIR(150, 40, 8, "none", { stroke: C.cyan, sw: 1.5 });
c6 += P(`M146,40 L149,43 L155,35`, "none", C.cyan, 1.6);
c6 += `</g>`;
bandC += station("cache", SX[5], CYT, SWD, CHH, "TOKEN CACHE", "train 50 948 171 084 B", "build-token-cache · validate", c6);

bandC += rail(52, 1632, 930, { step: 22 });
for (let i = 0; i < 5; i += 1) {
  const x = SX[i] + SWD + 14;
  bandC += P(`M${x},924 L${x + 9},930 L${x},936 Z`, C.ruleLit);
}

// ======================================================== band D: ledger ====

let bandD = "";
bandD += T(52, 972, "D · LEDGER", { size: 9.6, fill: C.faint, ls: 2, weight: 700 });
const cells = [
  ["PLANNED TRIALS", "37 152", C.barG, 13],
  ["INDEXED SEGMENTS", "36 287", C.barA, 13],
  ["NO WINDOW", "865 · 2.33 %", C.barR, 13],
  ["STORE ROWS", "9 160 010", C.barG, 13],
  ["INDEXABLE ROWS", "3 774 361", C.barA, 13],
  ["WINDOWS", "2 322 881", C.barA, 13],
  ["TRAIN / VAL / TEST", "1 860 237 · 231 470 · 231 174", C.barC, 8.8],
  ["GROUPS", "18 280 of 18 576", C.barG, 13],
];
cells.forEach(([label, value, tone, size], i) => {
  const x = 60 + i * 197;
  bandD += R(x, 982, 184, 54, "#f7f9fb", { stroke: C.ruleLit, sw: 1, r: 3 });
  bandD += R(x, 982, 184, 3, tone);
  bandD += T(x + 11, 1001, label, { size: 8.5, fill: C.dim, ls: 0.9, weight: 600 });
  bandD += T(x + 11, 1022, value, { size, mono: true, weight: 500 });
});

// ============================================================== footer =====

let foot = "";
foot += LINE(52, 1044, 1632, 1044, C.rule, 1);
foot += T(52, 1065, "MEASURED", { size: 8.5, fill: C.dim, ls: 1.4, weight: 700 });
foot += T(
  122,
  1065,
  "one store built 2026-10-20 · source fxy/test/flow-matching-baseline @ de82a8a56 · counts from window_index.npz",
  { size: 9.4, fill: C.faint, mono: true },
);
foot += T(52, 1084, "READ", { size: 8.5, fill: C.dim, ls: 1.4, weight: 700 });
foot += T(
  122,
  1084,
  "a fall fails the trial; the standing policy runs at the clip end.  no window holds a clip opening: earliest row 21, earliest target 30.",
  { size: 9.4, fill: C.faint },
);
foot += T(1632, 1065, "hover a station", { size: 9.4, fill: C.faint, anchor: "end" });
foot += `<text id="ro" x="1632" y="1084" font-size="9.4" fill="${C.dim}" text-anchor="end" font-family="${MONO}">${esc(
  "37 152 trials · 8 workers · 64 envs · 125 of 250 steps kept",
)}</text>`;

// =========================================================== assemble =====

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-labelledby="ttl dsc">
<title id="ttl">X2 diffusion D1 data collection pipeline</title>
<desc id="dsc">Schedule multiplier, eight parallel collectors, one trial timeline of 250 control steps with 125 retained steps, then merge, inspect, replay, window index, statistics and token caches. 37,152 planned trials produce 36,287 indexed segments and 2,322,881 windows.</desc>
<defs>
  <filter id="cardSh" x="-3%" y="-3%" width="106%" height="106%">
    <feDropShadow dx="0" dy="5" stdDeviation="9" flood-color="#25303d" flood-opacity="0.10"/>
  </filter>
  <linearGradient id="bgG" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#f7f9fb"/>
  </linearGradient>
  <style>
    text{font-family:${SANS}}
    .st{cursor:pointer}
    .st .plate{transition:fill .18s ease,stroke .18s ease}
    .st:hover .plate,.st:focus .plate{fill:#eef3f9;stroke:#8d97a2}
    .st:focus{outline:none}
    .flow{animation:flow 2.8s linear infinite}
    @keyframes flow{to{stroke-dashoffset:-44}}
    .spin{animation:spin 40s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
    .comb{animation:comb 3.6s ease-in-out infinite}
    @keyframes comb{0%,100%{transform:translateX(0)}50%{transform:translateX(4.66px)}}
    @media (prefers-reduced-motion:reduce){
      .flow,.spin,.comb{animation:none}
      .st .plate{transition:none}
    }
  </style>
</defs>
<rect width="${W}" height="${H}" fill="${C.page}"/>
<rect x="16" y="16" width="${W - 32}" height="${H - 32}" rx="14" fill="url(#bgG)" stroke="${C.rule}" stroke-width="1" filter="url(#cardSh)"/>
${head}
${bandA}
${bandB}
${bandC}
${bandD}
${foot}
<script><![CDATA[
(function(){
  var ro=document.getElementById('ro');
  var d={
    inputs:'contract 125/250 · manifest 144 clips · 3 artifacts',
    starts:'3 start frames per clip · 153 distinct values',
    phase:'clean and OU, pair-local',
    seeds:'43 seeds, 0 to 42',
    trials:'37 152 planned = 18 576 groups x 2 phases',
    lanes:'8 workers · 64 envs · 18 clips · 4700 trial cap',
    merge:'8 stores merged · 2 241 shards',
    inspect:'schema, terminal evidence, finiteness',
    replay:'every stored action re-run',
    index:'41-row windows, stride 1 · 85 per 125 rows',
    stats:'train-split mean and std · projection bundle',
    cache:'41 x 167 f32 · train cache 50 948 171 084 B'};
  document.querySelectorAll('.st').forEach(function(g){
    var k=g.getAttribute('data-i');
    function on(){ if(d[k]) ro.textContent=d[k]; }
    g.addEventListener('mouseenter',on);
    g.addEventListener('focus',on);
  });
})();
]]></script>
</svg>
`;

writeFileSync(OUT, svg);
console.log(`wrote ${OUT} (${(svg.length / 1024).toFixed(1)} KiB)`);
