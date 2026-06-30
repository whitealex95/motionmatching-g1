// Jump-skill indexing for the obstacle-triggered jump (the EMM "jump bucket").
// A 1:1 port of emm_g1/jumps.py: locomotion and the jump live in SEPARATE buckets;
// the search runs over locomotion frames only and the jump is a distinct skill
// entered through its run-up ('ready') frames and ridden through flight + landing.
// Take-off / apex / landing of each jump clip are detected from the pelvis-height
// (rootPos z) hop, since the exported DB carries no jump-phase tags.

const READY_LEN = 6;     // run-up frames before take-off offered as jump entries
const HOP_RISE = 0.06;   // pelvis-z rise (m) above the clip's walking baseline => airborne
const POST_LAND = 25;    // frames ridden after landing (PHASE_TOUCHDOWN + PHASE_AFTER + 1)
const SKIP_SUBSTR = ['stop'];   // exclude non-continuing jumps (they halt; bad for a lane)

function median(a) {
  const s = a.slice().sort((x, y) => x - y);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

// A: loaded DB (has starts, stops, clip_is_jump, rootPos). clipNames: meta.clip_names.
// Returns { enter:[frames], landOf:Map, endOf:Map, apexOf:Map }.
export function jumpIndex(A, clipNames) {
  const starts = A.starts, stops = A.stops, cij = A.clip_is_jump;
  const enter = [], landOf = new Map(), endOf = new Map(), apexOf = new Map();
  for (let ci = 0; ci < starts.length; ci++) {
    if (cij[ci] !== 1) continue;
    if (SKIP_SUBSTR.some((s) => String(clipNames[ci]).includes(s))) continue;
    const rs = starts[ci], re = stops[ci], n = re - rs;
    const z = new Array(n);
    for (let i = 0; i < n; i++) z[i] = A.rootPos[(rs + i) * 3 + 2];
    const base = median(z);
    const above = z.map((v) => v > base + HOP_RISE);
    if (!above.some(Boolean)) continue;
    let pk = 0;
    for (let i = 1; i < n; i++) if (z[i] > z[pk]) pk = i;
    let to = pk; while (to > 0 && above[to - 1]) to--;
    let la = pk; while (la < n - 1 && above[la + 1]) la++;
    const takeoff = rs + to, land = rs + la, apex = rs + pk;
    const end = Math.min(land + POST_LAND, re - 1);
    for (let f = Math.max(rs, takeoff - READY_LEN); f < takeoff; f++) {
      enter.push(f); landOf.set(f, land); endOf.set(f, end); apexOf.set(f, apex);
    }
  }
  return { enter, landOf, endOf, apexOf };
}
