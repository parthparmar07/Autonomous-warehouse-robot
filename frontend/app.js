const { useState, useEffect, useRef, useCallback, useMemo } = React;

const API_URL = window.WAREHOUSE_API_URL || "http://127.0.0.1:8000";
const COMPARE_ALGORITHMS = ["nearest_task", "iql", "vdn", "qmix"];
const ALGO_LABEL = { nearest_task: "Planner", shortest_path: "Shortest path", iql: "IQL", vdn: "VDN", qmix: "QMIX", dqn: "DQN" };

const GRID_FREE = 0;
const GRID_SHELF = 1;
const GRID_SAFETY_CORRIDOR = 2;
const GRID_PICKUP = 3;
const GRID_DELIVERY = 4;

const ORIENTATION_ANGLE = { 0: -90, 1: 0, 2: 90, 3: 180 };

function robotColor(id) {
  const hue = (id * 137.508) % 360; // golden-angle spacing -> well-separated hues at any N
  return `hsl(${hue.toFixed(0)}, 62%, 48%)`;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpAngle(a, b, t) {
  const diff = (((b - a + 540) % 360) + 360) % 360 - 180;
  return a + diff * t;
}

async function apiPost(path, body) {
  const response = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail ? JSON.stringify(detail.detail) : `${path} failed (${response.status})`);
  }
  return response.json();
}

async function apiGet(path) {
  const response = await fetch(`${API_URL}${path}`);
  if (!response.ok) throw new Error(`${path} failed (${response.status})`);
  return response.json();
}

// ---------------------------------------------------------------------------
// Help text: tooltips, presets and friendly errors
// ---------------------------------------------------------------------------

const ALGO_INFO = {
  nearest_task: "Planner (baseline): a hand-written central dispatcher that sees the whole map. Best possible reference, but needs central control.",
  shortest_path: "Shortest path: like the planner but without reservations, so robots can block each other.",
  iql: "IQL: each robot learns on its own from the team reward. No teamwork in training.",
  vdn: "VDN: robots learn together; the team value is the sum of every robot's value.",
  qmix: "QMIX (our main model): robots learn together through a mixer that only rewards a robot for helping the team. Each robot still decides from its own sensors.",
  dqn: "DQN: compatibility name for independent Q-learning.",
};

const TIPS = {
  algorithm: "Which brain drives the robots. Planner = hand-written baseline; IQL, VDN and QMIX = trained neural networks.",
  fleet: "How many robots share the warehouse. Trained models exist for 5, 10 and 20 robots (mixed package flow).",
  flow: "Shelf to shipping: only outbound orders. Receiving, storage and shipping: packages arrive, are stored on racks, then shipped (the setting our 10- and 20-robot models were trained on).",
  arrival: "How often a new package arrives at receiving. Smaller = busier warehouse.",
  tasks: "How many outbound orders are open at once.",
  battery: "Starting charge of every robot. Lower = robots must plan charging sooner.",
  front: "How much each robot's sensor sees in front of it.",
  rear: "Whether the robot can also see behind it.",
  seed: "Random seed. Same seed = exactly the same warehouse, orders and batteries, so runs are comparable.",
  speed: "Milliseconds per simulation step. Lower = faster animation.",
  reset: "Restart the shift with the current settings.",
  step: "Advance the simulation by exactly one tick.",
  play: "Run the simulation continuously.",
  view3d: "3D view of the same live simulation. Drag to rotate, scroll to zoom.",
  steps: "Simulation ticks so far. A shift is 600 ticks.",
  delivered: "Orders (or handling jobs) completed so far.",
  queued: "Orders waiting to be picked up or delivered.",
  collisions: "Times two robots tried to enter the same cell. The move is blocked, so no robot actually crashes.",
  blocks: "Times a robot tried to enter a worker corridor and the safety layer stopped it. Trained models should stay at 0.",
  reward: "The training score of the shift: + for deliveries, − for energy, delays, conflicts and battery failures.",
  sensor: "Exactly what one robot 'sees': a 5×5 window around it. This is the only input the trained network gets.",
  fleetTable: "Live status of every robot: where it is, what it's doing, its battery and its current order.",
};

const PRESETS = [
  { key: "planner5", label: "Planner · 5 robots", tip: "Start here: the hand-written baseline in a normal warehouse.", cfg: { algorithm: "nearest_task", numAgents: 5, workload: "mixed", arrivalInterval: 20 } },
  { key: "qmix5", label: "QMIX · 5 robots", tip: "Our trained model doing the same job, using only local sensors (99.8% sealed completion).", cfg: { algorithm: "qmix", numAgents: 5, workload: "mixed", arrivalInterval: 20 } },
  { key: "qmix10", label: "QMIX · 10 robots, busy", tip: "Twice the robots and twice the demand: where QMIX beat IQL by 15 points.", cfg: { algorithm: "qmix", numAgents: 10, workload: "mixed", arrivalInterval: 10 } },
  { key: "iql10", label: "IQL · 10 robots, busy", tip: "Same busy warehouse with independent learners, for comparison with QMIX.", cfg: { algorithm: "iql", numAgents: 10, workload: "mixed", arrivalInterval: 10 } },
  { key: "qmix20", label: "QMIX · 20 robots, stress", tip: "20 robots and 2 chargers under heavy demand: the limit of the system.", cfg: { algorithm: "qmix", numAgents: 20, workload: "mixed", arrivalInterval: 8 } },
];

function friendlyError(message, algorithm, numAgents, workload) {
  if (!message) return null;
  if (message.includes("No compatible")) {
    const hint = workload === "outbound"
      ? "Switch Package flow to “Receiving, storage and shipping” — trained models exist there for 5, 10 and 20 robots — or pick the Planner."
      : "Trained models exist for QMIX and IQL at 5, 10 and 20 robots and VDN at 5. Pick one of those, or the Planner.";
    return `No trained ${algorithm.toUpperCase()} model for ${numAgents} robots with this package flow. ${hint}`;
  }
  if (message.includes("Failed to fetch")) return `Cannot reach the decision service at ${API_URL}. Start it with scripts\\demo.ps1.`;
  return message;
}

function Tip({ text, children, wide }) {
  return (
    <span className={`tip ${wide ? "wide" : ""}`} data-tip={text} tabIndex={0}>
      {children ?? <span className="tip-i">i</span>}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Canvas drawing primitives
// ---------------------------------------------------------------------------

function drawFreeCell(ctx, x, y, size) {
  ctx.fillStyle = "#fbfaf7";
  ctx.fillRect(x, y, size, size);
  ctx.strokeStyle = "#ece9e1";
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
}

function drawShelfCell(ctx, x, y, size) {
  // A top-down pallet rack: a warm wood-toned frame with a grid of stacked boxes,
  // rather than a flat block, so it reads as shelving rather than a generic obstacle.
  ctx.fillStyle = "#8a6a45";
  ctx.fillRect(x, y, size, size);
  const pad = size * 0.1;
  const inner = size - pad * 2;
  const cols = 2;
  const rows = 2;
  const gap = size * 0.06;
  const cellW = (inner - gap * (cols - 1)) / cols;
  const cellH = (inner - gap * (rows - 1)) / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const bx = x + pad + c * (cellW + gap);
      const by = y + pad + r * (cellH + gap);
      ctx.fillStyle = "#c9a06b";
      ctx.fillRect(bx, by, cellW, cellH);
      ctx.strokeStyle = "#6b4e30";
      ctx.lineWidth = Math.max(0.5, size * 0.02);
      ctx.strokeRect(bx, by, cellW, cellH);
    }
  }
}

function drawCorridorCell(ctx, x, y, size) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, size, size);
  ctx.clip();
  ctx.fillStyle = "#2b2306";
  ctx.fillRect(x, y, size, size);
  ctx.strokeStyle = "#f5c518";
  ctx.lineWidth = size * 0.24;
  for (let i = -size; i < size * 2; i += size * 0.5) {
    ctx.beginPath();
    ctx.moveTo(x + i, y + size);
    ctx.lineTo(x + i + size, y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawGrid(ctx, grid, cellPx) {
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) {
      const cell = grid[y][x];
      const px = x * cellPx;
      const py = y * cellPx;
      if (cell === GRID_SHELF) drawShelfCell(ctx, px, py, cellPx);
      else if (cell === GRID_SAFETY_CORRIDOR) drawCorridorCell(ctx, px, py, cellPx);
      else if (cell === GRID_PICKUP || cell === GRID_DELIVERY) {
        ctx.fillStyle = cell === GRID_PICKUP ? "#d6efe3" : "#e8e1fa";
        ctx.fillRect(px, py, cellPx, cellPx);
        ctx.strokeStyle = cell === GRID_PICKUP ? "#1a8a5f" : "#6d5bd0";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(px + 2, py + 2, cellPx - 4, cellPx - 4);
      }
      else if (cell === 5 || cell === 6) {
        ctx.fillStyle = cell === 5 ? "#f8dd9a" : "#c9e8f5";
        ctx.fillRect(px, py, cellPx, cellPx);
        ctx.strokeStyle = cell === 5 ? "#966800" : "#226b8c";
        ctx.strokeRect(px + 2, py + 2, cellPx - 4, cellPx - 4);
      }
      else drawFreeCell(ctx, px, py, cellPx);
    }
  }
}

function drawStations(ctx, state, size) {
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `700 ${Math.max(8, size * 0.33)}px Inter, sans-serif`;
  const layout = state.layout || {};
  for (const [key, prefix, color] of [["receiving_bays", "R", "#b55b18"], ["pickup_bays", "P", "#156b4b"], ["drop_bays", "D", "#5643a8"], ["charger_bays", "C", "#966800"], ["transfer_bays", "T", "#226b8c"]]) {
    ctx.fillStyle = color;
    (layout[key] || []).forEach(([x, y], index) => {
      ctx.fillText(`${prefix}${index + 1}`, (x + 0.5) * size, (y + 0.5) * size);
    });
  }
  // A badge counts reserved packages on a rack, not loose parcels on the floor.
  const waiting = new Map();
  if (state.inventory) {
    Object.values(state.inventory).filter(item => item.state === "shelf").forEach(item => {
      const [bx, by] = item.storage_bay;
      const shelf = [[bx+1,by],[bx-1,by],[bx,by+1],[bx,by-1]].find(([x,y]) => state.grid?.[y]?.[x] === 1);
      if (shelf) { const key = shelf.join(","); waiting.set(key, (waiting.get(key) || 0) + 1); }
    });
  } else {
    (state.task_queue || []).filter((task) => task.status === "pending" && task.shelf).forEach((task) => {
      const key = task.shelf.join(",");
      waiting.set(key, (waiting.get(key) || 0) + 1);
    });
  }
  waiting.forEach((count, key) => {
    const [x, y] = key.split(",").map(Number);
    ctx.fillStyle = "#fff0c2";
    ctx.fillRect((x + 0.12) * size, (y + 0.12) * size, size * 0.76, size * 0.76);
    ctx.fillStyle = "#66400b";
    ctx.fillText(String(count), (x + 0.5) * size, (y + 0.5) * size);
  });
  ctx.restore();
}

function drawRouteLine(ctx, fromX, fromY, toX, toY, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.55;
  ctx.lineWidth = 1.6;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.stroke();
  ctx.restore();
}

function drawTrail(ctx, points, cellPx, color) {
  if (points.length < 2) return;
  ctx.save();
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1];
    const b = points[i];
    const alpha = (i / points.length) * 0.3;
    ctx.strokeStyle = color;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = Math.max(1.5, cellPx * 0.1);
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(a.x * cellPx + cellPx / 2, a.y * cellPx + cellPx / 2);
    ctx.lineTo(b.x * cellPx + cellPx / 2, b.y * cellPx + cellPx / 2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawRobot(ctx, cx, cy, r, angleDeg, color, loaded, id, showLabel) {
  ctx.save();
  ctx.shadowColor = "rgba(20,20,20,0.25)";
  ctx.shadowBlur = r * 0.6;
  ctx.shadowOffsetY = r * 0.18;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  ctx.restore();

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate((angleDeg * Math.PI) / 180);
  ctx.beginPath();
  ctx.arc(0, 0, r * 0.82, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = Math.max(2, r * 0.24);
  ctx.strokeStyle = loaded ? "#e0a526" : "rgba(255,255,255,0.9)";
  ctx.stroke();

  const tipX = r * 1.05;
  const baseW = r * 0.42;
  ctx.beginPath();
  ctx.moveTo(tipX, 0);
  ctx.lineTo(0, baseW);
  ctx.lineTo(0, -baseW);
  ctx.closePath();
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  ctx.restore();

  if (showLabel && r > 7) {
    ctx.fillStyle = "#6b6b6f";
    ctx.font = `700 ${Math.max(8, r * 0.7)}px Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(id), cx, cy - r * 1.85);
  }
}

function drawPulse(ctx, cx, cy, cellPx, progress) {
  const radius = cellPx * (0.3 + progress * 1.6);
  ctx.save();
  ctx.globalAlpha = Math.max(0, 1 - progress);
  ctx.strokeStyle = "#e0a526";
  ctx.lineWidth = Math.max(1.5, cellPx * 0.12);
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Session hook: owns one backend session (reset/tick), tracks history for
// smooth interpolation, trails, delivery pulses and per-robot activity.
// ---------------------------------------------------------------------------

function useSimSession(sessionId) {
  const [state, setState] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const animRef = useRef({ prev: null, curr: null, tickStartedAt: 0, tickDuration: 400 });
  const trailsRef = useRef(new Map());
  const pulsesRef = useRef([]);
  const activityRef = useRef(new Map()); // robotId -> cells moved this episode
  const resetQueue = useRef(Promise.resolve()); // resets run in order: a slow older reset must not overwrite a newer one

  const reset = useCallback(
    (...args) => {
      const run = resetQueue.current.then(() => doReset(...args));
      resetQueue.current = run.catch(() => {});
      return run;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sessionId]
  );

  const doReset = useCallback(
    async (algorithm, numAgents, numTasks, seed, options = {}) => {
      setError(null);
      try {
        const result = await apiPost(`/reset?session_id=${sessionId}`, {
          algorithm,
          num_agents: Number(numAgents),
          num_tasks: Number(numTasks),
          seed: Number(seed),
          ...options,
        });
        animRef.current = { prev: result, curr: result, tickStartedAt: performance.now(), tickDuration: 400 };
        trailsRef.current = new Map();
        pulsesRef.current = [];
        activityRef.current = new Map();
        setHistory([{ reward: 0, completed: 0, collisions: 0, blocked: 0, step: 0 }]);
        setState(result);
        return result;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [sessionId]
  );

  const tick = useCallback(
    async (speedMs) => {
      try {
        const result = await apiPost(`/tick?session_id=${sessionId}`);
        const prev = animRef.current.curr || result;
        animRef.current = { prev, curr: result, tickStartedAt: performance.now(), tickDuration: speedMs };

        result.robots.forEach((robot) => {
          const trail = trailsRef.current.get(robot.id) || [];
          trail.push({ x: robot.x, y: robot.y });
          if (trail.length > 7) trail.shift();
          trailsRef.current.set(robot.id, trail);

          const before = prev.robots.find((r) => r.id === robot.id);
          if (before && (before.x !== robot.x || before.y !== robot.y)) {
            activityRef.current.set(robot.id, (activityRef.current.get(robot.id) || 0) + 1);
          }
          if (before && before.load_status === 1 && robot.load_status === 0) {
            pulsesRef.current.push({ x: robot.x, y: robot.y, startedAt: performance.now() });
          }
        });

        setHistory((h) => [
          ...h.slice(-59),
          {
            reward: result.metrics?.reward ?? 0,
            completed: result.completed_tasks ?? 0,
            collisions: result.metrics?.collision_count ?? 0,
            blocked: result.metrics?.safety_overrides ?? 0,
            step: result.metrics?.step_count ?? 0,
          },
        ]);
        setState(result);
        setError(null);
        return result;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [sessionId]
  );

  return { state, error, history, reset, tick, animRef, trailsRef, pulsesRef, activityRef };
}

// ---------------------------------------------------------------------------
// Warehouse canvas
// ---------------------------------------------------------------------------

function WarehouseCanvas({ animRef, trailsRef, pulsesRef, cellPx = 26, showLabels = true, showRoutes = true }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const render = () => {
      const { prev, curr, tickStartedAt, tickDuration } = animRef.current;
      if (curr && curr.grid) {
        const [width, height] = curr.grid_size;
        const targetW = width * cellPx;
        const targetH = height * cellPx;
        if (canvas.width !== targetW || canvas.height !== targetH) {
          canvas.width = targetW;
          canvas.height = targetH;
        }

        const t = Math.min(1, Math.max(0, (performance.now() - tickStartedAt) / tickDuration));

        drawGrid(ctx, curr.grid, cellPx);

        drawStations(ctx, curr, cellPx);

        trailsRef.current.forEach((points, id) => drawTrail(ctx, points, cellPx, robotColor(id)));

        curr.robots.forEach((robot) => {
          const before = (prev && prev.robots.find((r) => r.id === robot.id)) || robot;
          const x = lerp(before.x, robot.x, t);
          const y = lerp(before.y, robot.y, t);

          if (showRoutes && robot.target) {
            drawRouteLine(ctx, x * cellPx + cellPx / 2, y * cellPx + cellPx / 2, robot.target[0] * cellPx + cellPx / 2, robot.target[1] * cellPx + cellPx / 2, robotColor(robot.id));
          }

          const angle = lerpAngle(ORIENTATION_ANGLE[before.orientation] ?? 0, ORIENTATION_ANGLE[robot.orientation] ?? 0, t);
          drawRobot(ctx, x * cellPx + cellPx / 2, y * cellPx + cellPx / 2, cellPx * 0.36, angle, robotColor(robot.id), robot.load_status === 1, robot.id, showLabels);
        });

        const now = performance.now();
        pulsesRef.current = pulsesRef.current.filter((p) => now - p.startedAt < 700);
        pulsesRef.current.forEach((p) => {
          const progress = (now - p.startedAt) / 700;
          drawPulse(ctx, p.x * cellPx + cellPx / 2, p.y * cellPx + cellPx / 2, cellPx, progress);
        });
      }
      rafRef.current = requestAnimationFrame(render);
    };

    rafRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(rafRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cellPx]);

  return <canvas ref={canvasRef} />;
}

// ---------------------------------------------------------------------------
// 3D view (three.js) of the same live simulation state
// ---------------------------------------------------------------------------

const CELL_3D = {
  1: { color: "#8a6a45", h: 1.15 }, // shelf / rack
  2: { color: "#f5c518", h: 0.05 }, // worker safety corridor
  3: { color: "#1a8a5f", h: 0.06 }, // pickup / receiving bay
  4: { color: "#6d5bd0", h: 0.06 }, // drop / shipping bay
  5: { color: "#c99a3f", h: 0.35 }, // charger
  6: { color: "#3a6fd8", h: 0.06 }, // transfer bay
  7: { color: "#d8d6cf", h: 0.03 }, // staging / parking
};

function Warehouse3D({ animRef, height = 560 }) {
  const mountRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    const T = window.THREE;
    if (!T) {
      mount.textContent = "The 3D view needs three.js from unpkg.com — check the internet connection and reload.";
      return undefined;
    }
    const renderer = new T.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.shadowMap.enabled = true;
    mount.appendChild(renderer.domElement);
    const scene = new T.Scene();
    scene.background = new T.Color("#f3f2ee");
    const camera = new T.PerspectiveCamera(42, 1, 0.1, 500);
    scene.add(new T.HemisphereLight(0xffffff, 0x8f8f86, 0.9));
    const sun = new T.DirectionalLight(0xffffff, 0.65);
    sun.position.set(14, 30, 22);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -20, right: 20, top: 20, bottom: -20 });
    scene.add(sun);

    const world = new T.Group();
    scene.add(world);
    const robots = new Map();
    const view = { theta: Math.PI * 0.3, phi: 0.9, r: 30, target: new T.Vector3(10, 0, 10) };
    let builtFrom = null;

    const resize = () => {
      const w = mount.clientWidth || 640;
      renderer.setSize(w, height);
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    const el = renderer.domElement;
    let drag = null;
    el.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; el.setPointerCapture(e.pointerId); });
    el.addEventListener("pointermove", (e) => {
      if (!drag) return;
      view.theta += (e.clientX - drag.x) * 0.008;
      view.phi = Math.min(1.45, Math.max(0.2, view.phi - (e.clientY - drag.y) * 0.006));
      drag = { x: e.clientX, y: e.clientY };
    });
    el.addEventListener("pointerup", () => { drag = null; });
    el.addEventListener("wheel", (e) => { e.preventDefault(); view.r = Math.min(80, Math.max(8, view.r * (e.deltaY > 0 ? 1.08 : 0.92))); }, { passive: false });

    const buildWorld = (state) => {
      world.clear();
      const [w, h] = state.grid_size;
      const floor = new T.Mesh(new T.BoxGeometry(w, 0.1, h), new T.MeshLambertMaterial({ color: "#fbfaf7" }));
      floor.position.set(w / 2 - 0.5, -0.05, h / 2 - 0.5);
      floor.receiveShadow = true;
      world.add(floor);
      const geoms = {}, mats = {};
      state.grid.forEach((row, y) => row.forEach((code, x) => {
        const spec = CELL_3D[code];
        if (!spec) return;
        geoms[code] = geoms[code] || new T.BoxGeometry(code === 1 ? 0.9 : 0.94, spec.h, code === 1 ? 0.9 : 0.94);
        mats[code] = mats[code] || new T.MeshLambertMaterial({ color: spec.color });
        const m = new T.Mesh(geoms[code], mats[code]);
        m.position.set(x, spec.h / 2, y);
        m.castShadow = code === 1 || code === 5;
        m.receiveShadow = true;
        world.add(m);
      }));
      view.target.set(w / 2 - 0.5, 0, h / 2 - 0.5);
      view.r = Math.max(w, h) * 1.35;
      robots.forEach((r) => scene.remove(r.group));
      robots.clear();
    };

    const robotFor = (id) => {
      if (robots.has(id)) return robots.get(id);
      const group = new T.Group();
      const body = new T.Mesh(new T.CylinderGeometry(0.34, 0.38, 0.3, 24), new T.MeshLambertMaterial({ color: new T.Color().setStyle(robotColor(id)) }));
      body.position.y = 0.17;
      body.castShadow = true;
      const nose = new T.Mesh(new T.BoxGeometry(0.22, 0.12, 0.16), new T.MeshLambertMaterial({ color: "#17171a" }));
      nose.position.set(0.3, 0.22, 0);
      const cargo = new T.Mesh(new T.BoxGeometry(0.34, 0.26, 0.34), new T.MeshLambertMaterial({ color: "#d9a35b" }));
      cargo.position.y = 0.46;
      cargo.castShadow = true;
      group.add(body, nose, cargo);
      scene.add(group);
      const entry = { group, cargo };
      robots.set(id, entry);
      return entry;
    };

    let raf;
    const loop = () => {
      const { prev, curr, tickStartedAt, tickDuration } = animRef.current;
      if (curr && curr.grid) {
        if (builtFrom !== curr.grid) { buildWorld(curr); builtFrom = curr.grid; }
        const t = Math.min(1, Math.max(0, (performance.now() - tickStartedAt) / tickDuration));
        const seen = new Set();
        curr.robots.forEach((robot) => {
          const before = (prev && prev.robots.find((r) => r.id === robot.id)) || robot;
          const r = robotFor(robot.id);
          r.group.position.set(lerp(before.x, robot.x, t), 0, lerp(before.y, robot.y, t));
          const deg = lerpAngle(ORIENTATION_ANGLE[before.orientation] ?? 0, ORIENTATION_ANGLE[robot.orientation] ?? 0, t);
          r.group.rotation.y = (-deg * Math.PI) / 180;
          r.cargo.visible = robot.load_status === 1;
          seen.add(robot.id);
        });
        robots.forEach((r, id) => { if (!seen.has(id)) { scene.remove(r.group); robots.delete(id); } });
      }
      camera.position.set(
        view.target.x + view.r * Math.sin(view.phi) * Math.cos(view.theta),
        view.r * Math.cos(view.phi),
        view.target.z + view.r * Math.sin(view.phi) * Math.sin(view.theta)
      );
      camera.lookAt(view.target);
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      renderer.dispose();
      mount.innerHTML = "";
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [height]);

  return <div ref={mountRef} className="scene3d" style={{ height }} title="Drag to rotate · scroll to zoom" />;
}

const LEGEND = [
  [{ background: "#fbfaf7", border: "1px solid #ece9e1" }, "Floor", "Open aisle robots can drive on."],
  [{ background: "#8a6a45" }, "Shelf / rack", "Storage racks. Robots cannot drive through them."],
  [{ background: "#2b2306", border: "1px solid #f5c518" }, "Worker corridor", "Reserved for people. The safety layer blocks every robot move into it, whatever the AI decides."],
  [{ borderRadius: "50%", border: "2px solid #1a8a5f", background: "transparent" }, "P: storage bay", "Where packages are put on or taken off a rack."],
  [{ border: "2px solid #6d5bd0", background: "transparent" }, "D: shipping · R: receiving", "D = outbound shipping bay. R = receiving dock where new packages arrive."],
  [{ background: "#fff0c2", border: "1px solid #66400b" }, "C: charger", "Charging station (only 2 in the whole warehouse). T = handover bay."],
  [{ borderRadius: "50%", border: "2px solid #e0a526", background: "#3a6fd8" }, "Robot", "Each robot has its own colour and number. A gold ring (2D) or a brown box (3D) means it is carrying a package. The dashed line points to its current target."],
];

function Legend() {
  return (
    <div className="legend">
      {LEGEND.map(([style, label, tip]) => (
        <Tip key={label} text={tip}>
          <span className="legend-item">
            <span className="swatch" style={style} />
            {label}
          </span>
        </Tip>
      ))}
    </div>
  );
}

function Sparkline({ data, accessor, color = "#17171a", height = 56 }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    ctx.clearRect(0, 0, width, height);
    if (data.length < 2) return;
    const values = data.map(accessor);
    const min = Math.min(...values, 0);
    const max = Math.max(...values, 1);
    const span = max - min || 1;
    const toXY = (v, i) => [(i / (values.length - 1)) * width, height - ((v - min) / span) * (height - 8) - 4];

    ctx.beginPath();
    ctx.moveTo(0, height);
    values.forEach((v, i) => {
      const [x, y] = toXY(v, i);
      ctx.lineTo(x, y);
    });
    ctx.lineTo(width, height);
    ctx.closePath();
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, `${color}33`);
    gradient.addColorStop(1, `${color}02`);
    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.beginPath();
    values.forEach((v, i) => {
      const [x, y] = toXY(v, i);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.stroke();

    const [lx, ly] = toXY(values[values.length - 1], values.length - 1);
    ctx.beginPath();
    ctx.arc(lx, ly, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }, [data, accessor, color, height]);

  return <canvas ref={canvasRef} width={320} height={height} style={{ width: "100%", height: `${height}px` }} />;
}

// ---------------------------------------------------------------------------
// Stat tiles with real trend deltas (compared to ~5 ticks ago)
// ---------------------------------------------------------------------------

function Trend({ series, higherIsBetter = true }) {
  if (series.length < 6) return <span className="trend flat">--</span>;
  const now = series[series.length - 1];
  const before = series[Math.max(0, series.length - 6)];
  const delta = now - before;
  if (Math.abs(delta) < 1e-9) return <span className="trend flat"><IconMinus size={10} /> 0%</span>;
  const pct = before !== 0 ? Math.abs((delta / before) * 100) : Math.abs(delta) * 100;
  const isUp = delta > 0;
  const good = isUp === higherIsBetter;
  return (
    <span className={`trend ${good ? "up" : "down"}`}>
      {isUp ? <IconTrendUp size={10} /> : <IconTrendDown size={10} />} {pct.toFixed(0)}%
    </span>
  );
}

function StatTile({ icon, value, label, tone, series, higherIsBetter, tip }) {
  return (
    <div className={`stat ${tone || ""}`}>
      <div className="stat-top">
        <span className="stat-icon">{icon}</span>
        {series && <Trend series={series} higherIsBetter={higherIsBetter} />}
      </div>
      <div className="value">{value}</div>
      <div className="label">{label}{tip && <Tip text={tip} />}</div>
    </div>
  );
}

function StatsPanel({ state, history, playing }) {
  const metrics = (state && state.metrics) || {};
  const totalTasks = state ? (state.task_queue || []).length : 0;

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <span className="icon-badge">
            <IconActivity size={15} />
          </span>
          Live Metrics
        </div>
        <span className={`status-line ${playing ? "" : "paused"}`}>
          <span className="status-dot" /> {playing ? "Running" : "Paused"}
        </span>
      </div>
      <div className="stat-grid">
        <StatTile icon={<IconFootprints size={14} />} value={metrics.step_count ?? 0} label="Steps" tip={TIPS.steps} series={history.map((h) => h.step)} higherIsBetter={true} />
        <StatTile icon={<IconPackage size={14} />} value={state ? state.completed_tasks ?? 0 : 0} label={state?.workload_version ? "Handling jobs completed" : "Delivered"} tip={TIPS.delivered} tone="good" series={history.map((h) => h.completed)} higherIsBetter={true} />
        <StatTile icon={<IconList size={14} />} value={totalTasks} label="Tasks Queued" tip={TIPS.queued} />
        <StatTile icon={<IconAlertTriangle size={14} />} value={metrics.collision_count ?? 0} label="Near-collisions blocked" tip={TIPS.collisions} tone={metrics.collision_count > 0 ? "bad" : "good"} series={history.map((h) => h.collisions)} higherIsBetter={false} />
        <StatTile icon={<IconBan size={14} />} value={metrics.safety_overrides ?? 0} label="Corridor Blocks" tip={TIPS.blocks} tone={metrics.safety_overrides > 0 ? "warn" : "good"} series={history.map((h) => h.blocked)} higherIsBetter={false} />
        <StatTile icon={<IconStar size={14} />} value={Number(metrics.reward ?? 0).toFixed(1)} label="Episode Reward" tip={TIPS.reward} series={history.map((h) => h.reward)} higherIsBetter={true} />
      </div>

      <div style={{ marginTop: 16 }}>
        <div className="panel-header" style={{ marginBottom: 6 }}>
          <div className="panel-title" style={{ fontSize: "0.82rem" }}>
            <span className="icon-badge">
              <IconTruck size={13} />
            </span>
            Deliveries Over Time
          </div>
          <Tip text="Completed jobs (y) against simulation steps (x) over the last 60 steps." />
        </div>
        <Sparkline data={history} accessor={(d) => d.completed} color="#1a8a5f" />
      </div>
    </div>
  );
}

function FleetTable({ state, activityRef }) {
  if (!state) return null;
  const maxSteps = Math.max(1, state.metrics?.step_count ?? 1);
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <span className="icon-badge">
            <IconGrid size={14} />
          </span>
          Fleet Status <Tip text={TIPS.fleetTable} />
        </div>
      </div>
      <div className="table-scroll">
      <table className="fleet-table">
        <thead>
          <tr>
            <th><Tip text="Robot number and its colour on the map.">Robot</Tip></th>
            <th><Tip text="Grid cell (x, y). (0,0) is the top-left corner.">Position</Tip></th>
            <th><Tip text="What the robot is doing right now: going to a pickup, delivering, charging, parking…">Status</Tip></th>
            <th><Tip text="Battery charge. Below about 30% the robot heads for one of the 2 chargers.">Battery</Tip></th>
            <th><Tip text="Current order: from bay → to bay (P storage, D shipping, R receiving).">Order route</Tip></th>
            <th><Tip text="Share of steps this robot spent moving (busy vs waiting).">Activity</Tip></th>
          </tr>
        </thead>
        <tbody>
          {state.robots.map((robot) => {
            const moved = activityRef.current.get(robot.id) || 0;
            const task = state.task_queue.find((item) => item.task_id === robot.current_task);
            const pct = Math.min(100, Math.round((moved / maxSteps) * 100));
            return (
              <tr key={robot.id}>
                <td>
                  <span className="fleet-id">
                    <span className="fleet-dot" style={{ background: robotColor(robot.id), color: robotColor(robot.id) }} />#{robot.id}
                  </span>
                </td>
                <td style={{ color: "var(--ink-dim)" }}>
                  ({robot.x}, {robot.y})
                </td>
                <td>
                  <span className={`status-pill ${robot.load_status === 1 ? "carrying" : "idle"}`}>{robot.mode?.toLowerCase().replaceAll("_", " ") || "idle"}</span>
                </td>
                <td title={`${robot.energy_eu?.toFixed(2)} energy units`}><span className={`battery ${robot.energy_eu / robot.capacity_eu < 0.3 ? "low" : ""}`}>{Math.round(100 * robot.energy_eu / robot.capacity_eu)}%</span></td>
                <td>
                  {task ? `${task.pickup_bay || "Pickup"} → ${task.drop_bay || "Drop"}` : "—"}
                </td>
                <td>
                  <div className="activity-bar-wrap">
                    <div className="activity-bar-fill" style={{ width: `${pct}%` }} />
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stepper control
// ---------------------------------------------------------------------------

function Stepper({ value, onChange, min = 1, max = 999, icon, labelText, tip }) {
  const set = (v) => onChange(Math.max(min, Math.min(max, v)));
  return (
    <div className="field">
      <label>
        {icon} {labelText} {tip && <Tip text={tip} />}
      </label>
      <div className="stepper">
        <button type="button" onClick={() => set(Number(value) - 1)}>
          &minus;
        </button>
        <input type="number" value={value} onChange={(e) => set(Number(e.target.value) || min)} />
        <button type="button" onClick={() => set(Number(value) + 1)}>
          +
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Single-algorithm view
// ---------------------------------------------------------------------------

function SensorPanel({ state, sessionId }) {
  const [robotId, setRobotId] = useState(0);
  const [sensor, setSensor] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    let cancelled = false;
    if (!state) return;
    const id = Math.min(robotId, state.robots.length - 1);
    apiGet(`/sensor?session_id=${sessionId}&agent_id=${id}`).then(value => {
      if (!cancelled) { setSensor(value); setError(null); }
    }).catch(err => { if (!cancelled) setError(err.message); });
    return () => { cancelled = true; };
  }, [state, robotId, sessionId]);
  const describe = cell => !cell[0] ? ["Unknown", "#263446", "?"] : cell[1] ? ["Outside map", "#111827", "×"] : cell[7] ? ["Nearby robot", "#de694b", "R"] : cell[2] ? ["Shelf", "#ad8656", "S"] : cell[3] ? ["Safety corridor", "#cba525", "!"] : cell[4] ? ["Pickup", "#b6dfc9", "P"] : cell[5] ? ["Drop", "#d7cef3", "D"] : cell[6] ? ["Charger", "#f8dd9a", "C"] : ["Visible floor", "#f4f1e9", "·"];
  return <div className="panel">
    <div className="panel-header">
      <div className="panel-title"><span className="icon-badge"><IconSearch size={14} /></span>What one robot sees <Tip text={TIPS.sensor} wide /></div>
      <select className="compact-select" aria-label="Robot to inspect" value={Math.min(robotId, (state?.robots.length || 1) - 1)} onChange={e => setRobotId(Number(e.target.value))}>
        {(state?.robots || []).map(r => <option key={r.id} value={r.id}>Robot #{r.id}</option>)}
      </select>
    </div>
    {error && <p role="alert">{error}</p>}
    {sensor && <><p style={{ fontSize: 12 }}>Front ↑ · {sensor.age_ticks} tick(s) old · simulated range sensing</p>
      <div aria-label="Local sensor grid" style={{ display: "grid", gridTemplateColumns: `repeat(${sensor.frame.length}, 35px)`, gap: 3, justifyContent: "center", margin: "12px 0" }}>
        {sensor.frame.flatMap((row, y) => row.map((cell, x) => {
          const [label, background, symbol] = describe(cell);
          const self = x === Math.floor(row.length / 2) && y === Math.floor(sensor.frame.length / 2);
          return <div key={`${x}-${y}`} title={self ? "This robot" : label} style={{ background, color: cell[0] ? "#172032" : "#fff", height: 35, display: "grid", placeItems: "center", border: self ? "2px solid #1a8a5f" : "1px solid #bbb", borderRadius: 3 }}>{self ? "↑" : symbol}</div>;
        }))}
      </div><p style={{ fontSize: 12 }}>Dark ? cells are unseen, occluded, or dropped readings. This is the actual local input; it is not a camera image. The central arrow is the robot.</p></>}
  </div>;
}

function StartHere({ onGuide }) {
  const [hidden, setHidden] = useState(() => { try { return localStorage.getItem("wfc-start-hidden") === "1"; } catch { return false; } });
  if (hidden) return null;
  const hide = () => { setHidden(true); try { localStorage.setItem("wfc-start-hidden", "1"); } catch { /* storage unavailable */ } };
  return (
    <div className="start-here">
      <div>
        <strong>New here? Three steps:</strong>
        <ol>
          <li>Pick a <b>demo preset</b> below. It loads the warehouse and starts playing.</li>
          <li>Watch the map: coloured circles are robots, and the yellow-black border is the worker corridor they must never enter.</li>
          <li>Read the <b>Live Metrics</b> on the right. Hover any <span className="tip-i inline">i</span> for an explanation.</li>
        </ol>
      </div>
      <div className="start-actions">
        <button className="pill solid" onClick={onGuide}>Open the guide</button>
        <button className="pill" onClick={hide}>Got it</button>
      </div>
    </div>
  );
}

function SingleView({ algorithms, pendingPreset, clearPreset, openGuide }) {
  const session = useSimSession("ui-single");
  const [algorithm, setAlgorithm] = useState("qmix");
  const [numAgents, setNumAgents] = useState(5);
  const [numTasks, setNumTasks] = useState(10);
  const [seed, setSeed] = useState(42);
  const [playing, setPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(240);
  const [zoom, setZoom] = useState(1);
  const [frontFov, setFrontFov] = useState(360);
  const [rearFov, setRearFov] = useState(0);
  const [initialSoc, setInitialSoc] = useState(100);
  const [workload, setWorkload] = useState("mixed");
  const [arrivalInterval, setArrivalInterval] = useState(20);
  const [available, setAvailable] = useState(algorithms);
  const [view3d, setView3d] = useState(false);
  const autoPlay = useRef(false);

  useEffect(() => { apiGet(`/algorithms?num_agents=${numAgents}&workload=${workload}`).then(setAvailable).catch(() => setAvailable([])); }, [numAgents, workload]);

  const applyPreset = (preset) => {
    const c = preset.cfg;
    setAlgorithm(c.algorithm); setNumAgents(c.numAgents); setWorkload(c.workload); setArrivalInterval(c.arrivalInterval);
    setInitialSoc(100); setFrontFov(360); setRearFov(0);
    autoPlay.current = true;
  };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { if (pendingPreset) { applyPreset(pendingPreset); clearPreset(); } }, [pendingPreset]);

  // Settings apply automatically (debounced): no hidden "press Reset" step.
  const config = JSON.stringify({ algorithm, numAgents, numTasks, seed, workload, arrivalInterval, frontFov, rearFov, initialSoc });
  const restart = useCallback(async () => {
    setPlaying(false);
    const result = await session.reset(algorithm, numAgents, numTasks, seed, { workload, arrival_interval: arrivalInterval, sensor_front_fov: frontFov, sensor_rear_fov: rearFov, initial_soc: initialSoc / 100 });
    if (result && autoPlay.current) setPlaying(true);
    autoPlay.current = false;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config]);
  useEffect(() => { const id = setTimeout(restart, 300); return () => clearTimeout(id); }, [restart]);

  useEffect(() => {
    if (!playing) return undefined;
    const id = setInterval(() => session.tick(speedMs), speedMs);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, speedMs]);

  useEffect(() => { if (playing && (session.state?.done || session.state?.truncated)) setPlaying(false); }, [session.state, playing]);

  const canvasWrapRef = useRef(null);
  const toggleFullscreen = () => {
    const el = canvasWrapRef.current;
    if (!document.fullscreenElement) el.requestFullscreen?.();
    else document.exitFullscreen?.();
  };

  const activeAlgo = available.find((a) => a.name === algorithm);
  const error = friendlyError(session.error, algorithm, numAgents, workload);
  const step = session.state?.metrics?.step_count ?? 0;

  return (
    <>
      <StartHere onGuide={openGuide} />
      {error && <div className="error-banner">{error}</div>}

      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <span className="icon-badge"><IconStar size={14} /></span>
            Demo presets <Tip text="One click loads a ready-made scenario and starts it. Run them in order for the presentation." />
          </div>
        </div>
        <div className="preset-row">
          {PRESETS.map((p) => (
            <Tip key={p.key} text={p.tip}>
              <button className="preset" onClick={() => applyPreset(p)}>{p.label}</button>
            </Tip>
          ))}
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <span className="icon-badge"><IconSliders size={14} /></span>
            Simulation Controls <Tip text="Changes apply automatically: the shift restarts with the new settings." />
          </div>
          <span className="shift-progress" title="A shift is 600 steps">Step {step} / 600</span>
        </div>
        <div className="controls-row">
          <div className="field">
            <label><IconCube size={13} /> Algorithm <Tip text={TIPS.algorithm} /></label>
            <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
              {(available.length ? available : [{ name: algorithm, trained: true }]).map((a) => (
                <option key={a.name} value={a.name} disabled={!a.trained}>
                  {a.name === "nearest_task" ? "planner (nearest_task)" : a.name}{!a.trained ? " (not trained for this size)" : ""}
                </option>
              ))}
            </select>
          </div>
          <Stepper value={numAgents} onChange={setNumAgents} min={1} max={20} icon={<IconTruck size={13} />} labelText="Fleet Size" tip={TIPS.fleet} />
          <div className="field">
            <label>Package flow <Tip text={TIPS.flow} /></label>
            <select value={workload} onChange={(e) => setWorkload(e.target.value)}>
              <option value="mixed">Receiving, storage and shipping</option>
              <option value="outbound">Shelf to shipping</option>
            </select>
          </div>
          {workload === "mixed" ? (
            <div className="field">
              <label>Incoming packages <Tip text={TIPS.arrival} /></label>
              <select value={arrivalInterval} onChange={(e) => setArrivalInterval(Number(e.target.value))}>
                {[80, 40, 20, 12, 10, 8].map((n) => <option key={n} value={n}>One every {n} steps</option>)}
              </select>
            </div>
          ) : (
            <Stepper value={numTasks} onChange={setNumTasks} min={1} max={40} icon={<IconPackage size={13} />} labelText="Tasks in Flight" tip={TIPS.tasks} />
          )}
          <Stepper value={initialSoc} onChange={setInitialSoc} min={0} max={100} labelText="Initial battery %" tip={TIPS.battery} />
          <div className="field">
            <label>Front sensor <Tip text={TIPS.front} /></label>
            <select value={frontFov} onChange={(e) => setFrontFov(Number(e.target.value))}>
              <option value={360}>360° range sensing</option><option value={120}>120° front</option><option value={90}>90° front</option><option value={0}>Off</option>
            </select>
          </div>
          <div className="field">
            <label>Rear sensor <Tip text={TIPS.rear} /></label>
            <select value={rearFov} onChange={(e) => setRearFov(Number(e.target.value))}>
              <option value={0}>Off</option><option value={90}>90° rear</option><option value={120}>120° rear</option>
            </select>
          </div>
          <div className="field">
            <label><IconRefreshCw size={13} /> Seed <Tip text={TIPS.seed} /></label>
            <div className="seed-field">
              <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
              <button className="icon-btn" type="button" title="Random seed" onClick={() => setSeed(Math.floor(Math.random() * 100000))}><IconRefreshCw size={14} /></button>
            </div>
          </div>
          <div className="field">
            <label>Speed ({speedMs} ms/step) <Tip text={TIPS.speed} /></label>
            <input type="range" min="40" max="1200" step="20" value={speedMs} onChange={(e) => setSpeedMs(Number(e.target.value))} />
          </div>
          <div className="buttons">
            <Tip text={TIPS.reset}><button className="pill" onClick={restart}><IconRotateCcw size={14} /> Restart</button></Tip>
            <Tip text={TIPS.step}><button className="pill" onClick={() => session.tick(speedMs)} disabled={playing}><IconStepForward size={14} /> Step</button></Tip>
            <Tip text={TIPS.play}><button className="pill solid" onClick={() => setPlaying((p) => !p)}>{playing ? <IconPause size={14} /> : <IconPlay size={14} />} {playing ? "Pause" : "Play"}</button></Tip>
          </div>
        </div>
        <p className="algo-description">
          <b>{ALGO_INFO[algorithm] || activeAlgo?.description}</b>
          {session.state?.workload_version && <span> · Stored now: {session.state.package_counts?.stored ?? 0} · Shipped: {session.state.package_counts?.shipped ?? 0}</span>}
          {activeAlgo && (["nearest_task", "shortest_path"].includes(activeAlgo.name)
            ? <span className="badge">privileged map planner</span>
            : activeAlgo.trained ? <span className="badge trained">trained model</span> : <span className="badge">no trained model</span>)}
        </p>
      </div>

      <div className="layout">
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <span className="icon-badge"><IconGrid size={14} /></span>
              Warehouse Floor <Tip wide text="Top-down supervisor view of the whole warehouse. The robots themselves only see the 5×5 window shown in 'What one robot sees'." />
            </div>
            <div className="header-actions">
              <div className="view-toggle">
                <button className={view3d ? "" : "active"} onClick={() => setView3d(false)}>2D View</button>
                <button className={view3d ? "active" : ""} onClick={() => setView3d(true)} title={TIPS.view3d}>3D View</button>
              </div>
              {!view3d && <button className="icon-btn" title="Zoom in" onClick={() => setZoom((z) => (z >= 1.6 ? 1 : z + 0.2))}><IconSearch size={15} /></button>}
              <button className="icon-btn" title="Fullscreen" onClick={toggleFullscreen}><IconMaximize size={15} /></button>
            </div>
          </div>
          <div className="canvas-wrap" ref={canvasWrapRef}>
            {view3d ? (
              <Warehouse3D animRef={session.animRef} />
            ) : (
              <div style={{ transform: `scale(${zoom})`, transformOrigin: "center" }}>
                <WarehouseCanvas animRef={session.animRef} trailsRef={session.trailsRef} pulsesRef={session.pulsesRef} cellPx={26} />
              </div>
            )}
          </div>
          <Legend />
          <p className="algo-description">
            {view3d
              ? "Drag to rotate, scroll to zoom. Racks are brown blocks, the worker corridor is the yellow strip, chargers are gold. A brown box on a robot means it is carrying a package."
              : "Hover any legend item for details. Numbers above robots are their IDs; dashed lines point to each robot's current target."}
          </p>
        </div>
        <div>
          <StatsPanel state={session.state} history={session.history} playing={playing} />
          <SensorPanel state={session.state} sessionId="ui-single" />
        </div>
      </div>
      <FleetTable state={session.state} activityRef={session.activityRef} />
    </>
  );
}

// ---------------------------------------------------------------------------
// Compare-all-algorithms view
// ---------------------------------------------------------------------------

function CompareCard({ algorithm, session, rank, cellPx }) {
  const state = session.state;
  const metrics = (state && state.metrics) || {};
  if (session.error) {
    return (
      <div className="compare-card unavailable">
        <div className="compare-card-header"><h3>{ALGO_LABEL[algorithm]}</h3></div>
        <p>{session.error.includes("No compatible") ? "No trained model for this fleet size and package flow." : session.error}</p>
      </div>
    );
  }
  return (
    <div className="compare-card">
      <div className="compare-card-header">
        <h3><Tip text={ALGO_INFO[algorithm]}>{ALGO_LABEL[algorithm]}</Tip></h3>
        <span className={`rank ${rank === 1 ? "first" : ""}`}>#{rank}</span>
      </div>
      <div className="compare-mini-canvas-wrap">
        <WarehouseCanvas animRef={session.animRef} trailsRef={session.trailsRef} pulsesRef={session.pulsesRef} cellPx={cellPx} showLabels={false} showRoutes={false} />
      </div>
      <div className="compare-stats">
        <div className="mini-stat">
          <div className="value">{state ? state.completed_tasks ?? 0 : 0}</div>
          <div className="label">Delivered</div>
        </div>
        <div className="mini-stat">
          <div className="value">{metrics.collision_count ?? 0}</div>
          <div className="label">Conflicts</div>
        </div>
        <div className="mini-stat">
          <div className="value">{metrics.safety_overrides ?? 0}</div>
          <div className="label">Corridor blocks</div>
        </div>
        <div className="mini-stat">
          <div className="value">{Number(metrics.reward ?? 0).toFixed(0)}</div>
          <div className="label">Reward</div>
        </div>
      </div>
    </div>
  );
}

function Leaderboard({ ranked }) {
  return (
    <div className="panel leaderboard">
      <div className="panel-header">
        <div className="panel-title">
          <span className="icon-badge">
            <IconStar size={14} />
          </span>
          Leaderboard
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Algorithm</th>
            <th>Delivered</th>
            <th>Conflicts</th>
            <th>Reward</th>
          </tr>
        </thead>
        <tbody>
          {ranked.map((row, idx) => (
            <tr key={row.algorithm} className={idx === 0 ? "leader" : ""}>
              <td>{idx + 1}</td>
              <td>{ALGO_LABEL[row.algorithm]}</td>
              <td>{row.completed}</td>
              <td>{row.collisions}</td>
              <td>{row.reward.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CompareView() {
  const sessions = {
    nearest_task: useSimSession("cmp-nearest_task"),
    vdn: useSimSession("cmp-vdn"),
    iql: useSimSession("cmp-iql"),
    qmix: useSimSession("cmp-qmix"),
  };
  const [numAgents, setNumAgents] = useState(5);
  const [numTasks, setNumTasks] = useState(10);
  const [seed, setSeed] = useState(42);
  const [playing, setPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(240);
  const [workload, setWorkload] = useState("mixed");
  const [arrivalInterval, setArrivalInterval] = useState(20);
  const [error, setError] = useState(null);

  const resetAll = useCallback(async () => {
    setPlaying(false);
    setError(null);
    try {
      await Promise.all(COMPARE_ALGORITHMS.map((algo) => sessions[algo].reset(algo, numAgents, numTasks, seed, { workload, arrival_interval: arrivalInterval })));
    } catch (err) {
      setError(err.message);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [numAgents, numTasks, seed, workload, arrivalInterval]);

  useEffect(() => {
    const id = setTimeout(resetAll, 300);
    return () => clearTimeout(id);
  }, [resetAll]);

  const sessionsRef = useRef(sessions);
  sessionsRef.current = sessions; // tickAll must see the latest state, not the first render's
  const tickAll = useCallback(() => {
    COMPARE_ALGORITHMS.forEach((algo) => { const sess = sessionsRef.current[algo]; if (sess.state && !sess.error) sess.tick(speedMs); });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [speedMs]);

  useEffect(() => {
    if (!playing) return undefined;
    const id = setInterval(tickAll, speedMs);
    return () => clearInterval(id);
  }, [playing, speedMs, tickAll]);

  const ranked = useMemo(() => {
    const rows = COMPARE_ALGORITHMS.filter(algo => sessions[algo].state && !sessions[algo].error).map((algo) => {
      const state = sessions[algo].state;
      const metrics = (state && state.metrics) || {};
      return {
        algorithm: algo,
        completed: state ? state.completed_tasks ?? 0 : 0,
        collisions: metrics.collision_count ?? 0,
        reward: Number(metrics.reward ?? 0),
      };
    });
    return rows.sort((a, b) => b.completed - a.completed || a.collisions - b.collisions);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessions.nearest_task.state, sessions.vdn.state, sessions.iql.state, sessions.qmix.state]);

  const rankOf = (algo) => ranked.findIndex((r) => r.algorithm === algo) + 1;

  return (
    <>
      {error && <div className="error-banner">{error}</div>}
      {COMPARE_ALGORITHMS.map((a) => sessions[a].error).filter((e) => e && !e.includes("No compatible")).length > 0 && (
        <div className="error-banner">{friendlyError(COMPARE_ALGORITHMS.map((a) => sessions[a].error).find((e) => e && !e.includes("No compatible")), "", numAgents, workload)}</div>
      )}

      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <span className="icon-badge">
              <IconLayers size={14} />
            </span>
            Compare all four, live, same seed <Tip text="Every algorithm runs on an identical copy of the warehouse. The leaderboard ranks by jobs completed, then fewest conflicts." />
          </div>
          <span className="shift-progress">Step {COMPARE_ALGORITHMS.map((a) => sessions[a].state?.metrics?.step_count).find((v) => v != null) ?? 0} / 600</span>
        </div>
        <div className="controls-row">
          <Stepper value={numAgents} onChange={setNumAgents} min={1} max={20} icon={<IconTruck size={13} />} labelText="Fleet Size" tip={TIPS.fleet} />
          <div className="field"><label>Package flow <Tip text={TIPS.flow} /></label><select value={workload} onChange={(e) => setWorkload(e.target.value)}><option value="mixed">Receiving, storage and shipping</option><option value="outbound">Shelf to shipping</option></select></div>
          {workload === "mixed"
            ? <div className="field"><label>Incoming packages <Tip text={TIPS.arrival} /></label><select value={arrivalInterval} onChange={(e) => setArrivalInterval(Number(e.target.value))}>{[80, 40, 20, 12, 10, 8].map((n) => <option key={n} value={n}>One every {n} steps</option>)}</select></div>
            : <Stepper value={numTasks} onChange={setNumTasks} min={1} max={40} icon={<IconPackage size={13} />} labelText="Tasks in Flight" tip={TIPS.tasks} />}
          <div className="field">
            <label>
              <IconRefreshCw size={13} /> Seed
            </label>
            <div className="seed-field">
              <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
              <button className="icon-btn" type="button" title="Randomize seed" onClick={() => setSeed(Math.floor(Math.random() * 100000))}>
                <IconRefreshCw size={14} />
              </button>
            </div>
          </div>
          <div className="field">
            <label>Speed ({speedMs} ms/step)</label>
            <input type="range" min="80" max="1200" step="20" value={speedMs} onChange={(e) => setSpeedMs(Number(e.target.value))} />
          </div>
          <div className="buttons">
            <button className="pill" onClick={resetAll}>
              <IconRotateCcw size={14} /> Reset all
            </button>
            <button className="pill" onClick={tickAll} disabled={playing}>
              <IconStepForward size={14} /> Step all
            </button>
            <button className="pill solid" onClick={() => setPlaying((p) => !p)}>
              {playing ? <IconPause size={14} /> : <IconPlay size={14} />} {playing ? "Pause" : "Play"}
            </button>
          </div>
        </div>
        <p className="algo-description">
          All four run on the same warehouse, fleet size and seed, stepping together, so the fairest live comparison. Try 10 robots with packages every 10 steps:
          that is where QMIX pulled ahead of IQL in the sealed tests. A card says so when a model was not trained for the chosen size.
        </p>
      </div>

      <div className="layout">
        <div className="compare-grid">
          {COMPARE_ALGORITHMS.map((algo) => (
            <CompareCard key={algo} algorithm={algo} session={sessions[algo]} rank={rankOf(algo)} cellPx={14} />
          ))}
        </div>
        <Leaderboard ranked={ranked} />
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Guide: what, where, who, how
// ---------------------------------------------------------------------------

function GuideSection({ title, children }) {
  return (
    <div className="panel guide-section">
      <div className="panel-title guide-title">{title}</div>
      <div className="guide-body">{children}</div>
    </div>
  );
}

function GuideView({ runPreset }) {
  const flow = [
    ["Browser dashboard", "You choose a scenario and press Play."],
    ["FastAPI service", "Receives each step request (src/deployment/api.py)."],
    ["Warehouse simulator", "Moves robots, packages and batteries (src/environment)."],
    ["Robot sensors", "Each robot gets only its own 5×5 view: 348 numbers."],
    ["Trained network", "The shared GRU network scores the 7 possible actions."],
    ["Safety layer", "Removes any move into a shelf or worker corridor. Not learned, cannot be overridden."],
    ["Physics check", "The simulator refuses illegal moves a second time, then returns the new state."],
  ];
  return (
    <>
      <GuideSection title="What is this?">
        <p>A live simulation of a <b>robot warehouse</b>. Robots pick up incoming packages at receiving docks (<b>R</b>), store them on racks (<b>P</b>), and later carry them to shipping bays (<b>D</b>). They share narrow aisles and only <b>2 chargers</b>, and must never enter the yellow-black <b>worker corridor</b> around the edge.</p>
        <p>The robots are driven either by a hand-written <b>planner</b> or by <b>neural networks trained with multi-agent reinforcement learning</b> (QMIX, VDN, IQL). The trained robots decide using only their own local sensors; nobody tells them where the others are.</p>
      </GuideSection>

      <GuideSection title="Start in 30 seconds">
        <ol className="guide-steps">
          <li>Run <code>powershell -ExecutionPolicy Bypass -File scripts\demo.ps1</code>. It starts the service and opens this page.</li>
          <li>Go to <b>Simulator</b> and click a <b>demo preset</b>. It loads and starts playing.</li>
          <li>Use <b>2D / 3D View</b> to switch views. In 3D, drag to rotate and scroll to zoom.</li>
          <li>Open <b>Compare</b> to run the planner, IQL, VDN and QMIX side by side on the same warehouse.</li>
        </ol>
        <div className="preset-row">
          {PRESETS.map((p) => <button key={p.key} className="preset" onClick={() => runPreset(p)} title={p.tip}>▶ {p.label}</button>)}
        </div>
      </GuideSection>

      <GuideSection title="Where to look (the screen)">
        <table className="guide-table"><tbody>
          <tr><td><b>Demo presets</b></td><td>One-click scenarios in presentation order.</td></tr>
          <tr><td><b>Simulation Controls</b></td><td>Algorithm, fleet size, package flow, batteries and sensors. Changes apply automatically.</td></tr>
          <tr><td><b>Warehouse Floor</b></td><td>The whole warehouse from above (2D) or in perspective (3D).</td></tr>
          <tr><td><b>Live Metrics</b></td><td>Progress of the current shift: deliveries, blocked conflicts, corridor blocks, reward.</td></tr>
          <tr><td><b>What one robot sees</b></td><td>The real input of the neural network for the selected robot. Dark cells are unknown.</td></tr>
          <tr><td><b>Fleet Status</b></td><td>Every robot's position, task, battery and current order.</td></tr>
        </tbody></table>
      </GuideSection>

      <GuideSection title="Who is who">
        <table className="guide-table"><tbody>
          <tr><td><b>Robots</b></td><td>Coloured circles numbered #0, #1… A gold ring (2D) or a brown box on top (3D) means the robot is carrying a package.</td></tr>
          {Object.entries(ALGO_INFO).filter(([k]) => ["nearest_task", "iql", "vdn", "qmix"].includes(k)).map(([k, v]) => (
            <tr key={k}><td><b>{k === "nearest_task" ? "Planner" : k.toUpperCase()}</b></td><td>{v}</td></tr>
          ))}
          <tr><td><b>Trained sizes</b></td><td>QMIX and IQL: 5, 10 and 20 robots (Receiving, storage and shipping). VDN: 5 robots. The planner works at any size.</td></tr>
        </tbody></table>
      </GuideSection>

      <GuideSection title="What the numbers mean">
        <table className="guide-table"><tbody>
          <tr><td><b>Handling jobs completed</b></td><td>{TIPS.delivered}</td></tr>
          <tr><td><b>Near-collisions blocked</b></td><td>{TIPS.collisions}</td></tr>
          <tr><td><b>Corridor blocks</b></td><td>{TIPS.blocks}</td></tr>
          <tr><td><b>Episode reward</b></td><td>{TIPS.reward}</td></tr>
          <tr><td><b>Sealed results</b></td><td>On 20 unseen test warehouses, QMIX completed 99.8% of orders with 5 robots. When demand grew with the fleet it beat IQL by 15 points at 10 robots. 0 corridor intrusions in 376,539 adversarial attempts.</td></tr>
        </tbody></table>
      </GuideSection>

      <GuideSection title="How one decision flows through the system">
        <div className="flow">
          {flow.map(([t, d], i) => (
            <React.Fragment key={t}>
              <div className={`flow-box ${t === "Safety layer" ? "safety" : ""}`}><b>{t}</b><span>{d}</span></div>
              {i < flow.length - 1 && <span className="flow-arrow">→</span>}
            </React.Fragment>
          ))}
        </div>
      </GuideSection>

      <GuideSection title="Honest limits">
        <ul>
          <li>This is a simulation. The safety layer is software, not a certified safety system for real robots.</li>
          <li>With 20 robots and only 2 chargers, a robot occasionally runs its battery flat.</li>
          <li>The live demo uses demo seeds; the reported results come from separate sealed test scenarios.</li>
        </ul>
      </GuideSection>
    </>
  );
}

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

const NAV_ITEMS = [
  { key: "simulator", label: "Simulator", icon: <IconCube size={17} /> },
  { key: "compare", label: "Compare", icon: <IconGrid size={17} /> },
  { key: "guide", label: "Guide", icon: <IconList size={17} /> },
];

function Sidebar({ page, setPage }) {
  return (
    <div className="sidebar">
      <div className="brand">
        <span className="logo">
          <IconCube size={17} />
        </span>
        <span className="name">WFC</span>
      </div>
      <div className="nav">
        {NAV_ITEMS.map((item) => (
          <button key={item.key} className={`nav-item ${page === item.key ? "active" : ""}`} onClick={() => setPage(item.key)}>
            {item.icon}
            {item.label}
          </button>
        ))}
      </div>
      <div className="sidebar-footer">
        <div className="dots">
          {Array.from({ length: 12 }).map((_, i) => (
            <span key={i} />
          ))}
        </div>
        <strong>Team 25 · QMIX fleet</strong>
        <span>New here? Open the Guide.</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// App shell
// ---------------------------------------------------------------------------

function App() {
  const [page, setPage] = useState("simulator");
  const [pendingPreset, setPendingPreset] = useState(null);
  const runPreset = (preset) => { setPendingPreset(preset); setPage("simulator"); };
  const [algorithms, setAlgorithms] = useState([]);
  const [apiError, setApiError] = useState(null);

  useEffect(() => {
    apiGet("/algorithms")
      .then(setAlgorithms)
      .catch((err) => setApiError(`Could not reach the API at ${API_URL}: ${err.message}`));
  }, []);

  const mode = page === "compare" ? "compare" : page === "guide" ? "guide" : "single";
  const setMode = (m) => setPage(m === "compare" ? "compare" : "simulator");

  const eyebrow = mode === "compare" ? "COMPARE" : mode === "guide" ? "GUIDE" : "SIMULATOR";
  const title = "Warehouse Fleet Control";
  const subtitle =
    mode === "compare"
      ? "Run every algorithm on the same seed and floor plan, side by side, live."
      : mode === "guide"
        ? "What you are looking at, who is who, and how to run the demo."
        : "Live warehouse simulation; every robot decision comes from the FastAPI fleet-decision service.";

  return (
    <div className="app-shell">
      <Sidebar page={page} setPage={setPage} />
      <div className="main">
        <div className="page-header">
          <div>
            <span className="kicker">{eyebrow}</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
          <div className="mode-toggle">
            <button className={mode === "single" ? "active" : ""} onClick={() => setMode("single")}>
              Single Run
            </button>
            <button className={mode === "compare" ? "active" : ""} onClick={() => setMode("compare")}>
              Compare Runs
            </button>
          </div>
        </div>

        {apiError && <div className="error-banner">{apiError}</div>}

        {page === "simulator" && <SingleView algorithms={algorithms} pendingPreset={pendingPreset} clearPreset={() => setPendingPreset(null)} openGuide={() => setPage("guide")} />}
        {page === "compare" && <CompareView />}
        {page === "guide" && <GuideView runPreset={runPreset} />}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
