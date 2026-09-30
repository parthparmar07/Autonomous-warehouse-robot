const { useState, useEffect, useRef, useCallback, useMemo } = React;

const API_URL = "http://127.0.0.1:8000";
const COMPARE_ALGORITHMS = ["nearest_task", "dqn", "iql", "qmix"];

const GRID_FREE = 0;
const GRID_SHELF = 1;
const GRID_SAFETY_CORRIDOR = 2;

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
      else drawFreeCell(ctx, px, py, cellPx);
    }
  }
}

function drawTaskMarker(ctx, cx, cy, size, kind) {
  ctx.lineWidth = 2;
  if (kind === "pickup") {
    ctx.strokeStyle = "#1a8a5f";
    ctx.beginPath();
    ctx.arc(cx, cy, size * 0.26, 0, Math.PI * 2);
    ctx.stroke();
  } else {
    ctx.strokeStyle = "#6d5bd0";
    const s = size * 0.22;
    ctx.strokeRect(cx - s, cy - s, s * 2, s * 2);
  }
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

  const reset = useCallback(
    async (algorithm, numAgents, numTasks, seed) => {
      setError(null);
      try {
        const result = await apiPost(`/reset?session_id=${sessionId}`, {
          algorithm,
          num_agents: Number(numAgents),
          num_tasks: Number(numTasks),
          seed: Number(seed),
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

        (curr.task_queue || []).forEach((task) => {
          if (task.status === "pending") {
            const [tx, ty] = task.pickup;
            drawTaskMarker(ctx, tx * cellPx + cellPx / 2, ty * cellPx + cellPx / 2, cellPx, "pickup");
          } else if (task.status === "picked") {
            const [tx, ty] = task.delivery;
            drawTaskMarker(ctx, tx * cellPx + cellPx / 2, ty * cellPx + cellPx / 2, cellPx, "delivery");
          }
        });

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

function Legend() {
  return (
    <div className="legend">
      <span className="legend-item">
        <span className="swatch" style={{ background: "#fbfaf7", border: "1px solid #ece9e1" }} />
        Floor
      </span>
      <span className="legend-item">
        <span className="swatch" style={{ background: "#8a6a45" }} />
        Shelf
      </span>
      <span className="legend-item">
        <span className="swatch" style={{ background: "#2b2306", border: "1px solid #f5c518" }} />
        Safety corridor (masked out)
      </span>
      <span className="legend-item">
        <span className="swatch" style={{ borderRadius: "50%", border: "2px solid #1a8a5f", background: "transparent" }} />
        Pickup point
      </span>
      <span className="legend-item">
        <span className="swatch" style={{ border: "2px solid #6d5bd0", background: "transparent" }} />
        Delivery point
      </span>
      <span className="legend-item">
        <span className="swatch" style={{ borderRadius: "50%", border: "2px solid #e0a526", background: "#3a6fd8" }} />
        Robot (gold ring = carrying)
      </span>
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

function StatTile({ icon, value, label, tone, series, higherIsBetter }) {
  return (
    <div className={`stat ${tone || ""}`}>
      <div className="stat-top">
        <span className="stat-icon">{icon}</span>
        {series && <Trend series={series} higherIsBetter={higherIsBetter} />}
      </div>
      <div className="value">{value}</div>
      <div className="label">{label}</div>
    </div>
  );
}

function StatsPanel({ state, history }) {
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
        <span className="status-line">
          <span className="status-dot" /> Running
        </span>
      </div>
      <div className="stat-grid">
        <StatTile icon={<IconFootprints size={14} />} value={metrics.step_count ?? 0} label="Steps" series={history.map((h) => h.step)} higherIsBetter={true} />
        <StatTile icon={<IconPackage size={14} />} value={state ? state.completed_tasks ?? 0 : 0} label="Delivered" tone="good" series={history.map((h) => h.completed)} higherIsBetter={true} />
        <StatTile icon={<IconList size={14} />} value={totalTasks} label="Tasks Queued" />
        <StatTile icon={<IconAlertTriangle size={14} />} value={metrics.collision_count ?? 0} label="Collisions" tone={metrics.collision_count > 0 ? "bad" : "good"} series={history.map((h) => h.collisions)} higherIsBetter={false} />
        <StatTile icon={<IconBan size={14} />} value={metrics.safety_overrides ?? 0} label="Corridor Blocks" tone={metrics.safety_overrides > 0 ? "warn" : "good"} series={history.map((h) => h.blocked)} higherIsBetter={false} />
        <StatTile icon={<IconStar size={14} />} value={Number(metrics.reward ?? 0).toFixed(1)} label="Episode Reward" series={history.map((h) => h.reward)} higherIsBetter={true} />
      </div>

      <div style={{ marginTop: 16 }}>
        <div className="panel-header" style={{ marginBottom: 6 }}>
          <div className="panel-title" style={{ fontSize: "0.82rem" }}>
            <span className="icon-badge">
              <IconTruck size={13} />
            </span>
            Deliveries Over Time
          </div>
          <span className="chart-dropdown">Steps</span>
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
          Fleet Status
        </div>
        <button className="see-all-link">
          View All <IconChevronRight size={13} />
        </button>
      </div>
      <table className="fleet-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Position</th>
            <th>Status</th>
            <th>Activity</th>
          </tr>
        </thead>
        <tbody>
          {state.robots.map((robot) => {
            const moved = activityRef.current.get(robot.id) || 0;
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
                  <span className={`status-pill ${robot.load_status === 1 ? "carrying" : "idle"}`}>{robot.load_status === 1 ? <IconBox size={11} /> : null}{robot.load_status === 1 ? "carrying" : "idle"}</span>
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
  );
}

// ---------------------------------------------------------------------------
// Stepper control
// ---------------------------------------------------------------------------

function Stepper({ value, onChange, min = 1, max = 999, icon, labelText }) {
  const set = (v) => onChange(Math.max(min, Math.min(max, v)));
  return (
    <div className="field">
      <label>
        {icon} {labelText}
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

function SingleView({ algorithms }) {
  const session = useSimSession("ui-single");
  const [algorithm, setAlgorithm] = useState("nearest_task");
  const [numAgents, setNumAgents] = useState(5);
  const [numTasks, setNumTasks] = useState(10);
  const [seed, setSeed] = useState(42);
  const [playing, setPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(400);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    session.reset(algorithm, numAgents, numTasks, seed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!playing) return undefined;
    const id = setInterval(() => session.tick(speedMs), speedMs);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, speedMs]);

  const handleReset = () => {
    setPlaying(false);
    session.reset(algorithm, numAgents, numTasks, seed);
  };

  const canvasWrapRef = useRef(null);
  const toggleFullscreen = () => {
    const el = canvasWrapRef.current;
    if (!document.fullscreenElement) el.requestFullscreen?.();
    else document.exitFullscreen?.();
  };

  const activeAlgo = algorithms.find((a) => a.name === algorithm);

  return (
    <>
      {session.error && <div className="error-banner">{session.error}</div>}

      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <span className="icon-badge">
              <IconSliders size={14} />
            </span>
            Simulation Controls
          </div>
        </div>
        <div className="controls-row">
          <div className="field">
            <label>
              <IconCube size={13} /> Algorithm
            </label>
            <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
              {(algorithms.length ? algorithms : [{ name: algorithm }]).map((a) => (
                <option key={a.name} value={a.name}>
                  {a.name}
                </option>
              ))}
            </select>
          </div>
          <Stepper value={numAgents} onChange={setNumAgents} min={1} max={20} icon={<IconTruck size={13} />} labelText="Fleet Size" />
          <Stepper value={numTasks} onChange={setNumTasks} min={1} max={40} icon={<IconPackage size={13} />} labelText="Tasks in Flight" />
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
            <button className="pill" onClick={handleReset}>
              <IconRotateCcw size={14} /> Reset
            </button>
            <button className="pill" onClick={() => session.tick(speedMs)} disabled={playing}>
              <IconStepForward size={14} /> Step
            </button>
            <button className="pill solid" onClick={() => setPlaying((p) => !p)}>
              {playing ? <IconPause size={14} /> : <IconPlay size={14} />} {playing ? "Pause" : "Play"}
            </button>
          </div>
        </div>
        {activeAlgo && (
          <p className="algo-description">
            {activeAlgo.description}
            {activeAlgo.trained ? <span className="badge trained">checkpoint loaded</span> : <span className="badge">no checkpoint / untrained</span>}
          </p>
        )}
      </div>

      <div className="layout">
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <span className="icon-badge">
                <IconGrid size={14} />
              </span>
              Warehouse Floor Plan
            </div>
            <div className="header-actions">
              <div className="view-toggle">
                <button className="active">2D View</button>
                <button disabled title="Not available in this build">
                  3D View
                </button>
              </div>
              <button className="icon-btn" title="Zoom in" onClick={() => setZoom((z) => (z >= 1.6 ? 1 : z + 0.2))}>
                <IconSearch size={15} />
              </button>
              <button className="icon-btn" title="Fullscreen" onClick={toggleFullscreen}>
                <IconMaximize size={15} />
              </button>
            </div>
          </div>
          <div className="canvas-wrap" ref={canvasWrapRef}>
            <div style={{ transform: `scale(${zoom})`, transformOrigin: "center" }}>
              <WarehouseCanvas animRef={session.animRef} trailsRef={session.trailsRef} pulsesRef={session.pulsesRef} cellPx={26} />
            </div>
          </div>
          <Legend />
        </div>
        <div>
          <StatsPanel state={session.state} history={session.history} />
          <FleetTable state={session.state} activityRef={session.activityRef} />
        </div>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Compare-all-algorithms view
// ---------------------------------------------------------------------------

function CompareCard({ algorithm, session, rank, cellPx }) {
  const state = session.state;
  const metrics = (state && state.metrics) || {};
  return (
    <div className="compare-card">
      <div className="compare-card-header">
        <h3>{algorithm.replace("_", " ")}</h3>
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
          <div className="label">Collisions</div>
        </div>
        <div className="mini-stat">
          <div className="value">{metrics.safety_overrides ?? 0}</div>
          <div className="label">Blocked</div>
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
            <th>Collisions</th>
            <th>Reward</th>
          </tr>
        </thead>
        <tbody>
          {ranked.map((row, idx) => (
            <tr key={row.algorithm} className={idx === 0 ? "leader" : ""}>
              <td>{idx + 1}</td>
              <td style={{ textTransform: "capitalize" }}>{row.algorithm.replace("_", " ")}</td>
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
    dqn: useSimSession("cmp-dqn"),
    iql: useSimSession("cmp-iql"),
    qmix: useSimSession("cmp-qmix"),
  };
  const [numAgents, setNumAgents] = useState(5);
  const [numTasks, setNumTasks] = useState(10);
  const [seed, setSeed] = useState(42);
  const [playing, setPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(350);
  const [error, setError] = useState(null);

  const resetAll = useCallback(async () => {
    setPlaying(false);
    setError(null);
    try {
      await Promise.all(COMPARE_ALGORITHMS.map((algo) => sessions[algo].reset(algo, numAgents, numTasks, seed)));
    } catch (err) {
      setError(err.message);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [numAgents, numTasks, seed]);

  useEffect(() => {
    resetAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const tickAll = useCallback(() => {
    COMPARE_ALGORITHMS.forEach((algo) => sessions[algo].tick(speedMs));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [speedMs]);

  useEffect(() => {
    if (!playing) return undefined;
    const id = setInterval(tickAll, speedMs);
    return () => clearInterval(id);
  }, [playing, speedMs, tickAll]);

  const ranked = useMemo(() => {
    const rows = COMPARE_ALGORITHMS.map((algo) => {
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
  }, [sessions.nearest_task.state, sessions.dqn.state, sessions.iql.state, sessions.qmix.state]);

  const rankOf = (algo) => ranked.findIndex((r) => r.algorithm === algo) + 1;

  return (
    <>
      {error && <div className="error-banner">{error}</div>}
      {COMPARE_ALGORITHMS.map((a) => sessions[a].error).filter(Boolean).length > 0 && (
        <div className="error-banner">{COMPARE_ALGORITHMS.map((a) => sessions[a].error).find(Boolean)}</div>
      )}

      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <span className="icon-badge">
              <IconLayers size={14} />
            </span>
            Compare all four, live, same seed
          </div>
        </div>
        <div className="controls-row">
          <Stepper value={numAgents} onChange={setNumAgents} min={1} max={20} icon={<IconTruck size={13} />} labelText="Fleet Size" />
          <Stepper value={numTasks} onChange={setNumTasks} min={1} max={40} icon={<IconPackage size={13} />} labelText="Tasks in Flight" />
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
          All four run on the same floor plan, fleet size and seed, ticking together — the fairest side-by-side comparison the live demo can give you. The
          nearest-task baseline uses BFS pathfinding; the learned checkpoints use whatever's currently saved in <code>artifacts/</code>.
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
// Sidebar
// ---------------------------------------------------------------------------

const NAV_ITEMS = [
  { key: "simulator", label: "Simulator", icon: <IconCube size={17} /> },
  { key: "compare", label: "Compare", icon: <IconGrid size={17} /> },
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
        <strong>Built with FastAPI</strong>
        <span>Live simulation &amp; pathfinding</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// App shell
// ---------------------------------------------------------------------------

function App() {
  const [page, setPage] = useState("simulator");
  const [algorithms, setAlgorithms] = useState([]);
  const [apiError, setApiError] = useState(null);

  useEffect(() => {
    apiGet("/algorithms")
      .then(setAlgorithms)
      .catch((err) => setApiError(`Could not reach the API at ${API_URL}: ${err.message}`));
  }, []);

  const mode = page === "compare" ? "compare" : "single";
  const setMode = (m) => setPage(m === "compare" ? "compare" : "simulator");

  const eyebrow = mode === "compare" ? "COMPARE" : "SIMULATOR";
  const title = "Warehouse Fleet Control";
  const subtitle =
    mode === "compare"
      ? "Run every algorithm on the same seed and floor plan, side by side, live."
      : "Live NumPy simulator, driven through the FastAPI fleet-decision service.";

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

        {page === "simulator" && <SingleView algorithms={algorithms} />}
        {page === "compare" && <CompareView />}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
