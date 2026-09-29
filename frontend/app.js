const { useState, useEffect, useRef, useCallback, useMemo } = React;

const API_URL = "http://127.0.0.1:8000";
const COMPARE_ALGORITHMS = ["nearest_task", "dqn", "iql", "qmix"];

const GRID_FREE = 0;
const GRID_SHELF = 1;
const GRID_SAFETY_CORRIDOR = 2;

const ORIENTATION_ANGLE = { 0: -90, 1: 0, 2: 90, 3: 180 };

function robotColor(id) {
  const hue = (id * 137.508) % 360; // golden-angle spacing -> well-separated hues at any N
  return `hsl(${hue.toFixed(0)}, 72%, 60%)`;
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
  ctx.fillStyle = "#eef2f9";
  ctx.fillRect(x, y, size, size);
  ctx.strokeStyle = "#d7deea";
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
}

function drawShelfCell(ctx, x, y, size) {
  ctx.fillStyle = "#5b3a22";
  ctx.fillRect(x, y, size, size);
  ctx.strokeStyle = "#3d2716";
  ctx.lineWidth = Math.max(1, size * 0.06);
  ctx.beginPath();
  ctx.moveTo(x, y + size * 0.33);
  ctx.lineTo(x + size, y + size * 0.33);
  ctx.moveTo(x, y + size * 0.66);
  ctx.lineTo(x + size, y + size * 0.66);
  ctx.stroke();
}

function drawCorridorCell(ctx, x, y, size) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, size, size);
  ctx.clip();
  ctx.fillStyle = "#3a2e12";
  ctx.fillRect(x, y, size, size);
  ctx.strokeStyle = "#f5a524";
  ctx.lineWidth = size * 0.22;
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
    ctx.strokeStyle = "#34d399";
    ctx.beginPath();
    ctx.arc(cx, cy, size * 0.28, 0, Math.PI * 2);
    ctx.stroke();
  } else {
    ctx.strokeStyle = "#818cf8";
    const s = size * 0.24;
    ctx.strokeRect(cx - s, cy - s, s * 2, s * 2);
  }
}

function drawTrail(ctx, points, cellPx, color) {
  if (points.length < 2) return;
  ctx.save();
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1];
    const b = points[i];
    const alpha = (i / points.length) * 0.35;
    ctx.strokeStyle = color;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = Math.max(1.5, cellPx * 0.12);
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
  ctx.translate(cx, cy);
  ctx.rotate((angleDeg * Math.PI) / 180);
  ctx.beginPath();
  ctx.arc(0, 0, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = loaded ? Math.max(2, r * 0.3) : 1;
  ctx.strokeStyle = loaded ? "#fbbf24" : "rgba(0,0,0,0.25)";
  ctx.stroke();

  const tipX = r * 1.4;
  const baseW = r * 0.55;
  ctx.beginPath();
  ctx.moveTo(tipX, 0);
  ctx.lineTo(0, baseW);
  ctx.lineTo(0, -baseW);
  ctx.closePath();
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  ctx.restore();

  if (showLabel && r > 7) {
    ctx.fillStyle = "#c9d6ee";
    ctx.font = `700 ${Math.max(8, r * 0.75)}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(id), cx, cy - r * 1.9);
  }
}

function drawPulse(ctx, cx, cy, cellPx, progress) {
  // progress: 0 (just delivered) -> 1 (fully faded)
  const radius = cellPx * (0.3 + progress * 1.6);
  ctx.save();
  ctx.globalAlpha = Math.max(0, 1 - progress);
  ctx.strokeStyle = "#fde68a";
  ctx.lineWidth = Math.max(1.5, cellPx * 0.12);
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Session hook: owns one backend session (reset/tick), tracks history for
// smooth interpolation, trails and delivery pulses without funnelling every
// animation frame through React state.
// ---------------------------------------------------------------------------

function useSimSession(sessionId) {
  const [state, setState] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]); // [{reward, completed}] per tick, for sparklines
  const animRef = useRef({ prev: null, curr: null, tickStartedAt: 0, tickDuration: 400 });
  const trailsRef = useRef(new Map()); // robotId -> [{x,y}]
  const pulsesRef = useRef([]); // [{x,y,startedAt}]

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
        setHistory([]);
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

        // Trails: append each robot's new cell.
        result.robots.forEach((robot) => {
          const trail = trailsRef.current.get(robot.id) || [];
          trail.push({ x: robot.x, y: robot.y });
          if (trail.length > 7) trail.shift();
          trailsRef.current.set(robot.id, trail);
        });

        // Delivery pulses: any robot whose load_status just dropped to 0.
        result.robots.forEach((robot) => {
          const before = prev.robots.find((r) => r.id === robot.id);
          if (before && before.load_status === 1 && robot.load_status === 0) {
            pulsesRef.current.push({ x: robot.x, y: robot.y, startedAt: performance.now() });
          }
        });

        setHistory((h) => [...h.slice(-59), { reward: result.reward ?? 0, completed: result.completed_tasks ?? 0 }]);
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

  return { state, error, history, reset, tick, animRef, trailsRef, pulsesRef };
}

// ---------------------------------------------------------------------------
// Warehouse canvas: reads an animation ref directly and runs its own
// requestAnimationFrame loop, independent of React's render cycle, so robot
// motion is smooth regardless of how often the parent re-renders.
// ---------------------------------------------------------------------------

function WarehouseCanvas({ animRef, trailsRef, pulsesRef, cellPx = 26, showLabels = true }) {
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
        <span className="swatch" style={{ background: "#eef2f9", border: "1px solid #d7deea" }} />
        Floor
      </span>
      <span className="legend-item">
        <span className="swatch" style={{ background: "#5b3a22" }} />
        Shelf
      </span>
      <span className="legend-item">
        <span className="swatch" style={{ background: "#3a2e12", border: "1px solid #f5a524" }} />
        Safety corridor (masked out)
      </span>
      <span className="legend-item">
        <span className="swatch" style={{ borderRadius: "50%", border: "2px solid #34d399", background: "transparent" }} />
        Pickup point
      </span>
      <span className="legend-item">
        <span className="swatch" style={{ border: "2px solid #818cf8", background: "transparent" }} />
        Delivery point
      </span>
      <span className="legend-item">
        <span className="swatch" style={{ borderRadius: "50%", border: "2px solid #fbbf24", background: "#4f8cff" }} />
        Robot (ring = carrying load)
      </span>
    </div>
  );
}

function Sparkline({ data, accessor, color = "#4f8cff", height = 40 }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    ctx.clearRect(0, 0, width, height);
    if (data.length < 2) return;
    const values = data.map(accessor);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - ((v - min) / span) * (height - 6) - 3;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.stroke();
    const last = values[values.length - 1];
    const lastY = height - ((last - min) / span) * (height - 6) - 3;
    ctx.beginPath();
    ctx.arc(width, lastY, 3, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }, [data, accessor, color, height]);

  return <canvas ref={canvasRef} width={280} height={height} style={{ width: "100%", height: `${height}px` }} />;
}

function StatsPanel({ state, history }) {
  const metrics = (state && state.metrics) || {};
  const totalTasks = state ? (state.task_queue || []).length : 0;

  return (
    <div className="panel">
      <h2>Live metrics</h2>
      <div className="stat-grid">
        <div className="stat">
          <div className="value">{metrics.step_count ?? 0}</div>
          <div className="label">Steps</div>
        </div>
        <div className="stat good">
          <div className="value">{state ? state.completed_tasks ?? 0 : 0}</div>
          <div className="label">Delivered</div>
        </div>
        <div className="stat">
          <div className="value">{totalTasks}</div>
          <div className="label">Tasks queued</div>
        </div>
        <div className={`stat ${metrics.collision_count > 0 ? "bad" : "good"}`}>
          <div className="value">{metrics.collision_count ?? 0}</div>
          <div className="label">Collisions</div>
        </div>
        <div className={`stat ${metrics.safety_overrides > 0 ? "warn" : "good"}`}>
          <div className="value">{metrics.safety_overrides ?? 0}</div>
          <div className="label">Corridor attempts blocked</div>
        </div>
        <div className="stat">
          <div className="value">{Number(metrics.reward ?? 0).toFixed(1)}</div>
          <div className="label">Episode reward</div>
        </div>
      </div>
      <div className="sparkline-wrap">
        <div className="label">Deliveries over time</div>
        <Sparkline data={history} accessor={(d) => d.completed} color="#34d399" />
      </div>
      <div className="sparkline-wrap">
        <div className="label">Cumulative reward</div>
        <Sparkline data={history} accessor={(d) => d.reward} color="#7c5cff" />
      </div>
    </div>
  );
}

function RobotList({ state }) {
  if (!state) return null;
  return (
    <div className="panel">
      <h2>Fleet</h2>
      <div className="robot-list">
        {state.robots.map((robot) => (
          <div className="robot-row" key={robot.id}>
            <span className="robot-dot" style={{ background: robotColor(robot.id), color: robotColor(robot.id) }} />
            <span>#{robot.id}</span>
            <span style={{ color: "var(--text-dim)" }}>
              ({robot.x}, {robot.y})
            </span>
            <span style={{ marginLeft: "auto" }}>{robot.load_status === 1 ? "carrying" : "idle"}</span>
          </div>
        ))}
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

  const activeAlgo = algorithms.find((a) => a.name === algorithm);

  return (
    <>
      {session.error && <div className="error-banner">{session.error}</div>}

      <div className="panel">
        <h2>Simulation controls</h2>
        <div className="controls-row">
          <div className="field">
            <label>Algorithm</label>
            <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
              {(algorithms.length ? algorithms : [{ name: algorithm }]).map((a) => (
                <option key={a.name} value={a.name}>
                  {a.name}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Fleet size</label>
            <select value={numAgents} onChange={(e) => setNumAgents(e.target.value)}>
              {[3, 5, 10, 20].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Tasks in flight</label>
            <input type="number" min="1" max="40" value={numTasks} onChange={(e) => setNumTasks(e.target.value)} />
          </div>
          <div className="field">
            <label>Seed</label>
            <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} />
          </div>
          <div className="field">
            <label>Speed ({speedMs} ms/step)</label>
            <input type="range" min="80" max="1200" step="20" value={speedMs} onChange={(e) => setSpeedMs(Number(e.target.value))} />
          </div>
          <div className="buttons">
            <button className="secondary" onClick={handleReset}>
              Reset
            </button>
            <button className="secondary" onClick={() => session.tick(speedMs)} disabled={playing}>
              Step
            </button>
            <button className="primary" onClick={() => setPlaying((p) => !p)}>
              {playing ? "Pause" : "Play"}
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
          <h2>Floor plan</h2>
          <div className="canvas-wrap">
            <WarehouseCanvas animRef={session.animRef} trailsRef={session.trailsRef} pulsesRef={session.pulsesRef} cellPx={26} />
          </div>
          <Legend />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <StatsPanel state={session.state} history={session.history} />
          <RobotList state={session.state} />
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
        <WarehouseCanvas animRef={session.animRef} trailsRef={session.trailsRef} pulsesRef={session.pulsesRef} cellPx={cellPx} showLabels={false} />
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
      <h2>Leaderboard</h2>
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
        <h2>Compare all four, live, same seed</h2>
        <div className="controls-row">
          <div className="field">
            <label>Fleet size</label>
            <select value={numAgents} onChange={(e) => setNumAgents(e.target.value)}>
              {[3, 5, 10, 20].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Tasks in flight</label>
            <input type="number" min="1" max="40" value={numTasks} onChange={(e) => setNumTasks(e.target.value)} />
          </div>
          <div className="field">
            <label>Seed</label>
            <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} />
          </div>
          <div className="field">
            <label>Speed ({speedMs} ms/step)</label>
            <input type="range" min="80" max="1200" step="20" value={speedMs} onChange={(e) => setSpeedMs(Number(e.target.value))} />
          </div>
          <div className="buttons">
            <button className="secondary" onClick={resetAll}>
              Reset all
            </button>
            <button className="secondary" onClick={tickAll} disabled={playing}>
              Step all
            </button>
            <button className="primary" onClick={() => setPlaying((p) => !p)}>
              {playing ? "Pause" : "Play"}
            </button>
          </div>
        </div>
        <p className="algo-description">
          All four run on the same floor plan, fleet size and seed, ticking together -- the fairest side-by-side
          comparison the live demo can give you. The nearest-task baseline uses BFS pathfinding; the learned
          checkpoints use whatever's currently saved in <code>artifacts/</code>.
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
// App shell
// ---------------------------------------------------------------------------

function App() {
  const [mode, setMode] = useState("single");
  const [algorithms, setAlgorithms] = useState([]);
  const [apiError, setApiError] = useState(null);

  useEffect(() => {
    apiGet("/algorithms")
      .then(setAlgorithms)
      .catch((err) => setApiError(`Could not reach the API at ${API_URL}: ${err.message}`));
  }, []);

  return (
    <div className="app">
      <div className="app-header">
        <div>
          <h1>Warehouse Fleet Control</h1>
          <p>Live NumPy simulator, driven through the FastAPI fleet-decision service.</p>
        </div>
        <div className="mode-toggle">
          <button className={mode === "single" ? "active" : ""} onClick={() => setMode("single")}>
            Single
          </button>
          <button className={mode === "compare" ? "active" : ""} onClick={() => setMode("compare")}>
            Compare all
          </button>
        </div>
      </div>

      {apiError && <div className="error-banner">{apiError}</div>}

      {mode === "single" ? <SingleView algorithms={algorithms} /> : <CompareView />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
