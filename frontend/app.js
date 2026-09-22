const responseBox = document.getElementById('responseBox');
const predictBtn = document.getElementById('predictBtn');
const stepBtn = document.getElementById('stepBtn');
const stateBox = document.getElementById('stateBox');
const API_URL = 'http://127.0.0.1:8000';

const sampleObservations = Array.from({ length: 3 }, () => {
  const arr = Array.from({ length: 28 }, (_, i) => (i % 7) / 10);
  return arr;
});

predictBtn.addEventListener('click', async () => {
  try {
    const stateResponse = await fetch(`${API_URL}/state`);
    const state = await stateResponse.json();
    const observations = state.robots.map((robot) => {
      const values = Array(28).fill(0);
      values[25] = robot.target ? (robot.target[0] - robot.x) / 20 : 0;
      values[26] = robot.target ? (robot.target[1] - robot.y) / 20 : 0;
      values[27] = robot.load_status;
      return values;
    });
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ observations })
    });

    const data = await response.json();
    responseBox.textContent = JSON.stringify(data, null, 2);
    stateBox.textContent = JSON.stringify(state, null, 2);
  } catch (error) {
    responseBox.textContent = `Request failed: ${error.message}`;
  }
});

stepBtn.addEventListener('click', async () => {
  try {
    const prediction = await fetch(`${API_URL}/state`).then((response) => response.json());
    const observations = prediction.robots.map(() => Array(28).fill(0));
    const actions = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ observations })
    }).then((response) => response.json());
    const state = await fetch(`${API_URL}/step`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ actions: actions.actions })
    }).then((response) => response.json());
    responseBox.textContent = JSON.stringify(actions, null, 2);
    stateBox.textContent = JSON.stringify(state, null, 2);
  } catch (error) {
    responseBox.textContent = `Step failed: ${error.message}`;
  }
});
