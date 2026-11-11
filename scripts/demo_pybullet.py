"""Render a recorded trajectory (scripts/record_demo.py) as a 3D PyBullet scene.

Runs in the conda-forge env, not .venv (PyBullet has no pip wheels for Windows):
    micromamba run -n pybullet python scripts/demo_pybullet.py outputs/demo/trajectory.json --gif outputs/demo/demo.gif
Use --gui for an interactive window instead of a file.
"""
import argparse
import json
import math

import imageio.v2 as imageio
import numpy as np
import pybullet as p

# Cell codes from src/environment: 1 shelf, 2 worker corridor, 3 pickup/receiving, 4 drop, 5 charger, 6 transfer.
TILES = {2: ((0.98, 0.55, 0.15, 0.55), 0.01), 3: ((0.0, 0.5, 0.47, 1), 0.02), 4: ((0.1, 0.35, 0.8, 1), 0.02),
         5: ((0.2, 0.75, 0.3, 1), 0.02), 6: ((0.55, 0.3, 0.75, 1), 0.02)}
SHELF = ((0.45, 0.3, 0.18, 1), 0.6)
ROBOT_COLORS = [(0.07, 0.18, 0.24, 1), (0.85, 0.2, 0.2, 1), (0.95, 0.75, 0.1, 1), (0.3, 0.3, 0.9, 1), (0.9, 0.4, 0.7, 1)]
HEADING = {0: math.pi / 2, 1: 0.0, 2: -math.pi / 2, 3: math.pi}  # grid y is drawn as world -y, so north (y-1) faces +y


def box(half, color, pos, yaw=0.0):
    shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=pos,
                             baseOrientation=p.getQuaternionFromEuler((0, 0, yaw)))


def build_scene(grid):
    h, w = len(grid), len(grid[0])
    box((w / 2, h / 2, 0.01), (0.93, 0.94, 0.93, 1), (w / 2 - 0.5, -(h / 2 - 0.5), -0.01))
    for y, row in enumerate(grid):
        for x, code in enumerate(row):
            if code == 1:
                box((0.45, 0.45, SHELF[1] / 2), SHELF[0], (x, -y, SHELF[1] / 2))
            elif code in TILES:
                color, half_z = TILES[code]
                box((0.48, 0.48, half_z), color, (x, -y, half_z))
    return w, h


def main():
    a = argparse.ArgumentParser()
    a.add_argument('trajectory')
    a.add_argument('--gif', default='outputs/demo/demo.gif')
    a.add_argument('--every', type=int, default=3, help='render every k-th tick')
    a.add_argument('--gui', action='store_true')
    args = a.parse_args()
    data = json.load(open(args.trajectory, encoding='utf-8'))
    p.connect(p.GUI if args.gui else p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    w, h = build_scene(data['grid'])
    robots, cargo = [], []
    for i, _ in enumerate(data['frames'][0]['robots']):
        robots.append(box((0.32, 0.3, 0.12), ROBOT_COLORS[i % len(ROBOT_COLORS)], (0, 0, 0.14)))
        cargo.append(box((0.18, 0.18, 0.1), (0.8, 0.62, 0.35, 1), (0, 0, -5)))
    view = p.computeViewMatrix(cameraEyePosition=(w / 2 - 0.5, -h - 3, 15), cameraTargetPosition=(w / 2 - 0.5, -h / 2 - 1, 0),
                               cameraUpVector=(0, 0, 1))
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=16 / 10, nearVal=0.1, farVal=100)
    images = []
    for frame in data['frames'][::args.every]:
        for body, load, r in zip(robots, cargo, frame['robots']):
            orn = p.getQuaternionFromEuler((0, 0, HEADING[r['heading']]))
            p.resetBasePositionAndOrientation(body, (r['x'], -r['y'], 0.14), orn)
            p.resetBasePositionAndOrientation(load, (r['x'], -r['y'], 0.36) if r['loaded'] else (0, 0, -5), orn)
        if args.gui:
            p.stepSimulation()
            continue
        _, _, rgb, _, _ = p.getCameraImage(960, 600, view, proj, renderer=p.ER_TINY_RENDERER)
        images.append(np.reshape(rgb, (600, 960, 4))[:, :, :3].astype(np.uint8))
    p.disconnect()
    if images:
        imageio.mimsave(args.gif, images, duration=0.08, loop=0)
        print(f'wrote {args.gif}: {len(images)} frames from {data["algorithm"]} N={data["num_agents"]} seed={data["seed"]}')


if __name__ == '__main__':
    main()
