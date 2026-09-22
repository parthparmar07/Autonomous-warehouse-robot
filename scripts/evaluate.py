from __future__ import annotations

from src.environment.warehouse_env import WarehouseEnv


def main() -> None:
    env = WarehouseEnv(grid_size=(8, 8), num_agents=2, seed=7)
    env.reset(seed=7)
    actions = [0, 0]
    env.step(actions)
    print({"metrics": env.episode_metrics})


if __name__ == "__main__":
    main()
