"""Shared sequence-aware expert warm start followed by explicit joint RL updates.
Use --bc-episodes and --bc-epochs; the resulting training_kind is bc_then_rl.
"""
from src.training.runner import main
if __name__ == '__main__':
    main('qmix')
