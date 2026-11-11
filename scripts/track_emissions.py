"""Run any module under CodeCarbon: python -m scripts.track_emissions <label> <module> [args...]"""
import runpy
import sys
from pathlib import Path

from codecarbon import EmissionsTracker

ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    label, module, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    out = ROOT / 'outputs' / 'emissions'
    out.mkdir(parents=True, exist_ok=True)
    tracker = EmissionsTracker(project_name=label, output_dir=str(out), output_file='emissions.csv',
                               log_level='error', save_to_file=True)
    tracker.start()
    sys.argv = [module, *rest]
    try:
        runpy.run_module(module, run_name='__main__', alter_sys=True)
    finally:
        print(f'[codecarbon] {label}: {tracker.stop() or 0.0:.6f} kg CO2eq -> {out / "emissions.csv"}', flush=True)


if __name__ == '__main__':
    main()
