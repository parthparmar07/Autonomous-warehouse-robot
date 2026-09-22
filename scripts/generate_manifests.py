from __future__ import annotations

from src.evaluation.manifests import generate_manifest_set


if __name__ == "__main__":
    paths = generate_manifest_set("evaluation/manifests")
    print(f"Generated {len(paths)} manifests")
    for path in paths:
        print(path)
