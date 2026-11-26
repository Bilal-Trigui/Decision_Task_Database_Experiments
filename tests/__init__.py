from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def config_path(name):
    """configs/**/<name>.json: a settings file found by name in any subfolder of configs/ (exactly one)."""
    hits = sorted(p for p in (ROOT / "configs").rglob(f"{name}.json")
                  if not any(part.startswith(".") for part in p.relative_to(ROOT).parts))
    assert len(hits) == 1, f"{name}.json: {len(hits)} matches under configs/: {hits}"
    return hits[0]
