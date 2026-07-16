from collections import Counter
from pathlib import Path

from tv_gptoss120b.config import load_config
from tv_gptoss120b.specs import build_specs

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def test_specs_are_stable_and_exactly_partitioned() -> None:
    config = load_config(CONFIG)
    first = build_specs(config.data)
    second = build_specs(config.data)
    assert first == second
    assert len(first) == 512
    assert len({item.spec_id for item in first}) == 512
    assert Counter(item.split for item in first) == {"difficulty": 128, "sft": 256, "grpo": 128}
    assert min(item.declared_difficulty for item in first) == 6
    assert max(item.declared_difficulty for item in first) == 10
    cells = Counter((item.category, item.unit, item.declared_difficulty) for item in first)
    assert max(cells.values()) - min(cells.values()) <= 1


def test_spec_hashes_are_unique() -> None:
    config = load_config(CONFIG)
    specs = build_specs(config.data)
    assert len({item.spec_sha256 for item in specs}) == len(specs)
