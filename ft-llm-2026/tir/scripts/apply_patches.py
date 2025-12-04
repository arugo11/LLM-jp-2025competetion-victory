from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Iterable, Optional


PATCH_TARGETS: list[tuple[str, str]] = [
    (
        "model_hosting_container_standards.sagemaker.sessions.manager",
        "sagemaker_sessions.patch",
    ),
    (
        "nemo_skills.inference.server.serve_vllm",
        "serve_vllm_enforce_eager.patch",
    ),
]


def find_module_path(module_name: str) -> Optional[Path]:
    """
    Locate a module file on disk **without importing it** (to avoid side effects
    like /dev/shm access before patching).
    """
    parts = module_name.split(".")

    # Candidate base dirs where site-packages may live inside the current venv.
    base_candidates = []
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        val = paths.get(key)
        if val:
            base_candidates.append(Path(val))

    # Fallback: sys.path entries that look like site-packages.
    for p in sys.path:
        if "site-packages" in p or "dist-packages" in p:
            base_candidates.append(Path(p))

    for base in base_candidates:
        candidate = base.joinpath(*parts)
        if candidate.is_dir():
            # package; expect __init__.py inside
            init_py = candidate / "__init__.py"
            if init_py.exists():
                return init_py
        else:
            # module file
            module_py = candidate.with_suffix(".py")
            if module_py.exists():
                return module_py

    print(f"[skip] {module_name}: module not found in site-packages")
    return None


def apply_patch(target_path: Path, patch_path: Path) -> bool:
    if not patch_path.exists():
        print(f"[skip] patch file not found: {patch_path}")
        return False

    cmd = ["patch", str(target_path)]
    with patch_path.open("r", encoding="utf-8") as pf:
        result = subprocess.run(
            cmd,
            stdin=pf,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    if result.returncode == 0:
        print(f"[applied] {patch_path.name} -> {target_path}")
        return True

    already_patched = (
        "Reversed (or previously applied) patch detected" in result.stdout
        or "Reversed (or previously applied) patch detected" in result.stderr
        or "Skipping patch" in result.stdout
        or "Skipping patch" in result.stderr
    )
    if already_patched:
        print(f"[skip] {patch_path.name}: already applied")
        return True

    print(f"[error] failed to apply {patch_path.name} to {target_path}")
    if result.stdout.strip():
        print("stdout:\n" + result.stdout)
    if result.stderr.strip():
        print("stderr:\n" + result.stderr)
    return False


def main(targets: Iterable[tuple[str, str]] = PATCH_TARGETS) -> int:
    script_dir = Path(__file__).resolve().parent
    patches_dir = script_dir.parent / "patches"

    all_ok = True
    for module_name, patch_file in targets:
        target_path = find_module_path(module_name)
        if target_path is None:
            all_ok = False
            continue

        patch_path = patches_dir / patch_file
        ok = apply_patch(target_path, patch_path)
        all_ok = all_ok and ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
