"""Run FLA pytest targets on a Modal GPU. The only Modal entry point for this repo.

    modal run plans/modal/runner/modal_test_runner.py \
        --pytest-args "tests/ops/test_gdn_kernels.py -k chunk_bwd_dqkwg"

Environment knobs (read locally at launch): FLA_MODAL_GPU (default H100, e.g. "H100:8"),
FLA_MODAL_TIMEOUT_MIN (default 30).
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

REMOTE_REPO = "/root/fla"
TRITON_CACHE = "/root/.triton"
IGNORE = [".git", ".venv", "plans", "profile", "images", "**/__pycache__", ".pytest_cache", ".ruff_cache", "*.egg-info"]

image = modal.Image.debian_slim(python_version="3.12").apt_install("build-essential")
if modal.is_local():
    import tomllib

    REPO = Path(__file__).resolve().parents[3]
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    # Dependencies come from pyproject.toml, so this file never carries its own pin list.
    image = image.uv_pip_install(*project["dependencies"], *extras["cuda"], *extras["test"])
    # Mounted at container start, not baked in: source edits do not rebuild the image.
    image = image.add_local_dir(str(REPO), REMOTE_REPO, ignore=IGNORE)

app = modal.App("fla-tests")
triton_cache = modal.Volume.from_name("fla-triton-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=os.environ.get("FLA_MODAL_GPU", "H100"),
    timeout=60 * int(os.environ.get("FLA_MODAL_TIMEOUT_MIN", "30")),
    volumes={TRITON_CACHE: triton_cache},
)
def run_pytest(pytest_args: list[str], source: str) -> int:
    import torch
    import triton

    print(f"source: {source}")
    print(f"torch {torch.__version__} | triton {triton.__version__} | cuda {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        print(f"gpu{i}: {torch.cuda.get_device_name(i)} sm{''.join(map(str, torch.cuda.get_device_capability(i)))}")
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *pytest_args]
    print("cmd: FLA_CI_ENV=0", shlex.join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=REMOTE_REPO, env={**os.environ, "FLA_CI_ENV": "0"}).returncode
    triton_cache.commit()
    return code


@app.local_entrypoint()
def main(pytest_args: str = "-q tests/ops/test_gdn_kernels.py -k chunk_bwd_dqkwg"):
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no")
    source = f"{git('rev-parse', '--abbrev-ref', 'HEAD')}@{git('rev-parse', '--short', 'HEAD')}" + (" +dirty" if dirty else "")
    code = run_pytest.remote(shlex.split(pytest_args), source)
    print(f"pytest exit code: {code}")
    raise SystemExit(code)
