#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="${ROOT_DIR}/patches/mlx-sys-metallib-relative.patch"
CARGO_HOME="${CARGO_HOME:-${HOME}/.cargo}"

mlx_commit="$(
  sed -nE 's@.*source = "git\+https://github.com/OminiX-ai/OminiX-MLX.git[^"]*#([0-9a-f]{40})".*@\1@p' \
    "${ROOT_DIR}/src-tauri/Cargo.lock" | head -n1
)"

if [ -z "${mlx_commit}" ]; then
  echo "Failed to resolve mlx-sys commit from Cargo.lock." >&2
  exit 1
fi

mlx_short="${mlx_commit:0:7}"
build_rs="$(find "${CARGO_HOME}/git/checkouts" -path "*/${mlx_short}/mlx-rs/mlx-sys/build.rs" -print -quit 2>/dev/null || true)"

if [ -z "${build_rs}" ]; then
  echo "Failed to find mlx-sys checkout for commit ${mlx_commit} in ${CARGO_HOME}." >&2
  echo "Please run one normal cargo/tauri build first to let Cargo fetch OminiX-MLX." >&2
  exit 1
fi

repo_root="${build_rs%/mlx-rs/mlx-sys/build.rs}"
if git -C "${repo_root}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
  git -C "${repo_root}" apply "${PATCH_FILE}"
  echo "Applied mlx patch: ${repo_root}"
elif git -C "${repo_root}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "mlx patch already applied: ${repo_root}"
else
  echo "Failed to apply mlx patch: ${repo_root}" >&2
  exit 1
fi

cd "${ROOT_DIR}"
tauri build --config src-tauri/tauri.mlx.conf.json
