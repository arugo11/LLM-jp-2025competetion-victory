#!/usr/bin/env bash
set -euo pipefail

target="${1:-${EXP_DIR:-}}"
readonly file_scan_limit=100000

[[ -n "${target}" ]] || { echo "usage: $0 /groups/gcg51557/experiments/NNNN_slug" >&2; exit 2; }
[[ -d "${target}" ]] || { echo "target is not a directory: ${target}" >&2; exit 2; }

canonical_target="$(realpath "${target}")"
case "${canonical_target}" in
  /groups/gcg51557/experiments/[0-9][0-9][0-9][0-9]_*) ;;
  *) echo "target is outside the canonical experiment root" >&2; exit 3 ;;
esac

target_group="$(stat -c '%G' "${canonical_target}")"
[[ "${target_group}" == "gcg51557" ]] || { echo "target group mismatch" >&2; exit 3; }

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT
count_file="${work_dir}/count"

# Stop before a login-node audit can become an unbounded recursive workload.
if ! timeout 30s find "${canonical_target}" -xdev -printf '.' \
  | head -c "$((file_scan_limit + 1))" \
  | wc -c >"${count_file}"; then
  echo "bounded file scan timed out or failed; use an approved rt_HC audit job" >&2
  exit 3
fi
inode_count="$(tr -d '[:space:]' <"${count_file}")"
if (( inode_count > file_scan_limit )); then
  echo "bounded file scan exceeded ${file_scan_limit}; use an approved rt_HC audit job" >&2
  exit 3
fi

bytes="$(timeout 30s du -sb "${canonical_target}" | awk 'NR == 1 {print $1}')"
[[ "${bytes}" =~ ^[0-9]+$ ]] || { echo "bounded byte scan failed" >&2; exit 3; }

echo "audit_mode=bounded-pre-qsub"
echo "storage_audit_target=${canonical_target}"
echo "created_at=$(date -Is)"
echo "target_group=${target_group}"
echo "bytes=${bytes}"
echo "inode=${inode_count}"
echo "scan_status=complete"
echo "file_scan_limit=${file_scan_limit}"
