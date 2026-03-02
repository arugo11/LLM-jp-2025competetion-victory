#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FONT_DIR="$SCRIPT_DIR/fonts"
INPUT="${1:-$SCRIPT_DIR/report.typ}"
OUTPUT="${2:-$SCRIPT_DIR/report.pdf}"

if ! command -v typst >/dev/null 2>&1; then
  echo "error: typst is not installed." >&2
  exit 1
fi

if [[ ! -d "$FONT_DIR" ]]; then
  echo "error: bundled font directory was not found: $FONT_DIR" >&2
  exit 1
fi

font_list=$(typst fonts --font-path "$FONT_DIR")
required_fonts=(
  "Harano Aji Mincho"
  "Harano Aji Gothic"
  "TeXGyreTermesX"
  "TeX Gyre Heros"
)

missing_fonts=()
for font in "${required_fonts[@]}"; do
  if ! grep -Fxq "$font" <<<"$font_list"; then
    missing_fonts+=("$font")
  fi
done

if (( ${#missing_fonts[@]} > 0 )); then
  echo "error: typst could not discover the bundled report fonts." >&2
  echo "missing families:" >&2
  printf '  - %s\n' "${missing_fonts[@]}" >&2
  echo "hint: build the report via report/build.sh so --font-path report/fonts is always applied." >&2
  exit 1
fi

typst compile \
  --root "$SCRIPT_DIR" \
  --font-path "$FONT_DIR" \
  "$INPUT" \
  "$OUTPUT"
