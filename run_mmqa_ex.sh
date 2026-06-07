#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

shopt -s nullglob
mkdir -p Evaluation
result_files=(MMQA/sql_results/sql_generation_MMQA_*.json)

if (( ${#result_files[@]} == 0 )); then
  echo "No MMQA SQL result JSON files found under MMQA/sql_results." >&2
  exit 1
fi

for input_path in "${result_files[@]}"; do
  case "$input_path" in
    *_ex_eval.json)
      continue
      ;;
  esac

  input_name="$(basename "$input_path")"
  output_path="Evaluation/${input_name%.json}_ex_eval.json"
  if [[ -s "$output_path" && "${FORCE:-0}" != "1" ]]; then
    echo "Skipping existing EX output: $output_path"
    continue
  fi

  echo "Evaluating EX: $input_path"
  python Evaluation/evaluate_ex.py \
    --input "$input_path" \
    --output "$output_path" \
    --engine sqlite \
    --sqlite-dir MMQA/Sqlite_database \
    --timeout 60 \
    --progress-every 500 \
    "$@"
done

echo "Done. EX evaluation files are saved under Evaluation/. Use FORCE=1 to overwrite existing outputs."
