#!/usr/bin/env bash
# Ambiente de ingestão com GPU (ADR-050): o venv do projeto com o onnxruntime-gpu (e as bibliotecas de CUDA e cuDNN em
# wheels) no lugar do onnxruntime de CPU. Os dois pacotes ocupam o mesmo módulo, então não convivem num venv: este fica
# em .cache/gpu-venv e serve só para `axon-scouting-corpus ingest|reindex`; o venv do projeto e a imagem de produção
# continuam com a CPU.
#
#   scripts/gpu-venv.sh
#   AXON_SCOUTING_EMBEDDING_DIR=.cache/e5 AXON_SCOUTING_EMBEDDING_DEVICE=cuda \
#     .cache/gpu-venv/bin/axon-scouting-corpus reindex corpus/corpus.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
export XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
VENV=.cache/gpu-venv
uv venv --allow-existing --python 3.14 "$VENV"
uv export --frozen --no-dev --no-emit-project --no-hashes -o "$VENV/requirements.txt" >/dev/null
uv pip install --python "$VENV/bin/python" -r "$VENV/requirements.txt"
uv pip install --python "$VENV/bin/python" --no-deps -e .
uv pip uninstall --python "$VENV/bin/python" onnxruntime
uv pip install --python "$VENV/bin/python" "onnxruntime-gpu[cuda,cudnn]>=1.24"
"$VENV/bin/python" - <<'PY'
import onnxruntime as ort
print("onnxruntime", ort.__version__, "provedores:", ort.get_available_providers())
PY
