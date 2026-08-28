#!/usr/bin/env bash
# RunPod startup: llama-server hosting Qwen3.8-27B-Uncensored (HauhauCS Aggressive) Q8_K_P.
#
# Why llama.cpp and not vLLM: the HauhauCS repo ships ONLY .gguf files -- no
# config.json, no safetensors. vLLM cannot discover a model there, which is why
# `vllm serve HauhauCS/...-GGUF` failed to "find or download" it. The repo is
# public and ungated; the format was the problem.
#
# Two things that also would have failed in the original command:
#   --max-model-len 128K   -> argparse wants an int. Use 131072.
#   --host 127.0.0.1       -> unreachable from outside the pod. Use 0.0.0.0.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/workspace/models}"
REPO="HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF"
BASE="https://huggingface.co/${REPO}/resolve/main"

MODEL_FILE="Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
DRAFT_FILE="Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf"
MMPROJ_FILE="mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf"

MODEL_SHA="4e7735df4d1e2ec721f2551f531b815702a2f89123238c564797eda4b0304bc2"
DRAFT_SHA="115e618e1f73cb50817ed5856f0551c6bf9c3d94df96f440eaca78dc63b8968b"
MMPROJ_SHA="5681b690bcb8eb10cd28d62d078cb4e01521a3ea4880a3fc7d54de72de2dd142"

# Native context is 262144. 131072 is what the original command intended.
CTX_SIZE="${CTX_SIZE:-131072}"
PORT="${PORT:-8000}"
# Vision needs the 931MB projector. Off by default: it costs VRAM and download.
WANT_VISION="${WANT_VISION:-0}"
# Speculative decoding with the HauhauCS FastMTP sidecar.
#
# OFF by default, and it must stay that way for this image. Verified on a live
# pod: enabling it makes llama-server refuse to start, in a crash loop --
#     check_tensor_dims: tensor 'output.weight' has wrong shape;
#     expected 5120, 248320, got 5120, 32768
# 32768 is the draft's vocab, 248320 is Qwen3.8's padded vocab. The model card
# names this exact symptom: the sidecar is fine, the EXECUTABLE is unpatched.
# FastMTP needs HauhauCS's llama.cpp patch on pinned commit 4df29be, which this
# image deliberately does not carry (it would mean compiling from source on
# every cold start).
#
# Turning this on without first building the patched llama.cpp will break the
# pod, not merely lose the speedup.
WANT_MTP="${WANT_MTP:-0}"

if [[ -z "${LLAMA_API_KEY:-}" ]]; then
  echo "FATAL: set LLAMA_API_KEY. An open port on a public pod is an open model." >&2
  exit 2
fi

mkdir -p "$MODELS_DIR"

fetch() {  # fetch <filename> <sha256>
  local f="$1" want="$2" dest="$MODELS_DIR/$1"
  if [[ -f "$dest" ]]; then
    echo "[models] $f present, verifying..."
    local got
    got="$(sha256sum "$dest" | awk '{print $1}')"
    if [[ "$got" == "$want" ]]; then
      echo "[models] $f OK (cached)"
      return
    fi
    echo "[models] $f checksum mismatch, refetching" >&2
    rm -f "$dest"
  fi
  echo "[models] downloading $f ..."
  # aria2c opens 16 parallel connections; curl uses ONE. Measured on this pod:
  # a single curl stream pulled 29.2 GiB in ~50 min, averaging ~10 MB/s while
  # swinging between 1.5 and 53 MB/s -- the signature of one connection at the
  # mercy of one CDN edge. Parallel chunks flatten that out.
  #
  # Both paths resume a partial file (-c / -C -), which matters because the
  # model lands on a persistent volume and RunPod network drops are common.
  if command -v aria2c >/dev/null 2>&1; then
    aria2c --console-log-level=warn --summary-interval=30 \
           -x16 -s16 -k16M -c --retry-wait=5 -m5 \
           --auto-file-renaming=false --allow-overwrite=true \
           -d "$MODELS_DIR" -o "$f" "$BASE/$f"
  else
    echo "[models] aria2c not found, falling back to single-stream curl" >&2
    curl -fL -C - --retry 5 --retry-delay 5 -o "$dest" "$BASE/$f"
  fi
  local got
  got="$(sha256sum "$dest" | awk '{print $1}')"
  if [[ "$got" != "$want" ]]; then
    echo "FATAL: $f checksum mismatch after download" >&2
    echo "  want $want" >&2
    echo "  got  $got" >&2
    exit 3
  fi
  echo "[models] $f OK"
}

fetch "$MODEL_FILE" "$MODEL_SHA"
[[ "$WANT_MTP" == "1" ]] && fetch "$DRAFT_FILE" "$DRAFT_SHA"
[[ "$WANT_VISION" == "1" ]] && fetch "$MMPROJ_FILE" "$MMPROJ_SHA"

# Locate llama-server: a prebuilt one on PATH, else a local build.
LLAMA_SERVER="${LLAMA_SERVER:-$(command -v llama-server || true)}"
if [[ -z "$LLAMA_SERVER" ]]; then
  for c in /opt/llama/bin/llama-server \
           /workspace/llama.cpp/build/bin/llama-server \
           /opt/llama.cpp/build/bin/llama-server; do
    [[ -x "$c" ]] && LLAMA_SERVER="$c" && break
  done
fi
if [[ -z "$LLAMA_SERVER" ]]; then
  echo "FATAL: no llama-server found." >&2
  echo "  Use a CUDA llama.cpp image, or build once into /workspace so it" >&2
  echo "  survives pod restarts:" >&2
  echo "    git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp" >&2
  echo "    cmake -S /workspace/llama.cpp -B /workspace/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release" >&2
  echo "    cmake --build /workspace/llama.cpp/build --config Release -j\"\$(nproc)\"" >&2
  exit 4
fi
echo "[serve] using $LLAMA_SERVER"

# Report the actual GPU. The image ships native code for sm_90a/100a/103a/
# 120a/121a; anything else falls back to JIT-ing PTX, which shows up as a long
# first-token delay rather than an error, so make it visible up front.
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total \
             --format=csv,noheader 2>/dev/null \
    | while IFS= read -r line; do echo "[gpu] $line"; done
fi

ARGS=(
  --model "$MODELS_DIR/$MODEL_FILE"
  --ctx-size "$CTX_SIZE"
  --n-gpu-layers all          # verified accepted by current arg.cpp; 64 layers
  --flash-attn on
  --no-mmap
  --parallel 1
  --batch-size 2048
  --ubatch-size 512
  --jinja                     # required: the chat template drives tool calling
  --host 0.0.0.0              # NOT 127.0.0.1, or the pod port maps to nothing
  --port "$PORT"
  --api-key "$LLAMA_API_KEY"
  --alias qwen3.8-27b-uncensored
)

# Sampler values from the model card's own serving profile.
ARGS+=( --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0
        --presence-penalty 0 --repeat-penalty 1.0 )

if [[ "$WANT_MTP" == "1" ]]; then
  ARGS+=( --spec-draft-model "$MODELS_DIR/$DRAFT_FILE"
          --spec-draft-ngl all
          --spec-type draft-mtp
          --spec-draft-n-max 3
          --spec-draft-p-min 0 )
fi

[[ "$WANT_VISION" == "1" ]] && ARGS+=( --mmproj "$MODELS_DIR/$MMPROJ_FILE" )

echo "[serve] ctx=$CTX_SIZE port=$PORT mtp=$WANT_MTP vision=$WANT_VISION"
echo "[serve] health:  curl -s localhost:$PORT/health"
echo "[serve] models:  curl -s -H 'Authorization: Bearer \$LLAMA_API_KEY' localhost:$PORT/v1/models"
exec "$LLAMA_SERVER" "${ARGS[@]}"
