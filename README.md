# RunPod: Qwen3.8-27B-Uncensored (HauhauCS Aggressive) Q8_K_P on llama.cpp

## Why the original vLLM command failed

```
vllm serve HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF ...
```

The repo is **public and ungated** (HTTP 200, `gated: false`) and the name is
correct, so it was not a typo or an auth problem. It contains **only `.gguf`
files** — no `config.json`, no `.safetensors`. vLLM cannot discover a model
there, hence "couldn't find or download".

The repo also ships `HauhauCS-FastMTP-llama.cpp.patch`: it is built for
llama.cpp, not vLLM.

Two more bugs in that command that would have surfaced next:

| flag | problem |
|---|---|
| `--max-model-len 128K` | int argument; argparse rejects `128K`. Use `131072`. |
| `--tokenizer Qwen/Qwen3.8-27B` | pointing a GGUF repo at a separate tokenizer does not make it loadable |

## What this serves

Q8_K_P, **31.46 GB**, sha256 `4e7735df...04bc2` (verified against the repo's
`SHA256SUMS`).

- 27B dense + vision encoder, 64 layers
- **262,144 native context** — the intended 128K is well within range
- Optional embedded-MTP speculative decoding (~2.2x document TG per the card)

## Run

```bash
export LLAMA_API_KEY='pick-a-long-random-string'
bash start_qwen38_q8.sh
```

Knobs (env): `CTX_SIZE` (131072), `PORT` (8000), `WANT_MTP` (1),
`WANT_VISION` (0), `MODELS_DIR` (/workspace/models), `LLAMA_SERVER`.

The script verifies sha256 before serving, resumes partial downloads
(`curl -C -`), and caches into `/workspace/models` so a pod restart does not
re-pull 31 GB.

Expose port 8000 as an **HTTP** port in the RunPod template. The script binds
`0.0.0.0` — the model card's example uses `127.0.0.1`, which would be
unreachable from outside the pod.

Verify:

```bash
curl -s localhost:8000/health
curl -s -H "Authorization: Bearer $LLAMA_API_KEY" localhost:8000/v1/models
```

### llama-server binary

Use a CUDA llama.cpp image if you can. Otherwise build **into `/workspace`** so
it survives restarts (a cold rebuild is 10-20 min of paid GPU time):

```bash
git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp
cmake -S /workspace/llama.cpp -B /workspace/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /workspace/llama.cpp/build --config Release -j"$(nproc)"
```

### MTP: two tiers

- **Embedded MTP** (`WANT_MTP=1`, the default) — plain upstream llama.cpp,
  `--spec-type draft-mtp`. ~2.23x document TG vs off.
- **HauhauCS FastMTP** — needs their patch on pinned commit `4df29be`, i.e. a
  from-source build every cold start unless you bake an image. ~3.02x. Worth it
  only for a long-lived pod.

All flags used here were checked against current upstream `common/arg.cpp`.

## Docker image

Built in GitHub Actions and pushed to `ghcr.io`, matching the pattern in
`negroISO/comfyui-base-runpod-bw` — **never build on the pod**. A four-architecture
CUDA build is 20-40 min of compile you would otherwise pay GPU rates for.

### First-time setup

```bash
cd E:\projects\runpod-llamacpp
git init && git add . && git commit -m "llama.cpp Qwen3.8 Q8_K_P RunPod image"
gh repo create qwen38-llamacpp-runpod --public --source=. --push
```

The ghcr **package is created automatically** by the first successful workflow
run — there is nothing to pre-create. The workflow already has
`permissions: packages: write`.

One manual step after that first run: a newly created package defaults to
**private**. Go to the package -> Package settings -> Change visibility ->
Public, or give RunPod a registry credential. A private package RunPod cannot
pull presents as a stuck "pulling image" with no useful error.

Image lands at:

```
ghcr.io/negroISO/qwen38-llamacpp:q8
```

Tags pushed: `q8`, `latest`, the commit SHA, and the branch name.

`workflow_dispatch` takes a `llama_ref` input, so you can pin or bump the
llama.cpp commit without editing the Dockerfile.

Locally, if you want: `docker build -t qwen38-llama .`

Base: `nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04` (current release; first line
shipping sm_103/B300 support), multi-stage down to the `runtime` image to drop
~5 GB of toolchain. Ubuntu 24.04 because llama.cpp needs CMake >= 3.31.8 (or
>= 4.0.2) for `f`-suffixed CUDA architectures.

### Why the arch list is set explicitly

llama.cpp's default for CUDA >= 12.9 is:

```
75-virtual 80-virtual 86-real 89-real 90-virtual 120a-real 121a-real
```

Two problems across the target hardware:

- `90-virtual` is **PTX only** — an H100/H200 JIT-compiles at every startup
  rather than loading native code.
- There is **nothing for sm_100**, so a B200 falls back to JIT-ing the sm_90
  PTX.

So the image sets:

```
90a-real;100a-real;120a-real;120-virtual
```

| target | hardware |
|---|---|
| `90a-real` | **H200**, and **H100** for the previous version |
| `100a-real` | **B200** |
| `120a-real` | **RTX PRO 6000 Blackwell** |
| `120-virtual` | PTX fallback for any other Blackwell (5090, B300, GB10) |

H100 and H200 are both compute 9.0 — same GH100 silicon, differing in HBM3e
capacity and bandwidth, not compute capability — so one target covers the pair.

All targets verified as valid in NVIDIA's CUDA 13.3 nvcc documentation.
`GGML_NATIVE=OFF` is mandatory — `ON` bakes the build machine's own arch in and
destroys portability.

Anything not listed still runs via the `120-virtual` PTX fallback (Blackwell) —
it just JIT-compiles on first load instead of starting instantly.

### Honest limit on B200

`ggml/src/ggml-cuda/common.cuh` says directly:

> While BW spans CC 1000, 1100 & 1200, we are integrating Tensor Core
> instructions available to **1200 family**

llama.cpp's Blackwell tensor-core work targets **sm_120 (consumer)**. sm_100
compiles and runs correctly, but there is no dedicated datacenter-Blackwell
fast path, so a B200 will not be proportionally faster than its raw FLOPs
suggest. **The RTX PRO 6000 Blackwell is the best-optimised target in this
set** — and at 96 GB it fits Q8_K_P at full context with room to spare, which
is exactly the configuration HauhauCS benchmarked on.

Build time is roughly 20-40 min for four architectures. Build once, push to a
registry, point the RunPod template at it — do not build on the pod.

## VRAM

Weights are **31.5 GB** before KV cache. MTP adds the 903 MB draft. KV cache at
131072 context is the variable that decides fit.

| GPU | VRAM | Q8_K_P @ 131072 ctx |
|---|---|---|
| **RTX PRO 6000 Blackwell** | 96 GB | comfortable — and the card HauhauCS benchmarked on (at 204800 ctx) |
| **B200** | 180 GB | ample; can raise `CTX_SIZE` toward the 262144 native limit |
| **H200** | 141 GB | ample |
| **H100** | 80 GB | fine at 131072 |

None of these four are tight for Q8_K_P — the constraint is cost, not capacity.
On price-per-token the RTX PRO 6000 is likely the best of the set here, since
it is both the cheapest and llama.cpp's best-optimised target.

---

## Client support — read before building the pod

This is the part that decides whether the goal is reachable.

| client | custom OpenAI endpoint? |
|---|---|
| **Codex** | **Yes** — configurable base URL |
| **Claude Code** | **No** — Anthropic API / Bedrock / Vertex only |
| **Reasonix** | unknown to me; check its own docs |

### Codex

Point it at the pod, OpenAI-compatible:

```bash
export OPENAI_BASE_URL="https://<pod-id>-8000.proxy.runpod.net/v1"
export OPENAI_API_KEY="$LLAMA_API_KEY"
```

Model name is `qwen3.8-27b-uncensored` (set via `--alias`).

### Claude Code

There is **no supported setting** that points Claude Code at a llama.cpp
server. It speaks the Anthropic Messages API; the only first-party redirects
are `CLAUDE_CODE_USE_BEDROCK` and `CLAUDE_CODE_USE_VERTEX`, neither of which
fits a self-hosted GGUF.

The community route is a third-party Anthropic<->OpenAI translation proxy. Be
aware that **tool calling is where those proxies typically break**, and tool
calling is most of what Claude Code does. Treat it as an experiment, not a
drop-in.

`--jinja` is on in the script because the chat template is what drives tool
calling; without it, tool use fails regardless of proxy.

### Worth weighing

A 27B at Q8 is not close to frontier models at agentic coding. If the goal is
**uncensored** output, that is a real reason to self-host. If the goal is
**cost**, price the pod hours against what you already pay — and note the 31 GB
pull on any cold start without a persistent volume.
