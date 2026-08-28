# llama-server for Qwen3.8-27B-Uncensored Q8_K_P.
# One image spanning RTX 5090 (sm_120) -> H100/H200 (sm_90) -> B200/B300 (sm_100/103).
#
# CUDA 13.3.1: current release, and the first line that ships sm_103 (B300)
# support. Ubuntu 24.04 for a newer CMake than 22.04 offers -- llama.cpp needs
# CMake >= 3.31.8 (or >= 4.0.2) to accept the "f"-suffixed CUDA architectures.
FROM nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build ccache \
        libcurl4-openssl-dev ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

# Pin a commit for reproducible images. Override at build time to move forward.
ARG LLAMA_REF=master
RUN git clone https://github.com/ggml-org/llama.cpp /src && \
    cd /src && git checkout "${LLAMA_REF}"

# The architecture list is the whole point of this image, so it is set
# explicitly rather than left to llama.cpp's default.
#
# llama.cpp's default (CUDA >= 12.9) is:
#     75-virtual 80-virtual 86-real 89-real 90-virtual 120a-real 121a-real
# Two problems for this hardware range:
#   * 90-virtual is PTX only -- an H100/H200 JIT-compiles at every startup
#     instead of loading native code.
#   * there is NOTHING for sm_100, so a B200 would fall back to JIT-ing the
#     sm_90 PTX.
#
# Targeted hardware:
#   90a-real   H200, and H100 for the previous version  (a = wgmma, TMA)
#   100a-real  B200
#   120a-real  RTX PRO 6000 Blackwell
#   120-virtual  PTX fallback for any other Blackwell (5090, B300, GB10)
#
# H100 and H200 are both sm_90a, so one target covers the pair.
#
# NOTE on B200: ggml-cuda/common.cuh states plainly that while Blackwell spans
# CC 1000/1100/1200, llama.cpp integrates the tensor-core instructions
# available to the *1200 family*. sm_100 builds and runs correctly but gets no
# dedicated datacenter-Blackwell fast path, so a B200 will not be
# proportionally faster than its FLOPs suggest. The RTX PRO 6000 is the
# best-optimised target in this set.
ARG CUDA_ARCHS="90a-real;100a-real;120a-real;120-virtual"

# GGML_NATIVE=OFF is mandatory: ON bakes in the *build* machine's -march and
# CUDA arch, which would defeat a portable image.
#
# libcuda.so.1 is the CUDA *driver* API. It ships with the NVIDIA driver on a
# GPU host, NOT with the toolkit, so a CI runner has no copy and linking any
# binary that touches cuMemCreate/cuMemMap/etc fails with:
#     ld: warning: libcuda.so.1 ... not found
#     ld: undefined reference to `cuMemCreate'
# The toolkit ships link-time stubs for exactly this case (in
# cuda-cudart-dev-13-3). At runtime the real driver takes over.
#
# DO NOT hardcode /usr/local/cuda/lib64/stubs: CUDA 13 versions that path, and
# a previous attempt at this fix pointed at a directory that did not exist. It
# failed silently because `ln -sf` happily creates a dangling symlink and
# LIBRARY_PATH entries that do not exist are ignored -- so the build ran for 50
# minutes and died with the same link error. Find the stub, verify it, and fail
# loudly if it is missing.
RUN set -eux; \
    stub="$(find /usr/local /usr/lib -name 'libcuda.so' -path '*stubs*' \
              -print -quit 2>/dev/null)"; \
    test -n "$stub" || { echo "FATAL: no libcuda.so stub found"; exit 1; }; \
    dir="$(dirname "$stub")"; \
    ln -sf "$stub" "$dir/libcuda.so.1"; \
    echo "$dir" > /etc/ld.so.conf.d/cuda-stubs.conf; \
    echo "CUDA_STUB_DIR=$dir"; \
    printf '%s\n' "$dir" > /tmp/cuda_stub_dir; \
    printf 'extern int cuMemCreate();\nint main(){return cuMemCreate();}\n' > /tmp/t.c; \
    gcc /tmp/t.c -L"$dir" -lcuda -o /tmp/t \
      || { echo "FATAL: cannot link against libcuda stub in $dir"; exit 1; }; \
    echo "stub link check: OK"; \
    rm -f /tmp/t /tmp/t.c

# --target llama-server, not a full build. LLAMA_BUILD_EXAMPLES=OFF does NOT
# exclude tools/ -- llama-batched-bench, llama-bench, llama-imatrix and friends
# still build, and they are ~120 extra link steps for binaries this image never
# runs. Building just the server target is faster and removes a whole class of
# unrelated link failures.
# LIBRARY_PATH is exported from the discovered stub dir here rather than via
# ENV, because ENV cannot read a shell variable produced by an earlier RUN.
RUN set -eux; \
    export LIBRARY_PATH="$(cat /tmp/cuda_stub_dir):${LIBRARY_PATH:-}"; \
    echo "linking against stubs in: $LIBRARY_PATH"; \
    cmake -S /src -B /src/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DLLAMA_CURL=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DCMAKE_EXE_LINKER_FLAGS="-L$(cat /tmp/cuda_stub_dir)" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L$(cat /tmp/cuda_stub_dir)"; \
    cmake --build /src/build --config Release -j"$(nproc)" --target llama-server

# Collect by hand rather than `cmake --install`: a --target build populates
# only part of the install manifest, and a silently-empty install would not
# surface until the container failed to start. Fail loudly here instead.
RUN mkdir -p /opt/llama/bin /opt/llama/lib && \
    cp /src/build/bin/llama-server /opt/llama/bin/ && \
    cp /src/build/bin/*.so* /opt/llama/lib/ && \
    test -x /opt/llama/bin/llama-server && \
    ls /opt/llama/lib/libggml-cuda.so* >/dev/null

# ---------------------------------------------------------------------------
# Runtime: the CUDA *runtime* image, not devel. Drops ~5GB of toolchain.
FROM nvidia/cuda:13.3.1-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4 libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# The build stage already gathered llama-server and every ggml backend .so
# into /opt/llama, so one copy is enough. (A second `COPY *.so` would also
# miss the real files, which are versioned: libggml-cuda.so.0.22.0.)
COPY --from=build /opt/llama /opt/llama
ENV PATH="/opt/llama/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/llama/lib:${LD_LIBRARY_PATH}"

# Fail at build time, not on the pod, if the binary cannot resolve its libs.
# The only expected "not found" is libcuda.so.1, which the NVIDIA container
# runtime injects at run time.
RUN ldd /opt/llama/bin/llama-server | grep -v "libcuda.so.1" | grep "not found" \
    && { echo "FATAL: unresolved libraries above"; exit 1; } || true

COPY start_qwen38_q8.sh /usr/local/bin/start_qwen38_q8.sh
RUN chmod +x /usr/local/bin/start_qwen38_q8.sh

# Model cache lives on the RunPod volume so a restart does not re-pull 31GB.
ENV MODELS_DIR=/workspace/models \
    CTX_SIZE=131072 \
    PORT=8000 \
    WANT_MTP=1 \
    WANT_VISION=0
EXPOSE 8000

# Fails fast if the build has no code for the attached GPU.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["/usr/local/bin/start_qwen38_q8.sh"]
