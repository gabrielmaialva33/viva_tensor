# Reproducible build for viva_tensor with the FP8 / sparse inference NIF.
#
# Two stages:
#   1) `builder`  pulls CUDA 12.9 + cuSPARSELt + Intel MKL + Zig 0.15.2 +
#                  Erlang/OTP 28 + Gleam 1.16, compiles the CUTLASS .a
#                  files and the Zig NIF, runs the Gleam test suite.
#   2) `runtime`  ships only the artefacts a downstream BEAM application
#                  needs to link against viva_tensor — the compiled .so,
#                  the .beam files, and the priv/ assets.
#
# Targets RTX 4090 (sm_89) by default; override with
#   docker build --build-arg CUDA_ARCH=sm_86 .  (for Ampere)
#
# Build:   docker build -t viva_tensor:2.2.101 .
# Run:     docker run --rm --gpus all viva_tensor:2.2.101 \
#            gleam run -m viva_tensor/bench/peak

# ----------------------------------------------------------------------
# Stage 1 — builder
# ----------------------------------------------------------------------
FROM nvidia/cuda:12.9.0-devel-ubuntu24.04 AS builder

ARG CUDA_ARCH=sm_89
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/zig:/opt/gleam:/usr/local/cuda/bin:/root/.mise/shims:$PATH

# System deps: build chain + Erlang/OTP 28
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl xz-utils git pkg-config \
    libstdc++-13-dev cmake ninja-build \
    erlang-base erlang-dev erlang-eunit erlang-tools \
    && rm -rf /var/lib/apt/lists/*

# Intel MKL (apt has it; use the lightweight runtime + headers package)
RUN apt-get update && apt-get install -y --no-install-recommends \
    intel-oneapi-mkl-devel \
    && rm -rf /var/lib/apt/lists/*

# Zig 0.15.2 (pinned — see CI workflow)
RUN curl -fsSL https://ziglang.org/download/0.15.2/zig-x86_64-linux-0.15.2.tar.xz \
    | tar -xJ -C /opt && mv /opt/zig-* /opt/zig

# Gleam 1.16
RUN curl -fsSL https://github.com/gleam-lang/gleam/releases/download/v1.16.0/gleam-v1.16.0-x86_64-unknown-linux-musl.tar.gz \
    | tar -xz -C /opt/gleam --strip-components=0

# cuSPARSELt (NVIDIA host repo)
RUN curl -fsSL https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.6.0.6-archive.tar.xz \
    | tar -xJ -C /opt && mv /opt/libcusparse_lt-* /opt/cusparselt

WORKDIR /src
COPY . .

# Build chain: pre-compiled CUTLASS object archives -> Zig NIF -> Gleam test
RUN make cutlass-libs CUDA_ARCH=${CUDA_ARCH} \
 && make zig \
 && gleam deps download \
 && gleam test

# ----------------------------------------------------------------------
# Stage 2 — runtime
# ----------------------------------------------------------------------
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/gleam:/root/.mise/shims:$PATH

# Minimal Erlang runtime + Gleam (no compilers in this layer).
RUN apt-get update && apt-get install -y --no-install-recommends \
    erlang-base erlang-tools ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# MKL runtime libraries (no headers).
RUN apt-get update && apt-get install -y --no-install-recommends \
    intel-oneapi-mkl \
    && rm -rf /var/lib/apt/lists/*

# Copy Gleam binary + project artefacts.
COPY --from=builder /opt/gleam /opt/gleam
COPY --from=builder /opt/cusparselt /opt/cusparselt
COPY --from=builder /src/build /workspace/build
COPY --from=builder /src/priv /workspace/priv
COPY --from=builder /src/gleam.toml /workspace/gleam.toml
COPY --from=builder /src/manifest.toml /workspace/manifest.toml
COPY --from=builder /src/src /workspace/src
COPY --from=builder /src/dev /workspace/dev

WORKDIR /workspace
ENV LD_LIBRARY_PATH=/opt/cusparselt/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

CMD ["gleam", "run", "-m", "viva_tensor/bench/peak"]
