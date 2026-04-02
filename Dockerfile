# syntax=docker/dockerfile:1
FROM alpine:3.21 AS builder

# Build deps: OpenCV headers, clang for bindgen, build tools
RUN apk add --no-cache \
    curl \
    build-base \
    clang19-dev \
    llvm19-dev \
    opencv-dev \
    pkgconf

# Install rustup with latest stable (Alpine's apk Rust 1.83 is too old for edition 2024)
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUSTFLAGS="-C target-feature=-crt-static"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable --profile minimal

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY tests/ tests/

# --- Test stage: docker buildx build --target test . ---
FROM builder AS test
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git \
    cargo test

# --- Release build ---
FROM builder AS release
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git \
    cargo build --release --bin oven-vision

# --- Runtime ---
FROM alpine:3.21

RUN apk add --no-cache opencv libstdc++

COPY --from=release /build/target/release/oven-vision /usr/local/bin/oven-vision

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/oven-vision"]
