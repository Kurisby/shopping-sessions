# Shopping Sessions: a native Rust pipeline (no dependencies beyond std).
# The scorer builds this natively on linux/amd64; build time is not timed.
FROM rust:1.97.0-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# x86-64-v3 = AVX2/BMI2/FMA, plus AES-NI for hashing: safe on every Cascade Lake or newer Xeon.
ENV RUSTFLAGS="-C target-cpu=x86-64-v3 -C target-feature=+aes"
RUN cargo build --release --locked && strip target/release/pipeline

FROM debian:bookworm-slim
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app
COPY --from=build /src/target/release/pipeline /app/pipeline
WORKDIR /app
CMD ["/app/pipeline"]
