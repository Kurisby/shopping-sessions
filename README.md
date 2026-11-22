# Shopping Sessions: native Rust pipeline

A single-binary Rust implementation (standard library only, no crates) of the Shopping Sessions rules.

- Memory-maps every CSV under `$DATA_DIR`, splits files into line-aligned chunks and parses them on all cores.
- Links are joined with union-find (person = byte-wise smallest ID) while the events are parsed.
- Events are partitioned by event_id hash and deduplicated (smallest `ingest_ts`, before validation), then repartitioned by person.
- Each person partition is sorted by time, sessionized (gap > 30 min), credited to the latest click within 7 days (ties: smallest click_id) and counted for assists.
- Both tables are written with a small built-in Parquet writer (PLAIN, uncompressed, `day` as `DATE`).

Build and run like the starter: `docker build -t shopping-sessions .`, then run with `/data`, `/output` and `/scratch` mounted. Outside Docker: `cargo build --release && target/release/pipeline DATA_DIR OUTPUT_DIR`.

Passes all 36 fixtures and matches the published dev answer exactly.
