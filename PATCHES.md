# Local Dependency Patches

This repository currently carries one local dependency patch through
`[patch.crates-io]` in `Cargo.toml`.

## candle-core 0.11.0

Path:

```text
vendor/candle-core-0.11.0
```

Reason:

`candle-core` 0.11.0 initializes Metal with:

```rust
let device = Device::all().swap_remove(ordinal);
```

If `Device::all()` returns an empty list, this panics instead of returning an
error. This was observed both from the PyO3 extension path and from a pure Rust
probe when Metal was not visible to the process.

Local patches:

```rust
let device = if ordinal == 0 {
    Device::system_default()
} else {
    Device::all().into_iter().nth(ordinal)
}
.ok_or_else(|| MetalError::Message(format!("no Metal device found for ordinal {ordinal}")))?;
```

The same file also removes one unused Metal import so `--features metal` builds
without dependency warnings.

Scope:

Only `src/metal_backend/mod.rs` differs from the crates.io `candle-core` 0.11.0
source, excluding crate metadata and lock files.

Long-term cleanup:

- replace the vendored directory with a minimal fork reference;
- or upgrade to a Candle release containing an equivalent fix;
- or submit this initialization fix upstream.
