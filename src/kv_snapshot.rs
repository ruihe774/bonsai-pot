//! Host-resident KV-cache snapshots.
//!
//! A [`KvSnapshot`] is an owned blob of the GPU's KV-cache for positions
//! `[0..pos)`. It is not tied to any [`Session`](crate::Session) lifetime and
//! can be persisted to disk via [`KvSnapshot::to_bytes`] /
//! [`KvSnapshot::from_bytes`].
//!
//! # Payload layout
//!
//! Bytes after the 32-byte header are organized as:
//! ```text
//! for kind in [K, V]:
//!   for il in 0..n_layer:
//!     d-section:  pos * (kv_dim/32) * 4  bytes (FP32 scales)
//!     qs-section: pos * kv_dim            bytes (i8 quants)
//! ```
//! This matches the `Q8_0` block structure used by the engine's KV buffers.
//!
//! A snapshot captured with `max_seq=512` can be restored into a model loaded
//! with `max_seq=2048` as long as `pos ≤ model.max_seq_len()`.

use std::result::Result as StdResult;
use std::sync::{Arc, OnceLock};

use crate::error::{PotError, Result};
use crate::model::Model;

// ---- on-disk header ---------------------------------------------------------

const MAGIC: &[u8; 8] = b"BONSAIKV";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 32;

/// Opaque host-resident copy of a Session's GPU KV-cache state.
///
/// `Clone` deep-copies the payload. `Send + Sync` (auto). Use
/// [`KvSnapshot::to_bytes`] / [`KvSnapshot::from_bytes`] to persist to disk.
#[derive(Debug, Clone)]
pub struct KvSnapshot {
    n_layer: u32,
    kv_dim: u32,
    max_seq: u32,
    pos: u32,
    /// K bytes then V bytes, in canonical packed order (see module docs).
    payload: Vec<u8>,
}

impl KvSnapshot {
    /// Number of tokens captured. `0` is a valid empty snapshot.
    #[must_use]
    pub const fn pos(&self) -> u32 {
        self.pos
    }
    #[must_use]
    pub const fn n_layer(&self) -> u32 {
        self.n_layer
    }
    #[must_use]
    pub const fn kv_dim(&self) -> u32 {
        self.kv_dim
    }
    #[must_use]
    pub const fn max_seq(&self) -> u32 {
        self.max_seq
    }

    /// Serialize to a flat byte blob (32-byte header + payload).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_BYTES + self.payload.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.n_layer.to_le_bytes());
        out.extend_from_slice(&self.kv_dim.to_le_bytes());
        out.extend_from_slice(&self.max_seq.to_le_bytes());
        out.extend_from_slice(&self.pos.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved
        out.extend_from_slice(&self.payload);
        out
    }

    /// Parse a blob produced by [`Self::to_bytes`].
    ///
    /// Validates magic, version, non-zero reserved field, and the declared
    /// (`n_layer`, `kv_dim`, `pos`) → payload-size invariant before allocating.
    /// Does **not** validate against any [`Model`]; call
    /// [`Session::restore`](crate::Session::restore) for that.
    ///
    /// # Errors
    ///
    /// Returns [`PotError::Config`] if the blob is too short, has the wrong
    /// magic, an unsupported version, a non-zero reserved field, or a payload
    /// length inconsistent with the declared header fields.
    ///
    /// # Panics
    ///
    /// Does not panic in practice: the fixed-size slice conversions used
    /// internally are guaranteed to succeed once the length precondition is
    /// met.
    #[allow(
        clippy::unwrap_used,
        reason = "fixed-size slice → array conversions are infallible after the length check above"
    )]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_BYTES {
            return Err(PotError::Config(
                "KvSnapshot blob too short: missing header",
            ));
        }
        if &bytes[0..8] != MAGIC {
            return Err(PotError::Config("KvSnapshot blob has wrong magic"));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != VERSION {
            return Err(PotError::Config("KvSnapshot blob has unsupported version"));
        }
        let n_layer = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let kv_dim = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let max_seq = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let pos = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        let reserved = u32::from_le_bytes(bytes[28..32].try_into().unwrap());
        if reserved != 0 {
            return Err(PotError::Config(
                "KvSnapshot blob has non-zero reserved field",
            ));
        }
        let expected = payload_bytes(n_layer, kv_dim, pos);
        if bytes.len() - HEADER_BYTES != expected {
            return Err(PotError::Config(
                "KvSnapshot blob payload length does not match header",
            ));
        }
        Ok(Self {
            n_layer,
            kv_dim,
            max_seq,
            pos,
            payload: bytes[HEADER_BYTES..].to_vec(),
        })
    }
}

// ---- layout helpers ---------------------------------------------------------

const fn nb(kv_dim: u32) -> u64 {
    (kv_dim / 32) as u64
}

/// Total payload bytes for a snapshot of `pos` tokens.
fn payload_bytes(n_layer: u32, kv_dim: u32, pos: u32) -> usize {
    let d_layer = u64::from(pos) * nb(kv_dim) * 4;
    let qs_layer = u64::from(pos) * u64::from(kv_dim);
    (2 * u64::from(n_layer) * (d_layer + qs_layer)) as usize
}

/// Byte offset into the snapshot payload for a given (kind, layer, section).
/// kind: 0 = K, 1 = V.  section: 0 = d (FP32 scales), 1 = qs (i8 quants).
fn payload_offset(n_layer: u32, kv_dim: u32, pos: u32, kind: u32, il: u32, sect: u32) -> usize {
    let d_layer = u64::from(pos) * nb(kv_dim) * 4;
    let qs_layer = u64::from(pos) * u64::from(kv_dim);
    let layer_bytes = d_layer + qs_layer;
    let base_kv = u64::from(kind) * u64::from(n_layer) * layer_bytes;
    let base_il = base_kv + u64::from(il) * layer_bytes;
    if sect == 0 {
        base_il as usize
    } else {
        (base_il + d_layer) as usize
    }
}

/// Byte offset of layer `il` in `kv_k`/`kv_v` (d-section start).
const fn buf_d_offset(max_seq: u32, kv_dim: u32, il: u32) -> u64 {
    il as u64 * max_seq as u64 * nb(kv_dim) * 4
}

/// Byte offset of layer `il` in `kv_k`/`kv_v` (qs-section start).
const fn buf_qs_offset(n_layer: u32, max_seq: u32, kv_dim: u32, il: u32) -> u64 {
    let d_total = (n_layer as u64) * (max_seq as u64) * nb(kv_dim) * 4;
    d_total + (il as u64) * (max_seq as u64) * (kv_dim as u64)
}

// ---- validate ---------------------------------------------------------------

pub const fn validate_against(snap: &KvSnapshot, model: &Model) -> Result<()> {
    let cfg = &model.cfg;
    if snap.n_layer != cfg.n_layer || snap.kv_dim != cfg.kv_dim {
        return Err(PotError::Config(
            "KvSnapshot does not match model: n_layer or kv_dim differs",
        ));
    }
    if snap.pos > model.max_seq {
        return Err(PotError::ContextOverflow {
            pos: 0,
            n: snap.pos,
            max: model.max_seq,
        });
    }
    Ok(())
}

// ---- GPU paths --------------------------------------------------------------

/// Read back the live `[0..pos)` slice of the GPU KV cache to a [`KvSnapshot`].
pub fn capture(model: &Model, pos: u32) -> Result<KvSnapshot> {
    type MapResult = StdResult<(), wgpu::BufferAsyncError>;
    let cfg = &model.cfg;
    let n_layer = cfg.n_layer;
    let kv_dim = cfg.kv_dim;
    let max_seq = model.max_seq;

    let snap_payload_bytes = payload_bytes(n_layer, kv_dim, pos);

    // Empty snapshot: skip GPU work entirely.
    if pos == 0 {
        return Ok(KvSnapshot {
            n_layer,
            kv_dim,
            max_seq,
            pos: 0,
            payload: Vec::new(),
        });
    }

    // One-shot staging buffer sized to the live prefix only.
    let staging = model.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("kv_snapshot_staging"),
        size: snap_payload_bytes as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Build one CommandEncoder with 4 × n_layer small copies:
    // for kind in {K,V} × for il in 0..n_layer × for sect in {d, qs}.
    let mut enc = model
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("kv_snapshot_enc"),
        });

    let d_size = u64::from(pos) * nb(kv_dim) * 4;
    let qs_size = u64::from(pos) * u64::from(kv_dim);

    for kind in 0u32..2 {
        let src_buf = if kind == 0 {
            &model.buffers.kv_k
        } else {
            &model.buffers.kv_v
        };
        for il in 0..n_layer {
            // d-section
            enc.copy_buffer_to_buffer(
                src_buf,
                buf_d_offset(max_seq, kv_dim, il),
                &staging,
                payload_offset(n_layer, kv_dim, pos, kind, il, 0) as u64,
                d_size,
            );
            // qs-section
            enc.copy_buffer_to_buffer(
                src_buf,
                buf_qs_offset(n_layer, max_seq, kv_dim, il),
                &staging,
                payload_offset(n_layer, kv_dim, pos, kind, il, 1) as u64,
                qs_size,
            );
        }
    }

    let slot: Arc<OnceLock<MapResult>> = Arc::new(OnceLock::new());
    let slot2 = slot.clone();
    enc.map_buffer_on_submit(
        &staging,
        wgpu::MapMode::Read,
        0..snap_payload_bytes as u64,
        move |res| {
            let _ = slot2.set(res);
        },
    );
    model.queue.submit([enc.finish()]);

    let slice = staging.slice(0..snap_payload_bytes as u64);
    if let Err(e) = model.device.poll(wgpu::PollType::wait_indefinitely()) {
        model.check_device()?;
        return Err(PotError::Poll(e));
    }
    match slot.get() {
        Some(Ok(())) => {}
        Some(Err(e)) => {
            model.check_device()?;
            return Err(PotError::BufferMap(e.clone()));
        }
        None => unreachable!("map_async callback did not fire before poll returned"),
    }

    let mut payload = vec![0u8; snap_payload_bytes];
    {
        let mapped = slice.get_mapped_range();
        payload.copy_from_slice(&mapped);
    }
    staging.unmap();

    Ok(KvSnapshot {
        n_layer,
        kv_dim,
        max_seq,
        pos,
        payload,
    })
}

/// Upload `snap`'s payload into the GPU KV cache.
pub fn apply(model: &Model, snap: &KvSnapshot) -> Result<()> {
    validate_against(snap, model)?;

    if snap.pos == 0 {
        return Ok(());
    }

    let cfg = &model.cfg;
    let n_layer = cfg.n_layer;
    let kv_dim = cfg.kv_dim;
    let max_seq = model.max_seq;

    // d_size: pos * nb(kv_dim) * 4 bytes — multiple of 4 by construction.
    // qs_size: pos * kv_dim bytes — kv_dim is a multiple of 32 (Q8_0 block),
    // so qs_size is always 4-aligned.
    let d_size = u64::from(snap.pos) * nb(kv_dim) * 4;
    let qs_size = u64::from(snap.pos) * u64::from(kv_dim);

    // Stage all per-layer d/qs writes into a single encoder so the whole
    // restore lands in one submit. Per-call write sizes may exceed the belt's
    // chunk_size (4 MiB) for the qs-section at large pos/kv_dim; the belt
    // allocates a one-off chunk for those, which is fine for a one-shot op.
    let mut enc = model
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("kv_restore"),
        });
    for kind in 0u32..2 {
        let dst_buf = if kind == 0 {
            &model.buffers.kv_k
        } else {
            &model.buffers.kv_v
        };
        for il in 0..n_layer {
            // d-section
            let src_off = payload_offset(n_layer, kv_dim, snap.pos, kind, il, 0);
            let dst_off = buf_d_offset(max_seq, kv_dim, il);
            model.belt_write(
                &mut enc,
                dst_buf,
                dst_off,
                &snap.payload[src_off..src_off + d_size as usize],
            );
            // qs-section
            let src_off = payload_offset(n_layer, kv_dim, snap.pos, kind, il, 1);
            let dst_off = buf_qs_offset(n_layer, max_seq, kv_dim, il);
            model.belt_write(
                &mut enc,
                dst_buf,
                dst_off,
                &snap.payload[src_off..src_off + qs_size as usize],
            );
        }
    }
    let cb = enc.finish();
    model.belt_finish();
    model.queue.submit(Some(cb));
    model.belt_recall();

    Ok(())
}

// ---- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snap(n_layer: u32, kv_dim: u32, max_seq: u32, pos: u32) -> KvSnapshot {
        let len = payload_bytes(n_layer, kv_dim, pos);
        let payload: Vec<u8> = (0..len).map(|i| (i & 0xFF) as u8).collect();
        KvSnapshot {
            n_layer,
            kv_dim,
            max_seq,
            pos,
            payload,
        }
    }

    #[test]
    fn header_roundtrip() {
        let snap = make_snap(36, 1024, 1024, 64);
        let bytes = snap.to_bytes();
        let snap2 = KvSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(snap2.pos, snap.pos);
        assert_eq!(snap2.n_layer, snap.n_layer);
        assert_eq!(snap2.kv_dim, snap.kv_dim);
        assert_eq!(snap2.max_seq, snap.max_seq);
        assert_eq!(snap2.payload, snap.payload);
    }

    #[test]
    fn empty_roundtrip() {
        let snap = make_snap(36, 1024, 1024, 0);
        assert_eq!(snap.payload.len(), 0);
        let bytes = snap.to_bytes();
        assert_eq!(bytes.len(), HEADER_BYTES);
        let snap2 = KvSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(snap2.pos, 0);
        assert_eq!(snap2.payload.len(), 0);
    }

    #[test]
    fn from_bytes_bad_magic() {
        let snap = make_snap(36, 1024, 1024, 4);
        let mut bytes = snap.to_bytes();
        bytes[0] = 0xFF;
        assert!(KvSnapshot::from_bytes(&bytes).is_err());
    }

    #[test]
    fn from_bytes_truncated() {
        let snap = make_snap(36, 1024, 1024, 4);
        let bytes = snap.to_bytes();
        assert!(KvSnapshot::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn payload_size_bonsai4b() {
        // For Bonsai-4B at pos=512: 2 × 36 × 512 × (1024 + 128) = 42_467_328 bytes.
        assert_eq!(payload_bytes(36, 1024, 512), 42_467_328);
    }

    #[test]
    fn payload_size_bonsai8b() {
        // Bonsai-8B: n_layer=36, kv_dim=1024 (n_kv_head=8 × head_dim=128).
        // Identical KV layout to 4B — payload depends only on (n_layer, kv_dim, pos).
        assert_eq!(payload_bytes(36, 1024, 512), 42_467_328);
    }

    #[test]
    fn payload_size_bonsai8b_full_useful_ctx() {
        // 8B at the practical ~32k cap (2× rope_orig_context=16384):
        // 2 × 36 × 32768 × (1024 + 128) = 2_717_908_992 bytes.
        // Verifies the arithmetic stays u64-clean above 2 GB.
        assert_eq!(payload_bytes(36, 1024, 32768), 2_717_908_992);
    }

    #[test]
    fn payload_offsets_no_overlap() {
        // Verify d and qs regions for consecutive layers don't overlap.
        let (n_layer, kv_dim, pos) = (36u32, 1024u32, 16u32);
        for kind in 0u32..2 {
            for il in 0..n_layer {
                let d_start = payload_offset(n_layer, kv_dim, pos, kind, il, 0);
                let d_end = d_start + (pos as usize * (kv_dim / 32) as usize * 4);
                let qs_start = payload_offset(n_layer, kv_dim, pos, kind, il, 1);
                let qs_end = qs_start + (pos as usize * kv_dim as usize);
                assert!(d_end <= qs_start, "d and qs overlap at kind={kind} il={il}");
                if il + 1 < n_layer {
                    let next_d = payload_offset(n_layer, kv_dim, pos, kind, il + 1, 0);
                    assert!(qs_end <= next_d, "qs/next-d overlap at kind={kind} il={il}");
                }
            }
        }
    }

    #[test]
    fn from_bytes_bad_version() {
        let snap = make_snap(36, 1024, 1024, 4);
        let mut bytes = snap.to_bytes();
        // VERSION field is at bytes[8..12].
        let bad = (VERSION + 1).to_le_bytes();
        bytes[8..12].copy_from_slice(&bad);
        assert!(KvSnapshot::from_bytes(&bytes).is_err());
    }

    #[test]
    fn from_bytes_nonzero_reserved() {
        let snap = make_snap(36, 1024, 1024, 4);
        let mut bytes = snap.to_bytes();
        // Reserved field is at bytes[28..32].
        bytes[28] = 1;
        assert!(KvSnapshot::from_bytes(&bytes).is_err());
    }

    #[test]
    fn from_bytes_payload_length_mismatch() {
        let snap = make_snap(36, 1024, 1024, 4);
        let bytes = snap.to_bytes();
        // Truncate the payload by one byte — header says longer than actual.
        assert!(KvSnapshot::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn getters_match_header() {
        let snap = make_snap(36, 1024, 512, 64);
        assert_eq!(snap.n_layer(), 36);
        assert_eq!(snap.kv_dim(), 1024);
        assert_eq!(snap.max_seq(), 512);
        assert_eq!(snap.pos(), 64);
    }

    #[test]
    fn clone_is_deep() {
        let snap = make_snap(4, 128, 256, 8);
        let mut clone = snap.clone();
        if !clone.payload.is_empty() {
            clone.payload[0] ^= 0xFF;
            assert_ne!(snap.payload[0], clone.payload[0]);
        }
    }

    #[test]
    fn payload_bytes_zero_pos() {
        // An empty snapshot (pos=0) has no payload.
        assert_eq!(payload_bytes(36, 1024, 0), 0);
    }
}
