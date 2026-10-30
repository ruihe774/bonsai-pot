//! Host-resident conversation snapshots.
//!
//! A [`KvSnapshot`] holds everything a session needs to continue from
//! position `pos`: the `Q8_0` KV cache rows `[0..pos)` of every full-attention
//! layer and the complete recurrent state (conv + delta-rule state) of every
//! Gated-DeltaNet layer. Recurrent state summarizes the whole prefix, so a
//! snapshot restores exactly the position it was captured at.
//!
//! # Payload layout (after the 40-byte header)
//! ```text
//! for buf in [K, V]:
//!   for each full-attention layer:
//!     scales: pos * (kv_dim/32) * 4 bytes (f32)
//!     qs:     pos * kv_dim          bytes (i8)
//! state: state_words * 4 bytes (f32; omitted when pos == 0)
//! ```
//! A snapshot taken with one `max_seq` restores into a model loaded with any
//! `max_seq >= pos`.

use alloc::vec::Vec;

use crate::error::{PotError, Result};
use crate::model::Model;
use crate::session::Session;

const MAGIC: &[u8; 8] = b"BONSAI2S";
const VERSION: u32 = 2;
const HEADER_BYTES: usize = 40;
/// Largest single staging buffer used by capture / restore.
const CHUNK_BYTES: u64 = 256 << 20;

/// Opaque host-resident copy of a session's KV cache and recurrent state.
#[derive(Debug, Clone)]
pub struct KvSnapshot {
    n_attn: u32,
    kv_dim: u32,
    n_gdn: u32,
    state_words: u32,
    max_seq: u32,
    pos: u32,
    payload: Vec<u8>,
}

impl KvSnapshot {
    /// Number of tokens captured. `0` is a valid empty snapshot.
    #[must_use]
    pub const fn pos(&self) -> u32 {
        self.pos
    }
    /// Full-attention layers captured.
    #[must_use]
    pub const fn n_attn_layers(&self) -> u32 {
        self.n_attn
    }
    /// Gated-DeltaNet layers captured.
    #[must_use]
    pub const fn n_gdn_layers(&self) -> u32 {
        self.n_gdn
    }
    #[must_use]
    pub const fn kv_dim(&self) -> u32 {
        self.kv_dim
    }
    /// `max_seq` of the session the snapshot was captured from.
    #[must_use]
    pub const fn max_seq(&self) -> u32 {
        self.max_seq
    }

    /// Serialize to a flat byte blob (header + payload).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_BYTES + self.payload.len());
        out.extend_from_slice(MAGIC);
        for v in [
            VERSION,
            self.n_attn,
            self.kv_dim,
            self.n_gdn,
            self.state_words,
            self.max_seq,
            self.pos,
            0,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.payload);
        out
    }

    /// Parse a blob produced by [`Self::to_bytes`]. Checks the header and the
    /// payload size; model compatibility is checked by
    /// [`Session::restore`](crate::Session::restore).
    ///
    /// # Errors
    ///
    /// Returns [`PotError::Config`] for a short blob, wrong magic, unsupported
    /// version, non-zero reserved field, or inconsistent payload length.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_BYTES {
            return Err(PotError::Config(
                "KvSnapshot blob too short: missing header",
            ));
        }
        if &bytes[0..8] != MAGIC {
            return Err(PotError::Config("KvSnapshot blob has wrong magic"));
        }
        let word = |i: usize| {
            u32::from_le_bytes([
                bytes[8 + 4 * i],
                bytes[9 + 4 * i],
                bytes[10 + 4 * i],
                bytes[11 + 4 * i],
            ])
        };
        if word(0) != VERSION {
            return Err(PotError::Config("KvSnapshot blob has unsupported version"));
        }
        if word(7) != 0 {
            return Err(PotError::Config(
                "KvSnapshot blob has non-zero reserved field",
            ));
        }
        let snap = Self {
            n_attn: word(1),
            kv_dim: word(2),
            n_gdn: word(3),
            state_words: word(4),
            max_seq: word(5),
            pos: word(6),
            payload: Vec::new(),
        };
        if (bytes.len() - HEADER_BYTES) as u64 != snap.expected_payload() {
            return Err(PotError::Config(
                "KvSnapshot blob payload length does not match header",
            ));
        }
        Ok(Self {
            payload: bytes[HEADER_BYTES..].to_vec(),
            ..snap
        })
    }

    const fn kv_bytes_per_buffer(&self) -> u64 {
        self.n_attn as u64 * self.pos as u64 * (self.kv_dim as u64 + self.kv_dim as u64 / 8)
    }

    const fn expected_payload(&self) -> u64 {
        if self.pos == 0 {
            0
        } else {
            2 * self.kv_bytes_per_buffer() + self.state_words as u64 * 4
        }
    }
}

/// (buffer, byte offset, byte length) of every payload segment, in order.
fn segments<'a>(session: &'a Session<'_>, pos: u32) -> Vec<(&'a wgpu::Buffer, u64, u64)> {
    let model = session.model;
    let kv = &model.kv;
    let n_attn = model.cfg.n_attn_layers();
    let mut v = Vec::new();
    if pos == 0 {
        return v;
    }
    for buf in [&session.state.kv_k, &session.state.kv_v] {
        for idx in 0..n_attn {
            let (o, l) = kv.d_range(idx, pos);
            v.push((buf, o, l));
            let (o, l) = kv.qs_range(idx, pos);
            v.push((buf, o, l));
        }
    }
    v.push((&session.state.state, 0, model.state.total * 4));
    v
}

fn validate_against(snap: &KvSnapshot, model: &Model) -> Result<()> {
    let cfg = &model.cfg;
    if snap.n_attn != cfg.n_attn_layers()
        || snap.n_gdn != cfg.n_gdn_layers()
        || snap.kv_dim != cfg.kv_dim()
        || u64::from(snap.state_words) != model.state.total
    {
        return Err(PotError::Config(
            "KvSnapshot does not match the model geometry",
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

/// Read the live state of `session` back into a [`KvSnapshot`].
pub fn capture(session: &Session<'_>) -> Result<KvSnapshot> {
    let model = session.model;
    let cfg = &model.cfg;
    let mut snap = KvSnapshot {
        n_attn: cfg.n_attn_layers(),
        kv_dim: cfg.kv_dim(),
        n_gdn: cfg.n_gdn_layers(),
        state_words: model.state.total as u32,
        max_seq: model.max_seq,
        pos: session.pos,
        payload: Vec::new(),
    };
    let segs = segments(session, session.pos);
    let total: u64 = segs.iter().map(|s| s.2).sum();
    debug_assert_eq!(total, snap.expected_payload());
    let mut payload = vec![0u8; total as usize];
    let chunk = CHUNK_BYTES.min(total.max(4));
    let staging = (total > 0).then(|| {
        model.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("snapshot_staging"),
            size: chunk,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });
    // Walk the segments, filling the staging buffer up to `chunk` bytes per round.
    let mut seg = 0usize;
    let mut seg_off = 0u64;
    let mut written = 0u64;
    while let Some(staging) = staging.as_ref().filter(|_| written < total) {
        let mut enc = model
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("snapshot"),
            });
        let mut used = 0u64;
        while used < chunk && seg < segs.len() {
            let (buf, off, len) = segs[seg];
            let n = (len - seg_off).min(chunk - used);
            enc.copy_buffer_to_buffer(buf, off + seg_off, staging, used, n);
            used += n;
            seg_off += n;
            if seg_off == len {
                seg += 1;
                seg_off = 0;
            }
        }
        enc.map_buffer_on_submit(staging, wgpu::MapMode::Read, 0..used, |_| {});
        model.queue.submit([enc.finish()]);
        if let Err(e) = model.device.poll(wgpu::PollType::wait_indefinitely()) {
            model.check_device()?;
            return Err(PotError::Poll(e));
        }
        model.check_device()?;
        {
            let mapped = staging.slice(0..used).get_mapped_range();
            payload[written as usize..(written + used) as usize].copy_from_slice(&mapped);
        }
        staging.unmap();
        written += used;
    }
    snap.payload = payload;
    Ok(snap)
}

/// Upload `snap` into `session` and set its position.
pub fn apply(session: &mut Session<'_>, snap: &KvSnapshot) -> Result<()> {
    let model = session.model;
    validate_against(snap, model)?;
    let segs = segments(session, snap.pos);
    let mut src = 0usize;
    for (buf, off, len) in segs {
        let mut done = 0u64;
        while done < len {
            let n = (len - done).min(CHUNK_BYTES);
            model
                .queue
                .write_buffer(buf, off + done, &snap.payload[src..src + n as usize]);
            src += n as usize;
            done += n;
        }
    }
    model.queue.submit([]);
    session.pos = snap.pos;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snap(pos: u32) -> KvSnapshot {
        let mut s = KvSnapshot {
            n_attn: 16,
            kv_dim: 1024,
            n_gdn: 48,
            state_words: 100,
            max_seq: 4096,
            pos,
            payload: Vec::new(),
        };
        s.payload = (0..s.expected_payload()).map(|i| (i % 251) as u8).collect();
        s
    }

    #[test]
    fn roundtrip() {
        for pos in [0u32, 1, 7] {
            let s = make_snap(pos);
            let b = KvSnapshot::from_bytes(&s.to_bytes()).unwrap();
            assert_eq!(b.pos(), pos);
            assert_eq!(b.n_attn_layers(), 16);
            assert_eq!(b.n_gdn_layers(), 48);
            assert_eq!(b.kv_dim(), 1024);
            assert_eq!(b.max_seq(), 4096);
            assert_eq!(b.payload, s.payload);
        }
    }

    #[test]
    fn payload_size() {
        // 16 layers × (1024 + 128) bytes × 2 buffers per position + state.
        assert_eq!(make_snap(3).expected_payload(), 2 * 16 * 1152 * 3 + 400);
        assert_eq!(make_snap(0).expected_payload(), 0);
    }

    #[test]
    fn rejects_bad_blobs() {
        let good = make_snap(2).to_bytes();
        assert!(KvSnapshot::from_bytes(&good[..10]).is_err());
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(KvSnapshot::from_bytes(&bad).is_err());
        let mut bad = good.clone();
        bad[8] = 1;
        assert!(KvSnapshot::from_bytes(&bad).is_err());
        let mut bad = good.clone();
        bad[36] = 1;
        assert!(KvSnapshot::from_bytes(&bad).is_err());
        assert!(KvSnapshot::from_bytes(&good[..good.len() - 1]).is_err());
    }
}
