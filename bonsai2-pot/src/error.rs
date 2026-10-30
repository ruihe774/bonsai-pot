use alloc::string::String;
use core::result::Result as StdResult;
#[cfg(feature = "std")]
use std::io;
#[cfg(feature = "std")]
use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PotError {
    #[cfg(feature = "std")]
    #[error("io error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("vocab.bin / vocab_offsets.bin: {0}")]
    Vocab(&'static str),
    #[error("no compatible GPU adapter found")]
    NoAdapter,
    #[error("no compatible GPU device found")]
    NoDevice,
    #[error("adapter does not support required feature: {0}")]
    FeatureUnsupported(&'static str),
    #[error("context overflow: pos {pos} + tokens {n} > max_seq {max}")]
    ContextOverflow { pos: u32, n: u32, max: u32 },
    #[error("prefill batch {n} exceeds max_prefill_tokens {max}")]
    PrefillTooLarge { n: u32, max: u32 },
    #[error("invalid config: {0}")]
    Config(&'static str),
    #[error("wgpu device lost ({reason:?}): {message}")]
    DeviceLost {
        reason: wgpu::DeviceLostReason,
        message: String,
    },
    #[error("device.poll failed: {0:?}")]
    Poll(wgpu::PollError),
}

pub type Result<T> = StdResult<T, PotError>;

#[cfg(test)]
mod tests {
    use alloc::string::{String, ToString as _};
    use core::error::Error as _;

    use super::PotError;

    #[cfg(feature = "std")]
    fn make_io_error() -> PotError {
        use std::io;
        use std::path::PathBuf;
        PotError::Io {
            path: PathBuf::from("/tmp/fake"),
            source: io::Error::new(io::ErrorKind::NotFound, "not found"),
        }
    }

    #[test]
    fn display_each_variant() {
        #[cfg(feature = "std")]
        assert!(make_io_error().to_string().contains("/tmp/fake"));
        assert!(
            PotError::Config("bad ini key")
                .to_string()
                .contains("bad ini key")
        );
        assert!(
            PotError::Vocab("bad magic")
                .to_string()
                .contains("bad magic")
        );
        assert!(PotError::NoAdapter.to_string().contains("GPU adapter"));
        assert!(
            PotError::FeatureUnsupported("SHADER_F16")
                .to_string()
                .contains("SHADER_F16")
        );
        assert!(
            PotError::ContextOverflow {
                pos: 1020,
                n: 8,
                max: 1024
            }
            .to_string()
            .contains("pos 1020")
        );
        assert!(
            PotError::ContextOverflow {
                pos: 1020,
                n: 8,
                max: 1024
            }
            .to_string()
            .contains("max_seq 1024")
        );
        assert!(
            PotError::PrefillTooLarge { n: 600, max: 512 }
                .to_string()
                .contains("600")
        );
        assert!(
            PotError::Config("bad value")
                .to_string()
                .contains("bad value")
        );
        assert!(
            PotError::DeviceLost {
                reason: wgpu::DeviceLostReason::Destroyed,
                message: "test".to_string()
            }
            .to_string()
            .contains("Destroyed")
        );
        assert!(
            PotError::DeviceLost {
                reason: wgpu::DeviceLostReason::Destroyed,
                message: "test".to_string()
            }
            .to_string()
            .contains("test")
        );
    }

    #[test]
    fn source_present_for_wrapped() {
        #[cfg(feature = "std")]
        assert!(make_io_error().source().is_some());

        assert!(PotError::Vocab("x").source().is_none());
        assert!(PotError::NoAdapter.source().is_none());
        assert!(PotError::FeatureUnsupported("x").source().is_none());
        assert!(
            PotError::ContextOverflow {
                pos: 0,
                n: 1,
                max: 1
            }
            .source()
            .is_none()
        );
        assert!(
            PotError::PrefillTooLarge { n: 1, max: 1 }
                .source()
                .is_none()
        );
        assert!(PotError::Config("x").source().is_none());
        assert!(
            PotError::DeviceLost {
                reason: wgpu::DeviceLostReason::Unknown,
                message: String::new()
            }
            .source()
            .is_none()
        );
    }
}
