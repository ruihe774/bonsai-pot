use std::io;
use std::path::PathBuf;
use std::result::Result as StdResult;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PotError {
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
    #[error("adapter does not support required feature: {0}")]
    FeatureUnsupported(&'static str),
    #[error("wgpu device request failed: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),
    #[error("buffer mapping failed: {0:?}")]
    BufferMap(wgpu::BufferAsyncError),
    #[error("context overflow: pos {pos} + tokens {n} > max_seq {max}")]
    ContextOverflow { pos: u32, n: u32, max: u32 },
    #[error("prefill batch {n} exceeds max_prefill_tokens {max}")]
    PrefillTooLarge { n: u32, max: u32 },
    #[error("invalid config: {0}")]
    Config(&'static str),
}

pub type Result<T> = StdResult<T, PotError>;

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::io;
    use std::path::PathBuf;

    use super::PotError;

    fn make_io_error() -> PotError {
        PotError::Io {
            path: PathBuf::from("/tmp/fake"),
            source: io::Error::new(io::ErrorKind::NotFound, "not found"),
        }
    }

    #[test]
    fn display_each_variant() {
        assert!(make_io_error().to_string().contains("/tmp/fake"));
        assert!(PotError::Config("bad ini key")
            .to_string()
            .contains("bad ini key"));
        assert!(PotError::Vocab("bad magic")
            .to_string()
            .contains("bad magic"));
        assert!(PotError::NoAdapter.to_string().contains("GPU adapter"));
        assert!(PotError::FeatureUnsupported("SHADER_F16")
            .to_string()
            .contains("SHADER_F16"));
        assert!(PotError::BufferMap(wgpu::BufferAsyncError)
            .to_string()
            .contains("buffer"));
        assert!(PotError::ContextOverflow {
            pos: 1020,
            n: 8,
            max: 1024
        }
        .to_string()
        .contains("pos 1020"));
        assert!(PotError::ContextOverflow {
            pos: 1020,
            n: 8,
            max: 1024
        }
        .to_string()
        .contains("max_seq 1024"));
        assert!(PotError::PrefillTooLarge { n: 600, max: 512 }
            .to_string()
            .contains("600"));
        assert!(PotError::Config("bad value")
            .to_string()
            .contains("bad value"));
    }

    #[test]
    fn source_present_for_wrapped() {
        assert!(make_io_error().source().is_some());

        assert!(PotError::Vocab("x").source().is_none());
        assert!(PotError::NoAdapter.source().is_none());
        assert!(PotError::FeatureUnsupported("x").source().is_none());
        assert!(PotError::BufferMap(wgpu::BufferAsyncError)
            .source()
            .is_none());
        assert!(PotError::ContextOverflow {
            pos: 0,
            n: 1,
            max: 1
        }
        .source()
        .is_none());
        assert!(PotError::PrefillTooLarge { n: 1, max: 1 }
            .source()
            .is_none());
        assert!(PotError::Config("x").source().is_none());
    }
}
