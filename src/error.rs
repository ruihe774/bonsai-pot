use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::io;
use std::path::PathBuf;
use std::result::Result as StdResult;

#[derive(Debug)]
pub enum PotError {
    Io { path: PathBuf, source: io::Error },
    MissingTensor(String),
    Vocab(&'static str),
    NoAdapter,
    FeatureUnsupported(&'static str),
    DeviceRequest(wgpu::RequestDeviceError),
    BufferMap(wgpu::BufferAsyncError),
    ContextOverflow { pos: u32, n: u32, max: u32 },
    PrefillTooLarge { n: u32, max: u32 },
    Config(&'static str),
}

impl Display for PotError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use PotError::{
            BufferMap, Config, ContextOverflow, DeviceRequest, FeatureUnsupported, Io,
            MissingTensor, NoAdapter, PrefillTooLarge, Vocab,
        };
        match self {
            Io { path, source } => write!(f, "io error reading {}: {}", path.display(), source),
            MissingTensor(name) => write!(f, "missing tensor in manifest: {name}"),
            Vocab(msg) => write!(f, "vocab.bin / vocab_offsets.bin: {msg}"),
            NoAdapter => write!(f, "no compatible GPU adapter found"),
            FeatureUnsupported(feat) => {
                write!(f, "adapter does not support required feature: {feat}")
            }
            DeviceRequest(e) => write!(f, "wgpu device request failed: {e}"),
            BufferMap(e) => write!(f, "buffer mapping failed: {e:?}"),
            ContextOverflow { pos, n, max } => write!(
                f,
                "context overflow: pos {pos} + tokens {n} > max_seq {max}"
            ),
            PrefillTooLarge { n, max } => {
                write!(f, "prefill batch {n} exceeds max_prefill_tokens {max}")
            }
            Config(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl Error for PotError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::DeviceRequest(e) => Some(e),
            _ => None,
        }
    }
}

impl From<wgpu::RequestDeviceError> for PotError {
    fn from(e: wgpu::RequestDeviceError) -> Self {
        Self::DeviceRequest(e)
    }
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
        assert!(PotError::MissingTensor("foo.weight".into())
            .to_string()
            .contains("foo.weight"));
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

        assert!(PotError::MissingTensor("x".into()).source().is_none());
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
