use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub enum PotError {
    Io { path: PathBuf, source: std::io::Error },
    ConfigParse(serde_json::Error),
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

impl fmt::Display for PotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use PotError::*;
        match self {
            Io { path, source } => write!(f, "io error reading {}: {}", path.display(), source),
            ConfigParse(e) => write!(f, "failed to parse config.json: {e}"),
            MissingTensor(name) => write!(f, "missing tensor in manifest: {name}"),
            Vocab(msg) => write!(f, "vocab.bin / vocab_offsets.bin: {msg}"),
            NoAdapter => write!(f, "no compatible GPU adapter found"),
            FeatureUnsupported(feat) => write!(f, "adapter does not support required feature: {feat}"),
            DeviceRequest(e) => write!(f, "wgpu device request failed: {e}"),
            BufferMap(e) => write!(f, "buffer mapping failed: {e:?}"),
            ContextOverflow { pos, n, max } => write!(f, "context overflow: pos {pos} + tokens {n} > max_seq {max}"),
            PrefillTooLarge { n, max } => write!(f, "prefill batch {n} exceeds max_prefill_tokens {max}"),
            Config(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for PotError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PotError::Io { source, .. } => Some(source),
            PotError::ConfigParse(e) => Some(e),
            PotError::DeviceRequest(e) => Some(e),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for PotError {
    fn from(e: serde_json::Error) -> Self { PotError::ConfigParse(e) }
}

impl From<wgpu::RequestDeviceError> for PotError {
    fn from(e: wgpu::RequestDeviceError) -> Self { PotError::DeviceRequest(e) }
}

pub type Result<T> = std::result::Result<T, PotError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    fn make_io_error() -> PotError {
        PotError::Io {
            path: std::path::PathBuf::from("/tmp/fake"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
        }
    }

    #[test]
    fn display_each_variant() {
        assert!(make_io_error().to_string().contains("/tmp/fake"));
        assert!(PotError::ConfigParse(serde_json::from_str::<u32>("bad").unwrap_err())
            .to_string().contains("config.json"));
        assert!(PotError::MissingTensor("foo.weight".into()).to_string().contains("foo.weight"));
        assert!(PotError::Vocab("bad magic").to_string().contains("bad magic"));
        assert!(PotError::NoAdapter.to_string().contains("GPU adapter"));
        assert!(PotError::FeatureUnsupported("SHADER_F16").to_string().contains("SHADER_F16"));
        assert!(PotError::BufferMap(wgpu::BufferAsyncError).to_string().contains("buffer"));
        assert!(PotError::ContextOverflow { pos: 1020, n: 8, max: 1024 }
            .to_string().contains("pos 1020"));
        assert!(PotError::ContextOverflow { pos: 1020, n: 8, max: 1024 }
            .to_string().contains("max_seq 1024"));
        assert!(PotError::PrefillTooLarge { n: 600, max: 512 }
            .to_string().contains("600"));
        assert!(PotError::Config("bad value").to_string().contains("bad value"));
    }

    #[test]
    fn source_present_for_wrapped() {
        assert!(make_io_error().source().is_some());
        let cfg_err = PotError::ConfigParse(serde_json::from_str::<u32>("bad").unwrap_err());
        assert!(cfg_err.source().is_some());

        assert!(PotError::MissingTensor("x".into()).source().is_none());
        assert!(PotError::Vocab("x").source().is_none());
        assert!(PotError::NoAdapter.source().is_none());
        assert!(PotError::FeatureUnsupported("x").source().is_none());
        assert!(PotError::BufferMap(wgpu::BufferAsyncError).source().is_none());
        assert!(PotError::ContextOverflow { pos: 0, n: 1, max: 1 }.source().is_none());
        assert!(PotError::PrefillTooLarge { n: 1, max: 1 }.source().is_none());
        assert!(PotError::Config("x").source().is_none());
    }

    #[test]
    fn from_serde_json_error_produces_config_parse() {
        let e: serde_json::Error = serde_json::from_str::<u32>("not-json").unwrap_err();
        let pot: PotError = e.into();
        assert!(matches!(pot, PotError::ConfigParse(_)));
    }
}
