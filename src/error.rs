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
