//! Metal frame-capture wiring (Apple only).
//!
//! Wraps `MTLCaptureManager` so a section of GPU work — typically a `--mode
//! bench` run — can be recorded into a `.gputrace` file and opened in the
//! Xcode Metal debugger.
//!
//! Requires `MTL_CAPTURE_ENABLED=1` in the environment (or the
//! `MetalCaptureEnabled` Info.plist key) — Metal refuses to start capture
//! otherwise.

use std::path::{Path, PathBuf};

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};

use crate::Model;

/// Active Metal capture. Drop to call `stopCapture`.
pub struct CaptureGuard {
    manager: Retained<MTLCaptureManager>,
}

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        self.manager.stopCapture();
    }
}

/// Start a Metal GPU-trace capture of all command queues on the model's
/// device, writing the trace to `output` (must end in `.gputrace`).
///
/// Returns a guard whose `Drop` calls `stopCapture`. Only command buffers
/// committed between `start_capture` and the guard's drop are recorded.
pub fn start_capture(model: &Model, output: &Path) -> Result<CaptureGuard, String> {
    let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };

    if !manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument) {
        return Err(
            "Metal does not support GPU-trace destination — set MTL_CAPTURE_ENABLED=1".into(),
        );
    }

    let mtl_device = unsafe { model.device.as_hal::<wgpu::hal::api::Metal>() }
        .ok_or_else(|| "device is not a Metal device".to_string())?
        .raw_device()
        .clone();

    let url = {
        let abs =
            std::path::absolute(output).map_err(|e| format!("absolute path: {e}"))?;
        let s = NSString::from_str(&abs.to_string_lossy());
        NSURL::fileURLWithPath(&s)
    };

    let descriptor = MTLCaptureDescriptor::new();
    let capture_object: &AnyObject = (&*mtl_device).as_ref();
    unsafe { descriptor.setCaptureObject(Some(capture_object)) };
    descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
    descriptor.setOutputURL(Some(&url));

    manager
        .startCaptureWithDescriptor_error(&descriptor)
        .map_err(|e| format!("startCapture failed: {e}"))?;

    Ok(CaptureGuard { manager })
}

/// Start a Metal GPU-trace capture iff `MTL_CAPTURE_ENABLED=1` is set.
///
/// `label` distinguishes traces from sibling profile sections (e.g. `"pp"` vs
/// `"tg"`); it is inserted into the filename so concurrent / sequential
/// sections in one run do not overwrite each other. Output path:
/// - default: `./bonsai-pot-{label}.gputrace`
/// - if `BONSAI_CAPTURE_PATH` is set, treat it as a stem; `{stem}-{label}.gputrace`
///   (stripping a trailing `.gputrace` from the env var first if present).
///
/// Any stale file/dir at the resolved path is removed first.
///
/// Returns `None` when capture is disabled or fails to start (the failure is
/// logged to stderr). The returned guard stops capture on drop — wrap exactly
/// the work you want recorded in its scope.
pub fn start_capture_if_enabled(model: &Model, label: &str) -> Option<CaptureGuard> {
    if std::env::var_os("MTL_CAPTURE_ENABLED").is_none_or(|v| v != "1") {
        return None;
    }
    let path = match std::env::var_os("BONSAI_CAPTURE_PATH") {
        Some(env) => {
            let base = PathBuf::from(env);
            let stem = base
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.strip_suffix(".gputrace").unwrap_or(s))
                .unwrap_or("bonsai-pot");
            base.with_file_name(format!("{stem}-{label}.gputrace"))
        }
        None => PathBuf::from(format!("bonsai-pot-{label}.gputrace")),
    };
    if path.exists() {
        std::fs::remove_dir_all(&path)
            .or_else(|_| std::fs::remove_file(&path))
            .unwrap_or_else(|e| {
                eprintln!(
                    "warning: could not remove stale capture at {}: {e}",
                    path.display()
                );
            });
    }
    match start_capture(model, &path) {
        Ok(g) => {
            eprintln!("metal capture: writing GPU trace to {}", path.display());
            Some(g)
        }
        Err(e) => {
            eprintln!("metal capture: failed to start ({e}); continuing without capture");
            None
        }
    }
}
