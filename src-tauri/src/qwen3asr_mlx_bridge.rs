use std::path::Path;

#[cfg(all(
    target_os = "macos",
    target_arch = "aarch64",
    handy_qwen3asr_bridge_built
))]
mod imp {
    use libloading::Library;
    use log::warn;
    use os_info::Type;
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_int, c_void};
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;

    const BRIDGE_FILE_NAME: &str = "libHandyQwen3ASRBridge.dylib";
    const BRIDGE_STAGE_DIR: &str = env!("HANDY_QWEN3ASR_BRIDGE_STAGE_DIR");

    type HandyQwen3AsrIsAvailable = unsafe extern "C" fn() -> c_int;
    type HandyQwen3AsrLoadModel =
        unsafe extern "C" fn(repo_id: *const c_char, local_model_dir: *const c_char) -> *mut c_void;
    type HandyQwen3AsrTranscribe = unsafe extern "C" fn(
        samples: *const f32,
        sample_count: usize,
        sample_rate: c_int,
        language: *const c_char,
    ) -> *mut c_void;
    type HandyQwen3AsrUnloadModel = unsafe extern "C" fn();
    type HandyQwen3AsrResponseSuccess = unsafe extern "C" fn(response: *const c_void) -> c_int;
    type HandyQwen3AsrResponseText =
        unsafe extern "C" fn(response: *const c_void) -> *const c_char;
    type HandyQwen3AsrResponseErrorMessage =
        unsafe extern "C" fn(response: *const c_void) -> *const c_char;
    type HandyQwen3AsrFreeResponse = unsafe extern "C" fn(response: *mut c_void);

    struct BridgeApi {
        _library: Library,
        is_available: HandyQwen3AsrIsAvailable,
        load_model: HandyQwen3AsrLoadModel,
        transcribe: HandyQwen3AsrTranscribe,
        unload_model: HandyQwen3AsrUnloadModel,
        response_success: HandyQwen3AsrResponseSuccess,
        response_text: HandyQwen3AsrResponseText,
        response_error_message: HandyQwen3AsrResponseErrorMessage,
        free_response: HandyQwen3AsrFreeResponse,
    }

    static BRIDGE_API: OnceLock<Result<BridgeApi, String>> = OnceLock::new();

    fn bridge_bundle_dir() -> Result<PathBuf, String> {
        let exe_path = std::env::current_exe()
            .map_err(|e| format!("Failed to determine current executable path: {e}"))?;
        let exe_dir = exe_path
            .parent()
            .ok_or_else(|| format!("Executable path has no parent: {}", exe_path.display()))?;

        let frameworks_dir = exe_dir
            .parent()
            .and_then(|contents_dir| contents_dir.parent().map(|_| contents_dir))
            .map(|contents_dir| contents_dir.join("Frameworks"));
        if let Some(frameworks_dir) = frameworks_dir.filter(|path| path.exists()) {
            return Ok(frameworks_dir);
        }

        let stage_dir = PathBuf::from(BRIDGE_STAGE_DIR);
        if stage_dir.exists() {
            return Ok(stage_dir);
        }

        Err(format!(
            "Qwen3ASR bridge bundle is missing from both app bundle Frameworks and stage dir: {}",
            stage_dir.display()
        ))
    }

    fn sync_bridge_resources(bundle_dir: &Path) -> Result<(), String> {
        let stage_dir = PathBuf::from(BRIDGE_STAGE_DIR);
        if !stage_dir.exists() || stage_dir == bundle_dir {
            return Ok(());
        }

        let source = stage_dir.join("mlx.metallib");
        if !source.exists() {
            return Ok(());
        }

        let destination = bundle_dir.join("mlx.metallib");
        std::fs::copy(&source, &destination).map_err(|e| {
            format!(
                "Failed to copy Qwen3ASR bridge resource {} to {}: {e}",
                source.display(),
                destination.display()
            )
        })?;

        Ok(())
    }

    impl BridgeApi {
        fn load() -> Result<Self, String> {
            let bundle_dir = bridge_bundle_dir()?;
            sync_bridge_resources(&bundle_dir)?;
            let dylib_path = bundle_dir.join(BRIDGE_FILE_NAME);
            unsafe {
                let library = Library::new(&dylib_path).map_err(|e| {
                    format!(
                        "Failed to load Qwen3ASR bridge dylib {}: {e}",
                        dylib_path.display()
                    )
                })?;

                let is_available = *library
                    .get::<HandyQwen3AsrIsAvailable>(b"handy_qwen3asr_is_available\0")
                    .map_err(|e| format!("Missing handy_qwen3asr_is_available symbol: {e}"))?;
                let load_model = *library
                    .get::<HandyQwen3AsrLoadModel>(b"handy_qwen3asr_load_model\0")
                    .map_err(|e| format!("Missing handy_qwen3asr_load_model symbol: {e}"))?;
                let transcribe = *library
                    .get::<HandyQwen3AsrTranscribe>(b"handy_qwen3asr_transcribe\0")
                    .map_err(|e| format!("Missing handy_qwen3asr_transcribe symbol: {e}"))?;
                let unload_model = *library
                    .get::<HandyQwen3AsrUnloadModel>(b"handy_qwen3asr_unload_model\0")
                    .map_err(|e| format!("Missing handy_qwen3asr_unload_model symbol: {e}"))?;
                let response_success = *library
                    .get::<HandyQwen3AsrResponseSuccess>(b"handy_qwen3asr_response_success\0")
                    .map_err(|e| format!("Missing handy_qwen3asr_response_success symbol: {e}"))?;
                let response_text = *library
                    .get::<HandyQwen3AsrResponseText>(b"handy_qwen3asr_response_text\0")
                    .map_err(|e| format!("Missing handy_qwen3asr_response_text symbol: {e}"))?;
                let response_error_message = *library
                    .get::<HandyQwen3AsrResponseErrorMessage>(
                        b"handy_qwen3asr_response_error_message\0",
                    )
                    .map_err(|e| {
                        format!("Missing handy_qwen3asr_response_error_message symbol: {e}")
                    })?;
                let free_response = *library
                    .get::<HandyQwen3AsrFreeResponse>(b"handy_qwen3asr_free_response\0")
                    .map_err(|e| format!("Missing handy_qwen3asr_free_response symbol: {e}"))?;

                Ok(Self {
                    _library: library,
                    is_available,
                    load_model,
                    transcribe,
                    unload_model,
                    response_success,
                    response_text,
                    response_error_message,
                    free_response,
                })
            }
        }
    }

    fn runtime_supported() -> bool {
        let info = os_info::get();
        if info.os_type() != Type::Macos {
            return false;
        }

        let version = info.version().to_string();
        let mut parts = version.split('.');
        let major = parts
            .next()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
        major >= 14
    }

    fn api() -> Result<&'static BridgeApi, String> {
        BRIDGE_API
            .get_or_init(BridgeApi::load)
            .as_ref()
            .map_err(Clone::clone)
    }

    fn parse_response(api: &BridgeApi, response_ptr: *mut c_void) -> Result<String, String> {
        if response_ptr.is_null() {
            return Err("Null response from Qwen3ASR Swift bridge".to_string());
        }

        let result = if unsafe { (api.response_success)(response_ptr.cast_const()) } == 1 {
            let text_ptr = unsafe { (api.response_text)(response_ptr.cast_const()) };
            if text_ptr.is_null() {
                Ok(String::new())
            } else {
                Ok(unsafe { CStr::from_ptr(text_ptr) }
                    .to_string_lossy()
                    .into_owned())
            }
        } else {
            let error_ptr = unsafe { (api.response_error_message)(response_ptr.cast_const()) };
            if error_ptr.is_null() {
                Err("Unknown Qwen3ASR Swift bridge error".to_string())
            } else {
                Err(unsafe { CStr::from_ptr(error_ptr) }
                    .to_string_lossy()
                    .into_owned())
            }
        };

        unsafe { (api.free_response)(response_ptr) };
        result
    }

    pub fn is_available() -> bool {
        if !runtime_supported() {
            return false;
        }

        match api() {
            Ok(api) => unsafe { (api.is_available)() == 1 },
            Err(error) => {
                warn!("Qwen3ASR bridge is unavailable: {error}");
                false
            }
        }
    }

    pub fn load_model(repo_id: &str, local_model_dir: &Path) -> Result<(), String> {
        if !runtime_supported() {
            return Err("Qwen3ASR requires macOS 14+ on Apple Silicon".to_string());
        }

        let api = api()?;
        let repo_id = CString::new(repo_id).map_err(|e| e.to_string())?;
        let model_dir = CString::new(local_model_dir.to_string_lossy().as_bytes())
            .map_err(|e| e.to_string())?;
        let response = unsafe { (api.load_model)(repo_id.as_ptr(), model_dir.as_ptr()) };
        parse_response(api, response).map(|_| ())
    }

    pub fn transcribe(
        samples: &[f32],
        sample_rate: i32,
        language: Option<&str>,
    ) -> Result<String, String> {
        if !runtime_supported() {
            return Err("Qwen3ASR requires macOS 14+ on Apple Silicon".to_string());
        }

        let api = api()?;
        let language = language
            .map(|value| CString::new(value).map_err(|e| e.to_string()))
            .transpose()?;
        let response = unsafe {
            (api.transcribe)(
                samples.as_ptr(),
                samples.len(),
                sample_rate as c_int,
                language
                    .as_ref()
                    .map(|value| value.as_ptr())
                    .unwrap_or(std::ptr::null()),
            )
        };
        parse_response(api, response)
    }

    pub fn unload_model() {
        if let Ok(api) = api() {
            unsafe { (api.unload_model)() };
        }
    }
}

#[cfg(not(all(
    target_os = "macos",
    target_arch = "aarch64",
    handy_qwen3asr_bridge_built
)))]
mod imp {
    use std::path::Path;

    pub fn is_available() -> bool {
        false
    }

    pub fn load_model(_repo_id: &str, _local_model_dir: &Path) -> Result<(), String> {
        Err("Qwen3ASR requires macOS 14+ on Apple Silicon".to_string())
    }

    pub fn transcribe(
        _samples: &[f32],
        _sample_rate: i32,
        _language: Option<&str>,
    ) -> Result<String, String> {
        Err("Qwen3ASR requires macOS 14+ on Apple Silicon".to_string())
    }

    pub fn unload_model() {}
}

pub fn is_available() -> bool {
    imp::is_available()
}

pub fn load_model(repo_id: &str, local_model_dir: &Path) -> Result<(), String> {
    imp::load_model(repo_id, local_model_dir)
}

pub fn transcribe(samples: &[f32], sample_rate: i32, language: Option<&str>) -> Result<String, String> {
    imp::transcribe(samples, sample_rate, language)
}

pub fn unload_model() {
    imp::unload_model()
}
