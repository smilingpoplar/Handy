use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use tauri::AppHandle;

static PYTHON_COMMAND_CACHE: OnceLock<String> = OnceLock::new();
static QWEN3_PYTHON_PATH: OnceLock<String> = OnceLock::new();
static QWEN3_SCRIPT_PATH: OnceLock<String> = OnceLock::new();
pub(crate) const QWEN3_DEFAULT_ENDPOINT: &str = "https://hf-mirror.com";

pub(crate) fn init_qwen3_python_path(
    app_handle: &AppHandle,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let app_data_dir = crate::portable::app_data_dir(app_handle)?;
    let runtime_dir = app_data_dir.join("qwen3_asr_mlx");
    let python_path = runtime_dir.join(".venv/bin/python3");
    let _ = QWEN3_PYTHON_PATH.set(python_path.to_string_lossy().to_string());
    let script_path = runtime_dir.join("qwen3_asr_server.py");
    if script_path.exists() {
        let _ = QWEN3_SCRIPT_PATH.set(script_path.to_string_lossy().to_string());
    }
    Ok(())
}

fn resolve_server_script_path() -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(cached) = QWEN3_SCRIPT_PATH.get() {
        let path = PathBuf::from(cached);
        if path.exists() {
            return Ok(path);
        }
    }

    let python_path = QWEN3_PYTHON_PATH.get().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            "Qwen3 Python path is not initialized",
        )
    })?;
    let runtime_dir = runtime_dir_from_python_path(python_path)?;

    let script_path = runtime_dir.join("qwen3_asr_server.py");
    if !script_path.exists() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Qwen3 server script not found in runtime dir: {}",
                runtime_dir.display()
            ),
        )));
    }
    let _ = QWEN3_SCRIPT_PATH.set(script_path.to_string_lossy().to_string());
    Ok(script_path)
}

fn runtime_dir_from_python_path(
    python_path: &str,
) -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
    let python = PathBuf::from(python_path);
    let bin_dir = python.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::Other, "Invalid Qwen3 python path: no parent")
    })?;
    let venv_dir = bin_dir.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::Other, "Invalid Qwen3 python path: no .venv")
    })?;
    let runtime_dir = if venv_dir.file_name().and_then(|s| s.to_str()) == Some(".venv") {
        venv_dir.parent().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                "Invalid Qwen3 python path: no runtime parent",
            )
        })?
    } else {
        venv_dir
    };
    Ok(runtime_dir.to_path_buf())
}

fn resolve_python_command() -> std::result::Result<String, Box<dyn std::error::Error>> {
    let fixed_python = QWEN3_PYTHON_PATH.get().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            "Qwen3 Python path is not initialized",
        )
    })?;

    let embedded = PathBuf::from(fixed_python);
    if embedded.exists() {
        let candidate = embedded.to_string_lossy().to_string();
        info!("Using Python interpreter for Qwen3: {}", candidate);
        return Ok(candidate);
    }

    Err(Box::new(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("Fixed Python path not found: {}", embedded.display()),
    )))
}

/// Get the Python command to use (embedded or system)
fn get_python_command() -> std::result::Result<(String, Vec<String>), Box<dyn std::error::Error>>
{
    if let Some(cached) = PYTHON_COMMAND_CACHE.get() {
        return Ok((cached.clone(), vec![]));
    }

    let python = resolve_python_command()?;
    let _ = PYTHON_COMMAND_CACHE.set(python.clone());
    Ok((python, vec![]))
}

pub(crate) fn get_qwen3_python_command(
) -> std::result::Result<(String, Vec<String>), Box<dyn std::error::Error>> {
    get_python_command()
}

#[derive(Debug, Deserialize)]
pub(crate) struct Qwen3ModelInfoFile {
    pub(crate) filename: String,
    pub(crate) size: u64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Qwen3ModelInfo {
    pub(crate) revision: String,
    pub(crate) files: Vec<Qwen3ModelInfoFile>,
    pub(crate) total: u64,
}

fn run_qwen3_model_info(
    python_cmd: &str,
    python_args: &[String],
    repo_id: &str,
) -> anyhow::Result<Qwen3ModelInfo> {
    let python_path = QWEN3_PYTHON_PATH
        .get()
        .ok_or_else(|| anyhow::anyhow!("Qwen3 Python path is not initialized"))?;
    let runtime_dir = runtime_dir_from_python_path(python_path)
        .map_err(|e| anyhow::anyhow!("Failed to resolve Qwen3 runtime dir: {}", e))?;
    let model_info_script_path = runtime_dir.join("model_info.py");
    if !model_info_script_path.exists() {
        return Err(anyhow::anyhow!(
            "Qwen3 model info script not found: {}",
            model_info_script_path.display()
        ));
    }
    info!(
        "Resolving Qwen3 model info via {} for repo {} (script: {})",
        python_cmd,
        repo_id,
        model_info_script_path.display()
    );

    let mut cmd = Command::new(python_cmd);
    cmd.env("PYTHONDONTWRITEBYTECODE", "1");
    cmd.env("HF_ENDPOINT", QWEN3_DEFAULT_ENDPOINT);
    for arg in python_args {
        cmd.arg(arg);
    }
    let output = cmd
        .arg("-B")
        .arg(&model_info_script_path)
        .arg(repo_id)
        .arg(QWEN3_DEFAULT_ENDPOINT)
        .output()
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to execute {}: {}",
                model_info_script_path.display(),
                e
            )
        })?;
    debug!(
        "{} exited with status {}",
        model_info_script_path.display(),
        output.status
    );

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        error!(
            "{} failed. status: {}, stderr: {}, stdout: {}",
            model_info_script_path.display(),
            output.status,
            if stderr.is_empty() {
                "<empty>"
            } else {
                stderr.as_str()
            },
            if stdout.is_empty() {
                "<empty>"
            } else {
                stdout.as_str()
            }
        );
        return Err(anyhow::anyhow!(
            "{} failed with status {}. stderr: {}. stdout: {}",
            model_info_script_path.display(),
            output.status,
            if stderr.is_empty() {
                "<empty>"
            } else {
                stderr.as_str()
            },
            if stdout.is_empty() {
                "<empty>"
            } else {
                stdout.as_str()
            }
        ));
    }

    let stdout = String::from_utf8(output.stdout).map_err(|e| {
        anyhow::anyhow!(
            "Invalid UTF-8 from {} stdout: {}",
            model_info_script_path.display(),
            e
        )
    })?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        error!("{} returned empty stdout", model_info_script_path.display());
        return Err(anyhow::anyhow!(
            "{} returned empty stdout",
            model_info_script_path.display()
        ));
    }

    if let Ok(model_info) = serde_json::from_str::<Qwen3ModelInfo>(trimmed) {
        info!(
            "Qwen3 model info resolved: revision={}, files={}, total={}",
            model_info.revision,
            model_info.files.len(),
            model_info.total
        );
        return Ok(model_info);
    }

    // Some environments may prepend warnings/logs before JSON output.
    if let Some(last_line) = trimmed.lines().rev().find(|line| !line.trim().is_empty()) {
        return serde_json::from_str::<Qwen3ModelInfo>(last_line.trim()).map(|model_info| {
            info!(
                "Qwen3 model info resolved (last-line JSON): revision={}, files={}, total={}",
                model_info.revision,
                model_info.files.len(),
                model_info.total
            );
            model_info
        }).map_err(|e| {
            error!(
                "Failed to parse Qwen3 model info JSON. last_line={}, full_stdout={}",
                last_line.trim(),
                trimmed
            );
            anyhow::anyhow!(
                "Failed to parse Qwen3 model info JSON from {} output: {}. Last line: {}",
                model_info_script_path.display(),
                e,
                last_line.trim()
            )
        });
    }

    error!(
        "{} produced no parseable JSON output: {}",
        model_info_script_path.display(),
        trimmed
    );
    Err(anyhow::anyhow!(
        "{} produced no parseable JSON output",
        model_info_script_path.display()
    ))
}

pub(crate) fn resolve_qwen3_model_info(repo_id: &str) -> anyhow::Result<Qwen3ModelInfo> {
    let (python_cmd, python_args) = get_qwen3_python_command()
        .map_err(|e| anyhow::anyhow!("Failed to resolve Python for Qwen3 model info: {}", e))?;
    let model_info = run_qwen3_model_info(&python_cmd, &python_args, repo_id)?;
    if model_info.files.is_empty() {
        return Err(anyhow::anyhow!(
            "Qwen3 model info contains no files (repo: {}, endpoint: {})",
            repo_id,
            QWEN3_DEFAULT_ENDPOINT
        ));
    }
    Ok(model_info)
}

pub(crate) fn build_qwen3_file_url(
    endpoint: &str,
    repo_id: &str,
    revision: &str,
    filename: &str,
) -> anyhow::Result<reqwest::Url> {
    let mut url = reqwest::Url::parse(endpoint)?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| anyhow::anyhow!("Endpoint cannot be a base URL"))?;
        for s in repo_id.split('/') {
            segments.push(s);
        }
        segments.push("resolve");
        segments.push(revision);
        for s in filename.split('/') {
            segments.push(s);
        }
    }
    Ok(url)
}

/// Qwen3 ASR Engine using MLX framework (macOS only)
pub struct Qwen3Engine {
    model_path: Option<String>,
    child_process: Option<Arc<Mutex<Child>>>,
    stdin: Option<Arc<Mutex<ChildStdin>>>,
    stdout: Option<Arc<Mutex<BufReader<ChildStdout>>>>,
}

/// Parameters for Qwen3 inference
#[derive(Debug, Clone, Serialize)]
pub struct Qwen3InferenceParams {
    pub language: Option<String>,
}

impl Default for Qwen3InferenceParams {
    fn default() -> Self {
        Self { language: None }
    }
}

/// Result from transcription
#[derive(Debug, Clone)]
pub struct Qwen3TranscriptionResult {
    pub text: String,
}

#[derive(Debug, Serialize)]
struct Qwen3TranscribeRequest {
    #[serde(rename = "type")]
    request_type: &'static str,
    audio_len_bytes: usize,
    params: serde_json::Value,
}

impl Qwen3Engine {
    pub fn new() -> Self {
        Self {
            model_path: None,
            child_process: None,
            stdin: None,
            stdout: None,
        }
    }

    pub fn load_model(
        &mut self,
        mlx_model_name: &str,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        info!("Loading Qwen3 model: {}", mlx_model_name);

        // Check MLX is available
        self.check_mlx_available()?;

        // Check mlx-audio is available
        self.check_mlx_audio_available()?;

        // Start persistent Python server process
        self.start_server(mlx_model_name)?;

        // Qwen3 model is managed externally by mlx-audio
        self.model_path = Some(mlx_model_name.to_string());
        info!("Qwen3 model loaded successfully: {}", mlx_model_name);
        Ok(())
    }

    pub fn unload_model(&mut self) {
        debug!("Unloading Qwen3 model");
        self.model_path = None;
        self.reset_server_state();
    }

    fn reset_server_state(&mut self) {
        // Best-effort kill for orphaned/stale child process.
        if let Some(child) = self.child_process.take() {
            match child.lock() {
                Ok(mut shared_child) => {
                    let _ = shared_child.kill();
                }
                Err(poisoned) => {
                    let mut shared_child = poisoned.into_inner();
                    let _ = shared_child.kill();
                }
            }
        }
        self.stdin = None;
        self.stdout = None;
    }

    fn ensure_server_running(
        &mut self,
        mlx_model_name: &str,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let has_pipes = self.stdin.is_some() && self.stdout.is_some();
        let mut child_alive = false;

        if let Some(child) = &self.child_process {
            let mut guard = child.lock().map_err(|e| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to lock Qwen3 child process: {}", e),
                )) as Box<dyn std::error::Error>
            })?;

            child_alive = match guard.try_wait() {
                Ok(None) => true,
                Ok(Some(status)) => {
                    warn!("Qwen3 ASR server exited unexpectedly: {}", status);
                    false
                }
                Err(e) => {
                    warn!("Failed to query Qwen3 ASR server status: {}", e);
                    false
                }
            };
        }

        if has_pipes && child_alive {
            return Ok(());
        }

        self.reset_server_state();
        self.start_server(mlx_model_name)
    }

    fn start_server(
        &mut self,
        mlx_model_name: &str,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let script_path = resolve_server_script_path()?;

        info!(
            "Starting Qwen3 ASR server process for model: {} with script {}",
            mlx_model_name,
            script_path.display()
        );
        let start_time = std::time::Instant::now();

        // Get Python command (embedded or system)
        let (python_cmd, python_args) = get_python_command()?;

        // Build command with all arguments
        let mut cmd = Command::new(&python_cmd);
        cmd.env("PYTHONDONTWRITEBYTECODE", "1");
        for arg in &python_args {
            cmd.arg(arg);
        }

        let mut child = cmd
            .env("HANDY_QWEN3_MODEL", mlx_model_name)
            .arg("-B")
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                error!("Failed to start Qwen3 server with '{}': {}", python_cmd, e);
                Box::new(e) as Box<dyn std::error::Error>
            })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            error!("Failed to get stdin from child process");
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to get stdin",
            ))
        })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            error!("Failed to get stdout from child process");
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to get stdout",
            ))
        })?;

        // Wait for "READY" signal from server with a hard timeout.
        let (ready_tx, ready_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        let _ = ready_tx.send(Err(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "Qwen3 server stdout closed before READY",
                        )));
                        return;
                    }
                    Ok(_) => {
                        let trimmed = line.trim().to_string();
                        if trimmed == "READY" {
                            let _ = ready_tx.send(Ok(reader));
                            return;
                        }
                        info!("Qwen3 server: {}", trimmed);
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                }
            }
        });

        // Timeout cannot be blocked by stdout read because readiness is reported via channel.
        let timeout = std::time::Duration::from_secs(30);
        let reader = match ready_rx.recv_timeout(timeout) {
            Ok(Ok(reader)) => {
                info!(
                    "Qwen3 ASR server is ready (startup took {:?})",
                    start_time.elapsed()
                );
                reader
            }
            Ok(Err(e)) => {
                let _ = child.kill();
                let mut stderr = String::new();
                if let Some(mut stderr_pipe) = child.stderr.take() {
                    use std::io::Read;
                    let _ = stderr_pipe.read_to_string(&mut stderr);
                }
                if !stderr.is_empty() {
                    error!("Qwen3 server startup failed. stderr: {}", stderr);
                }
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Qwen3 server startup failed: {}", e),
                )));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = child.kill();
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "Timeout waiting for Qwen3 server to be ready",
                )));
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let _ = child.kill();
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Qwen3 server readiness channel disconnected",
                )));
            }
        };

        self.child_process = Some(Arc::new(Mutex::new(child)));
        self.stdin = Some(Arc::new(Mutex::new(stdin)));
        self.stdout = Some(Arc::new(Mutex::new(reader)));

        Ok(())
    }

    pub fn transcribe_samples<P: serde::Serialize>(
        &mut self,
        audio: Vec<f32>,
        params: Option<P>,
    ) -> std::result::Result<Qwen3TranscriptionResult, Box<dyn std::error::Error>> {
        let model_path = self
            .model_path
            .as_ref()
            .ok_or_else(|| Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Model not loaded",
            )))?
            .clone();

        let transcribe_start = std::time::Instant::now();
        debug!("Transcribing {} samples with Qwen3", audio.len());

        // Serialize params directly as JSON object to avoid double JSON encode/decode.
        let params_value = match params {
            Some(p) => serde_json::to_value(p).unwrap_or_else(|_| json!({})),
            None => json!({}),
        };
        let input_data = Qwen3TranscribeRequest {
            request_type: "binary",
            audio_len_bytes: audio.len() * std::mem::size_of::<f32>(),
            params: params_value,
        };

        let mut last_error: Option<Box<dyn std::error::Error>> = None;
        let mut text: Option<String> = None;

        // Retry once if server died/disconnected mid-request.
        for attempt in 0..2 {
            if let Err(e) = self.ensure_server_running(&model_path) {
                last_error = Some(e);
                break;
            }

            let attempt_result = (|| -> std::result::Result<String, Box<dyn std::error::Error>> {
                // Write request to server
                {
                    let stdin = self.stdin.as_ref().ok_or_else(|| {
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "Server stdin not available",
                        )) as Box<dyn std::error::Error>
                    })?;

                    let mut stdin = stdin.lock().map_err(|e| {
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("Failed to lock stdin: {}", e),
                        )) as Box<dyn std::error::Error>
                    })?;

                    let json_line = format!("{}\n", serde_json::to_string(&input_data)?);
                    stdin.write_all(json_line.as_bytes()).map_err(|e| {
                        error!("Failed to write to server stdin: {}", e);
                        Box::new(e) as Box<dyn std::error::Error>
                    })?;

                    if cfg!(target_endian = "little") {
                        // Write raw f32 bytes directly to cut IPC overhead.
                        let audio_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                audio.as_ptr() as *const u8,
                                audio.len() * std::mem::size_of::<f32>(),
                            )
                        };
                        stdin.write_all(audio_bytes).map_err(|e| {
                            error!("Failed to write audio bytes to server stdin: {}", e);
                            Box::new(e) as Box<dyn std::error::Error>
                        })?;
                    } else {
                        // Fallback for rare big-endian targets.
                        let mut audio_bytes =
                            Vec::with_capacity(audio.len() * std::mem::size_of::<f32>());
                        for sample in &audio {
                            audio_bytes.extend_from_slice(&sample.to_le_bytes());
                        }
                        stdin.write_all(&audio_bytes).map_err(|e| {
                            error!("Failed to write audio bytes to server stdin: {}", e);
                            Box::new(e) as Box<dyn std::error::Error>
                        })?;
                    }

                    stdin.flush().map_err(|e| {
                        error!("Failed to flush stdin: {}", e);
                        Box::new(e) as Box<dyn std::error::Error>
                    })?;
                }

                // Read response from server
                let response_line = {
                    let stdout = self.stdout.as_ref().ok_or_else(|| {
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "Server stdout not available",
                        )) as Box<dyn std::error::Error>
                    })?;

                    let mut stdout = stdout.lock().map_err(|e| {
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("Failed to lock stdout: {}", e),
                        )) as Box<dyn std::error::Error>
                    })?;

                    let mut line = String::new();
                    let bytes = stdout.read_line(&mut line).map_err(|e| {
                        error!("Failed to read from server stdout: {}", e);
                        Box::new(e) as Box<dyn std::error::Error>
                    })?;
                    if bytes == 0 || line.trim().is_empty() {
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "Qwen3 server closed stdout unexpectedly",
                        )));
                    }
                    line
                };

                let parse_start = std::time::Instant::now();
                let result: serde_json::Value = serde_json::from_str(&response_line).map_err(|e| {
                    error!("Failed to parse response: {}. Response: {}", e, response_line);
                    Box::new(e) as Box<dyn std::error::Error>
                })?;
                debug!("JSON parsing took {:?}", parse_start.elapsed());

                if let Some(error) = result.get("error") {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Qwen3 error: {}", error),
                    )));
                }

                Ok(result
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string())
            })();

            match attempt_result {
                Ok(t) => {
                    text = Some(t);
                    break;
                }
                Err(e) => {
                    last_error = Some(e);
                    if attempt == 0 {
                        warn!("Qwen3 request failed, restarting server and retrying once");
                        self.reset_server_state();
                    }
                }
            }
        }

        let text = text.ok_or_else(|| {
            last_error.unwrap_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Qwen3 transcription failed for unknown reason",
                )) as Box<dyn std::error::Error>
            })
        })?;

        info!("Qwen3 transcription completed in {:?}", transcribe_start.elapsed());

        Ok(Qwen3TranscriptionResult { text })
    }

    fn check_mlx_available(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let _ = get_python_command()?;
        Ok(())
    }

    fn check_mlx_audio_available(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let (python_cmd, python_args) = get_python_command()?;

        let mut cmd = Command::new(&python_cmd);
        cmd.env("PYTHONDONTWRITEBYTECODE", "1");
        for arg in &python_args {
            cmd.arg(arg);
        }
        let output = cmd
            .args([
                "-c",
                "import importlib.util as u, sys; sys.exit(0 if u.find_spec('mlx_audio.stt.models.qwen3_asr') else 1)",
            ])
            .output()?;

        if !output.status.success() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Installed mlx-audio does not include qwen3_asr backend. Please use the mlx-audio version/build used by the reference project.",
            )));
        }

        Ok(())
    }
}

impl Default for Qwen3Engine {
    fn default() -> Self {
        Self::new()
    }
}

// Manual Clone implementation since Child doesn't implement Clone
impl Clone for Qwen3Engine {
    fn clone(&self) -> Self {
        // Create a new instance - the child process cannot be cloned
        // The new instance will need to start its own server if needed
        Self {
            model_path: self.model_path.clone(),
            child_process: None,
            stdin: None,
            stdout: None,
        }
    }
}
