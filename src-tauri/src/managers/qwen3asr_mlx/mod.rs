mod bridge;

pub(crate) use bridge::is_available;

#[cfg(mlx)]
mod imp {
    use super::bridge;
    use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
    use log::{debug, info};
    use serde::{Deserialize, Serialize};
    use std::path::{Path, PathBuf};

    pub(crate) const QWEN3_ASR_DEFAULT_ENDPOINT: &str = "https://hf-mirror.com";

    #[derive(Debug, Deserialize)]
    pub(crate) struct Qwen3ASRModelInfoFile {
        pub(crate) filename: String,
        pub(crate) size: u64,
    }

    #[derive(Debug, Deserialize)]
    pub(crate) struct Qwen3ASRModelInfo {
        pub(crate) revision: String,
        pub(crate) files: Vec<Qwen3ASRModelInfoFile>,
        pub(crate) total: u64,
    }

    #[derive(Debug, Deserialize)]
    struct HfHubSibling {
        rfilename: String,
        #[serde(default)]
        size: Option<u64>,
    }

    #[derive(Debug, Deserialize)]
    struct HfHubRepoInfo {
        sha: String,
        siblings: Vec<HfHubSibling>,
    }

    fn run_qwen3_asr_model_info(repo_id: &str) -> anyhow::Result<Qwen3ASRModelInfo> {
        let endpoint = QWEN3_ASR_DEFAULT_ENDPOINT.trim_end_matches('/').to_string();
        info!(
            "Resolving Qwen3ASR model info via hf-hub for repo {} (endpoint: {})",
            repo_id, endpoint
        );

        let api = ApiBuilder::new()
            .with_progress(false)
            .with_endpoint(endpoint.clone())
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to initialize hf-hub API client: {}", e))?;

        let repo = Repo::new(repo_id.to_string(), RepoType::Model);
        let mut response = api
            .repo(repo)
            .info_request()
            .query("blobs", "true")
            .call()
            .map_err(|e| anyhow::anyhow!("Failed to request Qwen3ASR model info: {}", e))?;

        let repo_info: HfHubRepoInfo = response
            .body_mut()
            .read_json()
            .map_err(|e| anyhow::anyhow!("Failed to parse hf-hub model info response: {}", e))?;

        let files: Vec<Qwen3ASRModelInfoFile> = repo_info
            .siblings
            .into_iter()
            .map(|sibling| Qwen3ASRModelInfoFile {
                filename: sibling.rfilename,
                size: sibling.size.unwrap_or(0),
            })
            .collect();
        let total = files.iter().map(|file| file.size).sum();

        let model_info = Qwen3ASRModelInfo {
            revision: repo_info.sha,
            files,
            total,
        };
        info!(
            "Qwen3ASR model info resolved: revision={}, files={}, total={}",
            model_info.revision,
            model_info.files.len(),
            model_info.total
        );
        Ok(model_info)
    }

    pub(crate) fn resolve_qwen3_asr_model_info(repo_id: &str) -> anyhow::Result<Qwen3ASRModelInfo> {
        let model_info = run_qwen3_asr_model_info(repo_id)?;
        if model_info.files.is_empty() {
            return Err(anyhow::anyhow!(
                "Qwen3ASR model info contains no files (repo: {}, endpoint: {})",
                repo_id,
                QWEN3_ASR_DEFAULT_ENDPOINT
            ));
        }
        Ok(model_info)
    }

    pub(crate) fn build_qwen3_asr_file_url(
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

    pub struct Qwen3ASR {
        repo_id: Option<String>,
        local_model_dir: Option<PathBuf>,
        model: Option<bridge::ModelHandle>,
    }

    #[derive(Debug, Clone, Serialize)]
    pub struct Qwen3ASRInferenceParams {
        pub language: Option<String>,
    }

    impl Qwen3ASRInferenceParams {
        pub fn new(language: Option<String>) -> Self {
            Self { language }
        }
    }

    impl Default for Qwen3ASRInferenceParams {
        fn default() -> Self {
            Self { language: None }
        }
    }

    #[derive(Debug, Clone)]
    pub struct Qwen3ASRTranscriptionResult {
        pub text: String,
    }

    impl Qwen3ASR {
        pub fn new() -> Self {
            Self {
                repo_id: None,
                local_model_dir: None,
                model: None,
            }
        }

        pub fn load_model(
            &mut self,
            repo_id: &str,
            local_model_dir: &Path,
        ) -> std::result::Result<(), Box<dyn std::error::Error>> {
            info!(
                "Loading Qwen3ASR model: repo={}, dir={}",
                repo_id,
                local_model_dir.display()
            );

            if !bridge::is_available() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Rust MLX Qwen3ASR bridge is not available on this platform",
                )));
            }

            let model = bridge::ModelHandle::load(local_model_dir).map_err(|e| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("failed to load Rust MLX Qwen3ASR model: {e}"),
                )) as Box<dyn std::error::Error>
            })?;

            self.repo_id = Some(repo_id.to_string());
            self.local_model_dir = Some(local_model_dir.to_path_buf());
            self.model = Some(model);
            info!("Qwen3ASR model loaded successfully: {}", repo_id);
            Ok(())
        }

        pub fn transcribe_samples(
            &mut self,
            audio: Vec<f32>,
            params: Option<Qwen3ASRInferenceParams>,
        ) -> std::result::Result<Qwen3ASRTranscriptionResult, Box<dyn std::error::Error>> {
            let model = self.model.as_mut().ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Qwen3ASR model is not loaded",
                )) as Box<dyn std::error::Error>
            })?;

            if self.repo_id.is_none() || self.local_model_dir.is_none() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Qwen3ASR model is not loaded",
                )));
            }

            let transcribe_start = std::time::Instant::now();
            debug!("Transcribing {} samples with Qwen3ASR", audio.len());

            let language = params.and_then(|value| value.language);
            let text = model
                .transcribe(&audio, 16_000, language.as_deref())
                .map_err(|e| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Qwen3ASR transcription failed: {e}"),
                    )) as Box<dyn std::error::Error>
                })?;

            info!(
                "Qwen3ASR transcription completed in {:?}",
                transcribe_start.elapsed()
            );

            Ok(Qwen3ASRTranscriptionResult { text })
        }
    }

    impl Default for Qwen3ASR {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Clone for Qwen3ASR {
        fn clone(&self) -> Self {
            Self {
                repo_id: self.repo_id.clone(),
                local_model_dir: self.local_model_dir.clone(),
                model: None,
            }
        }
    }
}

#[cfg(not(mlx))]
mod imp {
    use anyhow::Result;
    use reqwest::Url;
    use std::path::Path;

    pub(crate) const QWEN3_ASR_DEFAULT_ENDPOINT: &str = "https://hf-mirror.com";

    #[derive(Debug, Clone)]
    pub struct Qwen3ASRInferenceParams;

    impl Qwen3ASRInferenceParams {
        pub fn new(_language: Option<String>) -> Self {
            Self
        }
    }

    impl Default for Qwen3ASRInferenceParams {
        fn default() -> Self {
            Self
        }
    }

    #[derive(Debug, Clone)]
    pub struct Qwen3ASRTranscriptionResult {
        pub text: String,
    }

    #[derive(Debug)]
    pub(crate) struct Qwen3ASRModelInfoFile {
        pub(crate) filename: String,
        pub(crate) size: u64,
    }

    #[derive(Debug)]
    pub(crate) struct Qwen3ASRModelInfo {
        pub(crate) revision: String,
        pub(crate) files: Vec<Qwen3ASRModelInfoFile>,
        pub(crate) total: u64,
    }

    pub(crate) fn resolve_qwen3_asr_model_info(_repo_id: &str) -> Result<Qwen3ASRModelInfo> {
        Err(anyhow::anyhow!(
            "Qwen3ASR requires mlx on macOS 14+ Apple Silicon"
        ))
    }

    pub(crate) fn build_qwen3_asr_file_url(
        _endpoint: &str,
        _repo_id: &str,
        _revision: &str,
        _filename: &str,
    ) -> Result<Url> {
        Err(anyhow::anyhow!(
            "Qwen3ASR requires mlx on macOS 14+ Apple Silicon"
        ))
    }

    #[derive(Debug, Clone)]
    pub struct Qwen3ASR {
        _private: (),
    }

    impl Qwen3ASR {
        pub fn new() -> Self {
            Self { _private: () }
        }

        pub fn load_model(
            &mut self,
            _repo_id: &str,
            _local_model_dir: &Path,
        ) -> std::result::Result<(), Box<dyn std::error::Error>> {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Qwen3ASR requires mlx on macOS 14+ Apple Silicon",
            )))
        }

        pub fn transcribe_samples(
            &mut self,
            _audio: Vec<f32>,
            _params: Option<Qwen3ASRInferenceParams>,
        ) -> std::result::Result<Qwen3ASRTranscriptionResult, Box<dyn std::error::Error>> {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Qwen3ASR requires mlx on macOS 14+ Apple Silicon",
            )))
        }
    }

    impl Default for Qwen3ASR {
        fn default() -> Self {
            Self::new()
        }
    }
}

pub(crate) use imp::*;
