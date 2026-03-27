use super::{DownloadCleanup, DownloadProgress, EngineType, ModelInfo, ModelManager};
use crate::managers::qwen3asr_mlx::{
    build_qwen3_asr_file_url, is_available as qwen3asr_available, resolve_qwen3_asr_model_info,
    QWEN3_ASR_DEFAULT_ENDPOINT,
};
use anyhow::Result;
use log::info;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tauri::Emitter;

const MLX_URL_PREFIX: &str = "mlx://";
const QWEN3ASR_LANGUAGE_CODES: &[&str] = &[
    "zh", "zh-Hans", "zh-Hant", "yue", "en", "ja", "ko", "es", "fr", "de", "it", "pt", "ru", "ar",
    "hi", "th", "vi", "tr", "pl", "nl", "sv", "da", "fi", "cs", "el", "ro", "hu",
];

struct Qwen3AsrModelPreset {
    id: &'static str,
    name: &'static str,
    description: &'static str,
    url: &'static str,
    size_mb: u64,
    accuracy_score: f32,
    speed_score: f32,
}

const QWEN3ASR_MLX_MODELS: &[Qwen3AsrModelPreset] = &[
    Qwen3AsrModelPreset {
        id: "qwen3-asr",
        name: "Qwen3-ASR-0.6B-8bit (MLX)",
        description: "MLX backend, 0.6B model, 8-bit quantized. Multilingual ASR.",
        url: "mlx://mlx-community/Qwen3-ASR-0.6B-8bit",
        size_mb: 600,
        accuracy_score: 0.90,
        speed_score: 0.85,
    },
    Qwen3AsrModelPreset {
        id: "qwen3-asr-1.7b",
        name: "Qwen3-ASR-1.7B-8bit (MLX)",
        description: "MLX backend, 1.7B model, 8-bit quantized. Higher accuracy multilingual ASR.",
        url: "mlx://mlx-community/Qwen3-ASR-1.7B-8bit",
        size_mb: 1700,
        accuracy_score: 0.94,
        speed_score: 0.65,
    },
];

fn supported_languages() -> Vec<String> {
    QWEN3ASR_LANGUAGE_CODES
        .iter()
        .map(|code| (*code).to_string())
        .collect()
}

fn build_model_info(preset: &Qwen3AsrModelPreset, supported_languages: &[String]) -> ModelInfo {
    ModelInfo {
        id: preset.id.to_string(),
        name: preset.name.to_string(),
        description: preset.description.to_string(),
        filename: preset.id.to_string(),
        url: Some(preset.url.to_string()),
        sha256: None,
        size_mb: preset.size_mb,
        is_downloaded: false,
        is_downloading: false,
        partial_size: 0,
        is_directory: false,
        engine_type: EngineType::Qwen3ASR,
        accuracy_score: preset.accuracy_score,
        speed_score: preset.speed_score,
        supports_translation: false,
        is_recommended: false,
        supported_languages: supported_languages.to_vec(),
        supports_language_selection: true,
        is_custom: false,
    }
}

pub(super) fn insert_models(available_models: &mut HashMap<String, ModelInfo>) {
    // Qwen3ASR model (Apple Silicon macOS only, MLX-based)
    // Model files are managed in app_data/models as MLX model directories.
    if !qwen3asr_available() {
        return;
    }

    let supported_languages = supported_languages();
    for preset in QWEN3ASR_MLX_MODELS {
        available_models.insert(
            preset.id.to_string(),
            build_model_info(preset, &supported_languages),
        );
    }
}

fn is_mlx_url(url: &str) -> bool {
    url.starts_with(MLX_URL_PREFIX)
}

fn mlx_model_name_from_url(url: &str) -> Option<&str> {
    url.strip_prefix(MLX_URL_PREFIX)
}

impl ModelManager {
    fn find_file_recursive(dir: &Path, predicate: &impl Fn(&Path) -> bool) -> bool {
        let entries = match fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(_) => return false,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };

            if file_type.is_file() {
                if predicate(&path) {
                    return true;
                }
            } else if file_type.is_dir() && Self::find_file_recursive(&path, predicate) {
                return true;
            }
        }
        false
    }

    fn is_model_dir_complete(model_dir: &Path) -> bool {
        if !model_dir.exists() || !model_dir.is_dir() {
            return false;
        }
        let has_config = Self::find_file_recursive(model_dir, &|p| {
            p.file_name().and_then(|n| n.to_str()) == Some("config.json")
        });
        let has_tokenizer_json = Self::find_file_recursive(model_dir, &|p| {
            p.file_name().and_then(|n| n.to_str()) == Some("tokenizer.json")
        });
        let has_tokenizer_config = Self::find_file_recursive(model_dir, &|p| {
            p.file_name().and_then(|n| n.to_str()) == Some("tokenizer_config.json")
        });
        let has_vocab = Self::find_file_recursive(model_dir, &|p| {
            p.file_name().and_then(|n| n.to_str()) == Some("vocab.json")
        });
        let has_merges = Self::find_file_recursive(model_dir, &|p| {
            p.file_name().and_then(|n| n.to_str()) == Some("merges.txt")
        });
        let has_weights = Self::find_file_recursive(model_dir, &|p| {
            p.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
        });

        let has_tokenizer_assets =
            has_tokenizer_json || (has_tokenizer_config && has_vocab && has_merges);

        has_config && has_tokenizer_assets && has_weights
    }

    fn is_mlx_model(&self, model: &ModelInfo) -> bool {
        model.url.as_deref().is_some_and(is_mlx_url)
    }

    fn local_dir_for_model_info(&self, model: &ModelInfo) -> Option<PathBuf> {
        let url = model.url.as_deref()?;
        let mlx_model_name = mlx_model_name_from_url(url)?;
        Some(self.models_dir.join(mlx_model_name.replace("/", "--")))
    }

    fn mlx_model_name_for(&self, model_id: &str) -> Option<String> {
        let models = self.available_models.lock().unwrap();
        let model = models.get(model_id)?;
        let url = model.url.as_ref()?;
        mlx_model_name_from_url(url).map(|s| s.to_string())
    }

    fn local_mlx_model_dir(&self, model_id: &str) -> Result<PathBuf> {
        let mlx_model_name = self
            .mlx_model_name_for(model_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown MLX model: {}", model_id))?;
        Ok(self.models_dir.join(mlx_model_name.replace("/", "--")))
    }

    pub(super) fn refresh_mlx_download_status(&self, models: &mut HashMap<String, ModelInfo>) {
        for model in models.values_mut() {
            if self.is_mlx_model(model) {
                let downloaded = self
                    .local_dir_for_model_info(model)
                    .is_some_and(|model_dir| Self::is_model_dir_complete(&model_dir));
                model.is_downloaded = downloaded;
                model.is_downloading = false;
                model.partial_size = 0;
            }
        }
    }

    pub(super) async fn try_download_mlx_model(&self, model_id: &str, url: &str) -> Result<bool> {
        if !is_mlx_url(url) {
            return Ok(false);
        }
        self.download_mlx_model(model_id).await?;
        Ok(true)
    }

    pub(super) fn try_delete_mlx_model(
        &self,
        model_id: &str,
        model_info: &ModelInfo,
    ) -> Result<bool> {
        if !self.is_mlx_model(model_info) {
            return Ok(false);
        }
        self.delete_mlx_model(model_id)?;
        Ok(true)
    }

    pub(super) fn try_get_mlx_model_path(
        &self,
        model_id: &str,
        model_info: &ModelInfo,
    ) -> Result<Option<PathBuf>> {
        if !self.is_mlx_model(model_info) {
            return Ok(None);
        }

        let local_model_dir = self.local_mlx_model_dir(model_id)?;
        if Self::is_model_dir_complete(&local_model_dir) {
            return Ok(Some(local_model_dir));
        }
        let mlx_model_name = self
            .mlx_model_name_for(model_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown MLX model: {}", model_id))?;
        Err(anyhow::anyhow!(
            "Complete MLX model directory not found for {}: {}",
            mlx_model_name,
            local_model_dir.display()
        ))
    }

    /// Check if an MLX-managed model is cached locally.
    fn check_mlx_model_cached(&self, model_id: &str) -> bool {
        let model_dir = match self.local_mlx_model_dir(model_id) {
            Ok(path) => path,
            Err(_) => return false,
        };
        Self::is_model_dir_complete(&model_dir)
    }

    /// Delete model files from app-managed storage.
    fn delete_mlx_model(&self, model_id: &str) -> Result<()> {
        info!("Deleting model: {}", model_id);
        let local_model_dir = self.local_mlx_model_dir(model_id)?;
        if local_model_dir.exists() {
            fs::remove_dir_all(&local_model_dir)?;
        }

        self.update_download_status()?;
        let _ = self.app_handle.emit("model-deleted", model_id);
        Ok(())
    }

    /// Download an MLX-managed model directly from Hugging Face.
    async fn download_mlx_model(&self, model_id: &str) -> Result<()> {
        {
            let mut models = self.available_models.lock().unwrap();
            if let Some(model) = models.get_mut(model_id) {
                model.is_downloading = true;
                model.partial_size = 0;
            }
        }

        let mlx_model_name = self
            .mlx_model_name_for(model_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown MLX model: {}", model_id))?;
        let local_model_dir = self.local_mlx_model_dir(model_id)?;

        if self.check_mlx_model_cached(model_id) {
            let cached_size_bytes = self
                .get_model_info(model_id)
                .map(|m| m.size_mb * 1024 * 1024)
                .unwrap_or(0);
            let mut models = self.available_models.lock().unwrap();
            if let Some(model) = models.get_mut(model_id) {
                model.is_downloading = false;
                model.is_downloaded = true;
                model.partial_size = 0;
            }
            let _ = self.app_handle.emit(
                "model-download-progress",
                DownloadProgress {
                    model_id: model_id.to_string(),
                    downloaded: cached_size_bytes,
                    total: cached_size_bytes,
                    percentage: 100.0,
                },
            );
            let _ = self.app_handle.emit("model-download-complete", model_id);
            return Ok(());
        }

        let cancel_flag = Arc::new(AtomicBool::new(false));
        {
            let mut flags = self.cancel_flags.lock().unwrap();
            flags.insert(model_id.to_string(), cancel_flag.clone());
        }

        let mut cleanup = DownloadCleanup {
            available_models: &self.available_models,
            cancel_flags: &self.cancel_flags,
            model_id: model_id.to_string(),
            disarmed: false,
        };

        let model_plan = resolve_qwen3_asr_model_info(&mlx_model_name)?;
        let client = reqwest::Client::new();

        let mut total_bytes = if model_plan.total > 0 {
            model_plan.total
        } else {
            model_plan.files.iter().map(|f| f.size).sum()
        };

        let mut downloaded_bytes = 0u64;
        for file in &model_plan.files {
            let path = local_model_dir.join(&file.filename);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let existing = path.metadata().map(|m| m.len()).unwrap_or(0);
            let expected = if file.size > 0 { file.size } else { existing };
            downloaded_bytes += existing.min(expected);
            if total_bytes < downloaded_bytes {
                total_bytes = downloaded_bytes;
            }
        }

        let emit_progress = |downloaded: u64, total: u64| {
            let percentage = if total > 0 {
                (downloaded as f64 / total as f64) * 100.0
            } else {
                0.0
            };
            let _ = self.app_handle.emit(
                "model-download-progress",
                DownloadProgress {
                    model_id: model_id.to_string(),
                    downloaded,
                    total,
                    percentage,
                },
            );
        };

        emit_progress(downloaded_bytes, total_bytes);

        let revision = model_plan.revision.clone();
        for file in model_plan.files {
            let path = local_model_dir.join(&file.filename);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }

            let existing = path.metadata().map(|m| m.len()).unwrap_or(0);
            let previous_expected = if file.size > 0 { file.size } else { existing };
            let previous_counted = existing.min(previous_expected);

            if file.size > 0 && existing >= file.size {
                continue;
            }

            let url = build_qwen3_asr_file_url(
                QWEN3_ASR_DEFAULT_ENDPOINT,
                &mlx_model_name,
                &revision,
                &file.filename,
            )?
            .to_string();

            let result = self
                .download_file_with_resume(
                    &client,
                    &url,
                    &path,
                    &cancel_flag,
                    |file_downloaded, file_total| {
                        let effective_total = if total_bytes == 0 {
                            file_total
                        } else {
                            total_bytes
                        };
                        let aggregate_downloaded =
                            downloaded_bytes.saturating_sub(previous_counted) + file_downloaded;
                        let aggregate_total = effective_total.max(aggregate_downloaded);
                        emit_progress(aggregate_downloaded.min(aggregate_total), aggregate_total);
                    },
                )
                .await?;

            if result.cancelled {
                info!("Download cancelled for: {}", model_id);
                return Ok(());
            }

            if file.size > 0 {
                let final_size = path.metadata()?.len();
                if final_size < file.size {
                    return Err(anyhow::anyhow!(
                        "Incomplete file {}: expected {}, got {}",
                        file.filename,
                        file.size,
                        final_size
                    ));
                }
            }

            let discovered_total = if file.size > 0 {
                file.size
            } else {
                result.total
            };
            if discovered_total > previous_expected {
                total_bytes += discovered_total - previous_expected;
            }

            downloaded_bytes =
                downloaded_bytes.saturating_sub(previous_counted) + result.downloaded;
            if total_bytes < downloaded_bytes {
                total_bytes = downloaded_bytes;
            }
            emit_progress(downloaded_bytes, total_bytes);
        }

        if !self.check_mlx_model_cached(model_id) {
            return Err(anyhow::anyhow!("Model download verification failed"));
        }

        emit_progress(total_bytes, total_bytes);

        cleanup.disarmed = true;
        let mut models = self.available_models.lock().unwrap();
        if let Some(model) = models.get_mut(model_id) {
            model.is_downloading = false;
            model.is_downloaded = true;
            model.partial_size = 0;
        }
        self.cancel_flags.lock().unwrap().remove(model_id);
        let _ = self.app_handle.emit("model-download-complete", model_id);
        Ok(())
    }
}
