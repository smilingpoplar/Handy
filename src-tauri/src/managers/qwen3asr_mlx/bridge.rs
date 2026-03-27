#[cfg(mlx)]
mod imp {
    use log::debug;
    use os_info::Type;
    use qwen3asr_mlx::{Qwen3ASR as MlxQwen3ASR, SamplingConfig};
    use std::borrow::Cow;
    use std::path::Path;
    use std::sync::OnceLock;

    static RUNTIME_SUPPORTED: OnceLock<bool> = OnceLock::new();

    fn runtime_supported() -> bool {
        *RUNTIME_SUPPORTED.get_or_init(|| {
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
        })
    }

    fn fast_sampling_config() -> SamplingConfig {
        SamplingConfig {
            temperature: 0.0,
            max_tokens: 1024,
        }
    }

    fn warmup_model(model: &mut MlxQwen3ASR) -> Result<(), String> {
        let warmup_audio = vec![0.0_f32; 32_000];
        let config = fast_sampling_config();
        let language = map_language(None);
        model
            .transcribe_samples_with_config(&warmup_audio, language.as_ref(), &config)
            .map(|_| ())
            .map_err(|e| format!("Qwen3ASR warmup failed: {e}"))
    }

    fn map_language(language: Option<&str>) -> Cow<'_, str> {
        let raw = language
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("auto");

        match raw {
            // Keep auto-detect behavior without injecting a literal "auto" language token.
            "auto" => Cow::Borrowed(""),
            "zh" | "zh-hans" | "zh-hant" | "ZH" | "ZH-HANS" | "ZH-HANT" => Cow::Borrowed("Chinese"),
            "yue" | "YUE" => Cow::Borrowed("Cantonese"),
            "en" | "EN" => Cow::Borrowed("English"),
            "ja" | "JA" => Cow::Borrowed("Japanese"),
            "ko" | "KO" => Cow::Borrowed("Korean"),
            "es" | "ES" => Cow::Borrowed("Spanish"),
            "fr" | "FR" => Cow::Borrowed("French"),
            "de" | "DE" => Cow::Borrowed("German"),
            "it" | "IT" => Cow::Borrowed("Italian"),
            "pt" | "PT" => Cow::Borrowed("Portuguese"),
            "ru" | "RU" => Cow::Borrowed("Russian"),
            "ar" | "AR" => Cow::Borrowed("Arabic"),
            "hi" | "HI" => Cow::Borrowed("Hindi"),
            "th" | "TH" => Cow::Borrowed("Thai"),
            "vi" | "VI" => Cow::Borrowed("Vietnamese"),
            "tr" | "TR" => Cow::Borrowed("Turkish"),
            "pl" | "PL" => Cow::Borrowed("Polish"),
            "nl" | "NL" => Cow::Borrowed("Dutch"),
            "sv" | "SV" => Cow::Borrowed("Swedish"),
            "da" | "DA" => Cow::Borrowed("Danish"),
            "fi" | "FI" => Cow::Borrowed("Finnish"),
            "cs" | "CS" => Cow::Borrowed("Czech"),
            "el" | "EL" => Cow::Borrowed("Greek"),
            "ro" | "RO" => Cow::Borrowed("Romanian"),
            "hu" | "HU" => Cow::Borrowed("Hungarian"),
            _ => {
                let lowered = raw.to_ascii_lowercase();
                match lowered.as_str() {
                    "auto" => Cow::Borrowed(""),
                    "zh" | "zh-hans" | "zh-hant" => Cow::Borrowed("Chinese"),
                    "yue" => Cow::Borrowed("Cantonese"),
                    "en" => Cow::Borrowed("English"),
                    "ja" => Cow::Borrowed("Japanese"),
                    "ko" => Cow::Borrowed("Korean"),
                    "es" => Cow::Borrowed("Spanish"),
                    "fr" => Cow::Borrowed("French"),
                    "de" => Cow::Borrowed("German"),
                    "it" => Cow::Borrowed("Italian"),
                    "pt" => Cow::Borrowed("Portuguese"),
                    "ru" => Cow::Borrowed("Russian"),
                    "ar" => Cow::Borrowed("Arabic"),
                    "hi" => Cow::Borrowed("Hindi"),
                    "th" => Cow::Borrowed("Thai"),
                    "vi" => Cow::Borrowed("Vietnamese"),
                    "tr" => Cow::Borrowed("Turkish"),
                    "pl" => Cow::Borrowed("Polish"),
                    "nl" => Cow::Borrowed("Dutch"),
                    "sv" => Cow::Borrowed("Swedish"),
                    "da" => Cow::Borrowed("Danish"),
                    "fi" => Cow::Borrowed("Finnish"),
                    "cs" => Cow::Borrowed("Czech"),
                    "el" => Cow::Borrowed("Greek"),
                    "ro" => Cow::Borrowed("Romanian"),
                    "hu" => Cow::Borrowed("Hungarian"),
                    _ => Cow::Borrowed(raw),
                }
            }
        }
    }

    pub fn is_available() -> bool {
        runtime_supported()
    }

    pub struct ModelHandle {
        model: MlxQwen3ASR,
    }

    impl ModelHandle {
        pub fn load(local_model_dir: &Path) -> Result<Self, String> {
            if !runtime_supported() {
                return Err("Qwen3ASR requires macOS 14+ on Apple Silicon".to_string());
            }

            let mut model = MlxQwen3ASR::load(local_model_dir)
                .map_err(|e| format!("Failed to load qwen3-asr-mlx model: {e}"))?;
            warmup_model(&mut model)?;
            Ok(Self { model })
        }

        pub fn transcribe(
            &mut self,
            samples: &[f32],
            sample_rate: i32,
            language: Option<&str>,
        ) -> Result<String, String> {
            if !runtime_supported() {
                return Err("Qwen3ASR requires macOS 14+ on Apple Silicon".to_string());
            }

            let config = fast_sampling_config();
            let language = map_language(language);
            if sample_rate == 16_000 {
                self.model
                    .transcribe_samples_with_config(samples, language.as_ref(), &config)
                    .map_err(|e| format!("Qwen3ASR transcription failed: {e}"))
            } else {
                debug!(
                    "Resampling Qwen3ASR input from {}Hz to 16000Hz",
                    sample_rate
                );
                let resampled = qwen3asr_mlx::audio::resample(samples, sample_rate as u32, 16_000)
                    .map_err(|e| format!("Qwen3ASR resampling failed: {e}"))?;
                self.model
                    .transcribe_samples_with_config(&resampled, language.as_ref(), &config)
                    .map_err(|e| format!("Qwen3ASR transcription failed: {e}"))
            }
        }
    }
}

#[cfg(not(mlx))]
mod imp {
    pub fn is_available() -> bool {
        false
    }
}

pub fn is_available() -> bool {
    imp::is_available()
}

#[cfg(mlx)]
pub use imp::ModelHandle;
