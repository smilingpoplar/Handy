#!/usr/bin/env python3
"""
Qwen3 ASR Server Script using mlx-audio
Persistent server that keeps model loaded in memory for fast inference
"""

import sys
import json
import numpy as np
import time

# 设置 HuggingFace 镜像（中国区可加速）
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Global model cache
_stt_model = None

def resolve_model_name() -> str:
    """Resolve model name from environment."""
    env_model = os.environ.get("HANDY_QWEN3_MODEL", "").strip()
    if env_model:
        return env_model

    raise RuntimeError(
        "Qwen3 model is not specified. Pass it via HANDY_QWEN3_MODEL."
    )

def load_model():
    """Load model once and cache it globally"""
    global _stt_model
    if _stt_model is None:
        try:
            from mlx_audio.stt import utils as stt_utils
            # mlx-audio infers backend from repo-name tokens.
            # "Qwen3-ASR-0.6B-8bit" -> token "qwen3", map it to the mlx-audio qwen3_asr backend.
            stt_utils.MODEL_REMAPPING.setdefault("qwen3", "qwen3_asr")
            load_stt_model = stt_utils.load_model
        except Exception:
            # Backward compatibility with older mlx-audio APIs.
            from mlx_audio.stt import load as load_stt_model  # type: ignore
        model_name = resolve_model_name()
        print(f"Loading Qwen3 ASR model: {model_name}", file=sys.stderr, flush=True)
        start = time.time()
        _stt_model = load_stt_model(model_name)
        load_time = time.time() - start
        print(f"Model loaded in {load_time:.2f}s", file=sys.stderr, flush=True)
    return _stt_model

def transcribe_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    language: str = "auto",
) -> dict:
    """
    Transcribe audio using Qwen3 ASR with mlx-audio
    """
    try:
        # Get cached model
        stt_model = load_model()

        if language in ["auto", "", None]:
            # Preserve automatic language detection when language is not explicitly set.
            lang_param = None
        else:
            # Map common language codes to Qwen3 language names
            lang_map = {
                "zh": "Chinese",
                "zh-hans": "Chinese",
                "zh-hant": "Chinese",
                "yue": "Cantonese",
                "en": "English",
                "ja": "Japanese",
                "ko": "Korean",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "ru": "Russian",
                "ar": "Arabic",
                "hi": "Hindi",
                "th": "Thai",
                "vi": "Vietnamese",
                "tr": "Turkish",
                "pl": "Polish",
                "nl": "Dutch",
                "sv": "Swedish",
                "da": "Danish",
                "fi": "Finnish",
                "cs": "Czech",
                "el": "Greek",
                "ro": "Romanian",
                "hu": "Hungarian",
            }
            lang_lower = str(language).lower()
            lang_param = lang_map.get(lang_lower, language)

        # Run transcription directly from in-memory waveform
        # (mlx-audio supports np.ndarray / mx.array input).
        if lang_param is None:
            result_generator = stt_model.generate(audio)
        else:
            result_generator = stt_model.generate(audio, language=lang_param)

        # Extract text from result (handle generator or structured output)
        text = ""
        try:
            for chunk in result_generator:
                if hasattr(chunk, 'text'):
                    text += chunk.text
                elif isinstance(chunk, str):
                    text += chunk
        except Exception:
            if hasattr(result_generator, 'text'):
                text = result_generator.text
            else:
                text = str(result_generator)

        return {
            "text": text,
            "language": "auto",
            "confidence": 0.95,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "text": f"[转录错误: {str(e)}]",
            "language": "auto",
            "confidence": 0.0
        }

def main():
    """Main entry point - persistent server mode"""
    print("Qwen3 ASR Server starting...", file=sys.stderr, flush=True)

    # Pre-load model on startup, then run one in-memory warmup inference.
    try:
        load_model()
        warmup_start = time.time()
        warmup_audio = np.zeros(32000, dtype=np.float32)  # 2.0s @ 16kHz
        warmup_result = transcribe_audio(warmup_audio, 16000, "auto")
        if "error" in warmup_result:
            print(
                f"Warmup failed: {warmup_result['error']}",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"Warmup completed in {(time.time() - warmup_start):.2f}s",
                file=sys.stderr,
                flush=True,
            )
        print("READY", flush=True)  # Signal ready to parent process
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Process transcription requests
    stdin_buffer = sys.stdin.buffer
    while True:
        try:
            # Read metadata line first.
            line_bytes = stdin_buffer.readline()
            if not line_bytes:
                break  # EOF

            line = line_bytes.decode("utf-8").strip()
            if not line:
                continue

            # Parse request
            data = json.loads(line)

            if data.get("type") != "binary":
                raise ValueError(f"unsupported request type: {data.get('type')}")

            # Binary IPC: metadata JSON + raw float32 bytes.
            audio_len_bytes = int(data.get("audio_len_bytes", 0))
            if audio_len_bytes <= 0:
                raise ValueError(f"invalid audio_len_bytes: {audio_len_bytes}")
            audio_bytes = stdin_buffer.read(audio_len_bytes)
            if len(audio_bytes) != audio_len_bytes:
                raise EOFError(
                    f"incomplete audio payload: got {len(audio_bytes)}, expect {audio_len_bytes}"
                )
            # Use little-endian float32 to match Rust sender.
            audio = np.frombuffer(audio_bytes, dtype="<f4")
            params = data.get("params", {})

            if not isinstance(params, dict):
                params = {}

            sample_rate = params.get('sample_rate', 16000)
            language = params.get('language', 'auto')
            # Run transcription
            result = transcribe_audio(audio, sample_rate, language)

            # Output JSON result
            print(json.dumps(result), flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_response = {
                "error": str(e),
                "text": f"[处理错误: {str(e)}]",
                "language": "auto",
                "confidence": 0.0
            }
            print(json.dumps(error_response), flush=True)

if __name__ == "__main__":
    main()
