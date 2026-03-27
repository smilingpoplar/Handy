fn main() {
    println!("cargo:rustc-check-cfg=cfg(handy_qwen3asr_bridge_built)");

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    build_qwen3asr_bridge();

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    build_apple_intelligence_bridge();

    generate_tray_translations();

    tauri_build::build()
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn build_qwen3asr_bridge() {
    use std::path::Path;

    const PACKAGE_DIR: &str = "swift/handy_qwen3asr_bridge";
    const LIB_NAME: &str = "HandyQwen3ASRBridge";

    println!("cargo:rerun-if-changed={PACKAGE_DIR}");
    println!("cargo:rerun-if-env-changed=HANDY_BUILD_QWEN3ASR_MLX");

    if !env_var_enabled("HANDY_BUILD_QWEN3ASR_MLX") {
        println!("cargo:warning=Skipping Qwen3ASR bridge build because HANDY_BUILD_QWEN3ASR_MLX is not enabled.");
        return;
    }

    let manifest_path = Path::new(PACKAGE_DIR).join("Package.swift");
    if !manifest_path.exists() {
        panic!(
            "Qwen3ASR Swift bridge manifest is missing: {}",
            manifest_path.display()
        );
    }

    Qwen3ASRBridgeBuilder::new(PACKAGE_DIR, LIB_NAME).build();
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn env_var_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => false,
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct Qwen3ASRBridgeBuilder {
    package_dir: &'static str,
    lib_name: &'static str,
    out_dir: std::path::PathBuf,
    stage_root: std::path::PathBuf,
    scratch_dir: std::path::PathBuf,
    swiftpm_home: std::path::PathBuf,
    clang_cache: std::path::PathBuf,
    swiftpm_cache: std::path::PathBuf,
    swiftpm_config: std::path::PathBuf,
    swiftpm_security: std::path::PathBuf,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct Qwen3ASRBridgeArtifacts {
    primary_dylib: std::path::PathBuf,
    compatibility_dylibs: Vec<std::path::PathBuf>,
    metallib: std::path::PathBuf,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct MacOsSdkVersion {
    major: u64,
    minor: u64,
    raw: String,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Qwen3ASRBridgeBuilder {
    fn new(package_dir: &'static str, lib_name: &'static str) -> Self {
        let manifest_dir = std::path::PathBuf::from(
            std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
        );
        let out_dir =
            std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
        let stage_root = manifest_dir.join("target/qwen3asr_bridge");
        let scratch_dir = out_dir.join("qwen3asr-swiftpm");
        let swiftpm_home = scratch_dir.join("swiftpm-home");
        let clang_cache = scratch_dir.join("clang-module-cache");
        let swiftpm_cache = swiftpm_home.join("cache");
        let swiftpm_config = swiftpm_home.join("configuration");
        let swiftpm_security = swiftpm_home.join("security");

        Self {
            package_dir,
            lib_name,
            out_dir,
            stage_root,
            scratch_dir,
            swiftpm_home,
            clang_cache,
            swiftpm_cache,
            swiftpm_config,
            swiftpm_security,
        }
    }

    fn build(&self) {
        if !self.sdk_supports_bridge() {
            return;
        }

        self.prepare_dirs();
        self.run_swift_build();
        let artifacts = self.collect_bridge_artifacts();
        self.stage_bridge_bundle(&artifacts);

        println!("cargo:rustc-cfg=handy_qwen3asr_bridge_built");
        println!(
            "cargo:rustc-env=HANDY_QWEN3ASR_BRIDGE_STAGE_DIR={}",
            self.stage_root.display()
        );
    }

    fn sdk_supports_bridge(&self) -> bool {
        let sdk_version = match self.macos_sdk_version() {
            Ok(version) => version,
            Err(error) => {
                println!(
                    "cargo:warning=Skipping Qwen3ASR bridge build because the macOS SDK version could not be determined: {error}"
                );
                return false;
            }
        };

        if sdk_version.major < 14 {
            println!(
                "cargo:warning=Skipping Qwen3ASR bridge build because the available macOS SDK is {} (< 14).",
                sdk_version.raw
            );
            return false;
        }

        true
    }

    fn prepare_dirs(&self) {
        std::fs::create_dir_all(&self.swiftpm_cache)
            .expect("Failed to create SwiftPM cache directory");
        std::fs::create_dir_all(&self.swiftpm_config)
            .expect("Failed to create SwiftPM configuration directory");
        std::fs::create_dir_all(&self.swiftpm_security)
            .expect("Failed to create SwiftPM security directory");
        std::fs::create_dir_all(&self.clang_cache)
            .expect("Failed to create clang module cache directory");
    }

    fn run_swift_build(&self) {
        let status = std::process::Command::new("xcrun")
            .current_dir(self.package_dir)
            .env("HOME", &self.swiftpm_home)
            .env("CLANG_MODULE_CACHE_PATH", &self.clang_cache)
            .env("SWIFTPM_ENABLE_PLUGINS", "0")
            .args([
                "swift",
                "build",
                "-c",
                "release",
                "--triple",
                "arm64-apple-macosx14.0",
                "--scratch-path",
                self.scratch_dir
                    .to_str()
                    .expect("Failed to convert Qwen3ASR scratch path to string"),
                "--product",
                self.lib_name,
            ])
            .status()
            .expect("Failed to invoke swift build for Qwen3ASR bridge");

        if !status.success() {
            panic!("swift build failed for Qwen3ASR bridge");
        }
    }

    fn find_dylib(&self) -> Option<std::path::PathBuf> {
        find_file_recursive(&self.scratch_dir, &format!("lib{}.dylib", self.lib_name))
    }

    fn collect_bridge_artifacts(&self) -> Qwen3ASRBridgeArtifacts {
        let primary_dylib_source = self
            .find_dylib()
            .expect("Failed to locate built Qwen3ASR bridge dynamic library");
        let primary_dylib = self.copy_to_out_dir(&primary_dylib_source);
        let toolchain_rpaths = self.toolchain_rpaths(&primary_dylib);
        let compatibility_dylibs =
            self.copy_compatibility_dylibs(&primary_dylib, &toolchain_rpaths);
        self.strip_toolchain_rpaths(&primary_dylib, &toolchain_rpaths);
        let metallib = self.build_mlx_metallib();

        Qwen3ASRBridgeArtifacts {
            primary_dylib,
            compatibility_dylibs,
            metallib,
        }
    }

    fn copy_to_out_dir(&self, source_path: &std::path::Path) -> std::path::PathBuf {
        let destination = self.out_dir.join(
            source_path
                .file_name()
                .expect("Qwen3ASR bridge asset has no file name"),
        );
        std::fs::copy(source_path, &destination).unwrap_or_else(|error| {
            panic!(
                "Failed to copy Qwen3ASR bridge asset {} to {}: {}",
                source_path.display(),
                destination.display(),
                error
            )
        });
        destination
    }

    fn toolchain_rpaths(&self, dylib_path: &std::path::Path) -> Vec<String> {
        let output = std::process::Command::new("otool")
            .args(["-l", dylib_path.to_str().expect("Invalid dylib path")])
            .output()
            .expect("Failed to inspect Qwen3ASR bridge load commands");
        if !output.status.success() {
            panic!("otool -l failed for Qwen3ASR bridge dylib");
        }

        let text = String::from_utf8_lossy(&output.stdout);
        text.lines()
            .filter_map(|line| {
                let trimmed = line.trim();
                if !trimmed.starts_with("path ") {
                    return None;
                }
                let path = trimmed
                    .strip_prefix("path ")?
                    .split(" (offset")
                    .next()?
                    .to_string();
                if path.contains("/Xcode.app/")
                    && path.contains("/usr/lib/swift-")
                    && path.ends_with("/macosx")
                {
                    Some(path)
                } else {
                    None
                }
            })
            .collect()
    }

    fn copy_compatibility_dylibs(
        &self,
        dylib_path: &std::path::Path,
        toolchain_rpaths: &[String],
    ) -> Vec<std::path::PathBuf> {
        let output = std::process::Command::new("otool")
            .args(["-L", dylib_path.to_str().expect("Invalid dylib path")])
            .output()
            .expect("Failed to inspect Qwen3ASR bridge dylib dependencies");
        if !output.status.success() {
            panic!("otool -L failed for Qwen3ASR bridge dylib");
        }

        let dependency_names: Vec<String> = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| {
                let trimmed = line.trim();
                if !trimmed.starts_with("@rpath/libswiftCompatibility") {
                    return None;
                }
                Some(
                    trimmed
                        .split(" (")
                        .next()
                        .expect("Malformed otool dependency line")
                        .trim_start_matches("@rpath/")
                        .to_string(),
                )
            })
            .collect();

        let mut copied = Vec::new();
        for dependency_name in dependency_names {
            let source = toolchain_rpaths.iter().find_map(|rpath| {
                let candidate = std::path::Path::new(rpath).join(&dependency_name);
                candidate.exists().then_some(candidate)
            });
            let source = source.unwrap_or_else(|| {
                panic!(
                    "Failed to locate Swift compatibility dylib {} in {:?}",
                    dependency_name, toolchain_rpaths
                )
            });
            copied.push(self.copy_to_out_dir(&source));
        }

        copied
    }

    fn strip_toolchain_rpaths(&self, dylib_path: &std::path::Path, toolchain_rpaths: &[String]) {
        for rpath in toolchain_rpaths {
            let status = std::process::Command::new("install_name_tool")
                .args([
                    "-delete_rpath",
                    rpath,
                    dylib_path.to_str().expect("Invalid dylib path"),
                ])
                .status()
                .expect("Failed to invoke install_name_tool for Qwen3ASR bridge");
            if !status.success() {
                panic!("install_name_tool failed to strip Qwen3ASR bridge rpath {}", rpath);
            }
        }
    }

    fn stage_bridge_bundle(&self, artifacts: &Qwen3ASRBridgeArtifacts) {
        self.stage_bridge_bundle_at(&self.stage_root, artifacts);
    }

    fn stage_bridge_bundle_at(&self, bundle_dir: &std::path::Path, artifacts: &Qwen3ASRBridgeArtifacts) {
        if bundle_dir.exists() {
            for entry in std::fs::read_dir(bundle_dir)
                .expect("Failed to read Qwen3ASR bridge bundle directory")
                .flatten()
            {
                let path = entry.path();
                if path.is_dir() {
                    std::fs::remove_dir_all(&path).unwrap_or_else(|error| {
                        panic!(
                            "Failed to remove stale Qwen3ASR bridge directory {}: {}",
                            path.display(),
                            error
                        )
                    });
                } else {
                    std::fs::remove_file(&path).unwrap_or_else(|error| {
                        panic!(
                            "Failed to remove stale Qwen3ASR bridge file {}: {}",
                            path.display(),
                            error
                        )
                    });
                }
            }
        }

        std::fs::create_dir_all(bundle_dir)
            .expect("Failed to create Qwen3ASR bridge bundle directory");

        for asset in std::iter::once(&artifacts.primary_dylib).chain(artifacts.compatibility_dylibs.iter()) {
            let file_name = asset
                .file_name()
                .and_then(|name| name.to_str())
                .expect("Qwen3ASR bridge asset has invalid UTF-8 file name");
            let destination = bundle_dir.join(file_name);
            std::fs::copy(asset, &destination).unwrap_or_else(|error| {
                panic!(
                    "Failed to stage Qwen3ASR bridge asset {} to {}: {}",
                    asset.display(),
                    destination.display(),
                    error
                )
            });
        }

        let destination = bundle_dir.join("mlx.metallib");
        std::fs::copy(&artifacts.metallib, &destination).unwrap_or_else(|error| {
            panic!(
                "Failed to stage Qwen3ASR metallib {} to {}: {}",
                artifacts.metallib.display(),
                destination.display(),
                error
            )
        });
    }

    fn build_mlx_metallib(&self) -> std::path::PathBuf {
        let sdk_version = self
            .macos_sdk_version()
            .expect("Failed to determine macOS SDK version for mlx.metallib build");
        let mlx_root = self
            .find_mlx_source_root()
            .expect("Failed to locate mlx-swift Cmlx source root");
        let kernels_dir = mlx_root.join("mlx/backend/metal/kernels");
        let metallib_dir = self.scratch_dir.join("mlx-metallib");

        std::fs::create_dir_all(&metallib_dir)
            .expect("Failed to create Qwen3ASR mlx.metallib output directory");

        let kernel_sources =
            self.metal_kernel_sources(&kernels_dir, sdk_version.major, sdk_version.minor);
        if kernel_sources.is_empty() {
            panic!(
                "Failed to locate mlx-swift Metal kernel sources in {}",
                kernels_dir.display()
            );
        }

        let metallib_path = metallib_dir.join("mlx.metallib");
        if file_is_newer_than_all_sources(&metallib_path, &kernel_sources) {
            return metallib_path;
        }

        let include_root = &mlx_root;
        let mut air_files = Vec::with_capacity(kernel_sources.len());
        for source in kernel_sources {
            let relative = source
                .strip_prefix(&kernels_dir)
                .expect("Metal kernel source must be inside kernels dir");
            let air_file = metallib_dir.join(relative).with_extension("air");
            if let Some(parent) = air_file.parent() {
                std::fs::create_dir_all(parent)
                    .expect("Failed to create Metal AIR output directory");
            }

            let status = std::process::Command::new("xcrun")
                .args([
                    "-sdk",
                    "macosx",
                    "metal",
                    "-x",
                    "metal",
                    "-Wall",
                    "-Wextra",
                    "-fno-fast-math",
                    "-Wno-c++17-extensions",
                    "-Wno-c++20-extensions",
                    "-mmacosx-version-min=14.0",
                    "-c",
                ])
                .arg(&source)
                .arg(format!("-I{}", include_root.display()))
                .arg("-o")
                .arg(&air_file)
                .status()
                .unwrap_or_else(|error| {
                    panic!(
                        "Failed to invoke xcrun metal for mlx kernel {}: {}",
                        source.display(),
                        error
                    )
                });
            if !status.success() {
                panic!("xcrun metal failed for mlx kernel {}", source.display());
            }

            air_files.push(air_file);
        }

        let status = std::process::Command::new("xcrun")
            .args(["-sdk", "macosx", "metallib"])
            .args(&air_files)
            .arg("-o")
            .arg(&metallib_path)
            .status()
            .expect("Failed to invoke xcrun metallib for Qwen3ASR bridge");
        if !status.success() {
            panic!("xcrun metallib failed for Qwen3ASR bridge");
        }

        metallib_path
    }

    fn find_mlx_source_root(&self) -> Option<std::path::PathBuf> {
        find_file_recursive(&self.scratch_dir, "Cmlx+Util.swift").and_then(|path| {
            path.parent()
                .and_then(|parent| parent.parent())
                .map(|parent| parent.join("Cmlx/mlx"))
                .filter(|candidate| candidate.join("mlx/backend/metal/kernels").exists())
        })
    }

    fn metal_kernel_sources(
        &self,
        kernels_dir: &std::path::Path,
        _sdk_major: u64,
        _sdk_minor: u64,
    ) -> Vec<std::path::PathBuf> {
        let mut sources = find_files_with_extension_recursive(kernels_dir, "metal");
        sources.retain(|path| {
            let relative = path
                .strip_prefix(kernels_dir)
                .expect("Kernel source should be inside kernels dir");
            let relative = relative.to_string_lossy();

            if relative == "fence.metal" {
                return false;
            }

            if relative.contains("_nax.metal") {
                return false;
            }

            true
        });
        sources.sort();
        sources
    }

    fn macos_sdk_version(&self) -> Result<MacOsSdkVersion, String> {
        let output = std::process::Command::new("xcrun")
            .args(["--sdk", "macosx", "--show-sdk-version"])
            .output()
            .map_err(|error| error.to_string())?;
        if !output.status.success() {
            return Err("xcrun --show-sdk-version failed".to_string());
        }

        let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let mut parts = raw.split('.');
        let major = parts
            .next()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
        let minor = parts
            .next()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);

        Ok(MacOsSdkVersion { major, minor, raw })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn find_file_recursive(root: &std::path::Path, file_name: &str) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|name| name.ends_with(".dSYM"))
            {
                continue;
            }
            if let Some(found) = find_file_recursive(&path, file_name) {
                return Some(found);
            }
            continue;
        }

        if path.file_name().and_then(|n| n.to_str()) == Some(file_name) {
            return Some(path);
        }
    }
    None
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn find_files_with_extension_recursive(
    root: &std::path::Path,
    extension: &str,
) -> Vec<std::path::PathBuf> {
    let mut found = Vec::new();
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(_) => return found,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(find_files_with_extension_recursive(&path, extension));
            continue;
        }

        if path.extension().and_then(|value| value.to_str()) == Some(extension) {
            found.push(path);
        }
    }

    found
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn file_is_newer_than_all_sources(
    target: &std::path::Path,
    sources: &[std::path::PathBuf],
) -> bool {
    let target_mtime = match std::fs::metadata(target).and_then(|metadata| metadata.modified()) {
        Ok(mtime) => mtime,
        Err(_) => return false,
    };

    sources.iter().all(|source| {
        std::fs::metadata(source)
            .and_then(|metadata| metadata.modified())
            .map(|mtime| mtime <= target_mtime)
            .unwrap_or(false)
    })
}

/// Generate tray menu translations from frontend locale files.
///
/// Source of truth: src/i18n/locales/*/translation.json
/// The English "tray" section defines the struct fields.
fn generate_tray_translations() {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::Path;

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let locales_dir = Path::new("../src/i18n/locales");

    println!("cargo:rerun-if-changed=../src/i18n/locales");

    // Collect all locale translations
    let mut translations: BTreeMap<String, serde_json::Value> = BTreeMap::new();

    for entry in fs::read_dir(locales_dir).unwrap().flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let lang = path.file_name().unwrap().to_str().unwrap().to_string();
        let json_path = path.join("translation.json");

        println!("cargo:rerun-if-changed={}", json_path.display());

        let content = fs::read_to_string(&json_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

        if let Some(tray) = parsed.get("tray").cloned() {
            translations.insert(lang, tray);
        }
    }

    // English defines the schema
    let english = translations.get("en").unwrap().as_object().unwrap();
    let fields: Vec<_> = english
        .keys()
        .map(|k| (camel_to_snake(k), k.clone()))
        .collect();

    // Generate code
    let mut out = String::from(
        "// Auto-generated from src/i18n/locales/*/translation.json - do not edit\n\n",
    );

    // Struct
    out.push_str("#[derive(Debug, Clone)]\npub struct TrayStrings {\n");
    for (rust_field, _) in &fields {
        out.push_str(&format!("    pub {rust_field}: String,\n"));
    }
    out.push_str("}\n\n");

    // Static map
    out.push_str(
        "pub static TRANSLATIONS: Lazy<HashMap<&'static str, TrayStrings>> = Lazy::new(|| {\n",
    );
    out.push_str("    let mut m = HashMap::new();\n");

    for (lang, tray) in &translations {
        out.push_str(&format!("    m.insert(\"{lang}\", TrayStrings {{\n"));
        for (rust_field, json_key) in &fields {
            let val = tray.get(json_key).and_then(|v| v.as_str()).unwrap_or("");
            out.push_str(&format!(
                "        {rust_field}: \"{}\".to_string(),\n",
                escape_string(val)
            ));
        }
        out.push_str("    });\n");
    }

    out.push_str("    m\n});\n");

    fs::write(Path::new(&out_dir).join("tray_translations.rs"), out).unwrap();

    println!(
        "cargo:warning=Generated tray translations: {} languages, {} fields",
        translations.len(),
        fields.len()
    );
}

fn camel_to_snake(s: &str) -> String {
    s.chars()
        .enumerate()
        .fold(String::new(), |mut acc, (i, c)| {
            if c.is_uppercase() && i > 0 {
                acc.push('_');
            }
            acc.push(c.to_lowercase().next().unwrap());
            acc
        })
}

fn escape_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn build_apple_intelligence_bridge() {
    use std::env;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    const REAL_SWIFT_FILE: &str = "swift/apple_intelligence.swift";
    const STUB_SWIFT_FILE: &str = "swift/apple_intelligence_stub.swift";
    const BRIDGE_HEADER: &str = "swift/apple_intelligence_bridge.h";

    println!("cargo:rerun-if-changed={REAL_SWIFT_FILE}");
    println!("cargo:rerun-if-changed={STUB_SWIFT_FILE}");
    println!("cargo:rerun-if-changed={BRIDGE_HEADER}");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let object_path = out_dir.join("apple_intelligence.o");
    let static_lib_path = out_dir.join("libapple_intelligence.a");

    let sdk_path = String::from_utf8(
        Command::new("xcrun")
            .args(["--sdk", "macosx", "--show-sdk-path"])
            .output()
            .expect("Failed to locate macOS SDK")
            .stdout,
    )
    .expect("SDK path is not valid UTF-8")
    .trim()
    .to_string();

    // Check if the SDK supports FoundationModels (required for Apple Intelligence)
    let framework_path =
        Path::new(&sdk_path).join("System/Library/Frameworks/FoundationModels.framework");
    let has_foundation_models = framework_path.exists();

    let source_file = if has_foundation_models {
        println!("cargo:warning=Building with Apple Intelligence support.");
        REAL_SWIFT_FILE
    } else {
        println!("cargo:warning=Apple Intelligence SDK not found. Building with stubs.");
        STUB_SWIFT_FILE
    };

    if !Path::new(source_file).exists() {
        panic!("Source file {} is missing!", source_file);
    }

    let swiftc_path = String::from_utf8(
        Command::new("xcrun")
            .args(["--find", "swiftc"])
            .output()
            .expect("Failed to locate swiftc")
            .stdout,
    )
    .expect("swiftc path is not valid UTF-8")
    .trim()
    .to_string();

    let toolchain_swift_lib = Path::new(&swiftc_path)
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("lib/swift/macosx"))
        .expect("Unable to determine Swift toolchain lib directory");
    let sdk_swift_lib = Path::new(&sdk_path).join("usr/lib/swift");

    // Use macOS 11.0 as deployment target for compatibility
    // The @available(macOS 26.0, *) checks in Swift handle runtime availability
    // Weak linking for FoundationModels is handled via cargo:rustc-link-arg below
    let status = Command::new("xcrun")
        .args([
            "swiftc",
            "-target",
            "arm64-apple-macosx11.0",
            "-sdk",
            &sdk_path,
            "-O",
            "-import-objc-header",
            BRIDGE_HEADER,
            "-c",
            source_file,
            "-o",
            object_path
                .to_str()
                .expect("Failed to convert object path to string"),
        ])
        .status()
        .expect("Failed to invoke swiftc for Apple Intelligence bridge");

    if !status.success() {
        panic!("swiftc failed to compile {source_file}");
    }

    let status = Command::new("libtool")
        .args([
            "-static",
            "-o",
            static_lib_path
                .to_str()
                .expect("Failed to convert static lib path to string"),
            object_path
                .to_str()
                .expect("Failed to convert object path to string"),
        ])
        .status()
        .expect("Failed to create static library for Apple Intelligence bridge");

    if !status.success() {
        panic!("libtool failed for Apple Intelligence bridge");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=apple_intelligence");
    println!(
        "cargo:rustc-link-search=native={}",
        toolchain_swift_lib.display()
    );
    println!("cargo:rustc-link-search=native={}", sdk_swift_lib.display());
    println!("cargo:rustc-link-lib=framework=Foundation");

    if has_foundation_models {
        // Use weak linking so the app can launch on systems without FoundationModels
        println!("cargo:rustc-link-arg=-weak_framework");
        println!("cargo:rustc-link-arg=FoundationModels");
    }

    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
}
