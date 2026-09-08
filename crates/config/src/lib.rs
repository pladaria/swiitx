//! Shared configuration for Nixe applications.

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

use nixe_gpu::{
    DEFAULT_BIND_GROUPS_PER_DESCRIPTOR_TABLE, DEFAULT_PERSISTENT_PIPELINE_CACHE_BYTES,
    DEFAULT_PIPELINE_CACHE_ENTRIES, DEFAULT_PIPELINE_VARIANTS_PER_RESOURCE,
    DEFAULT_SHADER_CACHE_ENTRIES, GpuCacheConfiguration,
};
use nixe_input::{ControllerKind, InputSnapshot};
use nixe_loader_title::{DirectoryScanOptions, NacpLanguage};
use serde::Deserialize;

pub use nixe_input::GamepadProfile;

/// Configuration file name used during automatic discovery.
pub const CONFIG_FILE_NAME: &str = "nixe.toml";

/// Current configuration schema version.
pub const CONFIG_VERSION: u32 = 2;

/// Configuration shared by the CLI and desktop applications.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NixeConfig {
    /// Version of the configuration schema.
    pub version: u32,
    /// Title-library locations and discovery behavior.
    pub library: LibraryConfig,
    /// Host-backed filesystem locations exposed to the emulated system.
    pub filesystem: FileSystemConfig,
    /// System-wide preferences and caller-owned key location.
    pub system: SystemConfig,
    /// Cross-cutting diagnostic preferences consumed by application runtimes.
    pub diagnostics: DiagnosticsConfig,
    /// CPU backend selection and resource policy.
    pub cpu: CpuConfig,
    /// GPU shader, pipeline, and backend-derived cache policy.
    pub gpu: GpuCacheConfiguration,
    /// Host gamepad identification and emulated-controller mappings.
    pub input: InputConfig,
    source_path: PathBuf,
}

impl NixeConfig {
    /// Loads and validates a configuration file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_owned(),
            source,
        })?;
        let raw: RawConfig = toml::from_str(&text).map_err(|source| ConfigError::Parse {
            path: path.to_owned(),
            source,
        })?;
        if raw.version != CONFIG_VERSION {
            return Err(ConfigError::UnsupportedVersion {
                path: path.to_owned(),
                version: raw.version,
            });
        }
        validate_time_config(path, &raw.system.time)?;
        validate_input_config(path, &raw.input)?;
        let source_path = absolute_path(path).map_err(|source| ConfigError::Io {
            path: path.to_owned(),
            source,
        })?;
        let base_directory = source_path
            .parent()
            .expect("an absolute file path must have a parent");
        let cpu = cpu_configuration(raw.cpu);
        let gpu = gpu_cache_configuration(path, raw.gpu)?;

        Ok(Self {
            version: raw.version,
            library: LibraryConfig {
                paths: raw
                    .library
                    .paths
                    .into_iter()
                    .map(|path| resolve_path(base_directory, path))
                    .collect(),
                recursive_scan: raw.library.recursive_scan,
                follow_symlinks: raw.library.follow_symlinks,
            },
            filesystem: FileSystemConfig {
                sd_card: resolve_path(base_directory, raw.filesystem.sd_card),
            },
            system: SystemConfig {
                preferred_languages: raw.system.preferred_languages,
                keys: resolve_path(base_directory, raw.system.keys),
                initial_operation_mode: raw.system.initial_operation_mode,
                time: TimeConfig {
                    mode: raw.system.time.mode,
                    timezone: raw.system.time.timezone,
                    fixed_unix_timestamp: raw.system.time.fixed_unix_timestamp,
                },
            },
            diagnostics: DiagnosticsConfig {
                log_level: raw.diagnostics.log_level,
                guest_logs_level: raw.diagnostics.guest_logs_level,
                file_system_access_log: raw.diagnostics.file_system_access_log,
            },
            cpu,
            gpu,
            input: raw.input,
            source_path,
        })
    }

    /// Loads the first automatically discovered configuration file.
    pub fn load_discovered() -> Result<Option<Self>, ConfigError> {
        Self::discover_path().map(Self::load).transpose()
    }

    /// Finds the configuration selected by the environment or conventional paths.
    pub fn discover_path() -> Option<PathBuf> {
        if let Some(path) = env::var_os("NIXE_CONFIG").filter(|path| !path.is_empty()) {
            return Some(PathBuf::from(path));
        }

        let local = PathBuf::from(CONFIG_FILE_NAME);
        if local.is_file() {
            return Some(local);
        }

        user_config_path().filter(|path| path.is_file())
    }

    /// Returns the absolute path of the file from which this value was loaded.
    pub fn source_path(&self) -> &Path {
        &self.source_path
    }
}

/// Title-library locations and discovery behavior.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibraryConfig {
    /// Directories containing title packages.
    pub paths: Vec<PathBuf>,
    /// Whether directory scans descend into subdirectories.
    pub recursive_scan: bool,
    /// Whether discovery follows symbolic links to files and directories.
    pub follow_symlinks: bool,
}

impl LibraryConfig {
    /// Converts the shared setting to the title loader's scan options.
    pub const fn scan_options(&self) -> DirectoryScanOptions {
        DirectoryScanOptions::new()
            .with_recursive(self.recursive_scan)
            .with_follow_symlinks(self.follow_symlinks)
    }
}

/// Host-backed filesystem locations exposed through Horizon services.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileSystemConfig {
    /// Directory exposed to the guest as the removable `sdmc:` filesystem.
    pub sd_card: PathBuf,
}

/// System-wide preferences shared by applications.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemConfig {
    /// Languages to try in descending preference order.
    pub preferred_languages: Vec<NacpLanguage>,
    /// Directory containing caller-owned `prod.keys` and optional `title.keys`.
    pub keys: PathBuf,
    /// Operation mode reported to titles when the emulated system starts.
    pub initial_operation_mode: InitialOperationMode,
    /// Initial virtual-time policy.
    pub time: TimeConfig,
}

/// Virtual-time preferences shared by application frontends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeConfig {
    /// How the guest wall clock advances.
    pub mode: TimeMode,
    /// IANA time-zone location reported to Horizon clients.
    pub timezone: String,
    /// POSIX timestamp used when `mode` is `fixed`.
    pub fixed_unix_timestamp: Option<i64>,
}

/// Guest wall-clock policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TimeMode {
    /// Anchor to host time at process launch and advance monotonically.
    #[default]
    Realtime,
    /// Freeze wall and monotonic time at `fixed_unix_timestamp`.
    Fixed,
}

/// Initial physical presentation selected for the emulated console.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InitialOperationMode {
    /// The console starts outside its dock.
    Handheld,
    /// The console starts connected to its dock.
    Docked,
}

/// Minimum severity emitted by application loggers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticLogLevel {
    Error,
    Warn,
    #[default]
    Info,
    Debug,
    Trace,
}

/// Host severity policy applied to logs emitted by Horizon guests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GuestLogsLevel {
    /// Preserve the severity encoded by each guest log packet.
    #[default]
    Inherit,
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Off,
}

/// Cross-cutting diagnostics configuration shared by applications.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiagnosticsConfig {
    /// Minimum severity emitted by application loggers.
    pub log_level: DiagnosticLogLevel,
    /// Host severity policy applied to logs emitted by Horizon guests.
    pub guest_logs_level: GuestLogsLevel,
    /// Requests filesystem access logs from the guest SDK.
    pub file_system_access_log: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuConfig {
    pub backend: CpuBackendSelection,
    /// Enables host-parallel execution. Deterministic serialized workers
    /// remain the default.
    pub parallel_vcpus: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CpuBackendSelection {
    #[default]
    Jit,
    Interpreter,
}

/// Named mappings from host gamepads to the emulated controller.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputConfig {
    #[serde(default)]
    pub profiles: BTreeMap<String, GamepadProfile>,
}

impl InputConfig {
    /// Finds the unique profile matching an SDL device name and controller type.
    #[must_use]
    pub fn matching_profile(
        &self,
        device: &str,
        controller_type: ControllerKind,
    ) -> Option<(&str, &GamepadProfile)> {
        self.profiles
            .iter()
            .find(|(_, profile)| {
                profile.device == device && profile.controller_type == controller_type
            })
            .map(|(name, profile)| (name.as_str(), profile))
    }

    /// Matches only the first attached controller, never a later controller.
    #[must_use]
    pub fn matching_first_controller(
        &self,
        snapshot: &InputSnapshot,
    ) -> Option<(&str, &GamepadProfile)> {
        let controller = snapshot.controllers.first()?;
        self.matching_profile(&controller.name, controller.kind)
    }
}

/// Errors produced while locating or loading shared configuration.
#[derive(Debug)]
pub enum ConfigError {
    /// The configuration file could not be read or its path could not be resolved.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The TOML document does not match the configuration schema.
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
    /// The document uses a schema version this build cannot interpret.
    UnsupportedVersion { path: PathBuf, version: u32 },
    /// Virtual-time settings are internally inconsistent or not representable.
    InvalidTime { path: PathBuf, reason: String },
    /// Input profiles contain ambiguous device selectors.
    InvalidInput { path: PathBuf, reason: String },
    /// GPU cache capacities are zero or cannot be represented in bytes.
    InvalidGpu { path: PathBuf, reason: String },
}

impl Display for ConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    formatter,
                    "cannot read configuration {}: {source}",
                    path.display()
                )
            }
            Self::Parse { path, source } => {
                write!(
                    formatter,
                    "invalid configuration {}: {source}",
                    path.display()
                )
            }
            Self::UnsupportedVersion { path, version } => write!(
                formatter,
                "configuration {} uses unsupported version {version}; expected {CONFIG_VERSION}",
                path.display()
            ),
            Self::InvalidTime { path, reason } => write!(
                formatter,
                "configuration {} has invalid system.time settings: {reason}",
                path.display()
            ),
            Self::InvalidInput { path, reason } => write!(
                formatter,
                "configuration {} has invalid input settings: {reason}",
                path.display()
            ),
            Self::InvalidGpu { path, reason } => write!(
                formatter,
                "configuration {} has invalid GPU settings: {reason}",
                path.display()
            ),
        }
    }
}

impl Error for ConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parse { source, .. } => Some(source),
            Self::UnsupportedVersion { .. }
            | Self::InvalidTime { .. }
            | Self::InvalidInput { .. }
            | Self::InvalidGpu { .. } => None,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    version: u32,
    library: RawLibraryConfig,
    #[serde(default)]
    filesystem: RawFileSystemConfig,
    system: RawSystemConfig,
    #[serde(default)]
    diagnostics: RawDiagnosticsConfig,
    #[serde(default)]
    cpu: RawCpuConfig,
    #[serde(default)]
    gpu: RawGpuConfig,
    #[serde(default)]
    input: InputConfig,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCpuConfig {
    #[serde(default)]
    backend: CpuBackendSelection,
    #[serde(default)]
    parallel_vcpus: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGpuConfig {
    #[serde(default = "default_shader_cache_entries")]
    shader_cache_entries: usize,
    #[serde(default = "default_pipeline_cache_entries")]
    pipeline_cache_entries: usize,
    #[serde(default = "default_pipeline_variants_per_resource")]
    pipeline_variants_per_resource: usize,
    #[serde(default = "default_bind_groups_per_descriptor_table")]
    bind_groups_per_descriptor_table: usize,
    #[serde(default = "default_persistent_pipeline_cache_mib")]
    persistent_pipeline_cache_mib: u64,
}

impl Default for RawGpuConfig {
    fn default() -> Self {
        Self {
            shader_cache_entries: default_shader_cache_entries(),
            pipeline_cache_entries: default_pipeline_cache_entries(),
            pipeline_variants_per_resource: default_pipeline_variants_per_resource(),
            bind_groups_per_descriptor_table: default_bind_groups_per_descriptor_table(),
            persistent_pipeline_cache_mib: default_persistent_pipeline_cache_mib(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLibraryConfig {
    paths: Vec<PathBuf>,
    #[serde(default = "default_recursive_scan")]
    recursive_scan: bool,
    #[serde(default = "default_follow_symlinks")]
    follow_symlinks: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFileSystemConfig {
    #[serde(default = "default_sd_card_path")]
    sd_card: PathBuf,
}

impl Default for RawFileSystemConfig {
    fn default() -> Self {
        Self {
            sd_card: default_sd_card_path(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSystemConfig {
    #[serde(deserialize_with = "deserialize_languages")]
    preferred_languages: Vec<NacpLanguage>,
    keys: PathBuf,
    initial_operation_mode: InitialOperationMode,
    #[serde(default)]
    time: RawTimeConfig,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTimeConfig {
    #[serde(default)]
    mode: TimeMode,
    #[serde(default = "default_timezone")]
    timezone: String,
    fixed_unix_timestamp: Option<i64>,
}

impl Default for RawTimeConfig {
    fn default() -> Self {
        Self {
            mode: TimeMode::Realtime,
            timezone: default_timezone(),
            fixed_unix_timestamp: None,
        }
    }
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDiagnosticsConfig {
    #[serde(default)]
    log_level: DiagnosticLogLevel,
    #[serde(default)]
    guest_logs_level: GuestLogsLevel,
    #[serde(default)]
    file_system_access_log: bool,
}

const fn default_follow_symlinks() -> bool {
    true
}

const fn default_recursive_scan() -> bool {
    true
}

fn default_timezone() -> String {
    "UTC".to_owned()
}

fn default_sd_card_path() -> PathBuf {
    PathBuf::from("./storage/sdmc")
}

const fn default_shader_cache_entries() -> usize {
    DEFAULT_SHADER_CACHE_ENTRIES
}

const fn default_pipeline_cache_entries() -> usize {
    DEFAULT_PIPELINE_CACHE_ENTRIES
}

const fn default_pipeline_variants_per_resource() -> usize {
    DEFAULT_PIPELINE_VARIANTS_PER_RESOURCE
}

const fn default_bind_groups_per_descriptor_table() -> usize {
    DEFAULT_BIND_GROUPS_PER_DESCRIPTOR_TABLE
}

const fn default_persistent_pipeline_cache_mib() -> u64 {
    DEFAULT_PERSISTENT_PIPELINE_CACHE_BYTES / (1024 * 1024)
}

fn gpu_cache_configuration(
    path: &Path,
    raw: RawGpuConfig,
) -> Result<GpuCacheConfiguration, ConfigError> {
    let persistent_bytes = raw
        .persistent_pipeline_cache_mib
        .checked_mul(1024 * 1024)
        .ok_or_else(|| ConfigError::InvalidGpu {
            path: path.to_owned(),
            reason: "persistent_pipeline_cache_mib does not fit in bytes".to_owned(),
        })?;
    GpuCacheConfiguration::new(
        raw.shader_cache_entries,
        raw.pipeline_cache_entries,
        raw.pipeline_variants_per_resource,
        raw.bind_groups_per_descriptor_table,
        persistent_bytes,
    )
    .map_err(|error| ConfigError::InvalidGpu {
        path: path.to_owned(),
        reason: error.to_string(),
    })
}

fn cpu_configuration(raw: RawCpuConfig) -> CpuConfig {
    CpuConfig {
        backend: raw.backend,
        parallel_vcpus: raw.parallel_vcpus,
    }
}

fn validate_time_config(path: &Path, time: &RawTimeConfig) -> Result<(), ConfigError> {
    let valid_timezone = !time.timezone.is_empty()
        && time.timezone.len() < 0x24
        && time.timezone.is_ascii()
        && !time.timezone.starts_with('/')
        && !time.timezone.ends_with('/')
        && time
            .timezone
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
        && time
            .timezone
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'_' | b'-' | b'+'));
    if !valid_timezone {
        return Err(ConfigError::InvalidTime {
            path: path.to_owned(),
            reason: "timezone must be a representable IANA location name".to_owned(),
        });
    }
    // `chrono-tz` embeds the IANA database and gives configuration loading the
    // same typed location-name validation used by the Horizon service:
    // https://docs.rs/chrono-tz/0.10.4/chrono_tz/enum.Tz.html
    if time.timezone.parse::<chrono_tz::Tz>().is_err() {
        return Err(ConfigError::InvalidTime {
            path: path.to_owned(),
            reason: format!("timezone {:?} is not in the IANA database", time.timezone),
        });
    }
    match (time.mode, time.fixed_unix_timestamp) {
        (TimeMode::Realtime, None) | (TimeMode::Fixed, Some(0..)) => Ok(()),
        (TimeMode::Realtime, Some(_)) => Err(ConfigError::InvalidTime {
            path: path.to_owned(),
            reason: "fixed_unix_timestamp is only valid with mode = \"fixed\"".to_owned(),
        }),
        (TimeMode::Fixed, None) => Err(ConfigError::InvalidTime {
            path: path.to_owned(),
            reason: "mode = \"fixed\" requires fixed_unix_timestamp".to_owned(),
        }),
        (TimeMode::Fixed, Some(_)) => Err(ConfigError::InvalidTime {
            path: path.to_owned(),
            reason: "fixed_unix_timestamp must be non-negative".to_owned(),
        }),
    }
}

fn validate_input_config(path: &Path, input: &InputConfig) -> Result<(), ConfigError> {
    let mut selectors = HashMap::new();
    for (name, profile) in &input.profiles {
        let selector = (profile.device.as_str(), profile.controller_type);
        if let Some(previous) = selectors.insert(selector, name) {
            return Err(ConfigError::InvalidInput {
                path: path.to_owned(),
                reason: format!(
                    "profiles `{previous}` and `{name}` both match device {:?} with type {}",
                    profile.device,
                    profile.controller_type.identifier()
                ),
            });
        }
    }
    Ok(())
}

fn deserialize_languages<'de, D>(deserializer: D) -> Result<Vec<NacpLanguage>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let names = Vec::<String>::deserialize(deserializer)?;
    names
        .into_iter()
        .map(|name| {
            NacpLanguage::ALL
                .into_iter()
                .find(|language| language.icon_suffix() == name)
                .ok_or_else(|| serde::de::Error::custom(format!("unknown language `{name}`")))
        })
        .collect()
}

fn absolute_path(path: &Path) -> std::io::Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        Ok(env::current_dir()?.join(path))
    }
}

fn resolve_path(base_directory: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base_directory.join(path)
    }
}

#[cfg(target_os = "windows")]
fn user_config_path() -> Option<PathBuf> {
    env::var_os("APPDATA").map(|root| PathBuf::from(root).join("Nixe").join(CONFIG_FILE_NAME))
}

#[cfg(target_os = "macos")]
fn user_config_path() -> Option<PathBuf> {
    env::var_os("HOME").map(|root| {
        PathBuf::from(root)
            .join("Library/Application Support/Nixe")
            .join(CONFIG_FILE_NAME)
    })
}

#[cfg(all(unix, not(target_os = "macos")))]
fn user_config_path() -> Option<PathBuf> {
    env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|root| PathBuf::from(root).join(".config")))
        .map(|root| root.join("nixe").join(CONFIG_FILE_NAME))
}

#[cfg(not(any(unix, target_os = "windows")))]
fn user_config_path() -> Option<PathBuf> {
    None
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use nixe_input::{
        Axis, Button, ButtonSet, ControllerId, ControllerState, DPadState, FaceButtonLabels,
        MotionSensor, MotionState, StickState, TriggerState,
    };

    use super::*;

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    struct TemporaryConfig {
        directory: PathBuf,
        path: PathBuf,
    }

    impl TemporaryConfig {
        fn new(contents: &str) -> Self {
            let sequence = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
            let directory =
                env::temp_dir().join(format!("nixe-config-{}-{sequence}", std::process::id()));
            fs::create_dir(&directory).unwrap();
            let path = directory.join(CONFIG_FILE_NAME);
            fs::write(&path, contents).unwrap();
            Self { directory, path }
        }
    }

    impl Drop for TemporaryConfig {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.directory).unwrap();
        }
    }

    fn controller(name: &str, kind: ControllerKind) -> ControllerState {
        ControllerState {
            id: ControllerId::new(1),
            name: name.to_owned(),
            kind,
            buttons: ButtonSet::default(),
            button_labels: FaceButtonLabels::default(),
            dpad: DPadState::default(),
            left_stick: StickState::default(),
            right_stick: StickState::default(),
            triggers: TriggerState::default(),
            motion: MotionState::default(),
        }
    }

    #[test]
    fn loads_typed_values_and_resolves_relative_paths() {
        let file = TemporaryConfig::new(
            r#"
                version = 2
                [library]
                paths = ["./roms", "other"]
                recursive_scan = false
                follow_symlinks = false
                [filesystem]
                sd_card = "./custom-sd"
                [system]
                preferred_languages = ["Spanish", "AmericanEnglish"]
                keys = "./keys"
                initial_operation_mode = "docked"
            "#,
        );

        let config = NixeConfig::load(&file.path).unwrap();
        let base = file.path.parent().unwrap();

        assert_eq!(config.source_path(), file.path);
        assert_eq!(config.library.paths[0], base.join("./roms"));
        assert_eq!(config.library.paths[1], base.join("other"));
        assert!(!config.library.scan_options().recursive);
        assert!(!config.library.scan_options().follow_symlinks);
        assert_eq!(config.filesystem.sd_card, base.join("./custom-sd"));
        assert_eq!(
            config.system.preferred_languages,
            vec![NacpLanguage::Spanish, NacpLanguage::AmericanEnglish]
        );
        assert_eq!(config.system.keys, base.join("./keys"));
        assert_eq!(
            config.system.initial_operation_mode,
            InitialOperationMode::Docked
        );
        assert_eq!(config.system.time.mode, TimeMode::Realtime);
        assert_eq!(config.system.time.timezone, "UTC");
        assert_eq!(config.system.time.fixed_unix_timestamp, None);
        assert_eq!(config.diagnostics.log_level, DiagnosticLogLevel::Info);
        assert_eq!(config.diagnostics.guest_logs_level, GuestLogsLevel::Inherit);
        assert!(!config.diagnostics.file_system_access_log);
        assert_eq!(config.gpu, GpuCacheConfiguration::default());
        assert!(config.input.profiles.is_empty());
    }

    #[test]
    fn guest_diagnostics_are_typed_and_default_to_non_intrusive_policy() {
        let explicit = TemporaryConfig::new(
            r#"
                version = 2
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
                initial_operation_mode = "handheld"
                [diagnostics]
                guest_logs_level = "warn"
                file_system_access_log = true
            "#,
        );
        let config = NixeConfig::load(&explicit.path).unwrap();
        assert_eq!(config.diagnostics.guest_logs_level, GuestLogsLevel::Warn);
        assert!(config.diagnostics.file_system_access_log);

        for level in ["inherit", "trace", "debug", "info", "warn", "error", "off"] {
            let file = TemporaryConfig::new(&format!(
                r#"
                    version = 2
                    [library]
                    paths = []
                    [system]
                    preferred_languages = []
                    keys = "keys"
                    initial_operation_mode = "handheld"
                    [diagnostics]
                    guest_logs_level = "{level}"
                "#
            ));
            NixeConfig::load(&file.path).unwrap();
        }
    }

    #[test]
    fn defaults_recursive_scanning_to_true() {
        let file = TemporaryConfig::new(
            r#"
                version = 2
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
                initial_operation_mode = "handheld"
            "#,
        );

        let config = NixeConfig::load(&file.path).unwrap();

        assert!(config.library.recursive_scan);
        assert!(config.library.follow_symlinks);
        assert!(config.library.scan_options().follow_symlinks);
        assert_eq!(
            config.filesystem.sd_card,
            file.path.parent().unwrap().join("./storage/sdmc")
        );
    }

    #[test]
    fn cpu_backend_selection_is_typed() {
        let default_file = TemporaryConfig::new(
            r#"
                version = 2
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
                initial_operation_mode = "handheld"
            "#,
        );
        assert_eq!(
            NixeConfig::load(&default_file.path).unwrap().cpu.backend,
            CpuBackendSelection::Jit
        );
        assert!(
            !NixeConfig::load(&default_file.path)
                .unwrap()
                .cpu
                .parallel_vcpus
        );
        let explicit_file = TemporaryConfig::new(
            r#"
                version = 2
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
                initial_operation_mode = "handheld"
                [cpu]
                backend = "interpreter"
                parallel_vcpus = true
            "#,
        );
        assert_eq!(
            NixeConfig::load(&explicit_file.path).unwrap().cpu.backend,
            CpuBackendSelection::Interpreter
        );
        assert!(
            NixeConfig::load(&explicit_file.path)
                .unwrap()
                .cpu
                .parallel_vcpus
        );

        let invalid_backend = TemporaryConfig::new(
            r#"
                version = 2
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
                initial_operation_mode = "handheld"
                [cpu]
                backend = "native"
            "#,
        );
        assert!(matches!(
            NixeConfig::load(&invalid_backend.path),
            Err(ConfigError::Parse { .. })
        ));
    }

    #[test]
    fn gpu_cache_limits_are_typed_defaulted_and_validated() {
        let explicit = TemporaryConfig::new(
            r#"
                version = 2
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
                initial_operation_mode = "handheld"
                [gpu]
                shader_cache_entries = 512
                pipeline_cache_entries = 2048
                pipeline_variants_per_resource = 24
                bind_groups_per_descriptor_table = 12
                persistent_pipeline_cache_mib = 96
            "#,
        );
        let config = NixeConfig::load(&explicit.path).unwrap();
        assert_eq!(config.gpu.shader_entries(), 512);
        assert_eq!(config.gpu.pipeline_entries(), 2_048);
        assert_eq!(config.gpu.pipeline_variants_per_resource(), 24);
        assert_eq!(config.gpu.bind_groups_per_descriptor_table(), 12);
        assert_eq!(
            config.gpu.persistent_pipeline_cache_bytes(),
            96 * 1024 * 1024
        );

        for setting in [
            "shader_cache_entries = 0",
            "pipeline_cache_entries = 0",
            "pipeline_variants_per_resource = 0",
            "bind_groups_per_descriptor_table = 0",
            "persistent_pipeline_cache_mib = 0",
        ] {
            let invalid = TemporaryConfig::new(&format!(
                r#"
                    version = 2
                    [library]
                    paths = []
                    [system]
                    preferred_languages = []
                    keys = "keys"
                    initial_operation_mode = "handheld"
                    [gpu]
                    {setting}
                "#
            ));
            assert!(matches!(
                NixeConfig::load(&invalid.path),
                Err(ConfigError::InvalidGpu { .. })
            ));
        }
    }

    #[test]
    fn loads_and_matches_a_typed_gamepad_profile() {
        let file = TemporaryConfig::new(
            r#"
                version = 2
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
                initial_operation_mode = "handheld"
                [input.profiles.switch-pro]
                device = "Nintendo Switch Pro Controller"
                type = "switch-pro"
                a = "east"
                plus = "start"
                zl = "lefttrigger"
                leftx = "leftx"
                gyroscope = "gyroscope"
            "#,
        );

        let config = NixeConfig::load(&file.path).unwrap();
        let (name, profile) = config
            .input
            .matching_profile("Nintendo Switch Pro Controller", ControllerKind::SwitchPro)
            .unwrap();

        assert_eq!(name, "switch-pro");
        assert_eq!(profile.a, Some(Button::East));
        assert_eq!(profile.plus, Some(Button::Start));
        assert_eq!(profile.zl, Some(Axis::LeftTrigger));
        assert_eq!(profile.leftx, Some(Axis::LeftX));
        assert_eq!(profile.gyroscope, Some(MotionSensor::Gyroscope));
        assert!(
            config
                .input
                .matching_profile("Another controller", ControllerKind::SwitchPro)
                .is_none()
        );

        let snapshot = InputSnapshot {
            controllers: vec![
                controller("Another controller", ControllerKind::Standard),
                controller("Nintendo Switch Pro Controller", ControllerKind::SwitchPro),
            ],
        };
        assert!(config.input.matching_first_controller(&snapshot).is_none());

        let snapshot = InputSnapshot {
            controllers: vec![controller(
                "Nintendo Switch Pro Controller",
                ControllerKind::SwitchPro,
            )],
        };
        assert_eq!(
            config
                .input
                .matching_first_controller(&snapshot)
                .map(|(name, _)| name),
            Some("switch-pro")
        );
    }

    #[test]
    fn rejects_unknown_profile_identifiers_and_duplicate_selectors() {
        let invalid_identifier = TemporaryConfig::new(
            r#"
                version = 2
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
                initial_operation_mode = "handheld"
                [input.profiles.invalid]
                device = "Controller"
                type = "switch-pro"
                a = "right"
            "#,
        );
        assert!(matches!(
            NixeConfig::load(&invalid_identifier.path),
            Err(ConfigError::Parse { .. })
        ));

        let duplicate = TemporaryConfig::new(
            r#"
                version = 2
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
                initial_operation_mode = "handheld"
                [input.profiles.one]
                device = "Controller"
                type = "standard"
                [input.profiles.two]
                device = "Controller"
                type = "standard"
            "#,
        );
        assert!(matches!(
            NixeConfig::load(&duplicate.path),
            Err(ConfigError::InvalidInput { .. })
        ));
    }

    #[test]
    fn loads_fixed_time_and_validates_iana_location_names() {
        let file = TemporaryConfig::new(
            r#"
                version = 2
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
                initial_operation_mode = "handheld"
                [system.time]
                mode = "fixed"
                timezone = "Europe/Madrid"
                fixed_unix_timestamp = 1704067200
            "#,
        );

        let config = NixeConfig::load(&file.path).unwrap();
        assert_eq!(config.system.time.mode, TimeMode::Fixed);
        assert_eq!(config.system.time.timezone, "Europe/Madrid");
        assert_eq!(config.system.time.fixed_unix_timestamp, Some(1_704_067_200));

        for invalid_time in [
            "mode = \"fixed\"\ntimezone = \"Europe/Madrid\"",
            "mode = \"realtime\"\ntimezone = \"Mars/Olympus\"",
            "mode = \"realtime\"\ntimezone = \"UTC\"\nfixed_unix_timestamp = 0",
        ] {
            let file = TemporaryConfig::new(&format!(
                r#"
                    version = 2
                    [library]
                    paths = []
                    [system]
                    preferred_languages = []
                    keys = "keys"
                    initial_operation_mode = "handheld"
                    [system.time]
                    {invalid_time}
                "#
            ));
            assert!(matches!(
                NixeConfig::load(&file.path),
                Err(ConfigError::InvalidTime { .. })
            ));
        }
    }

    #[test]
    fn rejects_unknown_fields_languages_and_versions() {
        for contents in [
            r#"
                version = 2
                typo = true
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
                initial_operation_mode = "handheld"
            "#,
            r#"
                version = 2
                [library]
                paths = []
                [system]
                preferred_languages = ["Klingon"]
                keys = "keys"
                initial_operation_mode = "handheld"
            "#,
            r#"
                version = 2
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
                initial_operation_mode = "tabletop"
            "#,
            r#"
                version = 2
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
                initial_operation_mode = "handheld"
                [diagnostics]
                log_level = "verbose"
            "#,
        ] {
            let file = TemporaryConfig::new(contents);
            assert!(matches!(
                NixeConfig::load(&file.path),
                Err(ConfigError::Parse { .. })
            ));
        }

        let file = TemporaryConfig::new(
            r#"
                version = 1
                [library]
                paths = []
                [system]
                preferred_languages = []
                keys = "keys"
                initial_operation_mode = "handheld"
            "#,
        );
        assert!(matches!(
            NixeConfig::load(&file.path),
            Err(ConfigError::UnsupportedVersion { version: 1, .. })
        ));
    }
}
