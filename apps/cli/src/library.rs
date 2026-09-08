use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nixe_config::NixeConfig;
use nixe_loader_content::{NacpLoader, NcaKeySet};
use nixe_loader_executable::NroLoader;
use nixe_loader_storage::{FileStorage, FormatLoader};
use nixe_loader_title::{
    ApplicationId, DirectoryScanOptions, ResolvedTitle, TitleCatalog, TitleResolver,
};
use sha2::{Digest, Sha256};

/// A title library resolved from all configured ROM directories.
pub struct Library {
    titles: Vec<LibraryTitle>,
    /// Keys loaded while scanning, ready to be reused when launching a title.
    pub keys: NcaKeySet,
}

impl Library {
    /// Scans all configured paths and resolves installed and homebrew titles.
    pub fn scan(config: &NixeConfig) -> Result<Self, String> {
        let scan_started = Instant::now();
        if config.library.paths.is_empty() {
            return Err("library.paths does not contain any directories".to_owned());
        }

        let prod_keys = config.system.keys.join("prod.keys");
        let title_keys_path = config.system.keys.join("title.keys");
        let title_keys = title_keys_path.is_file().then_some(title_keys_path);
        let keys_started = Instant::now();
        let mut keys = NcaKeySet::from_files(&prod_keys, title_keys.as_deref())
            .map_err(|error| error.to_string())?;
        log::debug!("key set loaded in {:?}", keys_started.elapsed());

        let mut catalog = TitleCatalog::new();
        let mut homebrew = BTreeMap::new();
        let mut seen_files = BTreeSet::new();
        for root in &config.library.paths {
            let root_started = Instant::now();
            let paths = directory_files(root, config.library.scan_options())?;
            log::debug!(
                "library directory {} enumerated with {} file(s) in {:?}",
                root.display(),
                paths.len(),
                root_started.elapsed()
            );
            for path in paths {
                if !is_package(&path) && !has_extension(&path, "nro") {
                    continue;
                }
                let identity = fs::canonicalize(&path)
                    .map_err(|error| format!("cannot resolve {}: {error}", path.display()))?;
                if !seen_files.insert(identity) {
                    continue;
                }

                if has_extension(&path, "nro") {
                    let package_started = Instant::now();
                    let title = load_homebrew(&path, config)?;
                    log::trace!(
                        "homebrew {} catalogued in {:?}",
                        path.display(),
                        package_started.elapsed()
                    );
                    homebrew.entry(title.identifier.clone()).or_insert(title);
                } else {
                    let package_started = Instant::now();
                    let discovered = TitleCatalog::load_package_with_key_set(&path, &mut keys)
                        .map_err(|error| error.to_string())?;
                    log::trace!(
                        "package {} catalogued in {:?}",
                        path.display(),
                        package_started.elapsed()
                    );
                    for package in discovered.packages() {
                        catalog.add(package.clone());
                    }
                }
            }
        }

        let resolution_started = Instant::now();
        let installed = TitleResolver::resolve_all(&catalog).map_err(|error| error.to_string())?;
        log::debug!(
            "installed title relationships resolved in {:?}",
            resolution_started.elapsed()
        );
        let mut titles = installed
            .into_iter()
            .map(|title| installed_title(title, config))
            .chain(homebrew.into_values())
            .collect::<Vec<_>>();
        titles.sort_by(|left, right| left.identifier.cmp(&right.identifier));

        log::debug!(
            "library scan produced {} title(s) in {:?}",
            titles.len(),
            scan_started.elapsed()
        );
        Ok(Self { titles, keys })
    }

    /// Returns every title in identifier order.
    pub fn titles(&self) -> &[LibraryTitle] {
        &self.titles
    }

    /// Finds a title by identifier first, then by its displayed name.
    pub fn find(&self, candidate: &str) -> Option<&LibraryTitle> {
        self.titles
            .iter()
            .find(|title| title.identifier.matches(candidate))
            .or_else(|| self.titles.iter().find(|title| title.name == candidate))
    }
}

/// Identifier shared by list output and title lookup.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LibraryTitleId {
    Application(ApplicationId),
    Homebrew(String),
}

impl LibraryTitleId {
    fn matches(&self, candidate: &str) -> bool {
        self.to_string().eq_ignore_ascii_case(candidate)
    }
}

impl Display for LibraryTitleId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Application(identifier) => Display::fmt(identifier, formatter),
            Self::Homebrew(identifier) => write!(formatter, "nro:{identifier}"),
        }
    }
}

/// One launchable title and the source required to run it.
pub struct LibraryTitle {
    pub identifier: LibraryTitleId,
    pub name: String,
    pub source: LibraryTitleSource,
}

pub enum LibraryTitleSource {
    Installed(Box<ResolvedTitle>),
    Homebrew(PathBuf),
}

fn installed_title(title: ResolvedTitle, config: &NixeConfig) -> LibraryTitle {
    let name = title
        .control_metadata()
        .and_then(|control| {
            control
                .nacp
                .preferred_title(&config.system.preferred_languages)
        })
        .map_or_else(
            || title.application_id.to_string(),
            |(_, title)| title.name.clone(),
        );
    LibraryTitle {
        identifier: LibraryTitleId::Application(title.application_id),
        name,
        source: LibraryTitleSource::Installed(Box::new(title)),
    }
}

fn load_homebrew(path: &Path, config: &NixeConfig) -> Result<LibraryTitle, String> {
    let storage = FileStorage::open(path)
        .map_err(|error| format!("cannot open NRO {}: {error}", path.display()))?;
    let image = NroLoader::load(Arc::new(storage))
        .map_err(|error| format!("cannot load NRO {}: {error}", path.display()))?;
    let identifier = homebrew_identifier(path, image.executable().module_id())?;
    let name = image
        .assets()
        .and_then(|assets| assets.nacp())
        .map(|storage| NacpLoader::load(storage.clone()))
        .transpose()
        .map_err(|error| format!("cannot read NRO NACP from {}: {error}", path.display()))?
        .and_then(|control| {
            control
                .preferred_title(&config.system.preferred_languages)
                .map(|(_, title)| title.name.clone())
        })
        .unwrap_or_else(|| path_fallback_name(path));
    Ok(LibraryTitle {
        identifier: LibraryTitleId::Homebrew(identifier),
        name,
        source: LibraryTitleSource::Homebrew(path.to_owned()),
    })
}

fn directory_files(root: &Path, options: DirectoryScanOptions) -> Result<Vec<PathBuf>, String> {
    let metadata =
        fs::metadata(root).map_err(|error| format!("cannot access {}: {error}", root.display()))?;
    if !metadata.is_dir() {
        return Err(format!(
            "library path is not a directory: {}",
            root.display()
        ));
    }

    let mut directories = vec![root.to_owned()];
    let mut files = Vec::new();
    let mut visited = BTreeSet::new();
    while let Some(directory) = directories.pop() {
        if options.follow_symlinks {
            let identity = fs::canonicalize(&directory)
                .map_err(|error| format!("cannot resolve {}: {error}", directory.display()))?;
            if !visited.insert(identity) {
                continue;
            }
        }
        let entries = fs::read_dir(&directory)
            .map_err(|error| format!("cannot read directory {}: {error}", directory.display()))?;
        let mut nested = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| {
                format!("cannot read directory {}: {error}", directory.display())
            })?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?;
            let file_type = if options.follow_symlinks && file_type.is_symlink() {
                fs::metadata(&path)
                    .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?
                    .file_type()
            } else {
                file_type
            };
            if file_type.is_file() {
                files.push(path);
            } else if options.recursive && file_type.is_dir() {
                nested.push(path);
            }
        }
        nested.sort_by(|left, right| right.cmp(left));
        directories.extend(nested);
    }
    files.sort();
    Ok(files)
}

fn is_package(path: &Path) -> bool {
    ["nsp", "nsz", "xci", "xcz"]
        .into_iter()
        .any(|extension| has_extension(path, extension))
}

fn has_extension(path: &Path, expected: &str) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case(expected))
}

fn homebrew_identifier(path: &Path, module_id: &[u8; 32]) -> Result<String, String> {
    let bytes = if let Some(length) = module_id.iter().rposition(|byte| *byte != 0) {
        module_id[..=length].to_vec()
    } else {
        let mut file = File::open(path)
            .map_err(|error| format!("cannot open NRO {} for hashing: {error}", path.display()))?;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let count = file
                .read(&mut buffer)
                .map_err(|error| format!("cannot hash NRO {}: {error}", path.display()))?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        hasher.finalize().to_vec()
    };
    Ok(format_hex(&bytes))
}

fn path_fallback_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("unknown-homebrew")
        .to_owned()
}

fn format_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789ABCDEF";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(char::from(DIGITS[usize::from(byte >> 4)]));
        result.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_library(titles: Vec<LibraryTitle>) -> Library {
        Library {
            titles,
            keys: NcaKeySet::from_text("", None).unwrap(),
        }
    }

    fn homebrew(identifier: &str, name: &str) -> LibraryTitle {
        LibraryTitle {
            identifier: LibraryTitleId::Homebrew(identifier.to_owned()),
            name: name.to_owned(),
            source: LibraryTitleSource::Homebrew(PathBuf::from(format!("{name}.nro"))),
        }
    }

    #[cfg(unix)]
    #[test]
    fn follows_symlinks_without_revisiting_directories() {
        use std::os::unix::fs::symlink;
        let root = std::env::temp_dir().join(format!(
            "nixe-symlinks-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("library")).unwrap();
        fs::create_dir(root.join("external")).unwrap();
        fs::write(root.join("external/game.nro"), []).unwrap();
        fs::write(root.join("standalone.nro"), []).unwrap();
        symlink(root.join("external"), root.join("library/linked")).unwrap();
        symlink(root.join("external"), root.join("library/duplicate")).unwrap();
        symlink(root.join("library"), root.join("external/cycle")).unwrap();
        symlink(root.join("standalone.nro"), root.join("library/file.nro")).unwrap();
        let scan = |options| {
            directory_files(&root.join("library"), options)
                .unwrap_or_else(|_| panic!("directory discovery failed"))
        };
        let files = scan(DirectoryScanOptions::new());
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|path| path.ends_with("game.nro")));
        assert!(files.iter().any(|path| path.ends_with("file.nro")));
        assert!(scan(DirectoryScanOptions::new().with_follow_symlinks(false)).is_empty());
        assert_eq!(
            scan(DirectoryScanOptions::new().with_recursive(false)),
            vec![root.join("library/file.nro")]
        );
        symlink(root.join("missing"), root.join("library/broken")).unwrap();
        assert!(directory_files(&root.join("library"), DirectoryScanOptions::new()).is_err());
        assert!(scan(DirectoryScanOptions::new().with_follow_symlinks(false)).is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn uses_the_module_id_as_the_homebrew_identifier() {
        let module_id = std::array::from_fn(|index| u8::try_from(index).unwrap());
        assert_eq!(
            homebrew_identifier(Path::new("unused.nro"), &module_id).unwrap(),
            "000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F"
        );
    }

    #[test]
    fn title_identifiers_match_case_insensitively() {
        let identifier = LibraryTitleId::Homebrew("A1B2C3".to_owned());
        assert!(identifier.matches("nro:a1b2c3"));
        assert!(!identifier.matches("a1b2c3"));
    }

    #[test]
    fn finds_titles_by_identifier_before_falling_back_to_name() {
        let library = test_library(vec![
            homebrew("FIRST", "nro:SECOND"),
            homebrew("SECOND", "Friendly Name"),
        ]);

        assert_eq!(
            library.find("nro:SECOND").unwrap().name,
            "Friendly Name",
            "an identifier match must win over an earlier name match"
        );
        assert_eq!(library.find("nro:first").unwrap().name, "nro:SECOND");
        assert_eq!(
            library
                .find("Friendly Name")
                .unwrap()
                .identifier
                .to_string(),
            "nro:SECOND"
        );
        assert!(library.find("friendly name").is_none());
    }

    #[test]
    fn recognizes_supported_files_case_insensitively() {
        assert!(is_package(Path::new("game.XCI")));
        assert!(is_package(Path::new("game.nSp")));
        assert!(has_extension(Path::new("homebrew.NRO"), "nro"));
        assert!(!is_package(Path::new("ignored.nro")));
    }
}
