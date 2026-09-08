use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Controls how package files are discovered below a directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectoryScanOptions {
    /// Whether files in nested directories are included.
    pub recursive: bool,
    /// Whether symbolic links to files and directories are followed.
    pub follow_symlinks: bool,
}

impl DirectoryScanOptions {
    /// Creates directory scan options with recursive discovery enabled.
    pub const fn new() -> Self {
        Self {
            recursive: true,
            follow_symlinks: true,
        }
    }

    /// Sets whether symbolic links encountered during discovery are followed.
    pub const fn with_follow_symlinks(mut self, follow_symlinks: bool) -> Self {
        self.follow_symlinks = follow_symlinks;
        self
    }

    /// Sets whether files in nested directories are included.
    pub const fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }
}

impl Default for DirectoryScanOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Container format recognized while discovering title packages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackageFormat {
    /// Nintendo Submission Package backed by PFS0.
    Nsp,
    /// Compressed NSP variant with logical NCZ entries.
    Nsz,
    /// NX Card Image backed by nested HFS0 partitions.
    Xci,
    /// Compressed XCI variant with logical NCZ entries.
    Xcz,
}

pub(crate) struct DirectoryDiscoveryError {
    pub path: PathBuf,
    pub source: std::io::Error,
}

pub(crate) fn directory_files(
    path: &Path,
    options: DirectoryScanOptions,
) -> Result<Vec<PathBuf>, DirectoryDiscoveryError> {
    let mut directories = vec![path.to_owned()];
    let mut files = Vec::new();
    let mut visited = BTreeSet::new();

    while let Some(directory) = directories.pop() {
        if options.follow_symlinks {
            let identity =
                fs::canonicalize(&directory).map_err(|source| DirectoryDiscoveryError {
                    path: directory.clone(),
                    source,
                })?;
            if !visited.insert(identity) {
                continue;
            }
        }
        let entries = fs::read_dir(&directory).map_err(|source| DirectoryDiscoveryError {
            path: directory.clone(),
            source,
        })?;
        let mut nested_directories = Vec::new();

        for entry in entries {
            let entry = entry.map_err(|source| DirectoryDiscoveryError {
                path: directory.clone(),
                source,
            })?;
            let entry_path = entry.path();
            let file_type = entry
                .file_type()
                .map_err(|source| DirectoryDiscoveryError {
                    path: entry_path.clone(),
                    source,
                })?;
            let file_type = if options.follow_symlinks && file_type.is_symlink() {
                fs::metadata(&entry_path)
                    .map_err(|source| DirectoryDiscoveryError {
                        path: entry_path.clone(),
                        source,
                    })?
                    .file_type()
            } else {
                file_type
            };
            if file_type.is_file() {
                files.push(entry_path);
            } else if options.recursive && file_type.is_dir() {
                nested_directories.push(entry_path);
            }
        }

        // Reverse sorting makes the lexically first directory the next one
        // visited by the LIFO stack.
        nested_directories.sort_by(|left, right| right.cmp(left));
        directories.extend(nested_directories);
    }

    // A global sort defines discovery order independently of directory entry
    // enumeration and traversal order.
    files.sort();
    Ok(files)
}

pub(crate) fn package_format(path: &Path) -> Option<PackageFormat> {
    let extension = path.extension()?.to_str()?;
    if extension.eq_ignore_ascii_case("nsp") {
        Some(PackageFormat::Nsp)
    } else if extension.eq_ignore_ascii_case("nsz") {
        Some(PackageFormat::Nsz)
    } else if extension.eq_ignore_ascii_case("xci") {
        Some(PackageFormat::Xci)
    } else if extension.eq_ignore_ascii_case("xcz") {
        Some(PackageFormat::Xcz)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn recognizes_all_package_extensions_case_insensitively() {
        assert_eq!(package_format(Path::new("a.nsp")), Some(PackageFormat::Nsp));
        assert_eq!(package_format(Path::new("b.NSZ")), Some(PackageFormat::Nsz));
        assert_eq!(package_format(Path::new("c.Xci")), Some(PackageFormat::Xci));
        assert_eq!(package_format(Path::new("d.xCZ")), Some(PackageFormat::Xcz));
        assert_eq!(package_format(Path::new("standalone.ncz")), None);
    }
}
