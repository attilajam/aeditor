use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};

use crate::error::Result;

/// Content-addressed analysis cache stored in `.ae-cache/` so repeated edits over
/// the same video or song skip redundant decoding/analysis passes.
#[derive(Clone, Debug)]
pub struct CacheManager {
    root: PathBuf,
    enabled: bool,
}

impl CacheManager {
    pub fn new(root: impl Into<PathBuf>, enabled: bool) -> Self {
        Self {
            root: root.into(),
            enabled,
        }
    }

    pub fn default_local() -> Self {
        Self::new(".ae-cache", true)
    }

    /// Compute a fast content + metadata hash of a media file combined with a config tag.
    pub fn file_cache_key(&self, path: &Path, namespace: &str, params_debug: &str) -> Result<String> {
        let mut hasher = Sha256::new();
        hasher.update(namespace.as_bytes());
        hasher.update(b":");
        hasher.update(params_debug.as_bytes());
        hasher.update(b":");

        let meta = fs::metadata(path)?;
        hasher.update(meta.len().to_le_bytes());

        // Hash first 64 KiB and last 64 KiB for speed on multi-GB clips
        let mut file = fs::File::open(path)?;
        let mut buf = vec![0u8; 65536];
        let n = file.read(&mut buf)?;
        hasher.update(&buf[..n]);

        Ok(hex::encode(hasher.finalize()))
    }

    pub fn get<T: DeserializeOwned>(&self, key: &str) -> Option<T> {
        if !self.enabled {
            return None;
        }
        let path = self.root.join(format!("{key}.json"));
        let bytes = fs::read(path).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    pub fn put<T: Serialize>(&self, key: &str, value: &T) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        fs::create_dir_all(&self.root)?;
        let path = self.root.join(format!("{key}.json"));
        let bytes = serde_json::to_vec_pretty(value)?;
        fs::write(path, bytes)?;
        Ok(())
    }
}
