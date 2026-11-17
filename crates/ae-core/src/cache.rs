use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_and_atomic_replacement() {
        let root = std::env::temp_dir().join(format!("ae-cache-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let path = root.join("source");
        fs::write(&path, vec![0u8; 200_000]).unwrap();
        let cache = CacheManager::new(root.join("cache"), true);
        let key = cache.file_cache_key(&path, "test", "one").unwrap();
        let modified = fs::metadata(&path).unwrap().modified().unwrap();
        assert_ne!(key, cache.file_cache_key(&path, "test", "two").unwrap());
        let mut file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::End(-1)).unwrap();
        file.write_all(&[1]).unwrap();
        file.set_times(fs::FileTimes::new().set_modified(modified))
            .unwrap();
        assert_ne!(key, cache.file_cache_key(&path, "test", "one").unwrap());
        file.seek(SeekFrom::End(-1)).unwrap();
        file.write_all(&[0]).unwrap();
        file.set_times(fs::FileTimes::new().set_modified(modified))
            .unwrap();
        assert_eq!(key, cache.file_cache_key(&path, "test", "one").unwrap());
        file.seek(SeekFrom::Start(100_000)).unwrap();
        file.write_all(&[1]).unwrap();
        file.set_times(
            fs::FileTimes::new().set_modified(modified + std::time::Duration::from_secs(1)),
        )
        .unwrap();
        assert_ne!(key, cache.file_cache_key(&path, "test", "one").unwrap());
        cache.put(&key, &vec![1, 2]).unwrap();
        cache.put(&key, &vec![3]).unwrap();
        assert_eq!(cache.get::<Vec<i32>>(&key), Some(vec![3]));
        fs::remove_dir_all(root).unwrap();
    }
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
    pub fn file_cache_key(
        &self,
        path: &Path,
        namespace: &str,
        params_debug: &str,
    ) -> Result<String> {
        let mut hasher = Sha256::new();
        hasher.update(namespace.as_bytes());
        hasher.update(b":");
        hasher.update(params_debug.as_bytes());
        hasher.update(b":");

        let canonical = fs::canonicalize(path)?;
        let meta = fs::metadata(&canonical)?;
        hasher.update(canonical.as_os_str().as_encoded_bytes());
        hasher.update(meta.len().to_le_bytes());
        let modified = meta.modified()?;
        let (before_epoch, duration) = match modified.duration_since(std::time::UNIX_EPOCH) {
            Ok(duration) => (false, duration),
            Err(error) => (true, error.duration()),
        };
        hasher.update([before_epoch as u8]);
        hasher.update(duration.as_secs().to_le_bytes());
        hasher.update(duration.subsec_nanos().to_le_bytes());

        // Hash first 64 KiB and last 64 KiB for speed on multi-GB clips
        // shortcut: timestamp-preserved middle edits can escape detection, use a full hash when adversarial changes matter.
        let mut file = fs::File::open(&canonical)?;
        let mut buf = vec![0u8; 65536];
        let n = (meta.len().min(buf.len() as u64)) as usize;
        file.read_exact(&mut buf[..n])?;
        hasher.update(&buf[..n]);
        if meta.len() > buf.len() as u64 {
            file.seek(SeekFrom::End(-(buf.len() as i64)))?;
            file.read_exact(&mut buf)?;
            hasher.update(&buf);
        }

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
        static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
        let (temp_path, mut file) = loop {
            let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let temp_path = self
                .root
                .join(format!(".{key}.{}.{}.tmp", std::process::id(), id));
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)
            {
                Ok(file) => break (temp_path, file),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        };
        let result = file.write_all(&bytes).and_then(|()| {
            drop(file);
            fs::rename(&temp_path, path)
        });
        if result.is_err() {
            let _ = fs::remove_file(temp_path);
        }
        result?;
        Ok(())
    }
}
