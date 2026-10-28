use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{AeError, Result};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub filename: String,
    pub url: String,
    pub sha256: String,
    pub license: String,
    pub description: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelManifest {
    pub models: BTreeMap<String, ModelEntry>,
}

impl ModelManifest {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let manifest: Self = toml::from_str(&text)?;
        Ok(manifest)
    }

    pub fn verify_file(&self, key: &str, models_dir: &Path) -> Result<PathBuf> {
        let entry = self.models.get(key).ok_or_else(|| AeError::Model {
            model: key.to_string(),
            message: "model key not found in manifest.toml".to_string(),
        })?;
        let file_path = models_dir.join(&entry.filename);
        if !file_path.exists() {
            return Err(AeError::Model {
                model: key.to_string(),
                message: format!(
                    "model file '{}' not found in {}",
                    entry.filename,
                    models_dir.display()
                ),
            });
        }
        let bytes = std::fs::read(&file_path)?;
        let digest = hex::encode(Sha256::digest(&bytes));
        if !entry.sha256.is_empty() && entry.sha256 != "skip" && digest != entry.sha256 {
            return Err(AeError::Model {
                model: key.to_string(),
                message: format!(
                    "SHA-256 mismatch for {}: expected {}, got {}",
                    file_path.display(),
                    entry.sha256,
                    digest
                ),
            });
        }
        Ok(file_path)
    }
}
