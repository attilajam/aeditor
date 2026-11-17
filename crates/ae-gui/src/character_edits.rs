use std::path::Path;

use ae_core::{AeError, CacheManager, Rational, Result};
use serde::{Deserialize, Serialize};

use crate::character_model::{CharacterAnalysis, CharacterRecord, SceneCharacterTags};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CharacterEdits {
    pub analysis_key: String,
    pub operations: Vec<CharacterEdit>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CharacterEdit {
    Rename {
        id: u64,
        label: String,
    },
    Merge {
        from: u64,
        into: u64,
    },
    Assign {
        observation_id: u64,
        character_id: Option<u64>,
    },
    Ignore {
        observation_id: u64,
    },
    Interval {
        start: Rational,
        end: Rational,
        present: Vec<u64>,
        reviewed: bool,
    },
    Portrait {
        id: u64,
        observation_id: u64,
    },
}

fn invalid() -> AeError {
    AeError::Config("invalid character edit".into())
}

impl CharacterEdits {
    pub fn apply(&self, analysis: &mut CharacterAnalysis) -> Result<()> {
        analysis.validate()?;
        let mut a = analysis.clone();
        for edit in &self.operations {
            match edit {
                CharacterEdit::Rename { id, label } => {
                    a.records
                        .iter_mut()
                        .find(|r| r.id == *id)
                        .ok_or_else(invalid)?
                        .label = label.clone()
                }
                CharacterEdit::Merge { from, into } => {
                    if from == into || !a.records.iter().any(|r| r.id == *into) {
                        return Err(invalid());
                    }
                    let index = a
                        .records
                        .iter()
                        .position(|r| r.id == *from)
                        .ok_or_else(invalid)?;
                    let source = a.records.remove(index);
                    let target = a.records.iter_mut().find(|r| r.id == *into).unwrap();
                    for id in source.exemplars {
                        if !target.exemplars.contains(&id) && target.exemplars.len() < 8 {
                            target.exemplars.push(id);
                        }
                    }
                    for o in &mut a.observations {
                        if o.character_id == Some(*from) {
                            o.character_id = Some(*into);
                        }
                        for id in &mut o.possible {
                            if *id == *from {
                                *id = *into;
                            }
                        }
                        o.possible.sort_unstable();
                        o.possible.dedup();
                    }
                }
                CharacterEdit::Assign {
                    observation_id,
                    character_id,
                } => {
                    let o = a
                        .observations
                        .iter_mut()
                        .find(|o| o.id == *observation_id)
                        .ok_or_else(invalid)?;
                    if let Some(id) = character_id {
                        if !a.records.iter().any(|r| r.id == *id) {
                            a.records.push(CharacterRecord {
                                id: *id,
                                label: String::new(),
                                exemplars: vec![],
                                provisional: true,
                            });
                        }
                    }
                    o.character_id = *character_id;
                    o.possible.clear();
                    for r in &mut a.records {
                        r.exemplars.retain(|id| id != observation_id);
                    }
                    if let Some(id) = character_id {
                        let r = a.records.iter_mut().find(|r| r.id == *id).unwrap();
                        if r.exemplars.len() < 8 {
                            r.exemplars.push(*observation_id);
                        }
                    }
                }
                CharacterEdit::Ignore { observation_id } => {
                    let index = a
                        .observations
                        .iter()
                        .position(|o| o.id == *observation_id)
                        .ok_or_else(invalid)?;
                    a.observations.remove(index);
                    for r in &mut a.records {
                        r.exemplars.retain(|id| id != observation_id);
                    }
                }
                CharacterEdit::Interval {
                    start,
                    end,
                    present,
                    ..
                } => {
                    if start.denom() <= 0
                        || end.denom() <= 0
                        || start.is_negative()
                        || start >= end
                        || present
                            .iter()
                            .any(|id| !a.records.iter().any(|r| r.id == *id))
                    {
                        return Err(invalid());
                    }
                }
                CharacterEdit::Portrait { id, observation_id } => {
                    if !a
                        .observations
                        .iter()
                        .any(|o| o.id == *observation_id && o.character_id == Some(*id))
                    {
                        return Err(invalid());
                    }
                    let r = a
                        .records
                        .iter_mut()
                        .find(|r| r.id == *id)
                        .ok_or_else(invalid)?;
                    r.exemplars.retain(|x| x != observation_id);
                    r.exemplars.insert(0, *observation_id);
                    r.exemplars.truncate(8);
                }
            }
        }
        a.validate()?;
        *analysis = a;
        Ok(())
    }

    pub fn tags_for(
        &self,
        analysis: &CharacterAnalysis,
        start: Rational,
        end: Rational,
    ) -> SceneCharacterTags {
        let mut tags = analysis.tags_for(start, end);
        for edit in &self.operations {
            if let CharacterEdit::Merge { from, into } = edit {
                for id in tags.present.iter_mut().chain(tags.possible.iter_mut()) {
                    if id == from {
                        *id = *into;
                    }
                }
                tags.present.sort_unstable();
                tags.present.dedup();
                tags.possible.sort_unstable();
                tags.possible.dedup();
            }
            if let CharacterEdit::Interval {
                start: a,
                end: b,
                present,
                reviewed,
            } = edit
            {
                if *a == start && *b == end {
                    tags.present = present.clone();
                    tags.present.sort_unstable();
                    tags.present.dedup();
                    if *reviewed {
                        tags.possible.clear();
                        tags.complete = true;
                        tags.unresolved = false;
                    }
                } else if *a < end && start < *b {
                    tags.complete = false;
                    tags.unresolved = true;
                }
            }
        }
        tags
    }
}

fn check_key(key: &str) -> Result<()> {
    if key.len() != 64
        || !key
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(invalid());
    }
    Ok(())
}

pub fn load_edits(root: &Path, source_key: &str) -> Result<Option<CharacterEdits>> {
    check_key(source_key)?;
    match std::fs::read(root.join(format!("{source_key}.json"))) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub fn save_edits(root: &Path, source_key: &str, edits: &CharacterEdits) -> Result<()> {
    load_edits(root, source_key)?;
    CacheManager::new(root, true).put(source_key, edits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::character_model::CharacterObservation;

    fn analysis() -> CharacterAnalysis {
        CharacterAnalysis {
            records: vec![CharacterRecord {
                id: 1,
                label: String::new(),
                exemplars: vec![1, 2],
                provisional: false,
            }],
            observations: (1..=2)
                .map(|id| CharacterObservation {
                    id,
                    pts: Rational::from_secs(id as i64),
                    bounds: [0.0, 0.0, 0.5, 0.5],
                    score: 1.0,
                    feature: vec![0.0],
                    character_id: Some(1),
                    possible: vec![],
                })
                .collect(),
            scanned_ranges: vec![],
            sampled_pts: vec![],
        }
    }

    #[test]
    fn split_merge_and_transaction() {
        let mut a = analysis();
        let mut edits = CharacterEdits::default();
        edits.operations.push(CharacterEdit::Assign {
            observation_id: 2,
            character_id: Some(2),
        });
        edits.apply(&mut a).unwrap();
        assert_eq!(a.records[1].exemplars, vec![2]);
        edits.operations = vec![CharacterEdit::Merge { from: 2, into: 1 }];
        a.observations[0].possible = vec![2, 1];
        edits.apply(&mut a).unwrap();
        assert_eq!(a.observations[0].possible, vec![1]);
        let before = a.clone();
        edits.operations = vec![
            CharacterEdit::Ignore { observation_id: 1 },
            CharacterEdit::Rename {
                id: 99,
                label: String::new(),
            },
        ];
        assert!(edits.apply(&mut a).is_err());
        assert_eq!(a, before);
    }

    #[test]
    fn storage_and_interval() {
        let dir = tempfile::tempdir().unwrap();
        let key = "a".repeat(64);
        let mut edits = CharacterEdits::default();
        edits.operations.push(CharacterEdit::Interval {
            start: Rational::from_secs(0),
            end: Rational::from_secs(4),
            present: vec![1],
            reviewed: true,
        });
        save_edits(dir.path(), &key, &edits).unwrap();
        assert_eq!(load_edits(dir.path(), &key).unwrap(), Some(edits.clone()));
        assert!(
            edits
                .tags_for(&analysis(), Rational::from_secs(0), Rational::from_secs(4))
                .complete
        );
        let tags = edits.tags_for(&analysis(), Rational::from_secs(0), Rational::from_secs(2));
        assert!(!tags.complete && tags.unresolved);
        let path = dir.path().join(format!("{key}.json"));
        std::fs::write(&path, b"broken").unwrap();
        assert!(save_edits(dir.path(), &key, &edits).is_err());
        assert_eq!(std::fs::read(path).unwrap(), b"broken");
    }

    #[test]
    fn founder_can_split_and_reviewed_tags_follow_merges() {
        let mut source = analysis();
        let mut edits = CharacterEdits::default();
        edits.operations.push(CharacterEdit::Assign {
            observation_id: 1,
            character_id: Some(3),
        });
        edits.operations.push(CharacterEdit::Interval {
            start: Rational::ZERO,
            end: Rational::from_secs(4),
            present: vec![1],
            reviewed: true,
        });
        edits
            .operations
            .push(CharacterEdit::Merge { from: 1, into: 3 });
        edits.apply(&mut source).unwrap();
        assert_eq!(source.records.len(), 1);
        assert_eq!(source.records[0].id, 3);
        assert_eq!(
            edits
                .tags_for(&source, Rational::ZERO, Rational::from_secs(4))
                .present,
            vec![3]
        );
    }
}
