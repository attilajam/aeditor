use serde::{Deserialize, Serialize};

pub type CharacterId = u64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CharacterFilterMode {
    #[default]
    Neutral,
    With,
    Only,
    Without,
}

impl CharacterFilterMode {
    pub fn next(self) -> Self {
        match self {
            Self::Neutral => Self::With,
            Self::With => Self::Only,
            Self::Only => Self::Without,
            Self::Without => Self::Neutral,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CharacterFilter {
    pub character_id: CharacterId,
    pub mode: CharacterFilterMode,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneCharacterTags {
    pub present: Vec<CharacterId>,
    pub possible: Vec<CharacterId>,
    pub unresolved: bool,
    pub complete: bool,
    pub failed: bool,
}

use ae_core::{AeError, Rational, Result};
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CharacterObservation {
    pub id: u64,
    pub pts: Rational,
    pub bounds: [f32; 4],
    pub score: f32,
    pub feature: Vec<f32>,
    pub character_id: Option<u64>,
    pub possible: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CharacterRecord {
    pub id: u64,
    pub label: String,
    pub exemplars: Vec<u64>,
    pub provisional: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CharacterAnalysis {
    pub records: Vec<CharacterRecord>,
    pub observations: Vec<CharacterObservation>,
    pub scanned_ranges: Vec<(Rational, Rational)>,
    #[serde(default)]
    pub sampled_pts: Vec<Rational>,
}

impl CharacterAnalysis {
    pub fn validate(&self) -> Result<()> {
        let invalid = || AeError::Config("invalid character analysis".into());
        let mut records = HashSet::new();
        let mut observations = HashSet::new();
        let feature_len = self.observations.first().map(|o| o.feature.len());
        for r in &self.records {
            if !records.insert(r.id) || r.exemplars.len() > 8 {
                return Err(invalid());
            }
        }
        for o in &self.observations {
            let [x, y, w, h] = o.bounds;
            if !observations.insert(o.id)
                || o.pts.denom() <= 0
                || o.pts.is_negative()
                || !o.bounds.iter().all(|v| v.is_finite())
                || x < 0.0
                || y < 0.0
                || w <= 0.0
                || h <= 0.0
                || x + w > 1.0 + 1e-6
                || y + h > 1.0 + 1e-6
                || !o.score.is_finite()
                || !(0.0..=1.0).contains(&o.score)
                || o.feature.is_empty()
                || Some(o.feature.len()) != feature_len
                || !o.feature.iter().all(|v| v.is_finite())
                || o.character_id.is_some_and(|id| !records.contains(&id))
                || o.possible.iter().any(|id| !records.contains(id))
            {
                return Err(invalid());
            }
        }
        for r in &self.records {
            let mut seen = HashSet::new();
            if r.exemplars.iter().any(|id| {
                !seen.insert(id)
                    || !self
                        .observations
                        .iter()
                        .any(|o| o.id == *id && o.character_id == Some(r.id))
            }) {
                return Err(invalid());
            }
        }
        if self
            .scanned_ranges
            .iter()
            .any(|(a, b)| a.denom() <= 0 || b.denom() <= 0 || a.is_negative() || a >= b)
        {
            return Err(invalid());
        }
        if self
            .sampled_pts
            .iter()
            .any(|pts| pts.denom() <= 0 || pts.is_negative())
        {
            return Err(invalid());
        }
        Ok(())
    }

    pub fn tags_for(&self, start: Rational, end: Rational) -> SceneCharacterTags {
        let mut tags = SceneCharacterTags::default();
        if start >= end {
            return tags;
        }
        for o in self
            .observations
            .iter()
            .filter(|o| o.pts >= start && o.pts < end)
        {
            if let Some(id) = o.character_id {
                tags.present.push(id);
            } else {
                tags.unresolved = true;
            }
            tags.possible.extend(&o.possible);
        }
        tags.present.sort_unstable();
        tags.present.dedup();
        tags.possible.sort_unstable();
        tags.possible.dedup();
        let mut ranges = self.scanned_ranges.clone();
        ranges.sort_unstable();
        let mut covered = start;
        for (a, b) in ranges {
            if a > covered {
                break;
            }
            if b > covered {
                covered = b;
            }
        }
        tags.complete = covered >= end
            && (self.sampled_pts.is_empty()
                || self
                    .sampled_pts
                    .iter()
                    .any(|pts| *pts >= start && *pts < end));
        tags
    }

    /// Rebuild groups chronologically. Scores of at least 0.8 may seed a group.
    pub fn assign_groups<F>(&mut self, mut distance: F, threshold: f32, margin: f32) -> Result<()>
    where
        F: FnMut(&[f32], &[f32]) -> Result<f32>,
    {
        self.validate()?;
        if !threshold.is_finite() || threshold < 0.0 || !margin.is_finite() || margin < 0.0 {
            return Err(AeError::Config("invalid grouping threshold".into()));
        }
        self.records.clear();
        for o in &mut self.observations {
            o.character_id = None;
            o.possible.clear();
        }
        let mut order: Vec<usize> = (0..self.observations.len()).collect();
        order.sort_by_key(|&i| (self.observations[i].pts, self.observations[i].id));
        for i in order {
            if self.observations[i].score < 0.5 {
                continue;
            }
            let mut candidates = Vec::new();
            for r in &self.records {
                let mut best = f32::INFINITY;
                for (index, id) in r.exemplars.iter().enumerate() {
                    let exemplar = self.observations.iter().find(|o| o.id == *id).unwrap();
                    let value = distance(&self.observations[i].feature, &exemplar.feature)?;
                    if !value.is_finite() || value < 0.0 {
                        return Err(AeError::Config("invalid character distance".into()));
                    }
                    // Keep the founding exemplar as an anchor against identity drift.
                    best = best.min(value);
                    if index == 0 && value > threshold {
                        break;
                    }
                }
                candidates.push((best, r.id));
            }
            candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            if let Some(&(best, id)) = candidates.first().filter(|(best, _)| *best <= threshold) {
                if candidates.get(1).is_some_and(|next| next.0 - best < margin) {
                    self.observations[i].possible = candidates
                        .iter()
                        .filter(|c| c.0 - best < margin || c.0 <= threshold)
                        .map(|c| c.1)
                        .collect();
                } else {
                    self.observations[i].character_id = Some(id);
                    let r = self.records.iter_mut().find(|r| r.id == id).unwrap();
                    if r.exemplars.iter().any(|eid| {
                        self.observations
                            .iter()
                            .any(|o| o.id == *eid && o.pts != self.observations[i].pts)
                    }) {
                        r.provisional = false;
                    }
                    if r.exemplars.len() < 8 {
                        r.exemplars.push(self.observations[i].id);
                    }
                }
            } else if self.observations[i].score >= 0.8 {
                let id = self.observations[i].id;
                self.observations[i].character_id = Some(id);
                self.records.push(CharacterRecord {
                    id,
                    label: String::new(),
                    exemplars: vec![id],
                    provisional: true,
                });
            }
        }
        Ok(())
    }
}

pub fn character_matches(tags: &SceneCharacterTags, filters: &[CharacterFilter]) -> bool {
    let mut active = filters
        .iter()
        .filter(|f| f.mode != CharacterFilterMode::Neutral);
    if active.clone().next().is_none() {
        return tags.complete && !tags.failed && (!tags.present.is_empty() || tags.unresolved);
    }
    if tags.failed || !tags.complete {
        return false;
    }
    active.all(|filter| {
        let id = filter.character_id;
        match filter.mode {
            CharacterFilterMode::Neutral => true,
            CharacterFilterMode::With => tags.present.contains(&id),
            CharacterFilterMode::Only => {
                tags.present.contains(&id)
                    && tags.present.iter().all(|other| *other == id)
                    && tags.possible.iter().all(|other| *other == id)
                    && !tags.unresolved
            }
            CharacterFilterMode::Without => {
                !tags.present.contains(&id) && !tags.possible.contains(&id) && !tags.unresolved
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(id: u64, t: i64, feature: f32) -> CharacterObservation {
        CharacterObservation {
            id,
            pts: Rational::from_secs(t),
            bounds: [0.0, 0.0, 0.5, 0.5],
            score: 1.0,
            feature: vec![feature],
            character_id: None,
            possible: vec![],
        }
    }

    #[test]
    fn projection_and_validation() {
        let mut a = CharacterAnalysis {
            observations: vec![observation(1, 1, 0.0), observation(2, 2, 1.0)],
            scanned_ranges: vec![
                (Rational::from_secs(0), Rational::from_secs(1)),
                (Rational::from_secs(2), Rational::from_secs(3)),
            ],
            ..Default::default()
        };
        let tags = a.tags_for(Rational::from_secs(0), Rational::from_secs(2));
        assert!(tags.unresolved && !tags.complete);
        assert!(
            !a.tags_for(Rational::from_secs(0), Rational::from_secs(1))
                .unresolved
        );
        a.scanned_ranges
            .push((Rational::from_secs(1), Rational::from_secs(2)));
        assert!(
            a.tags_for(Rational::from_secs(0), Rational::from_secs(3))
                .complete
        );
        assert!(a.validate().is_ok());
        a.observations[0].character_id = Some(99);
        assert!(a.validate().is_err());
        a.observations[0].character_id = None;
        a.observations[0].feature[0] = f32::NAN;
        assert!(a.validate().is_err());
    }

    #[test]
    fn grouping_is_deterministic_and_preserves_ambiguity() {
        let mut a = CharacterAnalysis {
            observations: vec![
                observation(3, 3, 0.5),
                observation(2, 2, 1.0),
                observation(1, 1, 0.0),
                observation(4, 4, 0.05),
            ],
            ..Default::default()
        };
        a.assign_groups(|a, b| Ok((a[0] - b[0]).abs()), 0.6, 0.1)
            .unwrap();
        assert_eq!(a.observations[0].possible, vec![1, 2]);
        assert_eq!(a.observations[3].character_id, Some(1));
        assert!(!a.records[0].provisional);
        let first = a.clone();
        a.assign_groups(|a, b| Ok((a[0] - b[0]).abs()), 0.6, 0.1)
            .unwrap();
        assert_eq!(a, first);
    }

    #[test]
    fn grouping_rejects_failed_metrics_and_identity_drift() {
        let mut analysis = CharacterAnalysis {
            observations: vec![
                observation(1, 0, 0.0),
                observation(2, 1, 0.4),
                observation(3, 2, 0.8),
            ],
            ..Default::default()
        };
        assert!(
            analysis
                .assign_groups(|_, _| Ok(f32::NAN), 0.5, 0.1)
                .is_err()
        );
        analysis
            .assign_groups(|a, b| Ok((a[0] - b[0]).abs()), 0.5, 0.1)
            .unwrap();
        assert_eq!(analysis.observations[1].character_id, Some(1));
        assert_eq!(analysis.observations[2].character_id, Some(3));
        analysis.observations[2].feature.push(0.0);
        assert!(analysis.validate().is_err());
    }

    #[test]
    fn character_filter_truth_table() {
        use CharacterFilterMode::*;
        let filter = |mode| {
            [CharacterFilter {
                character_id: 1,
                mode,
            }]
        };
        let mut mode = Neutral;
        for expected in [With, Only, Without, Neutral] {
            mode = mode.next();
            assert_eq!(mode, expected);
        }
        let mut tags = SceneCharacterTags {
            present: vec![1],
            complete: true,
            ..Default::default()
        };
        assert!(character_matches(&tags, &filter(Neutral)));
        assert!(character_matches(&tags, &filter(With)));
        assert!(character_matches(&tags, &filter(Only)));
        assert!(!character_matches(&tags, &filter(Without)));
        tags.possible.push(1);
        assert!(character_matches(&tags, &filter(Only)));
        tags.possible.push(2);
        assert!(!character_matches(&tags, &filter(Only)));
        tags.present.clear();
        assert!(!character_matches(&tags, &filter(Without)));
        tags.possible.clear();
        assert!(character_matches(&tags, &filter(Without)));
        tags.unresolved = true;
        assert!(!character_matches(&tags, &filter(Without)));
        assert!(character_matches(&tags, &filter(Neutral)));
        tags.complete = false;
        assert!(!character_matches(&tags, &filter(Neutral)));
        assert!(!character_matches(&tags, &filter(Without)));
        tags.failed = true;
        assert!(!character_matches(&tags, &filter(Neutral)));
        tags.complete = true;
        assert!(!character_matches(&tags, &filter(Without)));
        tags.failed = false;
        tags.unresolved = false;
        tags.present.push(1);
        assert!(!character_matches(
            &tags,
            &[filter(With)[0], filter(Without)[0]]
        ));
    }
}
