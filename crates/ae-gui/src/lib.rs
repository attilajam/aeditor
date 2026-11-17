pub mod app;
pub mod beat_editor_model;
pub mod character_analysis;
pub mod character_edits;
pub mod character_model;
pub mod image_provider;
pub mod project_controller;
pub mod shot_pool_model;
pub mod timeline_model;

pub use app::AutoEditorEguiApp;
pub use beat_editor_model::{BeatEditorModel, BeatViewItem};
pub use character_model::{
    CharacterAnalysis, CharacterFilter, CharacterFilterMode, CharacterId, CharacterObservation,
    CharacterRecord, SceneCharacterTags, character_matches,
};
pub use image_provider::{GuiMediaCache, get_gui_progress, global_media_cache, set_gui_progress};
pub use project_controller::{CharacterViewItem, GuiStateSnapshot, ProjectController};
pub use shot_pool_model::{ShotPoolModel, ShotViewItem};
pub use timeline_model::{AlternativeShotItem, TimelineModel, TimelineSlotViewItem};

#[cfg(test)]
mod tests {
    use super::*;
    use ae_beats::generate_synthetic_bass_and_kick_wav;
    use ae_core::{Rational, Resolution};
    use ae_media::generate_synthetic_multi_shot_video;

    pub(super) static GUI_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn transition_style_import_validates_before_replacing_policy() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("style.toml");
        let mut controller = ProjectController::new();
        for content in [
            "strong_beat_transition = 'unknown'",
            "default_duration_ms = -1",
            "strong_beat_threshold = nan",
            "[invalid",
        ] {
            std::fs::write(&path, content).unwrap();
            assert!(controller.import_transition_style(&path).is_err());
            assert_eq!(
                controller.timeline_model.policy.strong_beat_transition,
                "cut"
            );
        }
        std::fs::write(
            &path,
            "strong_beat_transition = 'flash'\ndefault_duration_ms = 120",
        )
        .unwrap();
        controller.import_transition_style(&path).unwrap();
        assert_eq!(
            controller.timeline_model.policy.strong_beat_transition,
            "flash"
        );
        assert_eq!(controller.timeline_model.policy.default_duration_ms, 120);
    }

    #[test]
    fn test_sampled_import_refines_scene_edges_to_source_frames() {
        use ae_media::MediaBackend;

        let _guard = GUI_TEST_LOCK.lock().unwrap();
        for fps in [Rational::from_secs(30), Rational::new(30_000, 1_001)] {
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join("sampled_cuts.mp4");
            let cuts = generate_synthetic_multi_shot_video(
                &path,
                Resolution::new(64, 36),
                fps,
                &[
                    ([230, 30, 30], [230, 30, 30], 70),
                    ([30, 220, 40], [30, 220, 40], 82),
                    ([40, 60, 240], [40, 60, 240], 1700),
                ],
            )
            .unwrap();

            // Over 60 seconds selects 10 fps analysis; both cuts fall between sampled frames.
            let mut ctrl = ProjectController::new();
            ctrl.import_video(&path).unwrap();
            let shots = &ctrl.shot_pool.shots;
            assert_eq!(shots.len(), 3);
            for (i, cut) in cuts.iter().enumerate() {
                assert_eq!(shots[i].end, *cut);
                assert_eq!(shots[i].usable_end, *cut);
                assert_eq!(shots[i + 1].start, *cut);
                assert_eq!(shots[i + 1].usable_start, *cut);
            }

            let middle = &shots[1];
            let frames = ctrl
                .backend
                .decode_segment(
                    &path,
                    middle.usable_start,
                    middle.usable_duration(),
                    Resolution::new(64, 36),
                    fps,
                    1.0,
                    None,
                )
                .unwrap();
            assert_eq!(frames.len(), 82);
            for frame in &frames {
                let pixel = frame.pixel(32, 18);
                assert!(
                    pixel[1] > 180 && pixel[0] < 70 && pixel[2] < 80,
                    "neighboring shot leaked into frame at {}",
                    frame.pts
                );
            }
            let cache = global_media_cache().read().unwrap();
            for position in [0, 1, 2] {
                let thumbnail = cache.get_shot_thumbnail(1, position);
                assert!(thumbnail.pixel(thumbnail.width / 2, thumbnail.height / 2)[1] > 180);
            }
        }
    }

    #[test]
    fn test_egui_controller_shot_selection_beat_dragging_and_export() {
        let _guard = GUI_TEST_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let vid_path = tmp.path().join("gui_clip.mp4");
        let aud_path = tmp.path().join("gui_song.wav");
        let out_path = tmp.path().join("gui_export.mp4");

        let fps = Rational::from_secs(30);
        // 4 distinct shots (1s each)
        generate_synthetic_multi_shot_video(
            &vid_path,
            Resolution::new(320, 180),
            fps,
            &[
                ([230, 30, 30], [20, 20, 20], 30),
                ([30, 220, 40], [20, 20, 220], 30),
                ([40, 60, 240], [240, 220, 30], 30),
                ([230, 200, 30], [180, 20, 190], 30),
            ],
        )
        .unwrap();

        generate_synthetic_bass_and_kick_wav(
            &aud_path,
            2.2,
            &[(0.5, 65.41), (1.0, 82.41), (1.5, 73.42)],
            &[0.75],
        )
        .unwrap();

        let mut ctrl = ProjectController::new();
        ctrl.set_resolution_str("240x400").unwrap();
        ctrl.import_video(&vid_path).unwrap();
        ctrl.seed_character_detections_for_test();
        ctrl.import_audio(&aud_path).unwrap();

        let snap0 = ctrl.snapshot();
        assert_eq!(snap0.shots.len(), 4);
        assert_eq!(snap0.beats.iter().filter(|b| !b.is_ghost).count(), 3);
        assert_eq!(snap0.slots.len(), 3);

        // Verify HiDPI thumbnails (320x180) were generated into GuiMediaCache
        let cache = global_media_cache().read().unwrap();
        assert_eq!(cache.get_shot_thumbnail(0, 0).width, 320);
        assert!(cache.spectrogram.is_some());
        drop(cache);

        // 1. Exclude whatever shot is currently in slot 0 -> sequencer must pick a different enabled shot!
        let excluded_shot_id = snap0.slots[0].shot_id;
        ctrl.toggle_shot(excluded_shot_id).unwrap();
        let snap1 = ctrl.snapshot();
        assert!(
            snap1.slots.iter().all(|s| s.shot_id != excluded_shot_id),
            "excluded shot #{excluded_shot_id} must not appear in sequenced timeline!"
        );

        // 2. Drag beat #0 from 0.50s to 0.62s -> slot 0 end time must update to 0.62s and survive bass-band changes!
        ctrl.move_beat(0, 0.62).unwrap();
        ctrl.set_bass_band(35.0, 220.0).unwrap();
        let snap2 = ctrl.snapshot();
        assert!((snap2.beats[0].time_sec - 0.62).abs() < 1e-3);
        assert!(snap2.beats[0].is_manual);
        assert!((snap2.slots[0].dst_end_sec - 0.62).abs() < 1e-3);

        // 3. Test ranking alternative unused shots and swapping
        let alts = ctrl.rank_slot_alternatives(1, 3);
        assert!(!alts.is_empty());
        let chosen_alt = alts[0].shot_id;
        ctrl.swap_slot_shot(1, chosen_alt).unwrap();
        assert_eq!(ctrl.snapshot().slots[1].shot_id, chosen_alt);

        // 4. Test Instant RAM Preview frame resolution (0 ms wait, 320x180 unified single-pass RAM proxy)
        {
            let cache = global_media_cache().read().unwrap();
            assert!(
                !cache.proxy_reel.is_empty(),
                "proxy_reel must be populated on import"
            );
            let (preview_frame, resolved_slot, _resolved_shot) =
                cache.resolve_timeline_preview_frame(300); // 300ms -> slot 0
            assert_eq!(resolved_slot, 0);
            assert_eq!(preview_frame.width, 320);
            assert_eq!(preview_frame.height, 180);
        }

        // 5. Test Incremental Quick Render: first run encodes all 3 slots, second run after modifying slot 2 reuses 2 cached slots!
        let r1 = ctrl.quick_render_draft(3).unwrap();
        assert_eq!(r1.total_slots, 3);
        assert_eq!(r1.encoded_slots, 3);

        ctrl.set_slot_transition(2, "crossfade", 120).unwrap();
        let r2 = ctrl.quick_render_draft(3).unwrap();
        assert_eq!(
            r2.cached_slots, 2,
            "unchanged slots 0 and 1 must be cache hits"
        );
        assert_eq!(
            r2.encoded_slots, 1,
            "only modified slot 2 should be re-encoded"
        );

        // Each draft size must produce an actual file at the selected dimensions.
        for divisor in [1, 2, 3, 4] {
            use ae_media::MediaBackend;
            let draft = ctrl.quick_render_draft(divisor).unwrap();
            let expected = Resolution::new((240 / divisor).max(64), (400 / divisor).max(64));
            assert_eq!(draft.resolution, expected);
            let probe = ctrl.backend.probe_video(&draft.output_path).unwrap();
            assert_eq!(
                probe.oriented_dimensions(),
                (expected.width, expected.height)
            );
            let frame = crate::app::decode_draft_frame(
                &draft.output_path,
                300,
                ctrl.snapshot().timeline_duration_sec,
            )
            .unwrap();
            assert_eq!(
                (frame.width, frame.height),
                (expected.width, expected.height)
            );
        }

        // 6. Export final video from the GUI controller
        assert!(
            ctrl.export_video_with_quality(&out_path, false, 52, "fast")
                .is_err()
        );
        assert!(
            ctrl.export_video_with_quality(&out_path, false, 18, "invalid")
                .is_err()
        );
        ctrl.export_video_with_quality(&out_path, false, 35, "veryslow")
            .unwrap();
        let encoded = std::fs::read(&out_path).unwrap();
        let metadata = String::from_utf8_lossy(&encoded);
        assert!(
            metadata.contains("crf=35.0"),
            "selected compression did not reach x264"
        );
        assert!(
            metadata.contains("subme=10"),
            "selected veryslow preset did not reach x264"
        );

        // Resequencing a locked clip near its shot's end must not extend into the next scene.
        let shot_end = ctrl.shot_pool.shots[chosen_alt].usable_end;
        let slot = &mut ctrl.timeline_model.timeline.as_mut().unwrap().slots[1];
        slot.src_in = shot_end - Rational::from_millis(100);
        slot.src_out = shot_end;
        ctrl.resequence().unwrap();
        let restored = &ctrl.timeline_model.timeline.as_ref().unwrap().slots[1];
        assert_eq!(restored.shot_id, chosen_alt);
        assert!(
            restored.src_out <= shot_end,
            "locked clip crossed its scene boundary"
        );

        // 7. Round-robin reuse toggle round-trips through the controller snapshot and
        // never places the same clip in neighbouring slots.
        ctrl.set_allow_reuse(true).unwrap();
        assert!(ctrl.snapshot().allow_reuse);
        for w in ctrl.snapshot().slots.windows(2) {
            assert_ne!(
                w[0].shot_id, w[1].shot_id,
                "neighbouring slots must never share a clip"
            );
        }
        ctrl.set_allow_reuse(false).unwrap();
        assert!(!ctrl.snapshot().allow_reuse);
    }
}
