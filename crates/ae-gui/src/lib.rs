pub mod app;
pub mod beat_editor_model;
pub mod image_provider;
pub mod project_controller;
pub mod shot_pool_model;
pub mod timeline_model;

pub use app::AutoEditorEguiApp;
pub use beat_editor_model::{BeatEditorModel, BeatViewItem};
pub use image_provider::{
    GuiMediaCache, get_gui_progress, global_media_cache, set_gui_progress,
};
pub use project_controller::{GuiStateSnapshot, ProjectController};
pub use shot_pool_model::{ShotPoolModel, ShotViewItem};
pub use timeline_model::{AlternativeShotItem, TimelineModel, TimelineSlotViewItem};

#[cfg(test)]
mod tests {
    use super::*;
    use ae_beats::generate_synthetic_bass_and_kick_wav;
    use ae_core::{Rational, Resolution};
    use ae_media::generate_synthetic_multi_shot_video;

    #[test]
    fn test_egui_controller_shot_selection_beat_dragging_and_export() {
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

        // 6. Export final video from the GUI controller
        ctrl.export_video(&out_path, false).unwrap();
        assert!(out_path.exists());
    }
}
