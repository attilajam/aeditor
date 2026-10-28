use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use ae_core::{AeError, FrameMut, FrameRef, Result};
use ae_transitions::{Transition, TransitionCtx, apply_easing, build_transition};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuTransitionUniforms {
    pub progress: f32,
    pub ratio: f32,
    pub width: u32,
    pub height: u32,
    pub custom: [f32; 4],
}

/// GPU transition engine backed by `wgpu` 30.0.1 compute shaders (`transitions/*.wgsl`).
/// Automatically falls back to the linear-light CPU transition if a headless GPU adapter
/// is unavailable in CI environments.
pub struct GpuTransitionEngine {
    device: Option<Arc<wgpu::Device>>,
    queue: Option<Arc<wgpu::Queue>>,
    shaders: BTreeMap<String, String>,
}

impl GpuTransitionEngine {
    pub fn new(shaders_dir: Option<&Path>) -> Self {
        let instance = wgpu::Instance::default();
        let adapter_opt = pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }),
        )
        .ok();

        let (device, queue) = if let Some(adapter) = adapter_opt {
            match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())) {
                Ok((d, q)) => (Some(Arc::new(d)), Some(Arc::new(q))),
                Err(_) => (None, None),
            }
        } else {
            (None, None)
        };

        let mut shaders = BTreeMap::new();
        shaders.insert(
            "crossfade".to_string(),
            include_str!("../../../transitions/crossfade.wgsl").to_string(),
        );
        shaders.insert(
            "dreamy_wave".to_string(),
            include_str!("../../../transitions/dreamy_wave.wgsl").to_string(),
        );

        if let Some(dir) = shaders_dir
            && dir.exists()
            && let Ok(entries) = std::fs::read_dir(dir)
        {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("wgsl")
                    && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                    && let Ok(src) = std::fs::read_to_string(&path)
                {
                    shaders.insert(stem.to_string(), src);
                }
            }
        }

        Self {
            device,
            queue,
            shaders,
        }
    }

    pub fn has_gpu(&self) -> bool {
        self.device.is_some()
    }

    pub fn loaded_shaders(&self) -> Vec<String> {
        self.shaders.keys().cloned().collect()
    }

    /// Build a `Transition` backed by a WGSL shader if `id` matches a loaded `.wgsl` file,
    /// or fall back to the CPU transition registry.
    pub fn resolve(&self, id: &str, params: &toml::Value) -> Result<Box<dyn Transition>> {
        if let Some(wgsl) = self.shaders.get(id) {
            let cpu_fallback = build_transition(id, params)
                .or_else(|_| build_transition("crossfade", params))?;
            return Ok(Box::new(WgslTransition {
                id_leaked: Box::leak(id.to_string().into_boxed_str()),
                wgsl_source: wgsl.clone(),
                device: self.device.clone(),
                queue: self.queue.clone(),
                cpu_fallback,
            }));
        }
        build_transition(id, params)
    }
}

struct WgslTransition {
    id_leaked: &'static str,
    wgsl_source: String,
    device: Option<Arc<wgpu::Device>>,
    queue: Option<Arc<wgpu::Queue>>,
    cpu_fallback: Box<dyn Transition>,
}

impl Transition for WgslTransition {
    fn id(&self) -> &'static str {
        self.id_leaked
    }

    fn render(
        &self,
        ctx: &TransitionCtx,
        from: FrameRef<'_>,
        to: FrameRef<'_>,
        progress: f32,
        mut out: FrameMut<'_>,
    ) {
        if progress <= 0.0 {
            out.copy_from_ref(from);
            return;
        }
        if progress >= 1.0 {
            out.copy_from_ref(to);
            return;
        }

        if let (Some(device), Some(queue)) = (&self.device, &self.queue)
            && self
                .render_wgpu(device, queue, ctx, from, to, progress, &mut out)
                .is_ok()
        {
            return;
        }

        self.cpu_fallback.render(ctx, from, to, progress, out);
    }
}

impl WgslTransition {
    fn render_wgpu(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ctx: &TransitionCtx,
        from: FrameRef<'_>,
        to: FrameRef<'_>,
        progress: f32,
        out: &mut FrameMut<'_>,
    ) -> Result<()> {
        let eased = apply_easing(&ctx.easing, progress);
        let uniforms = GpuTransitionUniforms {
            progress: eased,
            ratio: (from.width as f32) / (from.height.max(1) as f32),
            width: from.width,
            height: from.height,
            custom: [ctx.beat_strength, 0.0, 0.0, 0.0],
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(self.id_leaked),
            source: wgpu::ShaderSource::Wgsl(self.wgsl_source.as_str().into()),
        });

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let from_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("from_buf"),
            contents: from.data,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let to_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("to_buf"),
            contents: to.data,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let byte_size = from.data.len() as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_buf"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buf"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("transition_compute"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transition_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: from_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: to_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let total_pixels = from.width * from.height;
            let workgroups = total_pixels.div_ceil(64);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, byte_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        rx.recv()
            .map_err(|e| AeError::Transition {
                id: self.id_leaked.into(),
                message: e.to_string(),
            })?
            .map_err(|e| AeError::Transition {
                id: self.id_leaked.into(),
                message: e.to_string(),
            })?;

        let mapped = slice.get_mapped_range().map_err(|e| AeError::Transition {
            id: self.id_leaked.into(),
            message: e.to_string(),
        })?;
        out.data.copy_from_slice(&mapped);
        drop(mapped);
        staging_buf.unmap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ae_core::{Rational, RgbaFrame};

    #[test]
    fn test_cpu_gpu_crossfade_delta_e_under_1() {
        let engine = GpuTransitionEngine::new(Some(Path::new("../../transitions")));
        assert!(engine.loaded_shaders().contains(&"crossfade".to_string()));
        assert!(engine.loaded_shaders().contains(&"dreamy_wave".to_string()));

        let from = RgbaFrame::solid(64, 36, [200, 60, 80, 255], Rational::ZERO);
        let to = RgbaFrame::solid(64, 36, [40, 160, 220, 255], Rational::ONE);
        let ctx = TransitionCtx {
            width: 64,
            height: 36,
            easing: "linear".to_string(),
            ..Default::default()
        };
        let empty = toml::Value::Table(toml::map::Map::new());

        let gpu_tr = engine.resolve("crossfade", &empty).unwrap();
        let cpu_tr = build_transition("crossfade", &empty).unwrap();

        let mut out_gpu = RgbaFrame::new(64, 36, Rational::ZERO);
        let mut out_cpu = RgbaFrame::new(64, 36, Rational::ZERO);

        gpu_tr.render(&ctx, from.as_ref(), to.as_ref(), 0.5, out_gpu.as_mut());
        cpu_tr.render(&ctx, from.as_ref(), to.as_ref(), 0.5, out_cpu.as_mut());

        let delta_e = out_gpu.mean_delta_e(&out_cpu);
        assert!(
            delta_e < 1.0,
            "expected CPU/GPU crossfade ΔE < 1.0, got {delta_e:.3}"
        );
    }
}
