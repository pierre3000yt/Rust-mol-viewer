//! VR Renderer for Stereo Rendering
//!
//! This module handles VR-specific rendering with dual render passes (left and right eye),
//! integrating OpenXR session management with the main wgpu renderer.

use crate::camera::{Camera, Eye};
use anyhow::{Context, Result};
use glam::{Mat4, Quat, Vec3, Vec4};
use log::{info, warn};
use mol_vr::{openxr, VrPerformanceMonitor, VrSession};
use wgpu;

/// VR renderer with stereo support
pub struct VrRenderer {
    pub vr_session: VrSession,

    /// The wgpu texture format used by the VR swapchain (e.g. Rgba8UnormSrgb).
    /// Render pipelines used in VR passes must target this format.
    pub swapchain_format: wgpu::TextureFormat,

    // Depth textures for each eye (stored to keep alive, accessed via views)
    _left_depth_texture: wgpu::Texture,
    pub left_depth_view: wgpu::TextureView,
    _right_depth_texture: wgpu::Texture,
    pub right_depth_view: wgpu::TextureView,

    // Camera buffers for each eye
    pub left_camera_buffer: wgpu::Buffer,
    pub right_camera_buffer: wgpu::Buffer,
    pub left_camera_bind_group: wgpu::BindGroup,
    pub right_camera_bind_group: wgpu::BindGroup,

    // Performance monitoring
    performance_monitor: VrPerformanceMonitor,
    last_warning_frame: usize,
}

impl VrRenderer {
    /// Create a VR renderer from a pre-created `VrSession`.
    ///
    /// The session is created in `Renderer::new()` (together with the Vulkan context that
    /// wgpu is then initialized from), so we receive it here already fully constructed.
    pub fn new(
        mut vr_session: VrSession,
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<Self> {
        info!("Initializing VR renderer...");

        // Create swapchains — format is auto-selected from OpenXR's supported list
        let swapchain_format = vr_session
            .create_swapchains(device)
            .context("Failed to create VR swapchains")?;
        info!("VR swapchains created with format: {:?}", swapchain_format);

        // Get swapchain resolution
        let (left_width, left_height) = vr_session
            .left_swapchain
            .as_ref()
            .map(|s| s.resolution)
            .unwrap_or((2048, 2048));

        let (right_width, right_height) = vr_session
            .right_swapchain
            .as_ref()
            .map(|s| s.resolution)
            .unwrap_or((2048, 2048));

        info!(
            "VR resolution — Left: {}x{}, Right: {}x{}",
            left_width, left_height, right_width, right_height
        );

        // Create depth textures for each eye
        let left_depth_texture =
            Self::create_depth_texture(device, left_width, left_height, "VR Left Depth");
        let left_depth_view =
            left_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let right_depth_texture =
            Self::create_depth_texture(device, right_width, right_height, "VR Right Depth");
        let right_depth_view =
            right_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create camera uniform buffers for each eye
        let left_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VR Left Camera Buffer"),
            size: std::mem::size_of::<crate::camera::CameraUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let right_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VR Right Camera Buffer"),
            size: std::mem::size_of::<crate::camera::CameraUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind groups for each eye
        let left_camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VR Left Camera Bind Group"),
            layout: camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: left_camera_buffer.as_entire_binding(),
            }],
        });

        let right_camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VR Right Camera Bind Group"),
            layout: camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: right_camera_buffer.as_entire_binding(),
            }],
        });

        info!("VR renderer initialized successfully");

        Ok(Self {
            vr_session,
            swapchain_format,
            _left_depth_texture: left_depth_texture,
            left_depth_view,
            _right_depth_texture: right_depth_texture,
            right_depth_view,
            left_camera_buffer,
            right_camera_buffer,
            left_camera_bind_group,
            right_camera_bind_group,
            performance_monitor: VrPerformanceMonitor::new(),
            last_warning_frame: 0,
        })
    }

    /// Create a depth texture for VR rendering
    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        label: &str,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    /// Update camera configuration from OpenXR views
    pub fn update_camera_from_vr(&mut self, camera: &mut Camera) -> Result<()> {
        // Get view configurations from OpenXR
        let view_configs = self.vr_session.get_view_configs()?;

        // Convert OpenXR FOV to our FOV format
        let left_fov = crate::camera::FovConfig {
            angle_left: view_configs[0].fov.angle_left,
            angle_right: view_configs[0].fov.angle_right,
            angle_up: view_configs[0].fov.angle_up,
            angle_down: view_configs[0].fov.angle_down,
        };

        let right_fov = crate::camera::FovConfig {
            angle_left: view_configs[1].fov.angle_left,
            angle_right: view_configs[1].fov.angle_right,
            angle_up: view_configs[1].fov.angle_up,
            angle_down: view_configs[1].fov.angle_down,
        };

        // Calculate IPD from view positions
        let left_pos = view_configs[0].position;
        let right_pos = view_configs[1].position;
        let ipd = (right_pos - left_pos).length();

        // Update camera stereo configuration
        camera.stereo_config = Some(crate::camera::StereoConfig {
            ipd,
            left_fov,
            right_fov,
        });

        // Update camera position from head pose (use average of both eyes)
        let head_position = (left_pos + right_pos) * 0.5;

        // Keep the current target, just update position to maintain viewing direction
        let current_forward = (camera.target - camera.position).normalize();
        let distance = camera.position.distance(camera.target);
        camera.position = head_position;
        camera.target = head_position + current_forward * distance;

        Ok(())
    }

    /// Render stereo frame for VR
    pub fn render_stereo<F>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &Camera,
        mut render_eye: F,
    ) -> Result<()>
    where
        F: FnMut(&wgpu::RenderPass, Eye),
    {
        // Update performance monitoring
        self.performance_monitor.tick();

        // Check for performance warnings (every 90 frames = ~1 second at 90 FPS)
        let stats = self.performance_monitor.stats();
        if stats.total_frames > self.last_warning_frame + 90 {
            if let Some(warning) = self.performance_monitor.get_warning() {
                warn!("{}", warning);
                self.last_warning_frame = stats.total_frames;
            }
        }

        // Acquire swapchain images
        let left_swapchain = self
            .vr_session
            .left_swapchain
            .as_mut()
            .context("Left swapchain not available")?;
        let right_swapchain = self
            .vr_session
            .right_swapchain
            .as_mut()
            .context("Right swapchain not available")?;

        let left_image_index = left_swapchain.acquire_image()?;
        let right_image_index = right_swapchain.acquire_image()?;

        // Create command encoder
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("VR Render Encoder"),
        });

        // Update camera uniforms for each eye
        let left_uniform = camera.uniform_stereo(Eye::Left);
        let right_uniform = camera.uniform_stereo(Eye::Right);

        queue.write_buffer(
            &self.left_camera_buffer,
            0,
            bytemuck::cast_slice(&[left_uniform]),
        );
        queue.write_buffer(
            &self.right_camera_buffer,
            0,
            bytemuck::cast_slice(&[right_uniform]),
        );

        // LEFT EYE RENDER PASS
        {
            let color_view = &left_swapchain.views[left_image_index];

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("VR Left Eye Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.02,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.left_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_bind_group(0, &self.left_camera_bind_group, &[]);
            render_eye(&render_pass, Eye::Left);
        }

        // RIGHT EYE RENDER PASS
        {
            let color_view = &right_swapchain.views[right_image_index];

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("VR Right Eye Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.02,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.right_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_bind_group(0, &self.right_camera_bind_group, &[]);
            render_eye(&render_pass, Eye::Right);
        }

        // Submit commands
        queue.submit(std::iter::once(encoder.finish()));

        // Release swapchain images
        self.vr_session.left_swapchain.as_mut().unwrap().release_image()?;
        self.vr_session.right_swapchain.as_mut().unwrap().release_image()?;

        Ok(())
    }

    /// Build a `CameraUniform` for one eye directly from an OpenXR view (6DoF).
    ///
    /// `mol_to_world` converts molecule coordinates (Angstroms, origin-centered) to
    /// VR stage space (meters). A good default:
    ///   `Mat4::from_scale_rotation_translation(Vec3::splat(0.002), Quat::IDENTITY, Vec3::new(0.0, 1.4, -1.5))`
    /// This scales 1 Å → 2 mm and places the molecule at eye height 1.4 m, 1.5 m ahead.
    ///
    /// `near`/`far` are in VR stage space (meters), e.g. 0.02 and 100.0.
    pub fn build_eye_uniform(
        view: &mol_vr::session::ViewConfig,
        near: f32,
        far: f32,
        mol_to_world: Mat4,
    ) -> crate::camera::CameraUniform {
        // Eye pose → world-from-eye transform
        let world_from_eye =
            Mat4::from_rotation_translation(view.orientation, view.position);
        // Invert: this transforms from world (stage) space into eye (view) space
        let eye_from_world = world_from_eye.inverse();

        // Full view: first bring molecule into stage space, then into eye space
        let view_matrix = eye_from_world * mol_to_world;

        // Asymmetric projection from OpenXR FOV angles (same formula as existing camera code)
        let l = near * view.fov.angle_left.tan();
        let r = near * view.fov.angle_right.tan();
        let b = near * view.fov.angle_down.tan();
        let t = near * view.fov.angle_up.tan();
        let w = r - l;
        let h = t - b;
        let d = far - near;
        let proj_matrix = Mat4::from_cols(
            Vec4::new(2.0 * near / w, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 * near / h, 0.0, 0.0),
            Vec4::new(
                (r + l) / w,
                (t + b) / h,
                -(far + near) / d,
                -1.0,
            ),
            Vec4::new(0.0, 0.0, -2.0 * far * near / d, 0.0),
        );

        let view_proj = proj_matrix * view_matrix;

        crate::camera::CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            view: view_matrix.to_cols_array_2d(),
            proj: proj_matrix.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            view_pos: view.position.into(),
            _padding: 0.0,
        }
    }

    /// Poll VR events and update session state
    pub fn poll_events(&mut self) -> Result<bool> {
        self.vr_session.poll_events()
    }

    /// Begin a VR frame
    pub fn begin_frame(&mut self) -> Result<openxr::FrameState> {
        self.vr_session.begin_frame()
    }

    /// End a VR frame and submit to compositor
    pub fn end_frame(&mut self, frame_state: &openxr::FrameState) -> Result<()> {
        self.vr_session.end_frame(frame_state)
    }

    /// Get current VR performance statistics
    pub fn performance_stats(&self) -> &mol_vr::VrPerformanceStats {
        self.performance_monitor.stats()
    }

    /// Check if VR performance is acceptable (>95% frames meeting 90 FPS target)
    pub fn is_performance_acceptable(&self) -> bool {
        self.performance_monitor.is_performance_acceptable()
    }
}
