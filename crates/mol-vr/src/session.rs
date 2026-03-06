//! OpenXR Session Management
//!
//! This module handles the lifecycle of an OpenXR VR session, including:
//! - Instance and session creation
//! - Swapchain management for stereo rendering
//! - Frame timing and synchronization
//! - Session state transitions

use anyhow::{Context, Result};
use ash::vk;
use ash::vk::Handle as _;
use glam::{Quat, Vec3};
use log::{debug, info, warn};
use openxr as xr;
use wgpu;

use crate::input::VrInput;
use crate::vulkan_init::WgpuVulkanContext;

/// OpenXR session state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Idle,
    Ready,
    Synchronized,
    Visible,
    Focused,
    Stopping,
    Exited,
}

impl From<xr::SessionState> for SessionState {
    fn from(state: xr::SessionState) -> Self {
        match state {
            xr::SessionState::IDLE => SessionState::Idle,
            xr::SessionState::READY => SessionState::Ready,
            xr::SessionState::SYNCHRONIZED => SessionState::Synchronized,
            xr::SessionState::VISIBLE => SessionState::Visible,
            xr::SessionState::FOCUSED => SessionState::Focused,
            xr::SessionState::STOPPING => SessionState::Stopping,
            xr::SessionState::EXITING => SessionState::Exited,
            _ => SessionState::Idle,
        }
    }
}

/// View configuration for stereo rendering
#[derive(Debug, Clone, Copy)]
pub struct ViewConfig {
    pub position: Vec3,
    pub orientation: Quat,
    pub fov: FovConfig,
}

/// Field-of-view configuration (asymmetric frustum)
#[derive(Debug, Clone, Copy)]
pub struct FovConfig {
    pub angle_left: f32,
    pub angle_right: f32,
    pub angle_up: f32,
    pub angle_down: f32,
}

impl From<xr::Fovf> for FovConfig {
    fn from(fov: xr::Fovf) -> Self {
        Self {
            angle_left: fov.angle_left,
            angle_right: fov.angle_right,
            angle_up: fov.angle_up,
            angle_down: fov.angle_down,
        }
    }
}

/// Swapchain for a single eye
pub struct VrSwapchain {
    pub handle: xr::Swapchain<xr::Vulkan>,
    pub resolution: (u32, u32),
    pub images: Vec<wgpu::Texture>,
    pub views: Vec<wgpu::TextureView>,
}

impl VrSwapchain {
    /// Create a new swapchain for the given eye
    pub fn new(
        session: &xr::Session<xr::Vulkan>,
        device: &wgpu::Device,
        resolution: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> Result<Self> {
        // Convert wgpu format to Vulkan format
        let vk_format = Self::wgpu_format_to_vulkan(format);

        // Create OpenXR swapchain
        let handle = session
            .create_swapchain(&xr::SwapchainCreateInfo {
                create_flags: xr::SwapchainCreateFlags::EMPTY,
                usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT
                    | xr::SwapchainUsageFlags::SAMPLED,
                format: vk_format,
                sample_count: 1,
                width: resolution.0,
                height: resolution.1,
                face_count: 1,
                array_size: 1,
                mip_count: 1,
            })
            .context("Failed to create OpenXR swapchain")?;

        // Enumerate swapchain images (OpenXR returns raw VkImage handles as u64)
        let xr_images = handle.enumerate_images()?;

        // Wrap OpenXR's VkImages as wgpu textures via wgpu::hal Vulkan backend
        let mut images = Vec::with_capacity(xr_images.len());
        let mut views = Vec::with_capacity(xr_images.len());

        let hal_desc = wgpu::hal::TextureDescriptor {
            label: Some("VR Swapchain Image"),
            size: wgpu::Extent3d {
                width: resolution.0,
                height: resolution.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::hal::TextureUses::COLOR_TARGET,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };

        let wgpu_desc = wgpu::TextureDescriptor {
            label: Some("VR Swapchain Texture"),
            size: wgpu::Extent3d {
                width: resolution.0,
                height: resolution.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };

        for &xr_image_raw in &xr_images {
            // Separate unsafe call from ? operator to avoid type inference ambiguity
            // as_hal in wgpu 22 passes &Device directly (not Option) and returns Option<R>
            let opt_hal_texture: Option<wgpu::hal::vulkan::Texture> = unsafe {
                device.as_hal::<wgpu::hal::api::Vulkan, _, wgpu::hal::vulkan::Texture>(|_d| {
                    wgpu::hal::vulkan::Device::texture_from_raw(
                        vk::Image::from_raw(xr_image_raw),
                        &hal_desc,
                        None, // OpenXR owns the image lifetime
                    )
                })
            };
            let hal_texture = opt_hal_texture
                .ok_or_else(|| anyhow::anyhow!("wgpu is not using the Vulkan backend"))?;

            let texture = unsafe {
                device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_texture, &wgpu_desc)
            };

            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            images.push(texture);
            views.push(view);
        }

        info!(
            "Created VR swapchain: {}x{}, {} images",
            resolution.0,
            resolution.1,
            images.len()
        );

        Ok(Self {
            handle,
            resolution,
            images,
            views,
        })
    }

    /// Convert wgpu texture format to Vulkan format
    fn wgpu_format_to_vulkan(format: wgpu::TextureFormat) -> u32 {
        // Vulkan format constants (from vulkan.h)
        const VK_FORMAT_B8G8R8A8_SRGB: u32 = 50;
        const VK_FORMAT_R8G8B8A8_SRGB: u32 = 43;
        const VK_FORMAT_B8G8R8A8_UNORM: u32 = 44;
        const VK_FORMAT_R8G8B8A8_UNORM: u32 = 37;

        match format {
            wgpu::TextureFormat::Bgra8UnormSrgb => VK_FORMAT_B8G8R8A8_SRGB,
            wgpu::TextureFormat::Rgba8UnormSrgb => VK_FORMAT_R8G8B8A8_SRGB,
            wgpu::TextureFormat::Bgra8Unorm => VK_FORMAT_B8G8R8A8_UNORM,
            wgpu::TextureFormat::Rgba8Unorm => VK_FORMAT_R8G8B8A8_UNORM,
            _ => {
                warn!("Unsupported texture format {:?}, defaulting to BGRA8_SRGB", format);
                VK_FORMAT_B8G8R8A8_SRGB
            }
        }
    }

    /// Acquire the next swapchain image
    pub fn acquire_image(&mut self) -> Result<usize> {
        let image_index = self.handle.acquire_image()?;
        self.handle.wait_image(xr::Duration::INFINITE)?;
        Ok(image_index as usize)
    }

    /// Release the current swapchain image
    pub fn release_image(&mut self) -> Result<()> {
        self.handle.release_image()?;
        Ok(())
    }
}

/// Main VR session manager
pub struct VrSession {
    pub instance: xr::Instance,
    pub system: xr::SystemId,
    pub session: xr::Session<xr::Vulkan>,
    pub session_state: SessionState,
    pub frame_waiter: xr::FrameWaiter,
    pub frame_stream: xr::FrameStream<xr::Vulkan>,
    pub stage_space: xr::Space,
    pub view_config_type: xr::ViewConfigurationType,

    // Swapchains for stereo rendering
    pub left_swapchain: Option<VrSwapchain>,
    pub right_swapchain: Option<VrSwapchain>,

    // View configuration
    pub views: Vec<xr::View>,

    // Input system
    pub input: VrInput,
}

impl VrSession {
    /// Create a new VR session.
    ///
    /// Returns `(VrSession, WgpuVulkanContext)`. The caller must wrap the `WgpuVulkanContext`
    /// into a `wgpu::Instance` / `wgpu::Device` via `from_hal` before it can be used for
    /// rendering. This split is necessary because OpenXR needs to inject its required Vulkan
    /// extensions (external memory/semaphore on Windows) before the device is created.
    pub fn new() -> Result<(Self, WgpuVulkanContext)> {
        info!("Initializing OpenXR session...");

        // ── Create OpenXR instance ─────────────────────────────────────────────
        let entry = unsafe { xr::Entry::load()? };

        let available_extensions = entry.enumerate_extensions()?;
        debug!("Available OpenXR extensions: {:?}", available_extensions);

        let mut enabled_extensions = xr::ExtensionSet::default();
        if available_extensions.khr_vulkan_enable2 {
            enabled_extensions.khr_vulkan_enable2 = true;
        } else {
            anyhow::bail!("OpenXR runtime does not support khr_vulkan_enable2 — cannot interop with wgpu");
        }

        let instance = entry.create_instance(
            &xr::ApplicationInfo {
                application_name: "PDB Visual VR",
                application_version: 1,
                engine_name: "mol-render",
                engine_version: 1,
                api_version: xr::Version::new(1, 0, 0),
            },
            &enabled_extensions,
            &[],
        )?;

        info!("OpenXR instance created: {:?}", instance.properties()?);

        // ── Get system ────────────────────────────────────────────────────────
        let system = instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)?;

        let system_props = instance.system_properties(system)?;
        info!("VR System: {}", system_props.system_name);

        let view_config_type = xr::ViewConfigurationType::PRIMARY_STEREO;
        let view_config_views =
            instance.enumerate_view_configuration_views(system, view_config_type)?;

        info!("View configuration:");
        for (i, view) in view_config_views.iter().enumerate() {
            info!(
                "  Eye {}: {}x{} (recommended), {}x{} (max)",
                i,
                view.recommended_image_rect_width,
                view.recommended_image_rect_height,
                view.max_image_rect_width,
                view.max_image_rect_height
            );
        }

        // ── Mandatory graphics requirements call ───────────────────────────────
        let vk_reqs = instance
            .graphics_requirements::<xr::Vulkan>(system)
            .context("Failed to get Vulkan graphics requirements")?;
        info!(
            "OpenXR Vulkan requirements: min={}.{}, max={}.{}",
            vk_reqs.min_api_version_supported.major(),
            vk_reqs.min_api_version_supported.minor(),
            vk_reqs.max_api_version_supported.major(),
            vk_reqs.max_api_version_supported.minor(),
        );

        // ── Create Vulkan context (with OpenXR-required extensions injected) ──
        // This replaces the old approach of extracting handles from wgpu AFTER device creation.
        // OpenXR's create_vulkan_instance / create_vulkan_device add VK_KHR_external_memory_win32
        // and VK_KHR_external_semaphore_win32 so the runtime can composite the frame.
        let vk_ctx = crate::vulkan_init::create_vulkan_context_for_openxr(&instance, system)
            .context("Failed to create Vulkan context for OpenXR interop")?;

        // ── Create OpenXR session ─────────────────────────────────────────────
        let vk_instance_raw = vk_ctx.instance.handle().as_raw();
        let vk_phd_raw = vk_ctx.physical_device.as_raw();
        let vk_device_raw = vk_ctx.device.handle().as_raw();

        info!(
            "Creating OpenXR session: instance=0x{:016x}, pdev=0x{:016x}, dev=0x{:016x}, qfi={}",
            vk_instance_raw, vk_phd_raw, vk_device_raw, vk_ctx.queue_family_index
        );

        let session_create_info = xr::vulkan::SessionCreateInfo {
            instance: vk_instance_raw as _,
            physical_device: vk_phd_raw as _,
            device: vk_device_raw as _,
            queue_family_index: vk_ctx.queue_family_index,
            queue_index: 0,
        };

        let (session, frame_waiter, frame_stream) = unsafe {
            instance
                .create_session::<xr::Vulkan>(system, &session_create_info)
                .context("xrCreateSession failed")?
        };
        info!("OpenXR session created successfully");

        info!("OpenXR session created");

        // Create reference space
        let stage_space: xr::Space = session.create_reference_space(
            xr::ReferenceSpaceType::STAGE,
            xr::Posef::IDENTITY,
        )?;

        // Create input system
        let input = VrInput::new(&instance, &session)?;

        let vr_session = Self {
            instance,
            system,
            session,
            session_state: SessionState::Idle,
            frame_waiter,
            frame_stream,
            stage_space,
            view_config_type,
            left_swapchain: None,
            right_swapchain: None,
            views: Vec::new(),
            input,
        };

        Ok((vr_session, vk_ctx))
    }

    /// Poll OpenXR events and update session state
    pub fn poll_events(&mut self) -> Result<bool> {
        let mut event_buffer = xr::EventDataBuffer::new();

        while let Some(event) = self.instance.poll_event(&mut event_buffer)? {
            match event {
                xr::Event::SessionStateChanged(state_change) => {
                    self.session_state = state_change.state().into();
                    info!("VR session state changed to: {:?}", self.session_state);

                    match self.session_state {
                        SessionState::Ready => {
                            // Begin session
                            self.session.begin(self.view_config_type)?;
                            info!("VR session started");
                        }
                        SessionState::Stopping => {
                            // End session
                            self.session.end()?;
                            info!("VR session ended");
                        }
                        SessionState::Exited => {
                            return Ok(false); // Signal to exit VR mode
                        }
                        _ => {}
                    }
                }
                xr::Event::InstanceLossPending(_) => {
                    warn!("OpenXR instance loss pending");
                    return Ok(false);
                }
                _ => {}
            }
        }

        Ok(true)
    }

    /// Begin a new frame
    pub fn begin_frame(&mut self) -> Result<xr::FrameState> {
        let frame_state = self.frame_waiter.wait()?;
        self.frame_stream.begin()?;

        // Locate views (get head pose and eye positions)
        if frame_state.should_render {
            let (_, views) = self.session.locate_views(
                self.view_config_type,
                frame_state.predicted_display_time,
                &self.stage_space,
            )?;
            self.views = views;
        }

        Ok(frame_state)
    }

    /// End the current frame and submit to compositor
    pub fn end_frame(&mut self, frame_state: &xr::FrameState) -> Result<()> {
        // Build projection layers
        let mut projection_views = Vec::new();

        if frame_state.should_render && !self.views.is_empty() {
            // Get swapchain images
            if let (Some(left_swap), Some(right_swap)) = (&self.left_swapchain, &self.right_swapchain) {
                for (i, view) in self.views.iter().enumerate() {
                    let swapchain = if i == 0 { left_swap } else { right_swap };

                    projection_views.push(xr::CompositionLayerProjectionView::new()
                        .pose(view.pose)
                        .fov(view.fov)
                        .sub_image(
                            xr::SwapchainSubImage::new()
                                .swapchain(&swapchain.handle)
                                .image_array_index(0)
                                .image_rect(xr::Rect2Di {
                                    offset: xr::Offset2Di { x: 0, y: 0 },
                                    extent: xr::Extent2Di {
                                        width: swapchain.resolution.0 as i32,
                                        height: swapchain.resolution.1 as i32,
                                    },
                                }),
                        ));
                }
            }
        }

        // Submit frame to compositor
        if !projection_views.is_empty() {
            let projection_layer = xr::CompositionLayerProjection::new()
                .space(&self.stage_space)
                .views(&projection_views);

            self.frame_stream.end(
                frame_state.predicted_display_time,
                xr::EnvironmentBlendMode::OPAQUE,
                &[&projection_layer],
            )?;
        } else {
            self.frame_stream.end(
                frame_state.predicted_display_time,
                xr::EnvironmentBlendMode::OPAQUE,
                &[],
            )?;
        }

        Ok(())
    }

    /// Get view configurations for stereo rendering
    pub fn get_view_configs(&self) -> Result<[ViewConfig; 2]> {
        if self.views.len() < 2 {
            anyhow::bail!("Not enough views for stereo rendering");
        }

        let left_view = &self.views[0];
        let right_view = &self.views[1];

        Ok([
            ViewConfig {
                position: Vec3::new(
                    left_view.pose.position.x,
                    left_view.pose.position.y,
                    left_view.pose.position.z,
                ),
                orientation: Quat::from_xyzw(
                    left_view.pose.orientation.x,
                    left_view.pose.orientation.y,
                    left_view.pose.orientation.z,
                    left_view.pose.orientation.w,
                ),
                fov: left_view.fov.into(),
            },
            ViewConfig {
                position: Vec3::new(
                    right_view.pose.position.x,
                    right_view.pose.position.y,
                    right_view.pose.position.z,
                ),
                orientation: Quat::from_xyzw(
                    right_view.pose.orientation.x,
                    right_view.pose.orientation.y,
                    right_view.pose.orientation.z,
                    right_view.pose.orientation.w,
                ),
                fov: right_view.fov.into(),
            },
        ])
    }

    /// Select the best swapchain format supported by the OpenXR runtime.
    /// Queries the runtime's supported list and prefers RGBA8_SRGB (Oculus default),
    /// falling back to RGBA8_UNORM, BGRA8_SRGB, BGRA8_UNORM in that order.
    pub fn select_swapchain_format(&self) -> wgpu::TextureFormat {
        // VkFormat constants (must match Vulkan spec)
        const VK_FORMAT_R8G8B8A8_SRGB: u32 = 43;
        const VK_FORMAT_R8G8B8A8_UNORM: u32 = 37;
        const VK_FORMAT_B8G8R8A8_SRGB: u32 = 50;
        const VK_FORMAT_B8G8R8A8_UNORM: u32 = 44;

        let supported = match self.session.enumerate_swapchain_formats() {
            Ok(formats) => {
                info!("OpenXR supported swapchain formats (VkFormat values): {:?}", formats);
                formats
            }
            Err(e) => {
                warn!("Failed to enumerate swapchain formats: {} — defaulting to Rgba8UnormSrgb", e);
                return wgpu::TextureFormat::Rgba8UnormSrgb;
            }
        };

        // Preferred order: RGBA8_SRGB first (Oculus/Meta prefer this over BGRA)
        let preferred: [(u32, wgpu::TextureFormat); 4] = [
            (VK_FORMAT_R8G8B8A8_SRGB,  wgpu::TextureFormat::Rgba8UnormSrgb),
            (VK_FORMAT_R8G8B8A8_UNORM, wgpu::TextureFormat::Rgba8Unorm),
            (VK_FORMAT_B8G8R8A8_SRGB,  wgpu::TextureFormat::Bgra8UnormSrgb),
            (VK_FORMAT_B8G8R8A8_UNORM, wgpu::TextureFormat::Bgra8Unorm),
        ];

        for (vk_fmt, wgpu_fmt) in &preferred {
            if supported.contains(vk_fmt) {
                info!("Selected swapchain format: {:?} (VkFormat {})", wgpu_fmt, vk_fmt);
                return *wgpu_fmt;
            }
        }

        warn!("None of the preferred swapchain formats are supported — defaulting to Rgba8UnormSrgb");
        wgpu::TextureFormat::Rgba8UnormSrgb
    }

    /// Create swapchains for stereo rendering
    pub fn create_swapchains(
        &mut self,
        device: &wgpu::Device,
    ) -> Result<wgpu::TextureFormat> {
        // Select the best format the OpenXR runtime actually supports
        let format = self.select_swapchain_format();

        // Get recommended resolution from view configuration
        let view_config_views = self.instance.enumerate_view_configuration_views(
            self.system,
            self.view_config_type,
        )?;

        if view_config_views.len() < 2 {
            anyhow::bail!("Not enough view configurations for stereo");
        }

        let left_resolution = (
            view_config_views[0].recommended_image_rect_width,
            view_config_views[0].recommended_image_rect_height,
        );

        let right_resolution = (
            view_config_views[1].recommended_image_rect_width,
            view_config_views[1].recommended_image_rect_height,
        );

        // Create swapchains
        self.left_swapchain = Some(VrSwapchain::new(&self.session, device, left_resolution, format)?);
        self.right_swapchain = Some(VrSwapchain::new(&self.session, device, right_resolution, format)?);

        Ok(format)
    }
}
