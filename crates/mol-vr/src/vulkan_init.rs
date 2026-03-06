//! Vulkan Context Initialization for OpenXR Interop
//!
//! Creates VkInstance and VkDevice using OpenXR's `khr_vulkan_enable2` patching functions,
//! which automatically add all required extensions (VK_KHR_external_semaphore_win32,
//! VK_KHR_external_memory_win32, etc.) that the OpenXR runtime needs for composition.
//!
//! This is the correct way to do wgpu + OpenXR interop: create the Vulkan context first
//! (letting OpenXR patch in its required extensions), then wrap it in wgpu via from_hal.

use anyhow::{Context, Result};
use ash::vk;
use ash::vk::Handle as _;
use log::{info, warn};
use openxr as xr;
use std::ffi::{CStr, CString};

/// Vulkan context created for OpenXR + wgpu interop.
/// The ash objects here are meant to be consumed by `wgpu::hal::vulkan::Instance::from_raw`
/// and `Adapter::device_from_raw`. After that, wgpu owns and keeps them alive.
pub struct WgpuVulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue_family_index: u32,
    /// Extensions enabled on the VkInstance (for wgpu::hal::vulkan::Instance::from_raw)
    pub instance_extensions: Vec<&'static CStr>,
    /// Extensions enabled on the VkDevice (for Adapter::device_from_raw)
    pub device_extensions: Vec<&'static CStr>,
    /// Vulkan API version used (e.g. vk::API_VERSION_1_1)
    pub api_version: u32,
}

/// Create a Vulkan instance and device compatible with both wgpu and the OpenXR runtime.
///
/// Uses `xrCreateVulkanInstanceKHR2` and `xrCreateVulkanDeviceKHR2` so that the Oculus
/// runtime (and others) can inject their required extensions before the objects are created.
pub fn create_vulkan_context_for_openxr(
    xr_instance: &xr::Instance,
    system: xr::SystemId,
) -> Result<WgpuVulkanContext> {
    info!("Creating Vulkan context for OpenXR (with required extension injection)...");

    // ─── Load Vulkan entry ─────────────────────────────────────────────────────
    let entry = unsafe { ash::Entry::load() }
        .context("Failed to load vulkan-1.dll — is the Vulkan runtime installed?")?;

    // ─── Instance extensions we want ──────────────────────────────────────────
    // OpenXR's create_vulkan_instance will ADD its own required extensions on top of these.
    // (VK_KHR_external_memory_capabilities, VK_KHR_external_semaphore_capabilities, etc.)
    let desired_instance_exts: &[&'static CStr] = &[
        ash::khr::surface::NAME,
        ash::khr::win32_surface::NAME,
        ash::ext::swapchain_colorspace::NAME,
        ash::khr::get_physical_device_properties2::NAME,
        ash::khr::external_memory_capabilities::NAME,
        ash::khr::external_semaphore_capabilities::NAME,
    ];

    // Filter to only extensions actually available on this system
    let available_inst_exts = unsafe { entry.enumerate_instance_extension_properties(None) }
        .context("vkEnumerateInstanceExtensionProperties failed")?;

    let mut enabled_inst_exts: Vec<&'static CStr> = desired_instance_exts
        .iter()
        .copied()
        .filter(|&name| {
            let available = available_inst_exts
                .iter()
                .any(|e| e.extension_name_as_c_str().ok() == Some(name));
            if !available {
                warn!("Instance extension {:?} not available, skipping", name);
            }
            available
        })
        .collect();

    info!(
        "Requesting {} instance extensions for OpenXR interop",
        enabled_inst_exts.len()
    );

    // ─── Create VkInstance via OpenXR ─────────────────────────────────────────
    // OpenXR patches the VkInstanceCreateInfo to add its required extensions.
    let inst_ext_ptrs: Vec<*const i8> = enabled_inst_exts
        .iter()
        .map(|name| name.as_ptr())
        .collect();

    let app_name = CString::new("PDB Visual VR").unwrap();
    let engine_name = CString::new("mol-render").unwrap();
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name.as_c_str())
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(engine_name.as_c_str())
        .engine_version(1)
        .api_version(vk::API_VERSION_1_1);

    let inst_create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&inst_ext_ptrs);

    // SAFETY: We transmute get_instance_proc_addr to the OpenXR expected function pointer type.
    // Both types have identical signatures and calling conventions.
    let get_inst_proc_addr =
        unsafe { std::mem::transmute(entry.static_fn().get_instance_proc_addr) };

    let raw_vk_instance = unsafe {
        xr_instance.create_vulkan_instance(
            system,
            get_inst_proc_addr,
            &inst_create_info as *const vk::InstanceCreateInfo as *const _,
        )
    }
    .context("xrCreateVulkanInstanceKHR2 call failed")?
    .map_err(|vk_result| {
        anyhow::anyhow!("VkCreateInstance failed: VkResult({})", vk_result)
    })?;

    // Wrap the raw VkInstance in ash
    let ash_instance = unsafe {
        ash::Instance::load(
            entry.static_fn(),
            vk::Instance::from_raw(raw_vk_instance as u64),
        )
    };
    info!("VkInstance created via OpenXR (handle: 0x{:016x})", raw_vk_instance as u64);

    // ─── Select physical device ───────────────────────────────────────────────
    // OpenXR picks the physical device that the runtime expects to compose from.
    let raw_vk_phd = unsafe {
        xr_instance.vulkan_graphics_device(system, raw_vk_instance)
    }
    .context("xrGetVulkanGraphicsDevice2KHR failed")?;

    let ash_phd = vk::PhysicalDevice::from_raw(raw_vk_phd as u64);
    info!("VkPhysicalDevice selected by OpenXR (handle: 0x{:016x})", raw_vk_phd as u64);

    // ─── Find graphics queue family ───────────────────────────────────────────
    let queue_families = unsafe {
        ash_instance.get_physical_device_queue_family_properties(ash_phd)
    };

    let queue_family_index = queue_families
        .iter()
        .enumerate()
        .find_map(|(i, props)| {
            if props.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                Some(i as u32)
            } else {
                None
            }
        })
        .ok_or_else(|| anyhow::anyhow!("No graphics queue family found on selected GPU"))?;

    info!("Selected graphics queue family index: {}", queue_family_index);

    // ─── Device extensions we want ────────────────────────────────────────────
    // OpenXR's create_vulkan_device will ADD its required extensions on top of these.
    // Key OpenXR-required ones: VK_KHR_external_semaphore_win32, VK_KHR_external_memory_win32
    let desired_device_exts: &[&'static CStr] = &[
        ash::khr::swapchain::NAME,
        ash::khr::external_memory::NAME,
        ash::khr::external_memory_win32::NAME,
        ash::khr::external_semaphore::NAME,
        ash::khr::external_semaphore_win32::NAME,
        ash::khr::dedicated_allocation::NAME,
        ash::khr::timeline_semaphore::NAME,
        ash::khr::get_memory_requirements2::NAME,
        ash::khr::maintenance1::NAME,
        ash::khr::maintenance2::NAME,
        ash::khr::maintenance3::NAME,
        ash::khr::multiview::NAME,
        ash::khr::draw_indirect_count::NAME,
    ];

    let available_dev_exts = unsafe {
        ash_instance.enumerate_device_extension_properties(ash_phd)
    }
    .context("vkEnumerateDeviceExtensionProperties failed")?;

    let mut enabled_dev_exts: Vec<&'static CStr> = desired_device_exts
        .iter()
        .copied()
        .filter(|&name| {
            let available = available_dev_exts
                .iter()
                .any(|e| e.extension_name_as_c_str().ok() == Some(name));
            if !available {
                warn!("Device extension {:?} not available, skipping", name);
            }
            available
        })
        .collect();

    info!(
        "Requesting {} device extensions for OpenXR interop",
        enabled_dev_exts.len()
    );

    // ─── Create VkDevice via OpenXR ───────────────────────────────────────────
    let dev_ext_ptrs: Vec<*const i8> = enabled_dev_exts
        .iter()
        .map(|name| name.as_ptr())
        .collect();

    let queue_priority = 1.0f32;
    let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(std::slice::from_ref(&queue_priority))];

    // Query supported features and enable those wgpu commonly needs
    let phd_features = unsafe { ash_instance.get_physical_device_features(ash_phd) };
    let device_features = vk::PhysicalDeviceFeatures {
        multi_draw_indirect: phd_features.multi_draw_indirect,
        sampler_anisotropy: phd_features.sampler_anisotropy,
        fragment_stores_and_atomics: phd_features.fragment_stores_and_atomics,
        independent_blend: phd_features.independent_blend,
        image_cube_array: phd_features.image_cube_array,
        ..Default::default()
    };

    let dev_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&dev_ext_ptrs)
        .enabled_features(&device_features);

    let raw_vk_device = unsafe {
        xr_instance.create_vulkan_device(
            system,
            get_inst_proc_addr,
            raw_vk_phd,
            &dev_create_info as *const vk::DeviceCreateInfo as *const _,
        )
    }
    .context("xrCreateVulkanDeviceKHR2 call failed")?
    .map_err(|vk_result| {
        anyhow::anyhow!("VkCreateDevice failed: VkResult({})", vk_result)
    })?;

    // Wrap the raw VkDevice in ash
    let ash_device = unsafe {
        ash::Device::load(
            ash_instance.fp_v1_0(),
            vk::Device::from_raw(raw_vk_device as u64),
        )
    };
    info!("VkDevice created via OpenXR (handle: 0x{:016x})", raw_vk_device as u64);

    Ok(WgpuVulkanContext {
        entry,
        instance: ash_instance,
        physical_device: ash_phd,
        device: ash_device,
        queue_family_index,
        instance_extensions: enabled_inst_exts,
        device_extensions: enabled_dev_exts,
        api_version: vk::API_VERSION_1_1,
    })
}
