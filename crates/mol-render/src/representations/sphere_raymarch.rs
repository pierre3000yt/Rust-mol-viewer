// Raymarched Spheres Renderer
// Uses analytic ray-sphere intersection with grid acceleration
// Much more efficient than surface SDF raymarching

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AtomGPU {
    position: [f32; 3],
    radius: f32,
    color: [f32; 3],
    _padding: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridCellGPU {
    offset: u32,
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridParamsGPU {
    origin: [f32; 3],
    cell_size: f32,
    dimensions: [u32; 3],
    atom_count: u32,
    inv_cell_size: f32,
    max_radius: f32,
    _padding1: f32,
    _padding2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightParamsGPU {
    direction: [f32; 3],
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    _padding: f32,
}

pub struct SphereRaymarchRenderer {
    pipeline: wgpu::RenderPipeline,
    data_bind_group_layout: wgpu::BindGroupLayout,
    data_bind_group: Option<wgpu::BindGroup>,

    atom_buffer: Option<wgpu::Buffer>,
    grid_buffer: Option<wgpu::Buffer>,
    atom_indices_buffer: Option<wgpu::Buffer>,
    grid_params_buffer: wgpu::Buffer,
    light_buffer: wgpu::Buffer,

    is_initialized: bool,
    atom_count: usize,
}

impl SphereRaymarchRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sphere Raymarch Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../../assets/shaders/sphere_raymarch.wgsl").into(),
            ),
        });

        let data_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Sphere Raymarch Data Layout"),
                entries: &[
                    // Atoms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Grid
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Atom indices
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Grid params
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Light params
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sphere Raymarch Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, &data_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sphere Raymarch Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,  // Opaque spheres
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Default grid params
        let grid_params = GridParamsGPU {
            origin: [0.0, 0.0, 0.0],
            cell_size: 8.0,
            dimensions: [1, 1, 1],
            atom_count: 0,
            inv_cell_size: 1.0 / 8.0,
            max_radius: 2.0,
            _padding1: 0.0,
            _padding2: 0.0,
        };

        let grid_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Raymarch Grid Params"),
            contents: bytemuck::cast_slice(&[grid_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Default light
        let light = LightParamsGPU {
            direction: [0.5, 1.0, 0.3],
            ambient: 0.2,
            diffuse: 0.7,
            specular: 0.3,
            shininess: 32.0,
            _padding: 0.0,
        };

        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Raymarch Light"),
            contents: bytemuck::cast_slice(&[light]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        log::info!("SphereRaymarchRenderer initialized");

        Self {
            pipeline,
            data_bind_group_layout,
            data_bind_group: None,
            atom_buffer: None,
            grid_buffer: None,
            atom_indices_buffer: None,
            grid_params_buffer,
            light_buffer,
            is_initialized: false,
            atom_count: 0,
        }
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        protein: &pdb_parser::Protein,
        cell_size: f32,
    ) -> anyhow::Result<()> {
        let start = std::time::Instant::now();

        if protein.atoms.is_empty() {
            self.is_initialized = false;
            return Ok(());
        }

        self.atom_count = protein.atoms.len();

        log::info!(
            "Preparing raymarched spheres for {} atoms...",
            protein.atoms.len()
        );

        // 1. Prepare atoms with colors
        let mut max_radius: f32 = 0.0;
        let atoms: Vec<AtomGPU> = protein
            .atoms
            .iter()
            .map(|atom| {
                let radius = atom.element.vdw_radius();
                max_radius = max_radius.max(radius);
                let color = atom.element.cpk_color();
                AtomGPU {
                    position: atom.position.into(),
                    radius,
                    color: [color[0], color[1], color[2]],
                    _padding: 0.0,
                }
            })
            .collect();

        // 2. Build acceleration grid
        let (min, max) = protein.bounding_box();
        let padding = max_radius + cell_size;
        let grid_min = min - Vec3::splat(padding);
        let grid_max = max + Vec3::splat(padding);
        let grid_size = grid_max - grid_min;

        let nx = ((grid_size.x / cell_size).ceil() as usize).max(1);
        let ny = ((grid_size.y / cell_size).ceil() as usize).max(1);
        let nz = ((grid_size.z / cell_size).ceil() as usize).max(1);
        let total_cells = nx * ny * nz;

        log::info!("  Grid: {}x{}x{} = {} cells", nx, ny, nz, total_cells);

        let inv_cell_size = 1.0 / cell_size;

        // Count atoms per cell
        let mut cell_counts: Vec<u32> = vec![0; total_cells];

        for atom in protein.atoms.iter() {
            let radius = atom.element.vdw_radius();
            let atom_min = atom.position - Vec3::splat(radius);
            let atom_max = atom.position + Vec3::splat(radius);

            let cell_min_x = ((atom_min.x - grid_min.x) * inv_cell_size).floor() as i32;
            let cell_min_y = ((atom_min.y - grid_min.y) * inv_cell_size).floor() as i32;
            let cell_min_z = ((atom_min.z - grid_min.z) * inv_cell_size).floor() as i32;

            let cell_max_x = ((atom_max.x - grid_min.x) * inv_cell_size).ceil() as i32;
            let cell_max_y = ((atom_max.y - grid_min.y) * inv_cell_size).ceil() as i32;
            let cell_max_z = ((atom_max.z - grid_min.z) * inv_cell_size).ceil() as i32;

            for cz in cell_min_z.max(0)..cell_max_z.min(nz as i32) {
                for cy in cell_min_y.max(0)..cell_max_y.min(ny as i32) {
                    for cx in cell_min_x.max(0)..cell_max_x.min(nx as i32) {
                        let idx = (cz as usize) * nx * ny + (cy as usize) * nx + (cx as usize);
                        if idx < total_cells {
                            cell_counts[idx] += 1;
                        }
                    }
                }
            }
        }

        // Calculate offsets
        let mut offsets: Vec<u32> = vec![0; total_cells];
        let mut current_offset = 0u32;
        for i in 0..total_cells {
            offsets[i] = current_offset;
            current_offset += cell_counts[i];
        }

        let total_indices = current_offset as usize;

        // Build grid cells
        let mut grid_cells: Vec<GridCellGPU> = Vec::with_capacity(total_cells);
        for i in 0..total_cells {
            grid_cells.push(GridCellGPU {
                offset: offsets[i],
                count: cell_counts[i],
            });
        }

        // Fill atom indices
        let mut atom_indices_data: Vec<u32> = vec![0; total_indices.max(1)];
        let mut cell_fill_counts: Vec<u32> = vec![0; total_cells];

        for (atom_idx, atom) in protein.atoms.iter().enumerate() {
            let radius = atom.element.vdw_radius();
            let atom_min = atom.position - Vec3::splat(radius);
            let atom_max = atom.position + Vec3::splat(radius);

            let cell_min_x = ((atom_min.x - grid_min.x) * inv_cell_size).floor() as i32;
            let cell_min_y = ((atom_min.y - grid_min.y) * inv_cell_size).floor() as i32;
            let cell_min_z = ((atom_min.z - grid_min.z) * inv_cell_size).floor() as i32;

            let cell_max_x = ((atom_max.x - grid_min.x) * inv_cell_size).ceil() as i32;
            let cell_max_y = ((atom_max.y - grid_min.y) * inv_cell_size).ceil() as i32;
            let cell_max_z = ((atom_max.z - grid_min.z) * inv_cell_size).ceil() as i32;

            for cz in cell_min_z.max(0)..cell_max_z.min(nz as i32) {
                for cy in cell_min_y.max(0)..cell_max_y.min(ny as i32) {
                    for cx in cell_min_x.max(0)..cell_max_x.min(nx as i32) {
                        let cell_idx = (cz as usize) * nx * ny + (cy as usize) * nx + (cx as usize);
                        if cell_idx < total_cells {
                            let idx = offsets[cell_idx] + cell_fill_counts[cell_idx];
                            atom_indices_data[idx as usize] = atom_idx as u32;
                            cell_fill_counts[cell_idx] += 1;
                        }
                    }
                }
            }
        }

        // 3. Create GPU buffers
        let atom_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Raymarch Atoms"),
            contents: bytemuck::cast_slice(&atoms),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let grid_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Raymarch Grid"),
            contents: bytemuck::cast_slice(&grid_cells),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let atom_indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Raymarch Atom Indices"),
            contents: bytemuck::cast_slice(&atom_indices_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Update grid params
        let grid_params = GridParamsGPU {
            origin: grid_min.into(),
            cell_size,
            dimensions: [nx as u32, ny as u32, nz as u32],
            atom_count: protein.atoms.len() as u32,
            inv_cell_size,
            max_radius,
            _padding1: 0.0,
            _padding2: 0.0,
        };

        queue.write_buffer(
            &self.grid_params_buffer,
            0,
            bytemuck::cast_slice(&[grid_params]),
        );

        // Create bind group
        let data_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sphere Raymarch Data"),
            layout: &self.data_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: atom_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: atom_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.grid_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.light_buffer.as_entire_binding(),
                },
            ],
        });

        self.atom_buffer = Some(atom_buffer);
        self.grid_buffer = Some(grid_buffer);
        self.atom_indices_buffer = Some(atom_indices_buffer);
        self.data_bind_group = Some(data_bind_group);
        self.is_initialized = true;

        log::info!(
            "Raymarched spheres prepared in {:?} ({} atoms, {} cells)",
            start.elapsed(),
            protein.atoms.len(),
            total_cells
        );

        Ok(())
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if !self.is_initialized || self.data_bind_group.is_none() {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(1, self.data_bind_group.as_ref().unwrap(), &[]);

        // Draw fullscreen triangle (3 vertices)
        render_pass.draw(0..3, 0..1);
    }

    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    pub fn atom_count(&self) -> usize {
        self.atom_count
    }
}
