use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

// GPU data structures - must match shader
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AtomGPU {
    position: [f32; 3],
    radius: f32, // VdW + probe radius
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridCellGPU {
    offset: u32,  // Start index in atom_indices buffer
    count: u32,   // Number of atoms in this cell
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridParamsGPU {
    origin: [f32; 3],
    cell_size: f32,
    dimensions: [u32; 3],
    atom_count: u32,
    inv_cell_size: f32,
    probe_radius: f32,
    smoothing_k: f32,
    _padding: f32,
    // Bounding box for early ray termination
    bbox_min: [f32; 3],
    _padding2: f32,
    bbox_max: [f32; 3],
    _padding3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SurfaceMaterialGPU {
    color: [f32; 3],
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    ao_strength: f32,
}

#[derive(Clone)]
pub struct RaymarchConfig {
    pub probe_radius: f32,
    pub smoothing_k: f32,
    pub cell_size: f32,
    pub color: [f32; 3],
    pub ambient: f32,
    pub diffuse: f32,
    pub specular: f32,
    pub shininess: f32,
    pub ao_strength: f32,
}

impl Default for RaymarchConfig {
    fn default() -> Self {
        Self {
            probe_radius: 1.4,
            smoothing_k: 1.0, // Smooth union blending (larger = smoother)
            cell_size: 8.0,   // Grid cell size in Angstroms (larger = faster but less precise)
            color: [0.7, 0.75, 0.85], // Light blue-gray
            ambient: 0.15,
            diffuse: 0.7,
            specular: 0.4,
            shininess: 32.0,
            ao_strength: 0.5,
        }
    }
}

pub struct SurfaceRaymarchRenderer {
    pipeline: wgpu::RenderPipeline,
    data_bind_group_layout: wgpu::BindGroupLayout,
    data_bind_group: Option<wgpu::BindGroup>,

    // Buffers
    atom_buffer: Option<wgpu::Buffer>,
    grid_buffer: Option<wgpu::Buffer>,
    atom_indices_buffer: Option<wgpu::Buffer>,
    grid_params_buffer: wgpu::Buffer,
    material_buffer: wgpu::Buffer,

    // State
    is_initialized: bool,
    atom_count: usize,
    config: RaymarchConfig,
}

impl SurfaceRaymarchRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Surface Raymarch Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../../assets/shaders/surface_raymarch.wgsl").into(),
            ),
        });

        // Create data bind group layout (atoms, grid, atom_indices, params, material)
        let data_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Raymarch Data Bind Group Layout"),
                entries: &[
                    // Atoms storage buffer
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
                    // Grid storage buffer
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
                    // Atom indices storage buffer
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
                    // Grid params uniform
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
                    // Material uniform
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

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Raymarch Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, &data_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Surface Raymarch Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[], // Fullscreen quad generated in shader
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        let config = RaymarchConfig::default();

        // Create placeholder buffers for params and material
        let grid_params = GridParamsGPU {
            origin: [0.0, 0.0, 0.0],
            cell_size: config.cell_size,
            dimensions: [1, 1, 1],
            atom_count: 0,
            inv_cell_size: 1.0 / config.cell_size,
            probe_radius: config.probe_radius,
            smoothing_k: config.smoothing_k,
            _padding: 0.0,
            bbox_min: [0.0, 0.0, 0.0],
            _padding2: 0.0,
            bbox_max: [1.0, 1.0, 1.0],
            _padding3: 0.0,
        };

        let grid_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Raymarch Grid Params Buffer"),
            contents: bytemuck::cast_slice(&[grid_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let material = SurfaceMaterialGPU {
            color: config.color,
            ambient: config.ambient,
            diffuse: config.diffuse,
            specular: config.specular,
            shininess: config.shininess,
            ao_strength: config.ao_strength,
        };

        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Raymarch Material Buffer"),
            contents: bytemuck::cast_slice(&[material]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        log::info!("SurfaceRaymarchRenderer initialized");
        log::info!("  Raymarching SDF enabled for real-time molecular surface");

        Self {
            pipeline,
            data_bind_group_layout,
            data_bind_group: None,
            atom_buffer: None,
            grid_buffer: None,
            atom_indices_buffer: None,
            grid_params_buffer,
            material_buffer,
            is_initialized: false,
            atom_count: 0,
            config,
        }
    }

    pub fn update_config(&mut self, queue: &wgpu::Queue, config: &RaymarchConfig) {
        self.config = config.clone();

        // Update material buffer
        let material = SurfaceMaterialGPU {
            color: config.color,
            ambient: config.ambient,
            diffuse: config.diffuse,
            specular: config.specular,
            shininess: config.shininess,
            ao_strength: config.ao_strength,
        };

        queue.write_buffer(&self.material_buffer, 0, bytemuck::cast_slice(&[material]));
    }

    pub fn prepare_surface(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        protein: &pdb_parser::Protein,
        config: &RaymarchConfig,
    ) -> anyhow::Result<()> {
        let start = std::time::Instant::now();

        if protein.atoms.is_empty() {
            log::warn!("No atoms to prepare for raymarching");
            self.is_initialized = false;
            return Ok(());
        }

        self.config = config.clone();
        self.atom_count = protein.atoms.len();

        log::info!(
            "Preparing raymarching surface for {} atoms...",
            protein.atoms.len()
        );

        // 1. Prepare atom data
        let atoms: Vec<AtomGPU> = protein
            .atoms
            .iter()
            .map(|atom| AtomGPU {
                position: atom.position.into(),
                radius: atom.element.vdw_radius() + config.probe_radius,
            })
            .collect();

        // 2. Build acceleration grid
        let (min, max) = protein.bounding_box();
        let padding = config.probe_radius + 3.0;
        let grid_min = min - Vec3::splat(padding);
        let grid_max = max + Vec3::splat(padding);
        let grid_size = grid_max - grid_min;

        let nx = ((grid_size.x / config.cell_size).ceil() as usize).max(1);
        let ny = ((grid_size.y / config.cell_size).ceil() as usize).max(1);
        let nz = ((grid_size.z / config.cell_size).ceil() as usize).max(1);
        let total_cells = nx * ny * nz;

        log::info!("  Grid dimensions: {}x{}x{} = {} cells", nx, ny, nz, total_cells);

        // First pass: count atoms per cell
        let inv_cell_size = 1.0 / config.cell_size;
        let mut cell_counts: Vec<u32> = vec![0; total_cells];

        for atom in protein.atoms.iter() {
            let atom_radius = atom.element.vdw_radius() + config.probe_radius;
            let atom_min = atom.position - Vec3::splat(atom_radius + config.cell_size);
            let atom_max = atom.position + Vec3::splat(atom_radius + config.cell_size);

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
                            cell_counts[cell_idx] += 1;
                        }
                    }
                }
            }
        }

        // Calculate offsets (prefix sum)
        let mut offsets: Vec<u32> = vec![0; total_cells];
        let mut current_offset = 0u32;
        for i in 0..total_cells {
            offsets[i] = current_offset;
            current_offset += cell_counts[i];
        }

        let total_indices = current_offset as usize;
        log::info!("  Total atom-cell assignments: {}", total_indices);

        // Initialize grid cells with offsets
        let mut grid_cells: Vec<GridCellGPU> = Vec::with_capacity(total_cells);
        for i in 0..total_cells {
            grid_cells.push(GridCellGPU {
                offset: offsets[i],
                count: cell_counts[i],
            });
        }

        // Second pass: fill atom indices buffer
        let mut atom_indices_data: Vec<u32> = vec![0; total_indices.max(1)];
        let mut cell_fill_counts: Vec<u32> = vec![0; total_cells];

        for (atom_idx, atom) in protein.atoms.iter().enumerate() {
            let atom_radius = atom.element.vdw_radius() + config.probe_radius;
            let atom_min = atom.position - Vec3::splat(atom_radius + config.cell_size);
            let atom_max = atom.position + Vec3::splat(atom_radius + config.cell_size);

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
            label: Some("Raymarch Atom Buffer"),
            contents: bytemuck::cast_slice(&atoms),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let grid_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Raymarch Grid Buffer"),
            contents: bytemuck::cast_slice(&grid_cells),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let atom_indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Raymarch Atom Indices Buffer"),
            contents: bytemuck::cast_slice(&atom_indices_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 4. Update grid params with bounding box for ray acceleration
        let grid_params = GridParamsGPU {
            origin: grid_min.into(),
            cell_size: config.cell_size,
            dimensions: [nx as u32, ny as u32, nz as u32],
            atom_count: protein.atoms.len() as u32,
            inv_cell_size,
            probe_radius: config.probe_radius,
            smoothing_k: config.smoothing_k,
            _padding: 0.0,
            bbox_min: grid_min.into(),
            _padding2: 0.0,
            bbox_max: grid_max.into(),
            _padding3: 0.0,
        };

        queue.write_buffer(
            &self.grid_params_buffer,
            0,
            bytemuck::cast_slice(&[grid_params]),
        );

        // 5. Update material
        let material = SurfaceMaterialGPU {
            color: config.color,
            ambient: config.ambient,
            diffuse: config.diffuse,
            specular: config.specular,
            shininess: config.shininess,
            ao_strength: config.ao_strength,
        };

        queue.write_buffer(&self.material_buffer, 0, bytemuck::cast_slice(&[material]));

        // 6. Create bind group
        let data_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raymarch Data Bind Group"),
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
                    resource: self.material_buffer.as_entire_binding(),
                },
            ],
        });

        self.atom_buffer = Some(atom_buffer);
        self.grid_buffer = Some(grid_buffer);
        self.atom_indices_buffer = Some(atom_indices_buffer);
        self.data_bind_group = Some(data_bind_group);
        self.is_initialized = true;

        log::info!(
            "Raymarching surface prepared in {:?}",
            start.elapsed()
        );
        log::info!("  {} atoms, {} grid cells", protein.atoms.len(), total_cells);
        log::info!("  Ready for real-time rendering");

        Ok(())
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if !self.is_initialized || self.data_bind_group.is_none() {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(1, self.data_bind_group.as_ref().unwrap(), &[]);

        // Draw fullscreen quad (6 vertices, generated in shader)
        render_pass.draw(0..6, 0..1);
    }

    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    pub fn atom_count(&self) -> usize {
        self.atom_count
    }
}
