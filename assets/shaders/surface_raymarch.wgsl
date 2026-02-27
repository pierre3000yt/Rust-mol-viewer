// Raymarching SDF Molecular Surface Shader
// Optimized version with bounding box acceleration
// Renders molecular surfaces in real-time using signed distance fields

// Camera uniform (same as other shaders)
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,  // Pre-computed inverse (avoids shader matrix inversion)
    position: vec3<f32>,
    _padding: f32,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Atom data for SDF calculation
struct Atom {
    position: vec3<f32>,
    radius: f32,  // VdW radius + probe radius (pre-computed)
};

// Acceleration grid cell - stores offset into atom_indices buffer
struct GridCell {
    offset: u32,   // Start index in atom_indices buffer
    count: u32,    // Number of atoms in this cell
};

// Grid parameters
struct GridParams {
    origin: vec3<f32>,
    cell_size: f32,
    dimensions: vec3<u32>,
    atom_count: u32,
    inv_cell_size: f32,
    probe_radius: f32,
    smoothing_k: f32,  // Smooth union parameter
    _padding: f32,
    // Bounding box for early ray termination
    bbox_min: vec3<f32>,
    _padding2: f32,
    bbox_max: vec3<f32>,
    _padding3: f32,
};

// Surface material
struct SurfaceMaterial {
    color: vec3<f32>,
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    ao_strength: f32,
};

@group(1) @binding(0)
var<storage, read> atoms: array<Atom>;

@group(1) @binding(1)
var<storage, read> grid: array<GridCell>;

@group(1) @binding(2)
var<storage, read> atom_indices: array<u32>;  // Linear buffer of atom indices

@group(1) @binding(3)
var<uniform> grid_params: GridParams;

@group(1) @binding(4)
var<uniform> material: SurfaceMaterial;

// Raymarching constants - optimized for performance
const MAX_STEPS: i32 = 48;       // Balanced steps for quality/performance
const MAX_DIST: f32 = 500.0;     // Maximum ray distance
const SURFACE_DIST: f32 = 0.15;  // Surface threshold (larger = faster but less precise)
const NORMAL_EPSILON: f32 = 0.15;

// Vertex output
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ray_origin: vec3<f32>,
    @location(1) ray_dir: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

// Fullscreen quad vertices (2 triangles)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Generate fullscreen quad
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );

    let pos = positions[vertex_index];

    var output: VertexOutput;
    output.clip_position = vec4<f32>(pos, 0.0, 1.0);
    output.uv = pos * 0.5 + 0.5;

    // Compute ray origin and direction from camera
    output.ray_origin = camera.position;

    // Calculate ray direction using PRE-COMPUTED inverse (avoids numerical issues)
    // Unproject near point (z=0 in wgpu NDC)
    let near_ndc = vec4<f32>(pos.x, pos.y, 0.0, 1.0);
    var near_world = camera.inv_view_proj * near_ndc;
    // Safe perspective divide
    let near_w = max(abs(near_world.w), 0.0001);
    near_world = near_world / near_w;

    // Unproject far point (z=1 in wgpu NDC)
    let far_ndc = vec4<f32>(pos.x, pos.y, 1.0, 1.0);
    var far_world = camera.inv_view_proj * far_ndc;
    let far_w = max(abs(far_world.w), 0.0001);
    far_world = far_world / far_w;

    // Ray direction is from near to far
    let ray_dir = far_world.xyz - near_world.xyz;
    let ray_len = length(ray_dir);
    output.ray_dir = select(vec3<f32>(0.0, 0.0, -1.0), ray_dir / ray_len, ray_len > 0.0001);

    return output;
}

// Smooth minimum for blending SDFs (creates smooth surface)
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

// Convert world position to grid cell index
fn world_to_grid(p: vec3<f32>) -> vec3<i32> {
    let local = (p - grid_params.origin) * grid_params.inv_cell_size;
    return vec3<i32>(floor(local));
}

// Get grid cell index (flattened)
fn grid_index(cell: vec3<i32>) -> u32 {
    let dims = grid_params.dimensions;
    return u32(cell.z) * dims.x * dims.y + u32(cell.y) * dims.x + u32(cell.x);
}

// Check if cell is within grid bounds
fn is_valid_cell(cell: vec3<i32>) -> bool {
    let dims = vec3<i32>(grid_params.dimensions);
    return cell.x >= 0 && cell.y >= 0 && cell.z >= 0 &&
           cell.x < dims.x && cell.y < dims.y && cell.z < dims.z;
}

// Compute SDF at point using acceleration grid
fn sdf_molecule(p: vec3<f32>) -> f32 {
    var d = MAX_DIST;
    let cell = world_to_grid(p);
    let k = grid_params.smoothing_k;

    // Check current cell and 26 neighbors (3x3x3 neighborhood)
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighbor = cell + vec3<i32>(dx, dy, dz);

                if (!is_valid_cell(neighbor)) {
                    continue;
                }

                let idx = grid_index(neighbor);
                let grid_cell = grid[idx];

                // Process atoms in this cell using linear index buffer
                let start = grid_cell.offset;
                let end = start + grid_cell.count;

                for (var i = start; i < end; i++) {
                    let atom_idx = atom_indices[i];
                    let atom = atoms[atom_idx];

                    let sphere_dist = length(p - atom.position) - atom.radius;

                    // Use smooth minimum for organic-looking surface
                    d = smin(d, sphere_dist, k);
                }
            }
        }
    }

    return d;
}

// Compute SDF gradient for normal calculation
// Tetrahedron technique from Inigo Quilez - 4 SDF calls, best quality
// https://iquilezles.org/articles/normalsSDF/
fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = NORMAL_EPSILON;
    // Tetrahedron vertices
    let k1 = vec3<f32>( 1.0, -1.0, -1.0);
    let k2 = vec3<f32>(-1.0, -1.0,  1.0);
    let k3 = vec3<f32>(-1.0,  1.0, -1.0);
    let k4 = vec3<f32>( 1.0,  1.0,  1.0);

    let n = k1 * sdf_molecule(p + k1 * e) +
            k2 * sdf_molecule(p + k2 * e) +
            k3 * sdf_molecule(p + k3 * e) +
            k4 * sdf_molecule(p + k4 * e);

    // Safe normalize to avoid NaN
    let len = length(n);
    return select(vec3<f32>(0.0, 1.0, 0.0), n / len, len > 0.0001);
}

// Ray-box intersection for bounding box acceleration
// Safe version that handles near-zero ray directions
fn intersect_box(ro: vec3<f32>, rd: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    // Safe inverse with epsilon to avoid division by zero
    let eps = vec3<f32>(0.0001);
    let safe_rd = select(rd, eps, abs(rd) < eps);
    let inv_rd = 1.0 / safe_rd;

    let t1 = (box_min - ro) * inv_rd;
    let t2 = (box_max - ro) * inv_rd;

    let tmin = min(t1, t2);
    let tmax = max(t1, t2);

    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);

    return vec2<f32>(t_near, t_far);
}

// Raymarching with sphere tracing and bounding box acceleration
fn raymarch(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    // First intersect with bounding box for early termination
    let bbox = intersect_box(ro, rd, grid_params.bbox_min, grid_params.bbox_max);

    // If ray misses bounding box entirely, no hit
    if (bbox.x > bbox.y || bbox.y < 0.0) {
        return -1.0;
    }

    // Start raymarching from where ray enters bounding box (or 0 if inside)
    var t = max(bbox.x, 0.0);
    let t_max = min(bbox.y, MAX_DIST);

    for (var i = 0; i < MAX_STEPS; i++) {
        let p = ro + rd * t;
        let d = sdf_molecule(p);

        if (d < SURFACE_DIST) {
            return t;
        }

        // Adaptive step size - smaller near surface
        t += max(d * 0.9, SURFACE_DIST * 0.5);

        if (t > t_max) {
            break;
        }
    }

    return -1.0;  // No hit
}

// Fast approximate ambient occlusion (single sample)
// Much cheaper than multi-sample AO
fn calc_ao_fast(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let ao_dist = 0.5;
    let d = sdf_molecule(p + n * ao_dist);
    return clamp(d / ao_dist, 0.0, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ro = in.ray_origin;
    let rd = normalize(in.ray_dir);

    // Raymarch to find surface
    let t = raymarch(ro, rd);

    if (t < 0.0) {
        // No hit - discard fragment (transparent background)
        discard;
    }

    // Hit point and normal
    let p = ro + rd * t;
    let n = calc_normal(p);

    // Lighting
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let view_dir = -rd;
    let half_dir = normalize(light_dir + view_dir);

    // Phong shading components
    let ambient = material.ambient;
    let diff = max(dot(n, light_dir), 0.0) * material.diffuse;
    let spec = pow(max(dot(n, half_dir), 0.0), material.shininess) * material.specular;

    // Fast ambient occlusion (single sample)
    let ao = calc_ao_fast(p, n);
    let ao_factor = mix(1.0, ao, material.ao_strength);

    // Final color
    let lighting = (ambient + diff) * ao_factor + spec;
    let color = material.color * lighting;

    // Gamma correction
    let gamma_color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma_color, 1.0);
}
