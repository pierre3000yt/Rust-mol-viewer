// Raymarched Spheres Shader - Individual Atom Primitives
// Uses analytic ray-sphere intersection (not iterative SDF)
// Much faster than surface SDF approach
// Based on Inigo Quilez primitives techniques

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position: vec3<f32>,
    _padding: f32,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Atom data with color
struct Atom {
    position: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    _padding: f32,
};

// Grid cell for spatial acceleration
struct GridCell {
    offset: u32,
    count: u32,
};

// Grid parameters
struct GridParams {
    origin: vec3<f32>,
    cell_size: f32,
    dimensions: vec3<u32>,
    atom_count: u32,
    inv_cell_size: f32,
    max_radius: f32,  // Largest atom radius for conservative bounds
    _padding1: f32,
    _padding2: f32,
};

// Lighting parameters
struct LightParams {
    direction: vec3<f32>,
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    _padding: f32,
};

@group(1) @binding(0)
var<storage, read> atoms: array<Atom>;

@group(1) @binding(1)
var<storage, read> grid: array<GridCell>;

@group(1) @binding(2)
var<storage, read> atom_indices: array<u32>;

@group(1) @binding(3)
var<uniform> grid_params: GridParams;

@group(1) @binding(4)
var<uniform> light: LightParams;

// Vertex output
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ray_origin: vec3<f32>,
    @location(1) ray_dir: vec3<f32>,
};

// Fullscreen triangle (more efficient than quad)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen triangle that covers the entire screen
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );

    let pos = positions[vertex_index];

    var output: VertexOutput;
    output.clip_position = vec4<f32>(pos, 0.0, 1.0);
    output.ray_origin = camera.position;

    // Unproject using pre-computed inverse
    let near_ndc = vec4<f32>(pos.x, pos.y, 0.0, 1.0);
    var near_world = camera.inv_view_proj * near_ndc;
    near_world = near_world / near_world.w;

    let far_ndc = vec4<f32>(pos.x, pos.y, 1.0, 1.0);
    var far_world = camera.inv_view_proj * far_ndc;
    far_world = far_world / far_world.w;

    output.ray_dir = normalize(far_world.xyz - near_world.xyz);

    return output;
}

// Analytic ray-sphere intersection
// Returns t (distance along ray) or -1.0 if no hit
fn ray_sphere_intersect(ro: vec3<f32>, rd: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    let oc = ro - center;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - radius * radius;
    let h = b * b - c;

    if (h < 0.0) {
        return -1.0;  // No intersection
    }

    let t = -b - sqrt(h);
    if (t < 0.0) {
        // Inside sphere or behind, try other root
        let t2 = -b + sqrt(h);
        if (t2 < 0.0) {
            return -1.0;
        }
        return t2;
    }
    return t;
}

// Get grid cell from world position
fn world_to_cell(p: vec3<f32>) -> vec3<i32> {
    let local = (p - grid_params.origin) * grid_params.inv_cell_size;
    return vec3<i32>(floor(local));
}

// Check if cell is valid
fn is_valid_cell(cell: vec3<i32>) -> bool {
    let dims = vec3<i32>(grid_params.dimensions);
    return cell.x >= 0 && cell.y >= 0 && cell.z >= 0 &&
           cell.x < dims.x && cell.y < dims.y && cell.z < dims.z;
}

// Get flat index for cell
fn cell_index(cell: vec3<i32>) -> u32 {
    let dims = grid_params.dimensions;
    return u32(cell.z) * dims.x * dims.y + u32(cell.y) * dims.x + u32(cell.x);
}

// Hit result
struct HitResult {
    hit: bool,
    t: f32,
    position: vec3<f32>,
    normal: vec3<f32>,
    color: vec3<f32>,
};

// Find closest sphere intersection using grid acceleration
fn trace_spheres(ro: vec3<f32>, rd: vec3<f32>, max_t: f32) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = max_t;

    // Calculate ray bounds in grid space
    let grid_min = grid_params.origin;
    let grid_max = grid_min + vec3<f32>(grid_params.dimensions) * grid_params.cell_size;

    // Intersect ray with grid bounds
    let eps = vec3<f32>(0.0001);
    let safe_rd = select(rd, eps, abs(rd) < eps);
    let inv_rd = 1.0 / safe_rd;

    let t1 = (grid_min - ro) * inv_rd;
    let t2 = (grid_max - ro) * inv_rd;
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);

    var t_enter = max(max(tmin.x, tmin.y), tmin.z);
    let t_exit = min(min(tmax.x, tmax.y), tmax.z);

    if (t_enter > t_exit || t_exit < 0.0) {
        return result;  // Ray misses grid entirely
    }

    t_enter = max(t_enter, 0.0);

    // DDA grid traversal
    let start_pos = ro + rd * t_enter;
    var cell = world_to_cell(start_pos);

    // Clamp to grid bounds
    let dims = vec3<i32>(grid_params.dimensions);
    cell = clamp(cell, vec3<i32>(0), dims - vec3<i32>(1));

    // Step direction
    let step = vec3<i32>(sign(rd));

    // Distance to next cell boundary
    let cell_min = grid_params.origin + vec3<f32>(cell) * grid_params.cell_size;
    let cell_max = cell_min + vec3<f32>(grid_params.cell_size);

    let t_delta = abs(vec3<f32>(grid_params.cell_size) * inv_rd);

    var t_next: vec3<f32>;
    t_next.x = select((cell_max.x - ro.x) * inv_rd.x, (cell_min.x - ro.x) * inv_rd.x, rd.x < 0.0);
    t_next.y = select((cell_max.y - ro.y) * inv_rd.y, (cell_min.y - ro.y) * inv_rd.y, rd.y < 0.0);
    t_next.z = select((cell_max.z - ro.z) * inv_rd.z, (cell_min.z - ro.z) * inv_rd.z, rd.z < 0.0);

    // Traverse grid
    let max_steps = 64;
    for (var i = 0; i < max_steps; i++) {
        if (!is_valid_cell(cell)) {
            break;
        }

        // Check atoms in current cell
        let idx = cell_index(cell);
        let grid_cell = grid[idx];

        for (var j = grid_cell.offset; j < grid_cell.offset + grid_cell.count; j++) {
            let atom_idx = atom_indices[j];
            let atom = atoms[atom_idx];

            let t = ray_sphere_intersect(ro, rd, atom.position, atom.radius);

            if (t > 0.0 && t < result.t) {
                result.hit = true;
                result.t = t;
                result.position = ro + rd * t;
                result.normal = normalize(result.position - atom.position);
                result.color = atom.color;
            }
        }

        // If we found a hit in this cell, we can stop
        // (atoms in further cells will be further away)
        if (result.hit && min(min(t_next.x, t_next.y), t_next.z) > result.t) {
            break;
        }

        // Move to next cell (DDA step)
        if (t_next.x < t_next.y && t_next.x < t_next.z) {
            cell.x += step.x;
            t_next.x += t_delta.x;
        } else if (t_next.y < t_next.z) {
            cell.y += step.y;
            t_next.y += t_delta.y;
        } else {
            cell.z += step.z;
            t_next.z += t_delta.z;
        }

        // Early exit if we've gone past max distance
        if (min(min(t_next.x, t_next.y), t_next.z) > max_t) {
            break;
        }
    }

    return result;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ro = in.ray_origin;
    let rd = normalize(in.ray_dir);

    // Trace spheres
    let hit = trace_spheres(ro, rd, 1000.0);

    if (!hit.hit) {
        discard;
    }

    // Lighting
    let light_dir = normalize(light.direction);
    let view_dir = -rd;
    let half_dir = normalize(light_dir + view_dir);

    let n_dot_l = max(dot(hit.normal, light_dir), 0.0);
    let n_dot_h = max(dot(hit.normal, half_dir), 0.0);

    let ambient = light.ambient;
    let diffuse = n_dot_l * light.diffuse;
    let specular = pow(n_dot_h, light.shininess) * light.specular;

    let color = hit.color * (ambient + diffuse) + vec3<f32>(specular);

    // Gamma correction
    let gamma_color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma_color, 1.0);
}
