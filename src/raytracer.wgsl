const C: f32 = 150.0;
const G: f32 = 10.0;
const PI: f32 = 3.14159265359;

struct Uniforms {
    camera_pos: vec3<f32>,
    _pad0: f32,
    camera_target: vec3<f32>,
    _pad1: f32,
    width: u32,
    height: u32,
    schwarzschild_radius: f32,
    gm: f32,
    global_time: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
}

struct CartesianState {
    pos: vec3<f32>,
    mom: vec3<f32>,
    time: f32,
    flight_time: f32,
}

struct StateDerivatives {
    d_pos: vec3<f32>,
    d_mom: vec3<f32>,
    d_time: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> output_buffer: array<vec4<f32>>;

fn hamiltonian_derivatives(state: CartesianState, rs: f32) -> StateDerivatives {
    let r = length(state.pos);

    if r < rs * 0.1 {
        var zero: StateDerivatives;
        return zero;
    }

    let n = state.pos / r;
    let h_factor = 1.0 - rs / r;

    let p_dot_n = dot(state.mom, n);
    let p_sq = dot(state.mom, state.mom);
    let metric_term = p_sq - (rs / r) * p_dot_n * p_dot_n;
    let energy_sq = h_factor * max(metric_term, 0.0);
    let energy = sqrt(energy_sq);

    let d_pos = state.mom - (rs / r) * p_dot_n * n;

    let common_factor = 0.5 * rs / (r * r);
    let term1 = 2.0 * p_dot_n * state.mom;
    let term2 = 3.0 * (p_dot_n * p_dot_n) * n;
    let term3 = (energy * energy) / (h_factor * h_factor) * n;

    let d_mom = common_factor * (term1 - term2 - term3);
    let d_time = energy / h_factor;

    var result: StateDerivatives;
    result.d_pos = d_pos;
    result.d_mom = d_mom;
    result.d_time = d_time;
    return result;
}

fn integrate_rk4_step(state: CartesianState, dt: f32, rs: f32) -> CartesianState {
    let k1 = hamiltonian_derivatives(state, rs);

    var s2 = state;
    s2.pos += k1.d_pos * dt * 0.5;
    s2.mom += k1.d_mom * dt * 0.5;
    s2.time += k1.d_time * dt * 0.5;
    let k2 = hamiltonian_derivatives(s2, rs);

    var s3 = state;
    s3.pos += k2.d_pos * dt * 0.5;
    s3.mom += k2.d_mom * dt * 0.5;
    s3.time += k2.d_time * dt * 0.5;
    let k3 = hamiltonian_derivatives(s3, rs);

    var s4 = state;
    s4.pos += k3.d_pos * dt;
    s4.mom += k3.d_mom * dt;
    s4.time += k3.d_time * dt;
    let k4 = hamiltonian_derivatives(s4, rs);

    var result: CartesianState;
    result.pos = state.pos + (dt / 6.0) * (k1.d_pos + 2.0 * k2.d_pos + 2.0 * k3.d_pos + k4.d_pos);
    result.mom = state.mom + (dt / 6.0) * (k1.d_mom + 2.0 * k2.d_mom + 2.0 * k3.d_mom + k4.d_mom);
    result.time = state.time + (dt / 6.0) * (k1.d_time + 2.0 * k2.d_time + 2.0 * k3.d_time + k4.d_time);
    result.flight_time = state.flight_time + dt;

    return result;
}

fn calculate_adaptive_dt(r: f32, rs: f32) -> f32 {
    let MIN_DT: f32 = 0.000005;
    let MAX_DT: f32 = 0.1;
    let ps = 1.5 * rs;

    if r < 3.0 * rs && r > rs {
        let dist_to_ps = abs(r - ps);
        let factor = clamp(dist_to_ps / rs, 0.01, 1.0);
        return MIN_DT + factor * 0.005;
    }

    if r <= 5.0 * rs {
        return MIN_DT * 10.0;
    }

    let alpha = (r - 5.0 * rs) / (50.0 * rs);
    return clamp(MIN_DT * 20.0 + alpha * MAX_DT, MIN_DT, MAX_DT);
}

fn init_cartesian_state(pos: vec3<f32>, vel: vec3<f32>, rs: f32) -> CartesianState {
    let r = length(pos);
    let n = normalize(pos);

    let v_dot_n = dot(vel, n);
    let factor = (rs / r) / (1.0 - rs / r);

    let mom = vel + n * (v_dot_n * factor);

    var state: CartesianState;
    state.pos = pos;
    state.mom = mom;
    state.time = 0.0;
    state.flight_time = 0.0;
    return state;
}

fn sample_skybox(dir: vec3<f32>) -> vec3<f32> {
    var color = vec3<f32>(0.0, 0.0, 0.0); // Deep space black

    let dir_blue = normalize(vec3<f32>(-0.2, 0.05, -1.0));
    let d_blue = max(dot(dir, dir_blue), 0.0);

    color += vec3<f32>(0.2, 0.5, 1.0) * pow(d_blue, 1000.0) * 5.0;
    color += vec3<f32>(0.1, 0.1, 0.4) * pow(d_blue, 100.0) * 0.5;
    let dir_red = normalize(vec3<f32>(0.25, -0.1, -1.0));
    let d_red = max(dot(dir, dir_red), 0.0);

    color += vec3<f32>(1.0, 0.3, 0.1) * pow(d_red, 800.0) * 4.0;
    color += vec3<f32>(0.4, 0.1, 0.05) * pow(d_red, 80.0) * 0.5;

    let dir_white = normalize(vec3<f32>(0.0, 0.3, -1.0));
    let d_white = max(dot(dir, dir_white), 0.0);

    color += vec3<f32>(1.0, 0.95, 0.8) * pow(d_white, 1200.0) * 6.0;

    return color;
}

fn get_volumetric_density(pos: vec3<f32>, rs: f32) -> f32 {
    let r = length(pos);
    let h = abs(pos.y);

    let isco = 3.0 * rs;

    var radial_density: f32;
    if r < isco {
        radial_density = pow(r / isco, 4.0) * 0.1;
    } else {
        radial_density = pow(isco / r, 3.0);
    }

    let scale_height = 0.015 * r;
    let vertical_density = exp(-h * h / (2.0 * scale_height * scale_height));

    let outer_fade = 1.0 - smoothstep(20.0 * rs, 30.0 * rs, r);

    return vertical_density * radial_density * 20.0 * outer_fade;
}

fn kelvin_to_rgb(temp: f32) -> vec3<f32> {
    let t = temp / 100.0;
    var r: f32; if t <= 66.0 { r = 1.0; } else { r = 3.29 * pow(t - 60.0, -0.133); }
    var g: f32; if t <= 66.0 { g = 0.39 * log(t) - 0.63; } else { g = 2.88 * pow(t - 60.0, -0.075); }
    var b: f32; if t >= 66.0 { b = 1.0; } else { if t <= 19.0 { b = 0.0; } else { b = 0.54 * log(t - 10.0) - 1.19; } }
    return vec3<f32>(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0));
}

fn calculate_relativistic_effects(hit_pos: vec3<f32>, ray_mom: vec3<f32>, rs: f32, gm: f32, base_temperature: f32) -> vec2<f32> {
    let r = length(hit_pos);
    let orbital_speed = sqrt(gm / r);
    let gas_dir = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), hit_pos));

    let ray_dir = normalize(ray_mom);
    let beta = min(orbital_speed / C, 0.99);
    let gamma = 1.0 / sqrt(1.0 - beta * beta);

    let cos_theta = dot(gas_dir, -ray_dir);
    let doppler_factor = 1.0 / (gamma * (1.0 - beta * cos_theta));

    let grav_factor = sqrt(max(1.0 - rs / r, 0.0));

    let total_shift = max(doppler_factor * grav_factor, 0.1);

    return vec2<f32>(base_temperature * total_shift, pow(total_shift, 4.0));
}

fn sample_volumetric_disk(pos: vec3<f32>, mom: vec3<f32>, rs: f32, gm: f32) -> vec4<f32> {
    let density = get_volumetric_density(pos, rs);

    if density <= 0.001 {
        return vec4<f32>(0.0);
    }

    let r = length(pos);
    let isco = 3.0 * rs;

    let inner_term = sqrt(isco / r);
    let boundary_term = max(1.0 - inner_term, 0.0);

    let temp_kelvin = 12000.0 * pow(isco / r, 0.75) * pow(boundary_term, 0.25);

    let effects = calculate_relativistic_effects(pos, mom, rs, gm, temp_kelvin);
    let brightness = effects.y;

    let color = kelvin_to_rgb(effects.x) * brightness;

    return vec4<f32>(color, density);
}

fn random_noise(x: f32, y: f32) -> f32 {
    return abs(fract(sin(x * 12.9898 + y * 78.233) * 43758.5453));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if x >= uniforms.width || y >= uniforms.height { return; }

    let idx = y * uniforms.width + x;

    let forward = normalize(uniforms.camera_target - uniforms.camera_pos);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(forward, up));
    let real_up = normalize(cross(right, forward));
    let aspect = f32(uniforms.width) / f32(uniforms.height);

    let u = (f32(x) / f32(uniforms.width)) * 2.0 - 1.0;
    let v = (f32(y) / f32(uniforms.height)) * 2.0 - 1.0;
    let uv_dir = normalize(right * u * aspect + real_up * v + forward);

    let noise = random_noise(f32(x), f32(y));
    let velocity = uv_dir * C;

    var state = init_cartesian_state(uniforms.camera_pos, velocity, uniforms.schwarzschild_radius);

    var accum_color = vec3<f32>(0.0);
    var accum_opacity = 0.0;

    let max_steps: u32 = 100000u;

    let rs = uniforms.schwarzschild_radius;
    let gm = uniforms.gm;
    var prev_pos = state.pos;

    for (var step: u32 = 0u; step < max_steps; step++) {
        let r = length(state.pos);
        if r <= rs * 1.01 { break; }
        if r > 2500.0 {
            accum_color += sample_skybox(normalize(state.pos)) * (1.0 - accum_opacity);
            break;
        }

        let dt_base = calculate_adaptive_dt(r, rs);

        var step_dt = dt_base;

        if r < rs * 5.0 {
            step_dt *= 0.02;
        } else if r < rs * 10.0 {
            step_dt *= 0.2;
        }

        if step == 0u { step_dt *= 0.5 + noise * 0.8; }

        let step_dist = length(state.pos - prev_pos);
        let sample = sample_volumetric_disk(state.pos, state.mom, rs, gm);

        if sample.w > 0.0 {
            let opacity = min(sample.w * step_dist * 0.5, 1.0);
            accum_color += sample.xyz * opacity * (1.0 - accum_opacity);
            accum_opacity += opacity;
            if accum_opacity >= 0.99 { break; }
        }

        prev_pos = state.pos;
        state = integrate_rk4_step(state, step_dt, rs);
    }

    output_buffer[idx] = vec4<f32>(accum_color, 1.0);
}