use macroquad::prelude::*;

pub const C: f32 = 150.0;
pub const G: f32 = 10.0;

#[derive(Clone, Copy, Debug)]
pub struct BlackHole {
    pub position: Vec3,
    pub schwarzschild_radius: f32,
    pub gm: f32,
}

impl BlackHole {
    pub fn new(position: Vec3, mass: f32) -> Self {
        let schwarzschild_radius = (2.0 * G * mass) / (C * C);
        Self {
            position,
            schwarzschild_radius,
            gm: G * mass,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CartesianState {
    pub pos: Vec3,
    pub mom: Vec3,
    pub time: f32,
    pub flight_time: f32,
}

#[derive(Clone, Copy, Debug)]
struct StateDerivatives {
    pub d_pos: Vec3,
    pub d_mom: Vec3,
    pub d_time: f32,
}

fn hamiltonian_derivatives(state: &CartesianState, rs: f32) -> StateDerivatives {
    let r = state.pos.length();
    
    if r < rs * 0.1 {
        return StateDerivatives {
            d_pos: Vec3::ZERO,
            d_mom: Vec3::ZERO,
            d_time: 0.0,
        };
    }

    let n = state.pos / r;
    let h_factor = 1.0 - rs / r;
    
    let p_dot_n = state.mom.dot(n);
    let p_sq = state.mom.length_squared();
    
    let metric_term = p_sq - (rs/r) * p_dot_n * p_dot_n;
    let energy_sq = h_factor * metric_term.max(0.0);
    let energy = energy_sq.sqrt();

    let d_pos = state.mom - (rs / r) * p_dot_n * n;

    let common_factor = 0.5 * rs / (r * r);
    let term1 = 2.0 * p_dot_n * state.mom;
    
    let term2 = 3.0 * (p_dot_n * p_dot_n) * n;
    
    let term3 = (energy * energy) / (h_factor * h_factor) * n;
    
    let d_mom = common_factor * (term1 - term2 - term3);

    let d_time = energy / h_factor;

    StateDerivatives {
        d_pos,
        d_mom,
        d_time,
    }
}

pub fn integrate_rk4_step(state: CartesianState, dt: f32, rs: f32) -> CartesianState {
    let k1 = hamiltonian_derivatives(&state, rs);
    
    let mid_pos_1 = state.pos + k1.d_pos * dt * 0.5;
    let mid_mom_1 = state.mom + k1.d_mom * dt * 0.5;
    let mid_time_1 = state.time + k1.d_time * dt * 0.5;
    let k2_state = CartesianState { pos: mid_pos_1, mom: mid_mom_1, time: mid_time_1, flight_time: 0.0 };
    let k2 = hamiltonian_derivatives(&k2_state, rs);

    let mid_pos_2 = state.pos + k2.d_pos * dt * 0.5;
    let mid_mom_2 = state.mom + k2.d_mom * dt * 0.5;
    let mid_time_2 = state.time + k2.d_time * dt * 0.5;
    let k3_state = CartesianState { pos: mid_pos_2, mom: mid_mom_2, time: mid_time_2, flight_time: 0.0 };
    let k3 = hamiltonian_derivatives(&k3_state, rs);

    let end_pos = state.pos + k3.d_pos * dt;
    let end_mom = state.mom + k3.d_mom * dt;
    let end_time = state.time + k3.d_time * dt;
    let k4_state = CartesianState { pos: end_pos, mom: end_mom, time: end_time, flight_time: 0.0 };
    let k4 = hamiltonian_derivatives(&k4_state, rs);

    let final_pos = state.pos + (dt / 6.0) * (k1.d_pos + 2.0 * k2.d_pos + 2.0 * k3.d_pos + k4.d_pos);
    let final_mom = state.mom + (dt / 6.0) * (k1.d_mom + 2.0 * k2.d_mom + 2.0 * k3.d_mom + k4.d_mom);
    let final_time = state.time + (dt / 6.0) * (k1.d_time + 2.0 * k2.d_time + 2.0 * k3.d_time + k4.d_time);

    CartesianState {
        pos: final_pos,
        mom: final_mom,
        time: final_time,
        flight_time: state.flight_time + dt,
    }
}

pub fn calculate_adaptive_dt(state: &CartesianState, rs: f32) -> f32 {
    const MIN_DT: f32 = 0.000005; 
    const MAX_DT: f32 = 0.1;
    
    let r = state.pos.length();
    
    let photon_sphere = 1.5 * rs;
    
    if r < 3.0 * rs && r > rs {
        let dist_to_ps = (r - photon_sphere).abs();
        let factor = (dist_to_ps / rs).clamp(0.01, 1.0);
        return MIN_DT + factor * 0.005;
    }
    
    if r <= 5.0 * rs {
        return MIN_DT * 10.0;
    }

    let alpha = (r - 5.0 * rs) / (50.0 * rs);
    (MIN_DT * 20.0 + alpha * MAX_DT).clamp(MIN_DT, MAX_DT)
}

pub fn init_cartesian_state(
    camera_pos: Vec3,
    camera_dir: Vec3, 
    black_hole_pos: Vec3,
    rs: f32,
) -> CartesianState {
    let rel_pos = camera_pos - black_hole_pos;
    let r = rel_pos.length();
    
    let vel = camera_dir * C;
    let n = rel_pos.normalize();
    let v_dot_n = vel.dot(n);
    let factor = (rs / r) / (1.0 - rs / r);
    
    let mom = vel + n * (v_dot_n * factor);

    CartesianState {
        pos: rel_pos,
        mom,
        time: 0.0,
        flight_time: 0.0,
    }
}

pub struct RelativisticAppearance {
    pub observed_temperature: f32,
    pub observed_intensity: f32,
}

pub fn calculate_relativistic_effects(
    hit_pos: Vec3,
    ray_dir: Vec3, 
    black_hole: &BlackHole,
    base_temperature_kelvin: f32,
) -> RelativisticAppearance {
    let r = hit_pos.length();
    let rs = black_hole.schwarzschild_radius;

    let orbital_speed = (black_hole.gm / r).sqrt();
    let gas_dir = Vec3::Y.cross(hit_pos).normalize();

    let beta = (orbital_speed / C).min(0.99);
    let gamma = 1.0 / (1.0 - beta * beta).sqrt();

    let cos_theta = gas_dir.dot(-ray_dir);
    let doppler_factor = 1.0 / (gamma * (1.0 - beta * cos_theta));

    let grav_factor = (1.0 - rs / r).max(0.0).sqrt();
    
    let total_shift = (doppler_factor * grav_factor).max(0.1);

    let observed_temperature = base_temperature_kelvin * total_shift;
    let observed_intensity = total_shift.powf(4.0);

    RelativisticAppearance {
        observed_temperature,
        observed_intensity,
    }
}

pub fn kelvin_to_rgb(temp: f32) -> Color {
    let t = temp / 100.0;
    let r = if t <= 66.0 {
        1.0
    } else {
        3.29 * (t - 60.0).powf(-0.133)
    };
    let g = if t <= 66.0 {
        0.39 * t.ln() - 0.63
    } else {
        2.88 * (t - 60.0).powf(-0.075)
    };
    let b = if t >= 66.0 {
        1.0
    } else {
        if t <= 19.0 {
            0.0
        } else {
            0.54 * (t - 10.0).ln() - 1.19
        }
    };
    Color::new(r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0), 1.0)
}