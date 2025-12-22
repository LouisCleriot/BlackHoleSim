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
pub struct SphericalState {
    pub r: f32,
    pub theta: f32,
    pub phi: f32,
    pub dr: f32,
    pub dtheta: f32,
    pub dphi: f32,
    pub flight_time: f32,
}

#[derive(Clone, Copy, Debug)]
struct SphericalDerivatives {
    pub ddr: f32,
    pub ddtheta: f32,
    pub ddphi: f32,
}

fn geodesic_derivatives(state: &SphericalState, gm: f32) -> SphericalDerivatives {
    let r = state.r;
    let sin_theta = state.theta.sin();
    let cos_theta = state.theta.cos();

    let cot_theta = if sin_theta.abs() > 1e-6 {
        cos_theta / sin_theta
    } else {
        0.0
    };

    let dr = state.dr;
    let dtheta = state.dtheta;
    let dphi = state.dphi;

    let dtheta_sq = dtheta * dtheta;
    let dphi_sq = dphi * dphi;
    let sin_theta_sq = sin_theta * sin_theta;

    let gr_term = r * (dtheta_sq + sin_theta_sq * dphi_sq);
    let ddr = -gm / (r * r) + gr_term - (3.0 * gm / (C * C)) * (dtheta_sq + sin_theta_sq * dphi_sq);

    let ddtheta = sin_theta * cos_theta * dphi_sq - (2.0 / r) * dr * dtheta;
    let ddphi = -(2.0 / r) * dr * dphi - 2.0 * cot_theta * dtheta * dphi;

    SphericalDerivatives {
        ddr,
        ddtheta,
        ddphi,
    }
}

pub fn integrate_rk4_step(state: SphericalState, dt: f32, gm: f32) -> SphericalState {
    // k1
    let k1_derivs = geodesic_derivatives(&state, gm);
    let k1_state = SphericalState {
        r: state.dr,
        theta: state.dtheta,
        phi: state.dphi,
        dr: k1_derivs.ddr,
        dtheta: k1_derivs.ddtheta,
        dphi: k1_derivs.ddphi,
        flight_time: 0.0,
    };

    // k2
    let mid_state_1 = SphericalState {
        r: state.r + k1_state.r * dt * 0.5,
        theta: state.theta + k1_state.theta * dt * 0.5,
        phi: state.phi + k1_state.phi * dt * 0.5,
        dr: state.dr + k1_state.dr * dt * 0.5,
        dtheta: state.dtheta + k1_state.dtheta * dt * 0.5,
        dphi: state.dphi + k1_state.dphi * dt * 0.5,
        flight_time: 0.0,
    };
    let k2_derivs = geodesic_derivatives(&mid_state_1, gm);
    let k2_state = SphericalState {
        r: mid_state_1.dr,
        theta: mid_state_1.dtheta,
        phi: mid_state_1.dphi,
        dr: k2_derivs.ddr,
        dtheta: k2_derivs.ddtheta,
        dphi: k2_derivs.ddphi,
        flight_time: 0.0,
    };

    // k3
    let mid_state_2 = SphericalState {
        r: state.r + k2_state.r * dt * 0.5,
        theta: state.theta + k2_state.theta * dt * 0.5,
        phi: state.phi + k2_state.phi * dt * 0.5,
        dr: state.dr + k2_state.dr * dt * 0.5,
        dtheta: state.dtheta + k2_state.dtheta * dt * 0.5,
        dphi: state.dphi + k2_state.dphi * dt * 0.5,
        flight_time: 0.0,
    };
    let k3_derivs = geodesic_derivatives(&mid_state_2, gm);
    let k3_state = SphericalState {
        r: mid_state_2.dr,
        theta: mid_state_2.dtheta,
        phi: mid_state_2.dphi,
        dr: k3_derivs.ddr,
        dtheta: k3_derivs.ddtheta,
        dphi: k3_derivs.ddphi,
        flight_time: 0.0,
    };

    // k4
    let end_state = SphericalState {
        r: state.r + k3_state.r * dt,
        theta: state.theta + k3_state.theta * dt,
        phi: state.phi + k3_state.phi * dt,
        dr: state.dr + k3_state.dr * dt,
        dtheta: state.dtheta + k3_state.dtheta * dt,
        dphi: state.dphi + k3_state.dphi * dt,
        flight_time: 0.0,
    };
    let k4_derivs = geodesic_derivatives(&end_state, gm);
    let k4_state = SphericalState {
        r: end_state.dr,
        theta: end_state.dtheta,
        phi: end_state.dphi,
        dr: k4_derivs.ddr,
        dtheta: k4_derivs.ddtheta,
        dphi: k4_derivs.ddphi,
        flight_time: 0.0,
    };

    let final_r =
        state.r + (dt / 6.0) * (k1_state.r + 2.0 * k2_state.r + 2.0 * k3_state.r + k4_state.r);
    let final_theta = state.theta
        + (dt / 6.0)
            * (k1_state.theta + 2.0 * k2_state.theta + 2.0 * k3_state.theta + k4_state.theta);
    let final_phi = state.phi
        + (dt / 6.0) * (k1_state.phi + 2.0 * k2_state.phi + 2.0 * k3_state.phi + k4_state.phi);
    let final_dr =
        state.dr + (dt / 6.0) * (k1_state.dr + 2.0 * k2_state.dr + 2.0 * k3_state.dr + k4_state.dr);
    let final_dtheta = state.dtheta
        + (dt / 6.0)
            * (k1_state.dtheta + 2.0 * k2_state.dtheta + 2.0 * k3_state.dtheta + k4_state.dtheta);
    let final_dphi = state.dphi
        + (dt / 6.0) * (k1_state.dphi + 2.0 * k2_state.dphi + 2.0 * k3_state.dphi + k4_state.dphi);

    SphericalState {
        r: final_r,
        theta: final_theta,
        phi: final_phi,
        dr: final_dr,
        dtheta: final_dtheta,
        dphi: final_dphi,
        flight_time: state.flight_time + dt,
    }
}

pub fn calculate_adaptive_dt(state: &SphericalState, schwarzschild_radius: f32) -> f32 {
    const MIN_DT: f32 = 0.00002;
    const MAX_DT: f32 = 0.05;
    const START_ADAPT_R_FACTOR: f32 = 5.0;
    const END_ADAPT_R_FACTOR: f32 = 50.0;

    let min_r = START_ADAPT_R_FACTOR * schwarzschild_radius;
    let max_r = END_ADAPT_R_FACTOR * schwarzschild_radius;

    if state.r <= min_r {
        return MIN_DT;
    }
    if state.r >= max_r {
        return MAX_DT;
    }

    let alpha = (state.r - min_r) / (max_r - min_r);
    let dt = MIN_DT + alpha * (MAX_DT - MIN_DT);
    dt.clamp(MIN_DT, MAX_DT)
}

pub fn cartesian_to_spherical_state(
    cart_pos: Vec3,
    cart_vel: Vec3,
    black_hole_pos: Vec3,
) -> SphericalState {
    let rel_pos = cart_pos - black_hole_pos;
    let r = rel_pos.length();

    if r < 1e-6 {
        return SphericalState {
            r: 0.0,
            theta: 0.0,
            phi: 0.0,
            dr: 0.0,
            dtheta: 0.0,
            dphi: 0.0,
            flight_time: 0.0,
        };
    }

    let theta = (rel_pos.z / r).acos();
    let phi = rel_pos.y.atan2(rel_pos.x);

    let dr = rel_pos.dot(cart_vel) / r;

    let rho_sq = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y;

    let dphi = if rho_sq > 1e-6 {
        (rel_pos.x * cart_vel.y - rel_pos.y * cart_vel.x) / rho_sq
    } else {
        0.0
    };

    let dtheta = if rho_sq > 1e-6 {
        let rho = rho_sq.sqrt();
        (rel_pos.z * dr - cart_vel.z * r) / (r * rho)
    } else {
        0.0
    };

    SphericalState {
        r,
        theta,
        phi,
        dr,
        dtheta,
        dphi,
        flight_time: 0.0,
    }
}

pub fn spherical_to_cartesian_pos(spherical_state: &SphericalState, black_hole_pos: Vec3) -> Vec3 {
    let r = spherical_state.r;
    let theta = spherical_state.theta;
    let phi = spherical_state.phi;

    let x = r * theta.sin() * phi.cos();
    let y = r * theta.sin() * phi.sin();
    let z = r * theta.cos();

    black_hole_pos + Vec3::new(x, y, z)
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
    let total_shift = doppler_factor * grav_factor;

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
