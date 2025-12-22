use macroquad::math::Vec3;
use crate::physics::{BlackHole, calculate_adaptive_dt, integrate_rk4_step, cartesian_to_spherical_state, spherical_to_cartesian_pos, SphericalState, C};

#[derive(Clone, Copy, PartialEq)]
pub enum BeamState {
    Active,
    Absorbed,
    Escaped,
}

pub struct LightBeam {
    pub position: Vec3,
    pub physics_state: SphericalState,
    pub path_history: Vec<Vec3>, 
    pub state: BeamState,
}

const MIN_SAFE_RADIUS: f32 = 0.1; 
const ESCAPE_DISTANCE: f32 = 1000.0;

impl LightBeam {
    pub fn new(position: Vec3, initial_velocity: Vec3, black_hole: &BlackHole) -> Self {
        let mut path_history = Vec::with_capacity(5000);
        path_history.push(position);
        
        let physics_state = cartesian_to_spherical_state(position, initial_velocity, black_hole.position);

        Self {
            position,
            physics_state,
            path_history,
            state: BeamState::Active,
        }
    }
    
    pub fn update(&mut self, black_hole: &BlackHole, frame_dt: f32) {
        if self.state != BeamState::Active {
            return;
        }

        if self.physics_state.r < MIN_SAFE_RADIUS || self.position.length() > ESCAPE_DISTANCE {
            self.state = BeamState::Escaped;
            return;
        }
        if self.physics_state.r <= black_hole.schwarzschild_radius {
            self.state = BeamState::Absorbed;
            return;
        }

        let mut integrated_time = 0.0;
        while integrated_time < frame_dt {
            let adaptive_dt = calculate_adaptive_dt(&self.physics_state, black_hole.schwarzschild_radius);
            let step_dt = adaptive_dt.min(frame_dt - integrated_time);
            
            self.physics_state = integrate_rk4_step(self.physics_state, step_dt, black_hole.gm);
            integrated_time += step_dt;

            if self.physics_state.r <= black_hole.schwarzschild_radius {
                self.state = BeamState::Absorbed;
                self.physics_state.r = black_hole.schwarzschild_radius;
                break; 
            }
        }

        self.position = spherical_to_cartesian_pos(&self.physics_state, black_hole.position);
        self.path_history.push(self.position);
        
        if self.path_history.len() > 5000 {
            self.path_history.remove(0);
        }
    }
}

pub struct BeamManager {
    pub beams: Vec<LightBeam>,
}

impl BeamManager {
    pub fn new() -> Self {
        Self { beams: Vec::new() }
    }
    
    pub fn spawn_initial_beams(&mut self, black_hole: &BlackHole, grid_size: usize, plane_width: f32) {
        let distance = plane_width * 1.5;

        let mut spawn_plane = |fixed_coord: f32, axis: char, vel: Vec3| {
            for i in 0..grid_size {
                for j in 0..grid_size {
                    let u_frac = (i as f32 / (grid_size.saturating_sub(1)) as f32) - 0.5;
                    let v_frac = (j as f32 / (grid_size.saturating_sub(1)) as f32) - 0.5;
                    
                    let u = u_frac * plane_width;
                    let v = v_frac * plane_width;
                    
                    if u.abs() < 1.0 && v.abs() < 1.0 { continue; }

                    let pos = match axis {
                        'x' => Vec3::new(fixed_coord, u, v),
                        'z' => Vec3::new(u, v, fixed_coord),
                        _ => unreachable!(),
                    };
                    self.beams.push(LightBeam::new(pos, vel, black_hole));
                }
            }
        };

        spawn_plane(-distance, 'z', Vec3::new(0.0, 0.0, C));
        spawn_plane(distance, 'z', Vec3::new(0.0, 0.0, -C));
        spawn_plane(-distance, 'x', Vec3::new(C, 0.0, 0.0));
        spawn_plane(distance, 'x', Vec3::new(-C, 0.0, 0.0));
    }
    
    pub fn update(&mut self, black_hole: &BlackHole, dt: f32) {
        for beam in &mut self.beams {
            beam.update(black_hole, dt);
        }
    }
}
