use crate::physics::{
    BlackHole, C, calculate_adaptive_dt, calculate_relativistic_effects,
    cartesian_to_spherical_state, integrate_rk4_step, kelvin_to_rgb, spherical_to_cartesian_pos,
};
use macroquad::prelude::*;
use rayon::prelude::*;

pub struct Renderer {
    // Orbit Camera
    pub camera_angle_h: f32,
    pub camera_angle_v: f32,
    pub camera_distance: f32,

    // FPV Camera
    pub fpv_mode: bool,
    pub fpv_pos: Vec3,
    pub fpv_yaw: f32,
    pub fpv_pitch: f32,

    pub last_mouse_pos: Option<Vec2>,
    pub show_stats: bool,
    pub global_time: f32,
}

impl Renderer {
    pub fn new() -> Self {
        Self {
            camera_angle_h: 0.1,
            camera_angle_v: 1.4,
            camera_distance: 600.0,

            fpv_mode: false,
            fpv_pos: vec3(0.0, 100.0, 600.0),
            fpv_yaw: -90.0f32.to_radians(),
            fpv_pitch: 0.0,

            last_mouse_pos: None,
            show_stats: true,
            global_time: 0.0,
        }
    }

    pub fn get_camera(&self) -> Camera3D {
        if self.fpv_mode {
            let front = vec3(
                self.fpv_yaw.cos() * self.fpv_pitch.cos(),
                self.fpv_pitch.sin(),
                self.fpv_yaw.sin() * self.fpv_pitch.cos(),
            )
            .normalize();

            Camera3D {
                position: self.fpv_pos,
                target: self.fpv_pos + front,
                up: vec3(0.0, 1.0, 0.0),
                ..Default::default()
            }
        } else {
            let cam_x =
                self.camera_angle_h.cos() * self.camera_angle_v.sin() * self.camera_distance;
            let cam_y =
                self.camera_angle_h.sin() * self.camera_angle_v.sin() * self.camera_distance;
            let cam_z = self.camera_angle_v.cos() * self.camera_distance;

            Camera3D {
                position: vec3(cam_x, cam_y, cam_z),
                up: vec3(0.0, 0.0, 1.0),
                target: vec3(0.0, 0.0, 0.0),
                ..Default::default()
            }
        }
    }

    pub fn draw_black_hole(&self, black_hole: &BlackHole) {
        draw_sphere(
            black_hole.position,
            black_hole.schwarzschild_radius,
            None,
            BLACK,
        );
    }

    pub fn draw_beams(&self, beam_manager: &crate::light_beam::BeamManager) {
        for beam in &beam_manager.beams {
            if beam.path_history.len() < 2 {
                continue;
            }
            for i in 1..beam.path_history.len() {
                let start = beam.path_history[i - 1];
                let end = beam.path_history[i];
                let mut color = Color::new(0.9, 0.9, 1.0, 0.9);
                if beam.state != crate::light_beam::BeamState::Active {
                    color.a *= (i as f32 / beam.path_history.len() as f32).powf(0.5);
                }
                draw_line_3d(start, end, color);
            }
        }
    }

    pub fn draw_grid(&self, size: f32, step: f32) {
        let color = Color::new(0.3, 0.3, 0.4, 0.6);
        let n = (size / step) as i32;
        let h = size / 2.0;
        for i in 0..=n {
            let p = -h + i as f32 * step;
            draw_line_3d(vec3(p, -h, 0.0), vec3(p, h, 0.0), color);
            draw_line_3d(vec3(-h, p, 0.0), vec3(h, p, 0.0), color);
        }
    }

    pub fn draw_stats(&self, beam_manager: &crate::light_beam::BeamManager, fps: f32) {
        if !self.show_stats {
            return;
        }

        let mode_text = if self.fpv_mode {
            "MODE: FPV (WASD to Fly)"
        } else {
            "MODE: ORBIT (Mouse to Rotate)"
        };
        draw_text(mode_text, 10.0, 20.0, 20.0, WHITE);

        let active = beam_manager
            .beams
            .iter()
            .filter(|b| b.state == crate::light_beam::BeamState::Active)
            .count();
        draw_text(
            &format!("FPS: {:.0} | Active Beams: {}", fps, active),
            10.0,
            40.0,
            20.0,
            WHITE,
        );
    }

    pub fn handle_input(&mut self) {
        let dt = get_frame_time();

        if is_mouse_button_down(MouseButton::Left) {
            let m: Vec2 = mouse_position().into();
            if let Some(last) = self.last_mouse_pos {
                let d = m - last;

                if self.fpv_mode {
                    let sensitivity = 0.005;
                    self.fpv_yaw += d.x * sensitivity;
                    self.fpv_pitch -= d.y * sensitivity;
                    self.fpv_pitch = self.fpv_pitch.clamp(-1.5, 1.5);
                } else {
                    self.camera_angle_h -= d.x * 0.01;
                    self.camera_angle_v =
                        (self.camera_angle_v - d.y * 0.01).clamp(0.01, std::f32::consts::PI - 0.01);
                }
            }
            self.last_mouse_pos = Some(m);
        } else {
            self.last_mouse_pos = None;
        }

        if !self.fpv_mode {
            let scroll = mouse_wheel().1;
            if scroll.abs() > 0.1 {
                self.camera_distance =
                    (self.camera_distance * (1.0 - scroll * 0.1)).clamp(50.0, 2000.0);
            }
        }

        if self.fpv_mode {
            let speed = if is_key_down(KeyCode::LeftShift) {
                200.0
            } else {
                50.0
            };
            let move_speed = speed * dt;

            let front = vec3(
                self.fpv_yaw.cos() * self.fpv_pitch.cos(),
                self.fpv_pitch.sin(),
                self.fpv_yaw.sin() * self.fpv_pitch.cos(),
            )
            .normalize();

            let right = front.cross(vec3(0.0, 1.0, 0.0)).normalize();
            let up = vec3(0.0, 1.0, 0.0);

            if is_key_down(KeyCode::W) {
                self.fpv_pos += front * move_speed;
            }
            if is_key_down(KeyCode::S) {
                self.fpv_pos -= front * move_speed;
            }
            if is_key_down(KeyCode::A) {
                self.fpv_pos -= right * move_speed;
            }
            if is_key_down(KeyCode::D) {
                self.fpv_pos += right * move_speed;
            }
            if is_key_down(KeyCode::Q) {
                self.fpv_pos -= up * move_speed;
            }
            if is_key_down(KeyCode::E) {
                self.fpv_pos += up * move_speed;
            }
        }

        if is_key_pressed(KeyCode::S) {
            self.show_stats = !self.show_stats;
        }
    }
}

fn sample_skybox(dir: Vec3) -> Vec3 {
    let noise_scale = 4.0;
    let n1 = (dir.x * noise_scale + dir.y * noise_scale * 0.5).sin();
    let n2 = (dir.z * noise_scale * 1.5 + dir.x * 0.5).cos();
    let noise_val = (n1 + n2).abs() * 0.5;

    let mut color = Vec3::new(0.001, 0.001, 0.003);

    let band_intensity = (1.0 - dir.y.abs()).powf(4.0);
    if band_intensity > 0.1 {
        let galaxy_col = Vec3::new(0.2, 0.1, 0.3) * noise_val * band_intensity * 2.0;
        color += galaxy_col;
    }

    let phi = dir.z.atan2(dir.x);
    let theta = dir.y.asin();
    let density = 20.0;
    let thickness = 0.02;
    let grid = (phi * density).sin().abs() < thickness || (theta * density).sin().abs() < thickness;

    if grid {
        color += Vec3::new(0.05, 0.05, 0.1);
    }

    let perfect_align_dir = Vec3::new(0.0, -0.164, -0.986).normalize();
    let dot_a = dir.dot(perfect_align_dir);
    if dot_a > 0.990 {
        let intensity = ((dot_a - 0.990) / 0.010).powf(3.0) * 120.0;
        color += Vec3::new(1.0, 0.6, 0.2) * intensity;
    }

    let off_axis_dir = Vec3::new(0.3, 0.2, -0.9).normalize();
    let dot_b = dir.dot(off_axis_dir);
    if dot_b > 0.995 {
        let intensity = ((dot_b - 0.995) / 0.005).powf(2.0) * 100.0;
        color += Vec3::new(1.0, 0.1, 0.1) * intensity;
    }

    let blue_dir = Vec3::new(-0.4, -0.1, -0.8).normalize();
    let dot_c = dir.dot(blue_dir);
    if dot_c > 0.996 {
        let intensity = ((dot_c - 0.996) / 0.004).powf(2.0) * 80.0;
        color += Vec3::new(0.4, 0.7, 1.0) * intensity;
    }

    color
}

fn get_volumetric_density(pos: Vec3, rs: f32) -> f32 {
    let r = pos.length();
    let z = pos.z.abs();

    let isco = 3.0 * rs;
    let outer_edge = 12.0 * rs;

    if r > outer_edge {
        return 0.0;
    }

    let edge_sharpness = 2.0;
    let transition = (1.0 / (1.0 + (-(r - isco) * edge_sharpness).exp())).clamp(0.0, 1.0);

    if transition < 0.01 {
        return 0.0;
    }

    let scale_height = 0.05 * r;
    let vertical_density = (-z * z / (2.0 * scale_height * scale_height)).exp();

    let mid = (isco + outer_edge) * 0.5;
    let width = (outer_edge - isco) * 0.5;
    let radial_density = (1.0 - ((r - mid) / width).powi(2)).max(0.0);

    vertical_density * radial_density * transition
}

fn sample_volumetric_disk(
    pos: Vec3,
    ray_dir: Vec3,
    black_hole: &BlackHole,
    flight_time: f32,
    global_time: f32,
) -> (Vec3, f32) {
    let density = get_volumetric_density(pos, black_hole.schwarzschild_radius);

    if density <= 0.001 {
        return (Vec3::ZERO, 0.0);
    }

    let r = pos.length();
    let rs = black_hole.schwarzschild_radius;
    let isco = 3.0 * rs;

    let base_temp = 12000.0 * (isco / r).powf(1.5);

    let appearance = calculate_relativistic_effects(pos, ray_dir, black_hole, base_temp);

    let col_rgb = kelvin_to_rgb(appearance.observed_temperature);
    let mut color_vec = Vec3::new(col_rgb.r, col_rgb.g, col_rgb.b);

    let brightness = appearance.observed_intensity.min(50.0);
    color_vec *= brightness;

    let orbital_speed_ang = (black_hole.gm / (r * r * r)).sqrt();
    let emission_time = global_time - flight_time * 0.5;

    let angle = pos.y.atan2(pos.x);
    let phase = angle + orbital_speed_ang * emission_time * 50.0;

    let bands = (phase * 10.0).sin();
    let turbulence = if bands > 0.0 { 1.5 } else { 0.5 };

    color_vec *= turbulence;

    (color_vec, density)
}

fn random_noise(x: f32, y: f32) -> f32 {
    ((x * 12.9898 + y * 78.233).sin() * 43758.5453)
        .fract()
        .abs()
}

fn apply_bloom(hdr_buffer: &Vec<Vec3>, width: usize, height: usize) -> Vec<Vec3> {
    let mut bright_buffer = vec![Vec3::ZERO; width * height];
    let threshold = 1.2;
    for (i, &pixel) in hdr_buffer.iter().enumerate() {
        let brightness = pixel.dot(Vec3::new(0.2126, 0.7152, 0.0722));
        if brightness > threshold {
            bright_buffer[i] = pixel;
        }
    }

    let mut temp_buffer = bright_buffer.clone();
    let kernel_radius = 2;

    for y in 0..height {
        for x in 0..width {
            let mut sum = Vec3::ZERO;
            let mut count = 0.0;

            for k in -(kernel_radius as i32)..=(kernel_radius as i32) {
                let px = (x as i32 + k).clamp(0, width as i32 - 1) as usize;
                sum += bright_buffer[y * width + px];
                count += 1.0;
            }
            temp_buffer[y * width + x] = sum / count;
        }
    }

    bright_buffer = temp_buffer.clone();

    for x in 0..width {
        for y in 0..height {
            let mut sum = Vec3::ZERO;
            let mut count = 0.0;

            for k in -(kernel_radius as i32)..=(kernel_radius as i32) {
                let py = (y as i32 + k).clamp(0, height as i32 - 1) as usize;
                sum += bright_buffer[py * width + x];
                count += 1.0;
            }
            temp_buffer[y * width + x] = sum / count;
        }
    }

    let mut final_buffer = Vec::with_capacity(width * height);
    for i in 0..hdr_buffer.len() {
        final_buffer.push(hdr_buffer[i] + temp_buffer[i] * 0.8);
    }

    final_buffer
}

fn tone_map(color: Vec3) -> Color {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;

    let mapped = (color * (a * color + Vec3::splat(b)))
        / (color * (c * color + Vec3::splat(d)) + Vec3::splat(e));

    Color::new(
        mapped.x.clamp(0.0, 1.0),
        mapped.y.clamp(0.0, 1.0),
        mapped.z.clamp(0.0, 1.0),
        1.0,
    )
}

pub fn render_observer_view(
    black_hole: &BlackHole,
    camera_pos: Vec3,
    target: Vec3,
    width: usize,
    height: usize,
    global_time: f32,
) -> Image {
    let mut hdr_buffer = vec![Vec3::ZERO; width * height];

    let forward = (target - camera_pos).normalize();
    let up = vec3(0.0, 1.0, 0.0);
    let right = forward.cross(up).normalize();
    let real_up = right.cross(forward).normalize();
    let aspect = width as f32 / height as f32;

    hdr_buffer
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, pixel) in row.iter_mut().enumerate() {
                let u = (x as f32 / width as f32) * 2.0 - 1.0;
                let v = (y as f32 / height as f32) * 2.0 - 1.0;
                let uv_dir = (right * u * aspect + real_up * v + forward).normalize();

                let noise = random_noise(x as f32, y as f32);

                let velocity = uv_dir * C;
                let mut state =
                    cartesian_to_spherical_state(camera_pos, velocity, black_hole.position);
                let mut prev_pos = camera_pos;

                let mut accum_color = Vec3::ZERO;
                let mut accum_opacity = 0.0;

                let max_steps = 5000;

                for step in 0..max_steps {
                    if state.r <= black_hole.schwarzschild_radius {
                        break;
                    }

                    let curr_pos = spherical_to_cartesian_pos(&state, black_hole.position);
                    let dt_base = calculate_adaptive_dt(&state, black_hole.schwarzschild_radius);
                    let dist_from_plane = curr_pos.z.abs();

                    let in_dense_zone = dist_from_plane < black_hole.schwarzschild_radius * 4.0
                        && state.r < black_hole.schwarzschild_radius * 14.0;

                    let mut integration_dt = if in_dense_zone {
                        dt_base * 0.5
                    } else {
                        dt_base * 2.5
                    };

                    if step == 0 {
                        integration_dt *= 0.5 + noise;
                    }

                    if in_dense_zone && dist_from_plane < black_hole.schwarzschild_radius * 1.5 {
                        let step_dist = (curr_pos - prev_pos).length();
                        let (emission, density) = sample_volumetric_disk(
                            curr_pos,
                            uv_dir,
                            black_hole,
                            state.flight_time,
                            global_time,
                        );

                        if density > 0.0 {
                            let step_opacity = (density * step_dist * 0.8).min(1.0);
                            accum_color += emission * step_opacity * (1.0 - accum_opacity);
                            accum_opacity += step_opacity;
                            if accum_opacity >= 0.98 {
                                break;
                            }
                        }
                    }

                    if state.r > 2000.0 {
                        let sky_vec = sample_skybox(curr_pos.normalize());
                        accum_color += sky_vec * (1.0 - accum_opacity);
                        break;
                    }

                    state = integrate_rk4_step(state, integration_dt, black_hole.gm);
                    prev_pos = curr_pos;
                }

                *pixel = accum_color;
            }
        });

    let bloomed_buffer = apply_bloom(&hdr_buffer, width, height);

    let mut image = Image::gen_image_color(width as u16, height as u16, BLACK);
    for (i, &hdr_color) in bloomed_buffer.iter().enumerate() {
        let ldr_color = tone_map(hdr_color);
        let x = (i % width) as u32;
        let y = (i / width) as u32;
        image.set_pixel(x, y, ldr_color);
    }

    image
}
