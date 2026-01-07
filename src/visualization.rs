use crate::gpu::{GpuContext, RaytracerUniforms};
use crate::physics::{
    BlackHole, calculate_adaptive_dt, calculate_relativistic_effects,
    init_cartesian_state, integrate_rk4_step, kelvin_to_rgb,
};
use macroquad::prelude::*;
use rayon::prelude::*;

pub struct Renderer {
    // FPV Camera
    pub fpv_pos: Vec3,
    pub fpv_yaw: f32,
    pub fpv_pitch: f32,

    pub last_mouse_pos: Option<Vec2>,
    pub global_time: f32,
}

impl Renderer {
    pub fn new() -> Self {
        Self {
            fpv_pos: vec3(0.0, 100.0, 1200.0),
            fpv_yaw: -90.0f32.to_radians(),
            fpv_pitch: 0.0,
            last_mouse_pos: None,
            global_time: 0.0,
        }
    }

    pub fn get_camera(&self) -> Camera3D {
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
    }

    pub fn handle_input(&mut self) {
        let dt = get_frame_time();

        // Mouse look
        if is_mouse_button_down(MouseButton::Left) {
            let m: Vec2 = mouse_position().into();
            if let Some(last) = self.last_mouse_pos {
                let d = m - last;
                let sensitivity = 0.005;
                self.fpv_yaw += d.x * sensitivity;
                self.fpv_pitch -= d.y * sensitivity;
                self.fpv_pitch = self.fpv_pitch.clamp(-1.5, 1.5);
            }
            self.last_mouse_pos = Some(m);
        } else {
            self.last_mouse_pos = None;
        }

        // Movement
        let speed = if is_key_down(KeyCode::LeftShift) {
            400.0
        } else {
            100.0
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
}

pub fn render_observer_view_gpu(
    gpu: &GpuContext,
    black_hole: &BlackHole,
    camera_pos: Vec3,
    target: Vec3,
    width: usize,
    height: usize,
    global_time: f32,
) -> Image {
    let uniforms = RaytracerUniforms {
        camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z],
        _pad0: 0.0,
        camera_target: [target.x, target.y, target.z],
        _pad1: 0.0,
        width: width as u32,
        height: height as u32,
        schwarzschild_radius: black_hole.schwarzschild_radius,
        gm: black_hole.gm,
        global_time,
        _pad2: 0.0,
        _pad3: 0.0,
        _pad4: 0.0,
    };

    let hdr_buffer = gpu.render(&uniforms);

    let hdr_vec3: Vec<Vec3> = hdr_buffer
        .iter()
        .map(|p| Vec3::new(p[0], p[1], p[2]))
        .collect();
    let bloomed = apply_bloom(&hdr_vec3, width, height);

    let mut image = Image::gen_image_color(width as u16, height as u16, BLACK);
    for (i, &hdr_color) in bloomed.iter().enumerate() {
        let ldr_color = tone_map(hdr_color);
        let x = (i % width) as u32;
        let y = (i / width) as u32;
        image.set_pixel(x, y, ldr_color);
    }

    image
}

pub fn render_observer_view_cpu(
    black_hole: &BlackHole,
    camera_pos: Vec3,
    target: Vec3,
    width: usize,
    height: usize,
    _global_time: f32,
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

                // Matches WGSL 'noise' used only for initial step offset
                let noise = random_noise(x as f32, y as f32);
                
                let mut state = init_cartesian_state(
                    camera_pos, 
                    uv_dir, 
                    black_hole.position, 
                    black_hole.schwarzschild_radius
                );
                
                let mut prev_pos = state.pos;
                let mut accum_color = Vec3::ZERO;
                let mut accum_opacity = 0.0;

                let max_steps = 10000; 
                let rs = black_hole.schwarzschild_radius;

                for step in 0..max_steps {
                    let r = state.pos.length();
                    
                    if r <= rs * 1.01 {
                        break;
                    }

                    if r > 2500.0 {
                        accum_color += sample_skybox(state.pos.normalize()) * (1.0 - accum_opacity);
                        break;
                    }

                    let dt_base = calculate_adaptive_dt(&state, rs);
                    let mut step_dt = dt_base;

                    if r < rs * 5.0 {
                        step_dt *= 0.02;
                    } else if r < rs * 10.0 {
                        step_dt *= 0.2;
                    }

                    if step == 0 { step_dt *= 0.5 + noise * 0.8; }

                    let step_dist = (state.pos - prev_pos).length();
                    
                    let (emission, density) = sample_volumetric_disk(
                        state.pos,
                        state.mom,
                        black_hole,
                    );

                    if density > 0.0 {
                        let opacity = (density * step_dist * 0.5).min(1.0);
                        accum_color += emission * opacity * (1.0 - accum_opacity);
                        accum_opacity += opacity;
                        if accum_opacity >= 0.99 {
                            break;
                        }
                    }

                    prev_pos = state.pos;
                    state = integrate_rk4_step(state, step_dt, rs);
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

fn get_volumetric_density(pos: Vec3, rs: f32) -> f32 {
    let r = pos.length();
    let h = pos.y.abs(); 

    let isco = 3.0 * rs;

    let radial_density = if r < isco {
        (r / isco).powf(4.0) * 0.1
    } else {
        (isco / r).powf(3.0)
    };

    let scale_height = 0.015 * r; 
    let vertical_density = (-h * h / (2.0 * scale_height * scale_height)).exp();

    let edge0 = 20.0 * rs;
    let edge1 = 30.0 * rs;
    let t = ((r - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    let smooth = t * t * (3.0 - 2.0 * t);
    let outer_fade = 1.0 - smooth;

    vertical_density * radial_density * 20.0 * outer_fade
}

fn sample_volumetric_disk(
    pos: Vec3,
    mom: Vec3,
    black_hole: &BlackHole,
) -> (Vec3, f32) {
    let density = get_volumetric_density(pos, black_hole.schwarzschild_radius);

    if density <= 0.001 {
        return (Vec3::ZERO, 0.0);
    }

    let r = pos.length();
    let rs = black_hole.schwarzschild_radius;
    let isco = 3.0 * rs;

    let inner_term = (isco / r).sqrt();
    let boundary_term = (1.0 - inner_term).max(0.0);

    let temp_kelvin = 12000.0 * (isco / r).powf(0.75) * boundary_term.powf(0.25); 

    let appearance = calculate_relativistic_effects(pos, mom.normalize(), black_hole, temp_kelvin);

    let col_rgb = kelvin_to_rgb(appearance.observed_temperature);
    let mut color_vec = Vec3::new(col_rgb.r, col_rgb.g, col_rgb.b);

    let brightness = appearance.observed_intensity; 
    color_vec *= brightness;

    (color_vec, density)
}

fn sample_skybox(dir: Vec3) -> Vec3 {
    let mut color = Vec3::ZERO; 

    let dir_blue = vec3(-0.2, 0.05, -1.0).normalize();
    let d_blue = dir.dot(dir_blue).max(0.0);

    color += vec3(0.2, 0.5, 1.0) * d_blue.powf(1000.0) * 5.0;
    color += vec3(0.1, 0.1, 0.4) * d_blue.powf(100.0) * 0.5;

    let dir_red = vec3(0.25, -0.1, -1.0).normalize();
    let d_red = dir.dot(dir_red).max(0.0);

    color += vec3(1.0, 0.3, 0.1) * d_red.powf(800.0) * 4.0;
    color += vec3(0.4, 0.1, 0.05) * d_red.powf(80.0) * 0.5;

    let dir_white = vec3(0.0, 0.3, -1.0).normalize();
    let d_white = dir.dot(dir_white).max(0.0);

    color += vec3(1.0, 0.95, 0.8) * d_white.powf(1200.0) * 6.0;

    color
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