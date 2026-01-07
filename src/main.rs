use macroquad::prelude::*;

mod gpu;
mod physics;
mod visualization;

use gpu::GpuContext;
use physics::BlackHole;
use visualization::{Renderer, render_observer_view_gpu};

#[macroquad::main("Black Hole Raytracer - GPU Accelerated")]
async fn main() {
    let black_hole = BlackHole::new(Vec3::ZERO, 50000.0);
    let mut renderer = Renderer::new();

    let render_scale = 1;
    let initial_width = (screen_width() as usize / render_scale).max(1);
    let initial_height = (screen_height() as usize / render_scale).max(1);
    let mut gpu_ctx = GpuContext::new(initial_width as u32, initial_height as u32);

    let mut paused = true;
    let time_scale = 1.0;
    let mut global_time = 0.0;
    let mut use_gpu = true;

    println!("Black Hole Raytracer - GPU Accelerated");
    println!("Controls:");
    println!("  WASD - Move camera");
    println!("  Q/E  - Move up/down");
    println!("  Mouse - Look around");
    println!("  Shift - Move faster");
    println!("  Space - Pause/unpause time");
    println!("  G     - Toggle GPU/CPU rendering");
    println!("  R     - Reset camera position");

    loop {
        renderer.handle_input();

        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }

        if is_key_pressed(KeyCode::R) {
            renderer.fpv_pos = vec3(0.0, 100.0, 600.0);
            renderer.fpv_yaw = -90.0f32.to_radians();
            renderer.fpv_pitch = 0.0;
        }

        if is_key_pressed(KeyCode::G) {
            use_gpu = !use_gpu;
        }

        let dt = get_frame_time();
        if !paused {
            global_time += dt * time_scale;
        }

        renderer.global_time = global_time;

        clear_background(Color::new(0.01, 0.01, 0.03, 1.0));

        let w = (screen_width() as usize / render_scale).max(1);
        let h = (screen_height() as usize / render_scale).max(1);

        gpu_ctx.resize(w as u32, h as u32);

        let cam = renderer.get_camera();

        let img = if use_gpu {
            render_observer_view_gpu(
                &gpu_ctx,
                &black_hole,
                cam.position,
                cam.target,
                w,
                h,
                global_time,
            )
        } else {
            visualization::render_observer_view_cpu(
                &black_hole,
                cam.position,
                cam.target,
                w,
                h,
                global_time,
            )
        };

        let tex = Texture2D::from_image(&img);
        tex.set_filter(FilterMode::Linear);
        draw_texture_ex(
            &tex,
            0.0,
            0.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(screen_width(), screen_height())),
                ..Default::default()
            },
        );

        let fps = get_fps() as f32;
        let mode_str = if use_gpu { "GPU" } else { "CPU" };
        draw_text(
            &format!("FPS: {:.0} | Mode: {} | Resolution: {}x{}", fps, mode_str, w, h),
            20.0,
            30.0,
            24.0,
            WHITE,
        );
        draw_text(
            "WASD to Fly, Mouse to Look, G to toggle GPU/CPU",
            20.0,
            55.0,
            18.0,
            GRAY,
        );

        if paused {
            let text = "PAUSED (Space to resume)";
            let text_size = 30.0;
            let text_dims = measure_text(text, None, text_size as u16, 1.0);
            draw_text(
                text,
                screen_width() * 0.5 - text_dims.width * 0.5,
                screen_height() * 0.5,
                text_size,
                Color::new(1.0, 1.0, 1.0, 0.8),
            );
        }

        next_frame().await
    }
}
