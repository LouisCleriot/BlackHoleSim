use macroquad::prelude::*;

mod light_beam;
mod physics;
mod visualization;

use light_beam::BeamManager;
use physics::BlackHole;
use visualization::Renderer;
use visualization::render_observer_view;

#[macroquad::main("3D Black Hole Light Bending")]
async fn main() {
    let black_hole = BlackHole::new(Vec3::ZERO, 50000.0);
    let mut beam_manager = BeamManager::new();
    let mut renderer = Renderer::new();

    beam_manager.spawn_initial_beams(&black_hole, 10, 400.0);

    let mut paused = true;
    let time_scale = 1.0;
    let mut global_time = 0.0;

    loop {
        renderer.handle_input();

        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }

        if is_key_pressed(KeyCode::R) {
            renderer.fpv_mode = false;
            beam_manager = BeamManager::new();
            beam_manager.spawn_initial_beams(&black_hole, 10, 400.0);
        }

        if is_key_pressed(KeyCode::P) {
            renderer.fpv_mode = !renderer.fpv_mode;

            if renderer.fpv_mode {
                let cam = renderer.get_camera();
                renderer.fpv_pos = cam.position;
            }
        }

        let dt = get_frame_time();
        if !paused {
            global_time += dt * time_scale;
            beam_manager.update(&black_hole, dt * time_scale);
        }

        renderer.global_time = global_time;

        clear_background(Color::new(0.01, 0.01, 0.03, 1.0));

        if renderer.fpv_mode {
            let render_scale = 8;
            let w = screen_width() as usize / render_scale;
            let h = screen_height() as usize / render_scale;

            let cam = renderer.get_camera();

            let img =
                render_observer_view(&black_hole, cam.position, cam.target, w, h, global_time);

            let tex = Texture2D::from_image(&img);
            tex.set_filter(FilterMode::Nearest);
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

            draw_text(
                "FPV MODE: WASD to Fly, Mouse to Look",
                20.0,
                40.0,
                30.0,
                WHITE,
            );
            draw_text("Press P to exit", 20.0, 70.0, 20.0, GRAY);
        } else {
            set_camera(&renderer.get_camera());
            renderer.draw_grid(500.0, 20.0);
            renderer.draw_black_hole(&black_hole);
            renderer.draw_beams(&beam_manager);
            set_default_camera();

            draw_text("ORBIT MODE: Mouse Drag to Rotate", 20.0, 40.0, 30.0, WHITE);
            draw_text("Press P for FPV Mode", 20.0, 70.0, 20.0, GRAY);
        }

        let fps = get_fps() as f32;
        renderer.draw_stats(&beam_manager, fps);

        if paused {
            let text = "PAUSED";
            let text_size = 40.0;
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
