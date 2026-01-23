//! Simple benchmarks for numrs
//!
//! Run with: cargo run --example benchmarks

use numrs::{Mat2, Mat3, Mat4, Vec2, Vec3, Vec4};
use std::time::Instant;

const ITERATIONS: usize = 100_000;

macro_rules! bench {
    ($name:expr, $block:block) => {
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            $block
        }
        let elapsed = start.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / ITERATIONS as f64;
        println!("{:30} {:8.2} ns/op", $name, ns_per_op);
    };
}

fn main() {
    println!("numrs Performance Benchmarks");
    println!("============================");
    println!("Iterations per benchmark: {}", ITERATIONS);
    println!();

    // Vec2 benchmarks
    println!("Vec2 Operations:");
    let v2 = Vec2::new(3.0, 4.0);
    let other2 = Vec2::new(1.0, 2.0);
    bench!("vec2/mag", {
        v2.mag();
    });
    bench!("vec2/normalize", {
        v2.normalize();
    });
    bench!("vec2/dot", {
        v2.dot(&other2);
    });
    println!();

    // Vec3 benchmarks
    println!("Vec3 Operations:");
    let v3 = Vec3::new(1.0, 2.0, 3.0);
    let other3 = Vec3::new(4.0, 5.0, 6.0);
    bench!("vec3/mag", {
        v3.mag();
    });
    bench!("vec3/normalize", {
        v3.normalize();
    });
    bench!("vec3/dot", {
        v3.dot(&other3);
    });
    bench!("vec3/cross", {
        v3.cross(&other3);
    });
    println!();

    // Vec4 benchmarks
    println!("Vec4 Operations:");
    let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    bench!("vec4/mag", {
        v4.mag();
    });
    bench!("vec4/normalize", {
        v4.normalize();
    });
    bench!("vec4/to_homogeneous", {
        v4.to_homogeneous();
    });
    bench!("vec4/truncate", {
        v4.truncate();
    });
    println!();

    // Mat2 benchmarks
    println!("Mat2 Operations:");
    let m2 = Mat2::new(1.0, 2.0, 3.0, 4.0);
    bench!("mat2/determinant", {
        m2.determinant();
    });
    bench!("mat2/inverse", {
        m2.inverse();
    });
    bench!("mat2/transpose", {
        m2.transpose();
    });
    println!();

    // Mat3 benchmarks
    println!("Mat3 Operations:");
    let m3 = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    bench!("mat3/determinant", {
        m3.determinant();
    });
    bench!("mat3/inverse", {
        m3.inverse();
    });
    bench!("mat3/transpose", {
        m3.transpose();
    });
    bench!("mat3/rotation_z", {
        Mat3::from_rotation_z(1.0);
    });
    println!();

    // Mat3 multiplication
    println!("Mat3 Multiplication:");
    let m3_rot = Mat3::from_rotation_z(1.0);
    let m3_scale = Mat3::from_scale(2.0, 3.0, 4.0);
    let v3_test = Vec3::new(1.0, 2.0, 3.0);
    bench!("mat3*vec3", {
        let _ = m3_rot * v3_test;
    });
    bench!("mat3*mat3", {
        let _ = m3_rot * m3_scale;
    });
    println!();

    // Mat4 benchmarks
    println!("Mat4 Operations:");
    let m4 = Mat4::from_rotation_y(1.0);
    bench!("mat4/determinant", {
        m4.determinant();
    });
    bench!("mat4/inverse", {
        m4.inverse();
    });
    bench!("mat4/transpose", {
        m4.transpose();
    });
    bench!("mat4/rotation_y", {
        Mat4::from_rotation_y(1.0);
    });
    bench!("mat4/translation", {
        Mat4::from_translation(1.0, 2.0, 3.0);
    });
    bench!("mat4/scale", {
        Mat4::from_scale(2.0, 3.0, 4.0, 1.0);
    });
    bench!("mat4/rotation_axis", {
        let axis = Vec3::new(1.0, 1.0, 1.0).normalize();
        Mat4::from_rotation_axis(&axis, 1.0);
    });
    println!();

    // Mat4 projection
    println!("Mat4 Projections:");
    bench!("mat4/perspective", {
        Mat4::from_perspective(std::f64::consts::PI / 4.0, 16.0 / 9.0, 0.1, 100.0);
    });
    bench!("mat4/orthographic", {
        Mat4::from_orthographic(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    });
    bench!("mat4/look_at", {
        Mat4::from_look_at(&Vec3::new(0.0, 0.0, 5.0), &Vec3::ZERO, &Vec3::UNIT_Y);
    });
    println!();

    // Mat4 multiplication
    println!("Mat4 Multiplication:");
    let m4_rot = Mat4::from_rotation_y(1.0);
    let m4_trans = Mat4::from_translation(1.0, 2.0, 3.0);
    let v4_test = Vec4::new(1.0, 2.0, 3.0, 1.0);
    bench!("mat4*vec4", {
        let _ = m4_rot * v4_test;
    });
    bench!("mat4*mat4", {
        let _ = m4_rot * m4_trans;
    });
    bench!("mat4/extract_3x3", {
        let _ = m4_rot.extract_3x3();
    });
    println!();

    // Transform pipeline
    println!("Transform Pipeline:");
    let v_pipeline = Vec4::new(1.0, 0.0, 0.0, 1.0);
    bench!("transform/chain_3", {
        let scale = Mat4::from_scale(2.0, 2.0, 2.0, 1.0);
        let rot = Mat4::from_rotation_y(1.0);
        let trans = Mat4::from_translation(0.0, 0.0, -5.0);
        let combined = trans * rot * scale;
        let _ = combined * v_pipeline;
    });
    bench!("transform/full_pipeline", {
        let model = Mat4::from_rotation_y(1.0) * Mat4::from_translation(0.0, 0.0, -5.0);
        let view = Mat4::from_look_at(&Vec3::new(0.0, 2.0, 8.0), &Vec3::ZERO, &Vec3::UNIT_Y);
        let projection = Mat4::from_perspective(std::f64::consts::PI / 3.0, 16.0 / 9.0, 0.1, 100.0);
        let mvp = projection * view * model;
        let _ = mvp * v_pipeline;
    });
    println!();

    println!("Done!");
}
