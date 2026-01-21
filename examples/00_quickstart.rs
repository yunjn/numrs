//! numrs - 快速上手示例
//!
//! 这个库提供了 2D/3D 线性代数类型，使用 f64 精度。

use numrs::{Mat2, Mat3, Mat4, Vec2, Vec3, Vec4};

fn main() {
    println!("=== numrs 快速上手 ===\n");

    // 2D 向量和矩阵
    println!("--- 2D ---");
    let v2 = Vec2::new(1.0, 2.0);
    let m2 = Mat2::IDENTITY;
    println!("Vec2: {}", v2);
    println!("Mat2: {}", m2);

    // 3D 向量和矩阵
    println!("\n--- 3D ---");
    let v3 = Vec3::new(1.0, 2.0, 3.0);
    let m3 = Mat3::IDENTITY;
    println!("Vec3: {}", v3);
    println!("Mat3: {}", m3);

    // 4D 向量和矩阵（齐次坐标）
    println!("\n--- 4D (齐次坐标) ---");
    let v4 = Vec4::new(1.0, 2.0, 3.0, 1.0);
    let m4 = Mat4::IDENTITY;
    println!("Vec4: {}", v4);
    println!("Mat4: {}", m4);

    // 基本的向量运算
    println!("\n--- 向量运算 ---");
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);
    println!("a + b = {}", a + b);
    println!("a * 2.0 = {}", a * 2.0);
    println!("a.dot(&b) = {}", a.dot(&b));
    println!("a.cross(&b) = {}", a.cross(&b));

    // 矩阵变换
    println!("\n--- 矩阵变换 ---");
    let rotation = Mat3::from_rotation_z(std::f64::consts::PI / 4.0);
    let v = Vec3::new(1.0, 0.0, 0.0);
    println!("旋转 45°: {} -> {}", v, rotation * v);
}
