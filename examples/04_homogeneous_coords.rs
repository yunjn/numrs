//! 齐次坐标示例
//!
//! 演示 Vec4 和齐次坐标的概念。

use numrs::{Vec3, Vec4};

fn main() {
    println!("=== 齐次坐标 (Vec4) ===\n");

    // 齐次坐标基础
    println!("--- 齐次坐标基础 ---");
    let v4 = Vec4::new(1.0, 2.0, 3.0, 1.0);
    println!("Vec4: {}", v4);
    println!("  这是 (x, y, z, w) 形式的齐次坐标");

    // w = 0 表示方向/向量
    println!("\n--- w = 0: 方向向量 ---");
    let direction = Vec4::new(1.0, 0.0, 0.0, 0.0);
    println!("方向向量: {}", direction);
    println!("  用于表示方向，不表示位置");

    // w = 1 表示点
    println!("\n--- w = 1: 点 ---");
    let point = Vec4::new(1.0, 2.0, 3.0, 1.0);
    println!("点: {}", point);
    println!("  表示空间中的位置");

    // w = 2 表示齐次坐标（需要归一化）
    println!("\n--- 齐次坐标归一化 ---");
    let homo = Vec4::new(2.0, 4.0, 6.0, 2.0);
    println!("齐次坐标: {}", homo);
    println!("  w = {} ≠ 1，需要归一化", homo.w);
    let normalized = homo.to_homogeneous();
    println!("  归一化后: {}", normalized);

    // w ≠ 1 的点
    println!("\n--- w 与透视除法 ---");
    let points = [
        Vec4::new(1.0, 2.0, 3.0, 1.0),
        Vec4::new(2.0, 4.0, 6.0, 2.0),
        Vec4::new(3.0, 6.0, 9.0, 3.0),
    ];
    println!("以下三个点表示同一个空间位置:");
    for p in &points {
        let cartesian = p.to_homogeneous();
        println!("  {} -> {}", p, cartesian);
    }

    // 从 Vec3 转换
    println!("\n--- Vec3 与 Vec4 转换 ---");
    let v3 = Vec3::new(1.0, 2.0, 3.0);

    // 转换为方向 (w = 0)
    let as_direction = Vec4::from(v3);
    println!("Vec3 作为方向: {} (w = {})", as_direction, as_direction.w);

    // 转换为点 (w = 1)
    let as_point = Vec4::from((v3, 1.0));
    println!("Vec3 作为点: {} (w = {})", as_point, as_point.w);

    // 截断为 Vec3
    println!("\n--- Vec4 截断为 Vec3 ---");
    let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let v3 = v4.truncate();
    println!("{} -> {}", v4, v3);
}
