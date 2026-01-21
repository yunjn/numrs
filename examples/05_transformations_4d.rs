//! 4D 变换示例
//!
//! 使用 Mat4 进行 3D 仿射变换（支持平移）。

use numrs::{Mat4, Vec4};
use std::f64::consts::PI;

fn main() {
    println!("=== 4D 变换 (Mat4) ===\n");

    // Mat4 可以表示平移
    println!("--- 平移变换 ---");
    let translation = Mat4::from_translation(2.0, 3.0, 4.0);
    println!("平移矩阵 (2, 3, 4):\n{}", translation);

    let point = Vec4::new(1.0, 1.0, 1.0, 1.0);
    let transformed = translation * point;
    println!("点 {} 平移后 = {}", point, transformed);
    println!(
        "  原点 (0,0,0,1) 平移后 = {}",
        translation * Vec4::new(0.0, 0.0, 0.0, 1.0)
    );

    // 方向向量不受平移影响
    let direction = Vec4::new(1.0, 0.0, 0.0, 0.0);
    let transformed_dir = translation * direction;
    println!("方向 {} 平移后 = {}", direction, transformed_dir);
    println!("  方向不受平移影响 (w = 0)");

    // 缩放
    println!("\n--- 缩放变换 ---");
    let scale = Mat4::from_scale(2.0, 3.0, 4.0, 1.0);
    println!("缩放矩阵 (2, 3, 4, 1):\n{}", scale);

    let v = Vec4::new(1.0, 1.0, 1.0, 1.0);
    println!("{} 缩放后 = {}", v, scale * v);

    // 旋转
    println!("\n--- 旋转变换 ---");
    let rot_x = Mat4::from_rotation_x(PI / 2.0);
    let rot_y = Mat4::from_rotation_y(PI / 2.0);
    let rot_z = Mat4::from_rotation_z(PI / 2.0);

    let v = Vec4::new(0.0, 1.0, 0.0, 1.0);
    println!("原始点: {}", v);
    println!("绕 X 旋转 90°: {}", rot_x * v);
    println!("绕 Y 旋转 90°: {}", rot_y * v);
    println!("绕 Z 旋转 90°: {}", rot_z * v);

    // 任意轴旋转
    println!("\n--- 任意轴旋转 ---");
    let axis = numrs::Vec3::new(1.0, 1.0, 1.0).normalize();
    let rot_axis = Mat4::from_rotation_axis(&axis, PI / 3.0);
    let v = Vec4::new(1.0, 0.0, 0.0, 1.0);
    println!("绕 ({:.3}, {:.3}, {:.3}) 旋转 60°:", axis.x, axis.y, axis.z);
    println!("  {} -> {}", v, rot_axis * v);

    // 组合变换
    println!("\n--- 组合变换 ---");
    let scale = Mat4::from_scale(2.0, 2.0, 2.0, 1.0);
    let rot = Mat4::from_rotation_y(PI / 4.0);
    let trans = Mat4::from_translation(0.0, 0.0, -5.0);

    // 变换顺序: 先缩放 -> 旋转 -> 平移
    let transform = trans * rot * scale;

    let v = Vec4::new(1.0, 0.0, 0.0, 1.0);
    println!("变换: T * R * S");
    println!("  {} -> {}", v, transform * v);

    // 验证矩阵乘法顺序
    println!("\n--- 矩阵乘法顺序 ---");
    let v = Vec4::new(1.0, 0.0, 0.0, 1.0);
    let a = Mat4::from_translation(1.0, 0.0, 0.0);
    let b = Mat4::from_scale(2.0, 1.0, 1.0, 1.0);

    println!("先平移后缩放 (S * T * v): {}", b * a * v);
    println!("先缩放后平移 (T * S * v): {}", a * b * v);
    println!("  顺序不同，结果不同！");
}
