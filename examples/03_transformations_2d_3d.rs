//! 2D/3D 变换示例
//!
//! 使用 Mat3 进行 2D 和 3D 变换。

use numrs::{Mat3, Vec3};
use std::f64::consts::PI;

fn main() {
    println!("=== 2D/3D 变换 (Mat3) ===\n");

    // 基础变换
    println!("--- 基础变换 ---");

    // 缩放
    let scale = Mat3::from_scale(2.0, 3.0, 1.0);
    let v = Vec3::new(1.0, 1.0, 1.0);
    println!("缩放 (2, 3, 1): {} -> {}", v, scale * v);

    // 旋转 - 绕 X 轴
    let rot_x = Mat3::from_rotation_x(PI / 2.0);
    let v = Vec3::new(0.0, 1.0, 0.0);
    println!("绕 X 旋转 90°: {} -> {}", v, rot_x * v);

    // 旋转 - 绕 Y 轴
    let rot_y = Mat3::from_rotation_y(PI / 2.0);
    let v = Vec3::new(1.0, 0.0, 0.0);
    println!("绕 Y 旋转 90°: {} -> {}", v, rot_y * v);

    // 旋转 - 绕 Z 轴
    let rot_z = Mat3::from_rotation_z(PI / 2.0);
    let v = Vec3::new(1.0, 0.0, 0.0);
    println!("绕 Z 旋转 90°: {} -> {}", v, rot_z * v);

    // 组合变换
    println!("\n--- 组合变换 ---");
    let scale = Mat3::from_scale(2.0, 2.0, 1.0);
    let rot = Mat3::from_rotation_z(PI / 4.0);

    let v = Vec3::new(1.0, 0.0, 0.0);

    // 先缩放后旋转
    let combined1 = rot * scale;
    println!("先缩放后旋转: {} -> {}", v, combined1 * v);

    // 先旋转后缩放
    let combined2 = scale * rot;
    println!("先旋转后缩放: {} -> {}", v, combined2 * v);

    println!("注意: 矩阵乘法顺序很重要！");

    // 变换原点
    println!("\n--- 变换原点 (0,0) ---");
    let rot = Mat3::from_rotation_z(PI / 4.0);
    let origin = Vec3::ZERO;
    println!("原点旋转后: {} -> {}", origin, rot * origin);
    println!("原点永远是原点，因为 Mat3 无法表示平移");
}
