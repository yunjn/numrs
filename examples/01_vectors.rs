//! 向量运算示例
//!
//! 演示 Vec2、Vec3、Vec4 的基本运算。

use numrs::{Vec2, Vec3, Vec4};

fn main() {
    println!("=== 向量运算 ===\n");

    // Vec2
    println!("--- Vec2 ---");
    let v2 = Vec2::new(3.0, 4.0);
    println!("v2 = {}", v2);
    println!("  长度: {}", v2.mag());
    println!("  长度平方: {}", v2.mag_sq());
    println!("  归一化: {}", v2.normalize());

    // Vec3
    println!("\n--- Vec3 ---");
    let v3 = Vec3::new(1.0, 2.0, 3.0);
    let w3 = Vec3::new(4.0, 5.0, 6.0);
    println!("v3 = {}", v3);
    println!("w3 = {}", w3);
    println!("  加法: v3 + w3 = {}", v3 + w3);
    println!("  减法: v3 - w3 = {}", v3 - w3);
    println!("  点积: v3.dot(&w3) = {}", v3.dot(&w3));
    println!("  叉积: v3.cross(&w3) = {}", v3.cross(&w3));
    println!("  标量乘法: v3 * 2.0 = {}", v3 * 2.0);
    println!("  逐元素乘法: v3 * w3 = {}", v3 * w3);

    // Vec4
    println!("\n--- Vec4 ---");
    let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    println!("v4 = {}", v4);
    println!("  长度: {}", v4.mag());
    println!("  归一化: {}", v4.normalize());

    // 角度计算
    println!("\n--- 角度计算 ---");
    let x = Vec3::UNIT_X;
    let y = Vec3::UNIT_Y;
    let _z = Vec3::UNIT_Z;
    println!(
        "x 与 y 的夹角: {} rad (π/2 = {:.4})",
        x.angle_to(&y),
        std::f64::consts::FRAC_PI_2
    );
    println!("x 与 x 的夹角: {} rad", x.angle_to(&x));
    println!(
        "x 与 -x 的夹角: {} rad (π = {:.4})",
        x.angle_to(&-x),
        std::f64::consts::PI
    );

    // 点类型
    println!("\n--- Point 类型 ---");
    let p1 = numrs::Point3D::new(0.0, 0.0, 0.0);
    let p2 = numrs::Point3D::new(3.0, 4.0, 0.0);
    println!("p1 = {}", p1);
    println!("p2 = {}", p2);
    println!("  两点距离: {}", p1.dist(&p2));
    println!("  距离平方: {}", p1.dist_sq(&p2));
}
