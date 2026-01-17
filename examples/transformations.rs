use numrs::{Mat3, Vec3};
use std::f64::consts::PI;

fn main() {
    // 创建一个绕 X 轴旋转 90 度的矩阵
    let rotation = Mat3::from_rotation_x(PI / 2.0);

    // 创建一个位于 Y 轴正方向的向量
    let point = Vec3::new(0.0, 1.0, 0.0);

    // 执行旋转：M * v
    // 绕 X 轴旋转 90 度后，(0, 1, 0) 应该变成 (0, 0, 1)
    let rotated_point = rotation * point;

    println!("Original point: {}", point);
    println!("Rotated point:  {}", rotated_point);
}
