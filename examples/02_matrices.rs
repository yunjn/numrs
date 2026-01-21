//! 矩阵运算示例
//!
//! 演示 Mat2、Mat3、Mat4 的基本运算。

use approx::assert_relative_eq;
use numrs::{Mat2, Mat3, Mat4};

fn main() {
    println!("=== 矩阵运算 ===\n");

    // Mat2 - 2x2 矩阵
    println!("--- Mat2 ---");
    let m2 = Mat2::new(1.0, 2.0, 3.0, 4.0);
    println!("m2 = \n{}", m2);
    println!("  转置: \n{}", m2.transpose());
    println!("  行列式: {}", m2.determinant());

    // Mat3 - 3x3 矩阵
    println!("\n--- Mat3 ---");
    let m3 = Mat3::new(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0);
    println!("m3 = \n{}", m3);
    println!("  行列式: {}", m3.determinant());

    // 矩阵求逆
    println!("\n--- 矩阵求逆 ---");
    let invertible = Mat3::new(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0);
    println!("可逆矩阵: \n{}", invertible);
    println!("  行列式: {}", invertible.determinant());

    let inv = invertible.inverse().expect("矩阵应该可逆");
    println!("  逆矩阵: \n{}", inv);

    let product = invertible * inv;
    println!("  M * M^-1 = \n{}", product);
    assert_relative_eq!(product, Mat3::IDENTITY, epsilon = 1e-10);
    println!("  验证: M * M^-1 ≈ I ✓");

    // 奇异矩阵
    println!("\n--- 奇异矩阵 ---");
    let singular = Mat3::new(1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0);
    println!("奇异矩阵 (行线性相关): \n{}", singular);
    println!("  行列式: {}", singular.determinant());
    println!("  逆矩阵: {:?}", singular.inverse());

    // Mat4 - 4x4 矩阵
    println!("\n--- Mat4 ---");
    let m4 = Mat4::IDENTITY;
    println!("单位矩阵: \n{}", m4);
    println!("  行列式: {}", m4.determinant());

    // 矩阵乘法
    println!("\n--- 矩阵乘法 ---");
    let a = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    let b = Mat3::IDENTITY;
    println!("A * I = A: {}", a * b == a);
    println!("I * A = A: {}", b * a == a);

    // 标量乘法
    println!("\n--- 标量乘法 ---");
    let m = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    println!("2.0 * M = \n{}", m * 2.0);
}
