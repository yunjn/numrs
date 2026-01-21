//! 投影矩阵示例
//!
//! 演示透视投影和正交投影。

use numrs::{Mat4, Vec4};
use std::f64::consts::PI;

fn main() {
    println!("=== 投影矩阵 ===\n");

    // 透视投影
    println!("--- 透视投影 ---");
    let fov = PI / 4.0; // 45 度
    let aspect = 16.0 / 9.0;
    let near = 0.1;
    let far = 100.0;

    let perspective = Mat4::from_perspective(fov, aspect, near, far);
    println!(
        "透视投影 (FOV=45°, aspect={:.2}, near={}, far={}):",
        aspect, near, far
    );
    println!("{}\n", perspective);

    // 透视除法演示
    println!("--- 透视除法 ---");
    let test_points = [
        Vec4::new(0.0, 0.0, -1.0, 1.0),   // 近裁剪面中心
        Vec4::new(0.0, 0.0, -50.0, 1.0),  // 中间
        Vec4::new(0.0, 0.0, -100.0, 1.0), // 远裁剪面
    ];

    println!("点              -> 裁剪空间               -> NDC");
    println!("{}", "-".repeat(70));
    for p in &test_points {
        let clip = perspective * *p;
        let ndc = clip.to_homogeneous();
        println!(
            "{} -> {} -> ({:.3}, {:.3}, {:.3})",
            p, clip, ndc.x, ndc.y, ndc.z
        );
    }

    // NDC 范围说明
    println!("\nNDC 范围: x, y ∈ [-1, 1], z ∈ [-1, 1] (OpenGL 标准)");

    // 正交投影
    println!("\n--- 正交投影 ---");
    let ortho = Mat4::from_orthographic(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    println!("正交投影 (-1~1, -1~1, -1~1):\n{}", ortho);

    // 正交投影不产生透视除法
    let v = Vec4::new(0.5, 0.5, 0.0, 1.0);
    let result = ortho * v;
    println!("\n点 {} 正交投影后 = {}", v, result);
    println!("  w 不变，无透视除法");

    // 不同的投影应用场景
    println!("\n--- 投影选择指南 ---");
    println!("透视投影:");
    println!("  - 3D 游戏、渲染");
    println!("  - 模拟人眼视觉效果");
    println!("  - 远处的物体看起来更小");

    println!("\n正交投影:");
    println!("  - CAD、工程图");
    println!("  - UI 界面");
    println!("  - 2D 游戏");
    println!("  - 不产生透视变形");

    // 视锥体
    println!("\n--- 视锥体中的点 ---");
    let proj = Mat4::from_perspective(PI / 3.0, 1.0, 1.0, 10.0);

    let points = [
        Vec4::new(0.0, 0.0, -1.0, 1.0),  // 相机位置（裁剪掉）
        Vec4::new(0.0, 0.0, -2.0, 1.0),  // 近裁剪面
        Vec4::new(0.0, 0.0, -5.0, 1.0),  // 中间
        Vec4::new(0.0, 0.0, -10.0, 1.0), // 远裁剪面
        Vec4::new(0.0, 0.0, -20.0, 1.0), // 裁剪掉
    ];

    for p in &points {
        let clip = proj * *p;
        let in_view = clip.w > 0.0
            && clip.x.abs() <= clip.w
            && clip.y.abs() <= clip.w
            && clip.z.abs() <= clip.w;
        let ndc = clip.to_homogeneous();
        println!(
            "{} -> {} -> NDC: ({:.2}, {:.2}, {:.2}) [{}]",
            p,
            clip,
            ndc.x,
            ndc.y,
            ndc.z,
            if in_view { "可见" } else { "不可见" }
        );
    }
}
