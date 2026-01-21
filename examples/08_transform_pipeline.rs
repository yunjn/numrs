//! 完整 3D 变换管线示例
//!
//! 演示从模型空间到裁剪空间的完整变换过程。

use numrs::{Mat4, Vec3, Vec4};
use std::f64::consts::PI;

fn main() {
    println!("=== 完整 3D 变换管线 ===\n");

    // 定义一个立方体的 8 个顶点
    let cube = [
        Vec4::new(-1.0, -1.0, -1.0, 1.0),
        Vec4::new(1.0, -1.0, -1.0, 1.0),
        Vec4::new(1.0, 1.0, -1.0, 1.0),
        Vec4::new(-1.0, 1.0, -1.0, 1.0),
        Vec4::new(-1.0, -1.0, 1.0, 1.0),
        Vec4::new(1.0, -1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(-1.0, 1.0, 1.0, 1.0),
    ];

    // 变换矩阵
    println!("--- 变换矩阵 ---");

    // 模型矩阵: 旋转 + 平移
    let model = Mat4::from_rotation_y(PI / 4.0) * Mat4::from_translation(0.0, 0.0, -5.0);
    println!("模型矩阵 (旋转 + 平移):\n{}\n", model);

    // 视图矩阵: 相机在 (0, 2, 8)，看向原点
    let view = Mat4::from_look_at(&Vec3::new(0.0, 2.0, 8.0), &Vec3::ZERO, &Vec3::UNIT_Y);
    println!("视图矩阵 (相机):\n{}\n", view);

    // 投影矩阵
    let projection = Mat4::from_perspective(PI / 3.0, 16.0 / 9.0, 0.1, 100.0);
    println!("投影矩阵:\n{}\n", projection);

    // MVP 矩阵
    let mvp = projection * view * model;
    println!("MVP (Model-View-Projection):\n{}\n", mvp);

    // 变换演示
    println!("--- 顶点变换 ---");
    println!(
        "{:<8} {:<30} {:<30} {:<30}",
        "顶点", "模型空间", "视图空间", "裁剪空间"
    );
    println!("{}", "-".repeat(100));

    for (i, v) in cube.iter().enumerate() {
        let model_space = model * *v;
        let view_space = view * model_space;
        let clip = mvp * *v;

        // 裁剪空间 -> NDC
        let ndc = if clip.w.abs() > 1e-10 {
            format!(
                "NDC: ({:.2}, {:.2}, {:.2})",
                clip.x / clip.w,
                clip.y / clip.w,
                clip.z / clip.w
            )
        } else {
            "w = 0".to_string()
        };

        println!(
            "V{:<5} {} {} {}",
            i + 1,
            format!("{}", model_space).replace(" ", ""),
            format!("{}", view_space).replace(" ", ""),
            format!("{} {}", format!("{}", clip).replace(" ", ""), ndc)
        );
    }

    // 提取 3x3 子矩阵
    println!("\n--- 提取 3x3 子矩阵 ---");
    let rot_x = Mat4::from_rotation_x(PI / 6.0);
    let rot_3x3 = rot_x.extract_3x3();
    println!("Mat4 旋转 X 30° 提取 3x3:\n{}\n", rot_3x3);

    // 变换管线总结
    println!("=== 变换管线总结 ===");
    println!("1. 模型变换 (Model):");
    println!("   世界坐标 = 模型矩阵 * 局部坐标");
    println!("   作用: 缩放、旋转、平移物体");

    println!("\n2. 视图变换 (View):");
    println!("   相机坐标 = 视图矩阵 * 世界坐标");
    println!("   作用: 将世界坐标转换到相机空间");

    println!("\n3. 投影变换 (Projection):");
    println!("   裁剪坐标 = 投影矩阵 * 相机坐标");
    println!("   作用: 将相机坐标转换为裁剪空间 (NDC)");

    println!("\n4. 透视除法:");
    println!("   NDC = 裁剪坐标 / w");
    println!("   作用: 完成透视投影");

    println!("\n5. 视口变换:");
    println!("   屏幕坐标 = 视口矩阵 * NDC");
    println!("   作用: 映射到屏幕像素坐标");
}
