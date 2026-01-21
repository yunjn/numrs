//! 相机/视图矩阵示例
//!
//! 演示 LookAt 视图矩阵的构建和使用。

use numrs::{Mat4, Vec3, Vec4};

fn main() {
    println!("=== 相机/视图矩阵 ===\n");

    // LookAt 基础
    println!("--- LookAt 视图矩阵 ---");
    let eye = Vec3::new(0.0, 0.0, 5.0); // 相机位置
    let target = Vec3::ZERO; // 看向原点
    let up = Vec3::UNIT_Y; // 上方向

    let view = Mat4::from_look_at(&eye, &target, &up);
    println!("相机位置: {}", eye);
    println!("看向目标: {}", target);
    println!("上方向: {}", up);
    println!("\n视图矩阵:\n{}\n", view);

    // 变换世界坐标到相机空间
    println!("--- 世界坐标 -> 相机空间 ---");
    let world_point = Vec4::new(0.0, 0.0, 0.0, 1.0);
    let view_space = view * world_point;
    println!("世界坐标 {} -> 相机空间 = {}", world_point, view_space);

    // 相机坐标系说明
    println!("\n--- 相机坐标系 ---");
    let z_axis = (eye - target).normalize();
    let x_axis = up.cross(&z_axis).normalize();
    let y_axis = z_axis.cross(&x_axis).normalize();

    println!("相机原点: {}", eye);
    println!("相机+X 轴: {}", x_axis);
    println!("相机+Y 轴: {}", y_axis);
    println!("相机-Z 轴 (前向): {}", z_axis);

    // 相机移动
    println!("\n--- 相机移动示例 ---");
    let cameras = [
        (Vec3::new(0.0, 0.0, 5.0), "相机在 Z 轴正方向"),
        (Vec3::new(0.0, 0.0, -5.0), "相机在 Z 轴负方向"),
        (Vec3::new(5.0, 0.0, 0.0), "相机在 X 轴正方向"),
        (Vec3::new(0.0, 5.0, 0.0), "相机在 Y 轴正方向"),
    ];

    for (pos, desc) in &cameras {
        let view = Mat4::from_look_at(pos, &Vec3::ZERO, &Vec3::UNIT_Y);
        let origin = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let transformed = view * origin;
        println!("{}:", desc);
        println!("  原点 -> {}", transformed);
    }

    // 相机旋转
    println!("\n--- 相机旋转示例 ---");
    let eye = Vec3::new(0.0, 2.0, 5.0);
    let targets = [
        (Vec3::new(0.0, 0.0, 0.0), "看向原点"),
        (Vec3::new(10.0, 0.0, 0.0), "看向 +X"),
        (Vec3::new(0.0, 10.0, 0.0), "看向 +Y"),
    ];

    for (target, desc) in &targets {
        let view = Mat4::from_look_at(&eye, target, &Vec3::UNIT_Y);
        let origin = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let transformed = view * origin;
        println!("{}:", desc);
        println!("  原点 -> {}", transformed);
    }

    // 完整视角示例
    println!("\n--- 第一人称视角模拟 ---");
    let player_pos = Vec3::new(0.0, 1.7, 5.0); // 玩家高度
    let look_dir = Vec3::new(0.0, 0.0, -1.0); // 看向前方

    // 构建相机矩阵
    let right = Vec3::UNIT_Y.cross(&look_dir).normalize();
    let up = look_dir.cross(&right).normalize();

    let view = Mat4::from_look_at(&player_pos, &(player_pos + look_dir), &up);
    println!("玩家位置: {}", player_pos);
    println!("玩家朝向: {}", look_dir);
    println!("视图矩阵:\n{}\n", view);

    // 玩家周围的物体
    let objects = [
        Vec4::new(0.0, 1.7, 0.0, 1.0),   // 玩家脚下
        Vec4::new(0.0, 1.7, -2.0, 1.0),  // 玩家前方 2 米
        Vec4::new(0.0, 1.7, -10.0, 1.0), // 玩家前方 10 米
        Vec4::new(2.0, 1.7, -5.0, 1.0),  // 玩家右侧
    ];

    println!("物体在相机空间的位置:");
    for obj in &objects {
        let view_space = view * *obj;
        let _ndc = view_space; // 尚未投影
        println!(
            "  {} -> ({:.2}, {:.2}, {:.2})",
            obj, view_space.x, view_space.y, view_space.z
        );
    }
}
