use numrs::{Mat3, Vec3};

fn main() {
    let m1 = Mat3::IDENTITY;
    let v1 = Vec3::new(1.0, 2.0, 3.0);

    let result = m1 * v1;
    println!("{}", result);

    let m2 = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    let v2 = Vec3::new(1.0, 2.0, 3.0);

    let result = m2 * v2;
    println!("{}", result);

    let m3 = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    let v3 = Vec3::new(1.0, 2.0, 3.0);

    let result = m3 * v3;
    println!("{}", result);

    println!("{}", m1 * 1. + m2 * 2.0 + m3 * 3.0);
}
