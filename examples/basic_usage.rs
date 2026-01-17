use numrs::Vec3;

fn main() {
    let v1 = Vec3::new(1.0, 2.0, 3.0);
    let v2 = Vec3::new(4.0, 5.0, 6.0);

    let sum = v1 + v2;
    println!("{:?}", sum);

    let scaled = v1 * 2.0;
    println!("{:?}", scaled);

    println!("{:?}", v1.normalize());
}
