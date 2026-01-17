use approx::assert_relative_eq;
use numrs::Mat3;

fn main() {
    let m = Mat3::new(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0);

    let det = m.determinant();
    println!("Determinant: {}", det);

    if det != 0.0 {
        let inv = m.inverse();
        println!("Inverse matrix: {:?}", inv);

        if let Some(inv) = m.inverse() {
            let identity = m * inv;
            println!("M * M^-1 = {}", identity);

            assert_relative_eq!(identity, Mat3::IDENTITY, epsilon = 1e-15);
            println!("Verification successful: Identity matrix restored.");
        }
    }
}
