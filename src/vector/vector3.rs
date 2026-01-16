use crate::EPSILON;
use crate::Mat3;
use derive_more::{Add, Constructor, Div, Mul, Neg, Sub};

#[derive(Add, Sub, Mul, Div, Neg, Clone, Copy, Debug, PartialOrd, Constructor)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const ONE: Vec3 = Vec3 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };
    pub const UNIT_X: Vec3 = Vec3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    pub const UNIT_Y: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    pub const UNIT_Z: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    pub fn mag(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn mag_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn map<F>(&self, f: F) -> Vec3
    where
        F: Fn(f64) -> f64,
    {
        let x = f(self.x);
        let y = f(self.y);
        let z = f(self.z);
        Vec3 { x, y, z }
    }

    pub fn normalize(&self) -> Vec3 {
        let mag = self.mag();
        if mag < EPSILON {
            Vec3::ZERO
        } else {
            Vec3 {
                x: self.x / mag,
                y: self.y / mag,
                z: self.z / mag,
            }
        }
    }

    pub fn angle_to(&self, other: &Vec3) -> f64 {
        let mag_self = self.mag();
        let mag_other = other.mag();

        if mag_self < EPSILON || mag_other < EPSILON {
            return 0.0;
        }

        let cos_theta = self.dot(other) / (mag_self * mag_other);
        cos_theta.clamp(-1.0, 1.0).acos()
    }

    pub fn angle_to_unit(&self, other: &Vec3) -> f64 {
        self.dot(other).clamp(-1.0, 1.0).acos()
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::fmt::Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

impl PartialEq for Vec3 {
    fn eq(&self, other: &Self) -> bool {
        (self.x - other.x).abs() < EPSILON
            && (self.y - other.y).abs() < EPSILON
            && (self.z - other.z).abs() < EPSILON
    }
}

impl std::ops::Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, vec: Vec3) -> Self::Output {
        Vec3 {
            x: self * vec.x,
            y: self * vec.y,
            z: self * vec.z,
        }
    }
}

impl std::ops::Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, vec: Vec3) -> Self::Output {
        Vec3 {
            x: self.x * vec.x,
            y: self.y * vec.y,
            z: self.z * vec.z,
        }
    }
}

impl std::ops::Mul<Mat3> for Vec3 {
    type Output = Vec3;
    fn mul(self, m: Mat3) -> Vec3 {
        Vec3::new(
            self.x * m.a + self.y * m.d + self.z * m.g,
            self.x * m.b + self.y * m.e + self.z * m.h,
            self.x * m.c + self.y * m.f + self.z * m.i,
        )
    }
}

pub type Point3D = Vec3;

impl Point3D {
    pub fn dist(&self, other: &Point3D) -> f64 {
        (*self - *other).mag()
    }
    pub fn dist_sq(&self, other: &Point3D) -> f64 {
        (*self - *other).mag_sq()
    }
}

#[cfg(test)]

mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn test_constants() {
        assert_eq!(Vec3::ZERO, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(Vec3::ONE, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(Vec3::UNIT_X, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(Vec3::UNIT_Y, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(Vec3::UNIT_Z, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_default() {
        assert_eq!(Vec3::default(), Vec3::ZERO);
    }

    #[test]
    fn test_neg() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        assert_eq!(-v, Vec3::new(-1.0, 2.0, -3.0));
    }

    #[test]
    fn test_add() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(v1 + v2, Vec3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_sub() {
        let v1 = Vec3::new(5.0, 7.0, 9.0);
        let v2 = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v1 - v2, Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn test_scalar_mul_left() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(2.0 * v, Vec3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_scalar_mul_right() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v * 2.0, Vec3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_scalar_div() {
        let v = Vec3::new(2.0, 4.0, 6.0);
        assert_eq!(v / 2.0, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_component_mul() {
        let v1 = Vec3::new(2.0, 3.0, 4.0);
        let v2 = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v1 * v2, Vec3::new(2.0, 6.0, 12.0));
    }

    #[test]
    fn test_mag() {
        let v = Vec3::new(1.0, 2.0, 2.0);
        assert!((v.mag() - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_mag_sq() {
        let v = Vec3::new(1.0, 2.0, 2.0);
        assert_eq!(v.mag_sq(), 9.0);
    }

    #[test]
    fn test_dot() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, -5.0, 6.0);
        assert_eq!(v1.dot(&v2), 12.0);
    }

    #[test]
    fn test_cross() {
        let x = Vec3::UNIT_X;
        let y = Vec3::UNIT_Y;
        let z = Vec3::UNIT_Z;

        assert_eq!(x.cross(&y), z);
        assert_eq!(y.cross(&z), x);
        assert_eq!(z.cross(&x), y);

        // Anticommutativity
        assert_eq!(x.cross(&y), -y.cross(&x));
    }

    #[test]
    fn test_map() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let mapped = v.map(|x| x * 2.0 + 1.0);
        assert_eq!(mapped, Vec3::new(3.0, 5.0, 7.0));
    }

    #[test]
    fn test_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = v.normalize();
        assert!((n.mag() - 1.0).abs() < 1e-10);
        assert!(n.x > 0.0 && n.y > 0.0);
    }

    #[test]
    fn test_normalize_zero() {
        let v = Vec3::ZERO;
        let n = v.normalize();
        assert_eq!(n, Vec3::ZERO);
    }

    #[test]
    fn test_angle_to() {
        let x = Vec3::UNIT_X;
        let y = Vec3::UNIT_Y;
        let z = Vec3::UNIT_Z;

        // Orthogonal vectors → π/2
        assert!((x.angle_to(&y) - FRAC_PI_2).abs() < 1e-10);
        assert!((y.angle_to(&z) - FRAC_PI_2).abs() < 1e-10);

        // Same direction → 0
        assert!(x.angle_to(&x) < EPSILON);

        // Opposite direction → π
        assert!((x.angle_to(&-x) - PI).abs() < 1e-10);

        // Non-unit vectors
        let a = Vec3::new(2.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 3.0, 0.0);
        assert!((a.angle_to(&b) - FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_angle_to_zero_vector() {
        let v = Vec3::UNIT_X;
        let zero = Vec3::ZERO;
        assert_eq!(v.angle_to(&zero), 0.0);
        assert_eq!(zero.angle_to(&v), 0.0);
    }

    #[test]
    fn test_dist() {
        let v1 = Point3D::new(1.0, 2.0, 3.0);
        let v2 = Point3D::new(4.0, 6.0, 3.0);
        assert!((v1.dist(&v2) - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_dist_sq() {
        let v1 = Point3D::new(1.0, 2.0, 3.0);
        let v2 = Point3D::new(4.0, 6.0, 3.0);
        assert_eq!(v1.dist_sq(&v2), 25.0);
    }

    #[test]
    fn test_display() {
        let v = Vec3::new(1.5, -2.5, 3.0);
        assert_eq!(format!("{}", v), "(1.5, -2.5, 3)");
    }

    #[test]
    fn test_partial_eq_epsilon() {
        let v1 = Vec3::new(1.0, 1.0, 1.0);
        let v2 = Vec3::new(1.0 + EPSILON / 2.0, 1.0, 1.0);
        assert_eq!(v1, v2); // Within tolerance

        let v3 = Vec3::new(1.0 + 2.0 * EPSILON, 1.0, 1.0);
        assert_ne!(v1, v3); // Outside tolerance
    }

    #[test]
    fn test_partial_ord() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(1.0, 2.0, 4.0);
        assert!(v1 < v2); // Lexicographic comparison
    }

    #[test]
    fn test_vec3_mul_mat3() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let m = Mat3::from_scale(2.0, 3.0, 4.0); // diag(2,3,4)

        // Row vector: [1,2,3] * diag(2,3,4) = [1*2, 2*3, 3*4] = [2,6,12]
        let result = v * m;
        assert_eq!(result, Vec3::new(2.0, 6.0, 12.0));

        // Identity
        assert_eq!(v * Mat3::IDENTITY, v);

        // Test with non-diagonal matrix
        // Let M = [[1,0,0],
        //          [1,1,0],
        //          [0,0,1]]
        let m = Mat3::new(1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let v = Vec3::new(2.0, 3.0, 4.0);
        // [2,3,4] * M = [
        //   2*1 + 3*1 + 4*0,
        //   2*0 + 3*1 + 4*0,
        //   2*0 + 3*0 + 4*1
        // ] = [5, 3, 4]
        let result = v * m;
        assert_eq!(result, Vec3::new(5.0, 3.0, 4.0));
    }

    #[test]
    fn test_vec3_mul_mat3_rotation_consistency() {
        // For row vectors, v * R should equal (R^T * v_col)^T
        // We'll test with Z-rotation
        let angle = FRAC_PI_2; // 90°
        let rot_z = Mat3::from_rotation_z(angle); // column-vector rotation matrix

        let v_row = Vec3::UNIT_X; // (1, 0, 0)
        let result_row = v_row * rot_z;

        // Column-vector way: rot_z * Vec3::UNIT_X = (0, 1, 0)
        // So row-vector result should be (0, 1, 0) as well?
        // Actually, no: row * R = (R^T * col)^T
        // R^T for Z-rotation is rotation by -angle
        let expected = Vec3::new(0.0, -1.0, 0.0); // because cos(-90)=0, sin(-90)=-1

        // But let's compute directly from your implementation:
        // rot_z = [[cos, -sin, 0],
        //          [sin,  cos, 0],
        //          [0,    0,   1]]
        // v_row * rot_z = [
        //   1*cos + 0*sin + 0*0,
        //   1*(-sin) + 0*cos + 0*0,
        //   0
        // ] = [cos, -sin, 0] = [0, -1, 0]
        assert!((result_row - expected).mag() < 1e-10);
    }
}
