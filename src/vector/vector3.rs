use derive_more::{Add, Constructor, Div, Mul, Neg, Sub};
use std::fmt;

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
}

impl Default for Vec3 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

const EPSILON: f64 = 1e-9;

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

pub type Point3D = Vec3;

impl Point3D {
    pub fn dist(&self, other: &Point3D) -> f64 {
        (*self - *other).mag().abs()
    }
    pub fn dist_sq(&self, other: &Point3D) -> f64 {
        (*self - *other).mag_sq().abs()
    }
}

#[cfg(test)]

mod tests {
    use super::*;

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
        let v1: Vec3 = Vec3::ZERO;
        let v2: Vec3 = Vec3::ONE;
        assert_eq!(v1 + v2, Vec3::ONE);
    }

    #[test]
    fn test_sub() {
        let v1: Vec3 = Vec3::ZERO;
        let v2: Vec3 = Vec3::ONE;
        assert_eq!(v2 - v1, Vec3::ONE);
    }

    #[test]
    fn test_mag() {
        let v: Vec3 = Vec3::new(1.0, 2.0, 2.0);
        assert_eq!(v.mag(), 3.0);
    }

    #[test]
    fn test_mag_sq() {
        let v: Vec3 = Vec3::new(1.0, 2.0, 2.0);
        assert_eq!(v.mag_sq(), 9.0);
    }

    #[test]
    fn test_dot() {
        let v1: Vec3 = Point3D::new(1.0, 2.0, 3.0);
        let v2: Vec3 = Point3D::new(4.0, -5.0, 6.0);
        assert_eq!(v1.dot(&v2), 12.0);
    }

    #[test]
    fn test_cross() {
        let v1: Vec3 = Vec3::new(1.0, 0.0, 0.0);
        let v2: Vec3 = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(v1.cross(&v2), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_map() {}

    #[test]
    fn test_normalize() {
        let v: Vec3 = Vec3::new(3.0, 4.0, 0.0);
        let n: Vec3 = v.normalize();
        assert!((n.mag() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dist() {
        let v1: Vec3 = Point3D::new(1.0, 2.0, 3.0);
        let v2: Vec3 = Point3D::new(4.0, 6.0, 3.0);
        assert_eq!(v1.dist(&v2), 5.0);
    }

    #[test]
    fn test_dist_sq() {
        let v1: Vec3 = Point3D::new(1.0, 2.0, 3.0);
        let v2: Vec3 = Point3D::new(4.0, 6.0, 3.0);
        assert_eq!(v1.dist_sq(&v2), 25.0);
    }

    #[test]
    fn test_scalar_mul() {
        let v: Vec3 = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(2.0 * v, Vec3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_scalar_div() {
        let v: Vec3 = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v / 2.0, Vec3::new(0.5, 1.0, 1.5));
    }

    #[test]
    fn test_component_mul() {
        let v1: Vec3 = Vec3::new(2.0, 3.0, 4.0);
        let v2: Vec3 = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v1 * v2, Vec3::new(2.0, 6.0, 12.0));
    }

    #[test]
    fn test_mul_mat3() {}
}
