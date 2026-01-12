use derive_more::{Add, Constructor, Div, Mul, Neg, Sub};
use std::fmt;

#[derive(Add, Sub, Div, Mul, Neg, Clone, Copy, Debug, PartialOrd, Constructor)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };
    pub const ONE: Vec2 = Vec2 { x: 1.0, y: 1.0 };
    pub const UNIT_X: Vec2 = Vec2 { x: 1.0, y: 0.0 };
    pub const UNIT_Y: Vec2 = Vec2 { x: 0.0, y: 1.0 };

    pub fn mag(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn mag_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    pub fn dot(&self, other: &Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    pub fn cross(&self, other: &Vec2) -> f64 {
        self.x * other.y - self.y * other.x
    }

    pub fn map<F>(&self, f: F) -> Vec2
    where
        F: Fn(f64) -> f64,
    {
        let x = f(self.x);
        let y = f(self.y);
        Vec2 { x, y }
    }

    pub fn normalize(&self) -> Vec2 {
        let mag = self.mag();
        if mag < EPSILON {
            Vec2::ZERO
        } else {
            Vec2 {
                x: self.x / mag,
                y: self.y / mag,
            }
        }
    }
}

impl Default for Vec2 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

const EPSILON: f64 = 1e-9;

impl PartialEq for Vec2 {
    fn eq(&self, other: &Self) -> bool {
        (self.x - other.x).abs() < EPSILON && (self.y - other.y).abs() < EPSILON
    }
}

impl std::ops::Mul<Vec2> for f64 {
    type Output = Vec2;
    fn mul(self, vec: Vec2) -> Self::Output {
        Vec2 {
            x: self * vec.x,
            y: self * vec.y,
        }
    }
}

impl std::ops::Mul<Vec2> for Vec2 {
    type Output = Vec2;
    fn mul(self, vector: Vec2) -> Vec2 {
        Vec2 {
            x: self.x * vector.x,
            y: self.y * vector.y,
        }
    }
}

pub type Point2D = Vec2;

impl Point2D {
    pub fn dist(&self, other: &Point2D) -> f64 {
        (*self - *other).mag().abs()
    }

    pub fn dist_sq(&self, other: &Point2D) -> f64 {
        (*self - *other).mag_sq().abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        assert_eq!(Vec2::default(), Vec2::ZERO);
    }

    #[test]
    fn test_neg() {
        let v = Vec2::new(1.0, -2.0);
        assert_eq!(-v, Vec2::new(-1.0, 2.0));
    }

    #[test]
    fn test_add() {
        let v1: Vec2 = Vec2::UNIT_X;
        let v2: Vec2 = Vec2::UNIT_Y;
        assert_eq!(v1 + v2, Vec2::ONE);
    }

    #[test]
    fn test_sub() {
        let v1: Vec2 = Vec2::ONE * 2.0;
        let v2: Vec2 = Vec2::ONE;
        assert_eq!(v1 - v2, Vec2::ONE);
    }

    #[test]
    fn test_scalar_mul() {
        let v: Vec2 = Vec2::ONE;
        assert_eq!(2.0 * v, Vec2::new(2.0, 2.0));
    }

    #[test]
    fn test_scalar_div() {
        let v: Vec2 = Vec2::new(2.0, 2.0);
        assert_eq!(v / 2.0, Vec2::ONE);
    }

    #[test]
    fn test_mag() {
        let v: Vec2 = Vec2::UNIT_X * 3.0 + Vec2::UNIT_Y * 4.0;
        assert_eq!(v.mag(), 5.0);
    }

    #[test]
    fn test_mag_sq() {
        let v: Vec2 = Vec2::UNIT_X * 3.0 + Vec2::UNIT_Y * 4.0;
        assert_eq!(v.mag_sq(), 25.0);
    }

    #[test]
    fn test_dot() {
        let v1: Vec2 = Vec2::new(1.0, 2.0);
        let v2: Vec2 = Vec2::new(3.0, 4.0);
        assert_eq!(v1.dot(&v2), 11.0);
    }

    #[test]
    fn test_cross() {
        let v1: Vec2 = Vec2::UNIT_X;
        let v2: Vec2 = Vec2::UNIT_Y;
        assert_eq!(v1.cross(&v2), 1.0);
    }

    #[test]
    fn test_map() {
        let v: Vec2 = Vec2::ZERO;
        let mapped: Vec2 = v.map(|x| x + 1.0);
        assert_eq!(mapped, Vec2::ONE);
    }

    #[test]
    fn test_normalize() {
        let v: Vec2 = Vec2::new(3.0, 4.0);
        let n: Vec2 = v.normalize();
        assert!((n.mag() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dist() {
        let v1: Vec2 = Point2D::new(1.0, 1.0);
        let v2: Vec2 = Point2D::new(1.0, 2.0);
        assert_eq!(v1.dist(&v2), 1.0);
    }

    #[test]
    fn test_dist_sq() {
        let v1: Vec2 = Point2D::new(0.0, 0.0);
        let v2: Vec2 = Point2D::new(1.0, 1.0);
        assert_eq!(v1.dist_sq(&v2), 2.0);
    }

    #[test]
    fn test_zero_length_normalize() {
        let v: Vec2 = Vec2::ZERO;
        let n: Vec2 = v.normalize();
        assert_eq!(n, Vec2::ZERO);
    }

    #[test]
    fn test_dot_perpendicular() {
        let v1: Vec2 = Vec2::UNIT_X;
        let v2: Vec2 = Vec2::UNIT_Y;
        assert_eq!(v1.dot(&v2), 0.0);
    }

    #[test]
    fn test_component_mul() {
        let v1: Vec2 = Vec2::new(2.0, 3.0);
        let v2: Vec2 = Vec2::new(4.0, 5.0);
        assert_eq!(v1 * v2, Vec2::new(8.0, 15.0));
    }
}
