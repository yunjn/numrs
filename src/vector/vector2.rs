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

    pub fn angle_to(&self, other: &Vec2) -> f64 {
        let mag_self = self.mag();
        let mag_other = other.mag();

        if mag_self < EPSILON || mag_other < EPSILON {
            return 0.0;
        }

        let cos_theta = self.dot(other) / (mag_self * mag_other);
        cos_theta.clamp(-1.0, 1.0).acos()
    }

    pub fn angle_to_signed(&self, other: &Vec2) -> f64 {
        let mag_self = self.mag();
        let mag_other = other.mag();

        if mag_self < EPSILON || mag_other < EPSILON {
            return 0.0;
        }

        let dot = self.dot(other);
        let det = self.cross(other);
        det.atan2(dot)
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
        (*self - *other).mag()
    }

    pub fn dist_sq(&self, other: &Point2D) -> f64 {
        (*self - *other).mag_sq()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn test_constants() {
        assert_eq!(Vec2::ZERO, Vec2::new(0.0, 0.0));
        assert_eq!(Vec2::ONE, Vec2::new(1.0, 1.0));
        assert_eq!(Vec2::UNIT_X, Vec2::new(1.0, 0.0));
        assert_eq!(Vec2::UNIT_Y, Vec2::new(0.0, 1.0));
    }

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
        let v1 = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(3.0, 4.0);
        assert_eq!(v1 + v2, Vec2::new(4.0, 6.0));
    }

    #[test]
    fn test_sub() {
        let v1 = Vec2::new(5.0, 7.0);
        let v2 = Vec2::new(2.0, 3.0);
        assert_eq!(v1 - v2, Vec2::new(3.0, 4.0));
    }

    #[test]
    fn test_scalar_mul_left() {
        let v = Vec2::new(1.0, 2.0);
        assert_eq!(3.0 * v, Vec2::new(3.0, 6.0));
    }

    #[test]
    fn test_scalar_mul_right() {
        let v = Vec2::new(1.0, 2.0);
        assert_eq!(v * 3.0, Vec2::new(3.0, 6.0));
    }

    #[test]
    fn test_scalar_div() {
        let v = Vec2::new(4.0, 6.0);
        assert_eq!(v / 2.0, Vec2::new(2.0, 3.0));
    }

    #[test]
    fn test_component_mul() {
        let v1 = Vec2::new(2.0, 3.0);
        let v2 = Vec2::new(4.0, 5.0);
        assert_eq!(v1 * v2, Vec2::new(8.0, 15.0));
    }

    #[test]
    fn test_mag() {
        let v = Vec2::new(3.0, 4.0);
        assert!((v.mag() - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_mag_sq() {
        let v = Vec2::new(3.0, 4.0);
        assert_eq!(v.mag_sq(), 25.0);
    }

    #[test]
    fn test_dot() {
        let v1 = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(3.0, 4.0);
        assert_eq!(v1.dot(&v2), 11.0);
    }

    #[test]
    fn test_cross() {
        let x = Vec2::UNIT_X;
        let y = Vec2::UNIT_Y;
        assert_eq!(x.cross(&y), 1.0); // X × Y = +1 (out of screen)
        assert_eq!(y.cross(&x), -1.0); // Y × X = -1
    }

    #[test]
    fn test_map() {
        let v = Vec2::new(1.0, 2.0);
        let mapped = v.map(|x| x * 2.0 + 1.0);
        assert_eq!(mapped, Vec2::new(3.0, 5.0));
    }

    #[test]
    fn test_normalize() {
        let v = Vec2::new(3.0, 4.0);
        let n = v.normalize();
        assert!((n.mag() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero() {
        let v = Vec2::ZERO;
        let n = v.normalize();
        assert_eq!(n, Vec2::ZERO);
    }

    #[test]
    fn test_angle_to() {
        let x = Vec2::UNIT_X;
        let y = Vec2::UNIT_Y;

        // Orthogonal → π/2
        assert!((x.angle_to(&y) - FRAC_PI_2).abs() < 1e-10);

        // Same direction → 0
        assert!(x.angle_to(&x) < EPSILON);

        // Opposite → π
        assert!((x.angle_to(&-x) - PI).abs() < 1e-10);

        // Non-unit vectors
        let a = Vec2::new(2.0, 0.0);
        let b = Vec2::new(0.0, 3.0);
        assert!((a.angle_to(&b) - FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_angle_to_signed() {
        let forward = Vec2::UNIT_X;
        let left = Vec2::UNIT_Y;
        let right = Vec2::new(0.0, -1.0);
        let back = -Vec2::UNIT_X;

        // Counter-clockwise (positive)
        assert!((forward.angle_to_signed(&left) - FRAC_PI_2).abs() < 1e-10);

        // Clockwise (negative)
        assert!((forward.angle_to_signed(&right) + FRAC_PI_2).abs() < 1e-10);

        // 180 degrees (π or -π, but atan2 returns π for (-1, 0))
        assert!((forward.angle_to_signed(&back) - PI).abs() < 1e-10);

        // Zero angle
        assert!(forward.angle_to_signed(&forward).abs() < EPSILON);
    }

    #[test]
    fn test_angle_to_zero_vector() {
        let v = Vec2::UNIT_X;
        let zero = Vec2::ZERO;
        assert_eq!(v.angle_to(&zero), 0.0);
        assert_eq!(v.angle_to_signed(&zero), 0.0);
    }

    #[test]
    fn test_dist() {
        let p1 = Point2D::new(1.0, 1.0);
        let p2 = Point2D::new(1.0, 2.0);
        assert!((p1.dist(&p2) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_dist_sq() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(1.0, 1.0);
        assert_eq!(p1.dist_sq(&p2), 2.0);
    }

    #[test]
    fn test_display() {
        let v = Vec2::new(1.5, -2.5);
        assert_eq!(format!("{}", v), "(1.5, -2.5)");
    }

    #[test]
    fn test_partial_eq_epsilon() {
        let v1 = Vec2::new(1.0, 1.0);
        let v2 = Vec2::new(1.0 + EPSILON / 2.0, 1.0);
        assert_eq!(v1, v2); // Within tolerance

        let v3 = Vec2::new(1.0 + 2.0 * EPSILON, 1.0);
        assert_ne!(v1, v3); // Outside tolerance
    }

    #[test]
    fn test_partial_ord() {
        let v1 = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(1.0, 3.0);
        assert!(v1 < v2); // Lexicographic order
    }

    #[test]
    fn test_dot_perpendicular() {
        let x = Vec2::UNIT_X;
        let y = Vec2::UNIT_Y;
        assert_eq!(x.dot(&y), 0.0);
    }
}
