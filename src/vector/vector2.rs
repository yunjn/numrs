use crate::Mat2;
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use derive_more::{Add, Constructor, Div, Mul, Neg, Sub};

#[repr(C)]
#[derive(Add, Sub, Div, Mul, Neg, Clone, Copy, Debug, PartialEq, PartialOrd, Constructor)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2::new(0.0, 0.0);
    pub const ONE: Vec2 = Vec2::new(1.0, 1.0);
    pub const UNIT_X: Vec2 = Vec2::new(1.0, 0.0);
    pub const UNIT_Y: Vec2 = Vec2::new(0.0, 1.0);

    #[inline]
    pub fn mag(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    #[inline]
    pub fn mag_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    #[inline]
    pub fn dot(&self, other: &Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    #[inline]
    pub fn cross(&self, other: &Vec2) -> f64 {
        self.x * other.y - self.y * other.x
    }

    #[inline]
    pub fn map<F>(&self, f: F) -> Vec2
    where
        F: Fn(f64) -> f64,
    {
        Vec2::new(f(self.x), f(self.y))
    }

    #[inline]
    pub fn normalize(&self) -> Vec2 {
        let mag = self.mag();
        if mag.abs_diff_eq(&0.0, f64::default_epsilon()) {
            Vec2::ZERO
        } else {
            let inv = 1.0 / mag;
            Vec2::new(self.x * inv, self.y * inv)
        }
    }

    pub fn angle_to(&self, other: &Vec2) -> f64 {
        let mag_self = self.mag();
        let mag_other = other.mag();
        if mag_self.abs_diff_eq(&0.0, 1e-6) || mag_other.abs_diff_eq(&0.0, 1e-6) {
            return 0.0;
        }
        let cos_theta = self.dot(other) / (mag_self * mag_other);
        cos_theta.clamp(-1.0, 1.0).acos()
    }

    pub fn angle_to_signed(&self, other: &Vec2) -> f64 {
        let mag_self = self.mag();
        let mag_other = other.mag();
        if mag_self.abs_diff_eq(&0.0, 1e-6) || mag_other.abs_diff_eq(&0.0, 1e-6) {
            return 0.0;
        }
        let dot = self.dot(other);
        let det = self.cross(other);
        det.atan2(dot)
    }
}

impl AbsDiffEq for Vec2 {
    type Epsilon = f64;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.x.abs_diff_eq(&other.x, epsilon) && self.y.abs_diff_eq(&other.y, epsilon)
    }
}

impl RelativeEq for Vec2 {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.x.relative_eq(&other.x, epsilon, max_relative)
            && self.y.relative_eq(&other.y, epsilon, max_relative)
    }
}

impl UlpsEq for Vec2 {
    #[inline]
    fn default_max_ulps() -> u32 {
        f64::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.x.ulps_eq(&other.x, epsilon, max_ulps) && self.y.ulps_eq(&other.y, epsilon, max_ulps)
    }
}

impl Default for Vec2 {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::fmt::Display for Vec2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl std::ops::Mul<Vec2> for f64 {
    type Output = Vec2;
    #[inline]
    fn mul(self, vec: Vec2) -> Self::Output {
        Vec2::new(self * vec.x, self * vec.y)
    }
}

impl std::ops::Mul<Vec2> for Vec2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, vector: Vec2) -> Vec2 {
        Vec2::new(self.x * vector.x, self.y * vector.y)
    }
}

impl std::ops::Mul<Mat2> for Vec2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, m: Mat2) -> Vec2 {
        Vec2::new(self.x * m.a + self.y * m.c, self.x * m.b + self.y * m.d)
    }
}

impl From<[f64; 2]> for Vec2 {
    #[inline]
    fn from(arr: [f64; 2]) -> Self {
        Self::new(arr[0], arr[1])
    }
}

impl From<Vec2> for [f64; 2] {
    #[inline]
    fn from(v: Vec2) -> Self {
        [v.x, v.y]
    }
}

pub type Point2D = Vec2;

impl Point2D {
    #[inline]
    pub fn dist(&self, other: &Point2D) -> f64 {
        (*self - *other).mag()
    }
    #[inline]
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
        approx::assert_relative_eq!(v.mag(), 5.0);
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
        approx::assert_relative_eq!(x.angle_to(&x), 0.0);

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
        approx::assert_relative_eq!(forward.angle_to_signed(&forward), 0.0);
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
        approx::assert_relative_eq!(p1.dist(&p2), 1.0);
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

    #[test]
    fn test_vec2_mul_mat2() {
        let v = Vec2::new(1.0, 0.0);
        // 顺时针转90度 (行向量逻辑)
        // M = [[0, -1], [1, 0]] (Col1: 0,1; Col2: -1,0)
        // [1, 0] * M = [1*0 + 0*1, 1*-1 + 0*0] = [0, -1]
        let m = Mat2::new(0.0, -1.0, 1.0, 0.0);
        let res = v * m;
        assert_eq!(res, Vec2::new(0.0, -1.0));
    }
}
