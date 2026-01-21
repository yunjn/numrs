use crate::Vec3;
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use derive_more::{Add, Constructor, Div, Mul, Neg, Sub};

#[repr(C)]
#[derive(Add, Sub, Mul, Div, Neg, Clone, Copy, Debug, PartialEq, PartialOrd, Constructor)]
pub struct Vec4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Vec4 {
    pub const ZERO: Vec4 = Vec4::new(0.0, 0.0, 0.0, 0.0);
    pub const ONE: Vec4 = Vec4::new(1.0, 1.0, 1.0, 1.0);
    pub const UNIT_X: Vec4 = Vec4::new(1.0, 0.0, 0.0, 0.0);
    pub const UNIT_Y: Vec4 = Vec4::new(0.0, 1.0, 0.0, 0.0);
    pub const UNIT_Z: Vec4 = Vec4::new(0.0, 0.0, 1.0, 0.0);
    pub const UNIT_W: Vec4 = Vec4::new(0.0, 0.0, 0.0, 1.0);

    #[inline]
    pub fn mag(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    #[inline]
    pub fn mag_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    #[inline]
    pub fn dot(&self, other: &Vec4) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    #[inline]
    pub fn map<F>(&self, f: F) -> Vec4
    where
        F: Fn(f64) -> f64,
    {
        Vec4::new(f(self.x), f(self.y), f(self.z), f(self.w))
    }

    #[inline]
    pub fn normalize(&self) -> Vec4 {
        let mag = self.mag();
        if mag.abs_diff_eq(&0.0, f64::default_epsilon()) {
            Vec4::ZERO
        } else {
            let inv = 1.0 / mag;
            Vec4::new(self.x * inv, self.y * inv, self.z * inv, self.w * inv)
        }
    }

    #[inline]
    pub fn truncate(&self) -> Vec3 {
        crate::Vec3::new(self.x, self.y, self.z)
    }

    #[inline]
    pub fn with_w(&self, w: f64) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, w)
    }

    #[inline]
    pub fn is_homogeneous(&self) -> bool {
        self.w.abs_diff_eq(&1.0, 1e-10)
    }

    #[inline]
    pub fn to_homogeneous(&self) -> Vec4 {
        if self.w.abs_diff_eq(&0.0, f64::default_epsilon()) {
            *self
        } else {
            Vec4::new(self.x / self.w, self.y / self.w, self.z / self.w, 1.0)
        }
    }
}

impl AbsDiffEq for Vec4 {
    type Epsilon = f64;
    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.x.abs_diff_eq(&other.x, epsilon)
            && self.y.abs_diff_eq(&other.y, epsilon)
            && self.z.abs_diff_eq(&other.z, epsilon)
            && self.w.abs_diff_eq(&other.w, epsilon)
    }
}

impl RelativeEq for Vec4 {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }
    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_rel: Self::Epsilon) -> bool {
        self.x.relative_eq(&other.x, epsilon, max_rel)
            && self.y.relative_eq(&other.y, epsilon, max_rel)
            && self.z.relative_eq(&other.z, epsilon, max_rel)
            && self.w.relative_eq(&other.w, epsilon, max_rel)
    }
}

impl UlpsEq for Vec4 {
    #[inline]
    fn default_max_ulps() -> u32 {
        f64::default_max_ulps()
    }
    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.x.ulps_eq(&other.x, epsilon, max_ulps)
            && self.y.ulps_eq(&other.y, epsilon, max_ulps)
            && self.z.ulps_eq(&other.z, epsilon, max_ulps)
            && self.w.ulps_eq(&other.w, epsilon, max_ulps)
    }
}

impl Default for Vec4 {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::fmt::Display for Vec4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl std::ops::Mul<Vec4> for f64 {
    type Output = Vec4;
    #[inline]
    fn mul(self, vec: Vec4) -> Self::Output {
        Vec4::new(self * vec.x, self * vec.y, self * vec.z, self * vec.w)
    }
}

impl std::ops::Mul<Vec4> for Vec4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, vec: Vec4) -> Self::Output {
        Vec4::new(
            self.x * vec.x,
            self.y * vec.y,
            self.z * vec.z,
            self.w * vec.w,
        )
    }
}

impl From<[f64; 4]> for Vec4 {
    #[inline]
    fn from(arr: [f64; 4]) -> Self {
        Self::new(arr[0], arr[1], arr[2], arr[3])
    }
}

impl From<Vec4> for [f64; 4] {
    #[inline]
    fn from(v: Vec4) -> Self {
        [v.x, v.y, v.z, v.w]
    }
}

impl From<Vec3> for Vec4 {
    #[inline]
    fn from(v: Vec3) -> Self {
        Vec4::new(v.x, v.y, v.z, 0.0)
    }
}

impl From<(Vec3, f64)> for Vec4 {
    #[inline]
    fn from((v, w): (Vec3, f64)) -> Self {
        Vec4::new(v.x, v.y, v.z, w)
    }
}

pub type Point4D = Vec4;

impl Point4D {
    #[inline]
    pub fn dist(&self, other: &Point4D) -> f64 {
        (*self - *other).mag()
    }
    #[inline]
    pub fn dist_sq(&self, other: &Point4D) -> f64 {
        (*self - *other).mag_sq()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(Vec4::ZERO, Vec4::new(0.0, 0.0, 0.0, 0.0));
        assert_eq!(Vec4::ONE, Vec4::new(1.0, 1.0, 1.0, 1.0));
        assert_eq!(Vec4::UNIT_X, Vec4::new(1.0, 0.0, 0.0, 0.0));
        assert_eq!(Vec4::UNIT_Y, Vec4::new(0.0, 1.0, 0.0, 0.0));
        assert_eq!(Vec4::UNIT_Z, Vec4::new(0.0, 0.0, 1.0, 0.0));
        assert_eq!(Vec4::UNIT_W, Vec4::new(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_default() {
        assert_eq!(Vec4::default(), Vec4::ZERO);
    }

    #[test]
    fn test_neg() {
        let v = Vec4::new(1.0, -2.0, 3.0, -4.0);
        assert_eq!(-v, Vec4::new(-1.0, 2.0, -3.0, 4.0));
    }

    #[test]
    fn test_add() {
        let v1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let v2 = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(v1 + v2, Vec4::new(6.0, 8.0, 10.0, 12.0));
    }

    #[test]
    fn test_sub() {
        let v1 = Vec4::new(6.0, 8.0, 10.0, 12.0);
        let v2 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v1 - v2, Vec4::new(5.0, 6.0, 7.0, 8.0));
    }

    #[test]
    fn test_scalar_mul_left() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(2.0 * v, Vec4::new(2.0, 4.0, 6.0, 8.0));
    }

    #[test]
    fn test_scalar_mul_right() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v * 2.0, Vec4::new(2.0, 4.0, 6.0, 8.0));
    }

    #[test]
    fn test_scalar_div() {
        let v = Vec4::new(2.0, 4.0, 6.0, 8.0);
        assert_eq!(v / 2.0, Vec4::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_component_mul() {
        let v1 = Vec4::new(2.0, 3.0, 4.0, 5.0);
        let v2 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v1 * v2, Vec4::new(2.0, 6.0, 12.0, 20.0));
    }

    #[test]
    fn test_mag() {
        let v = Vec4::new(3.0, 0.0, 0.0, 0.0);
        approx::assert_relative_eq!(v.mag(), 3.0);
    }

    #[test]
    fn test_mag_sq() {
        let v = Vec4::new(3.0, 0.0, 0.0, 0.0);
        assert_eq!(v.mag_sq(), 9.0);
    }

    #[test]
    fn test_dot() {
        let v1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let v2 = Vec4::new(5.0, -6.0, 7.0, 8.0);
        assert_eq!(v1.dot(&v2), 1.0 * 5.0 + 2.0 * -6.0 + 3.0 * 7.0 + 4.0 * 8.0);
    }

    #[test]
    fn test_map() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let mapped = v.map(|x| x * 2.0 + 1.0);
        assert_eq!(mapped, Vec4::new(3.0, 5.0, 7.0, 9.0));
    }

    #[test]
    fn test_normalize() {
        let v = Vec4::new(3.0, 4.0, 0.0, 0.0);
        let n = v.normalize();
        assert!((n.mag() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero() {
        let v = Vec4::ZERO;
        let n = v.normalize();
        assert_eq!(n, Vec4::ZERO);
    }

    #[test]
    fn test_truncate() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.truncate(), crate::Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_with_w() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.with_w(5.0), Vec4::new(1.0, 2.0, 3.0, 5.0));
    }

    #[test]
    fn test_is_homogeneous() {
        let v1 = Vec4::new(1.0, 2.0, 3.0, 1.0);
        let v2 = Vec4::new(1.0, 2.0, 3.0, 2.0);
        assert!(v1.is_homogeneous());
        assert!(!v2.is_homogeneous());
    }

    #[test]
    fn test_to_homogeneous() {
        let v = Vec4::new(2.0, 4.0, 6.0, 2.0);
        let h = v.to_homogeneous();
        assert!((h.x - 1.0).abs() < 1e-10);
        assert!((h.y - 2.0).abs() < 1e-10);
        assert!((h.z - 3.0).abs() < 1e-10);
        assert!((h.w - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_homogeneous_zero_w() {
        let v = Vec4::new(1.0, 2.0, 3.0, 0.0);
        let h = v.to_homogeneous();
        assert_eq!(h, v);
    }

    #[test]
    fn test_display() {
        let v = Vec4::new(1.5, -2.5, 3.0, 4.0);
        assert_eq!(format!("{}", v), "(1.5, -2.5, 3, 4)");
    }

    #[test]
    fn test_into_array() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let arr: [f64; 4] = v.into();
        assert_eq!(arr, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dist() {
        let v1 = Point4D::new(1.0, 2.0, 3.0, 4.0);
        let v2 = Point4D::new(5.0, 6.0, 7.0, 8.0);
        approx::assert_relative_eq!(v1.dist(&v2), v1.dist_sq(&v2).sqrt());
    }

    #[test]
    fn test_dist_sq() {
        let v1 = Point4D::new(1.0, 2.0, 3.0, 4.0);
        let v2 = Point4D::new(4.0, 6.0, 7.0, 8.0);
        let dx = 4.0 - 1.0;
        let dy = 6.0 - 2.0;
        let dz = 7.0 - 3.0;
        let dw = 8.0 - 4.0;
        assert_eq!(v1.dist_sq(&v2), dx * dx + dy * dy + dz * dz + dw * dw);
    }

    #[test]
    fn test_approx_tolerance() {
        let custom_epsilon = 1e-9;
        let v1 = Vec4::new(1.0, 1.0, 1.0, 1.0);
        let v2 = Vec4::new(1.0 + custom_epsilon / 2.0, 1.0, 1.0, 1.0);
        approx::assert_abs_diff_eq!(v1, v2, epsilon = custom_epsilon);
        let v3 = Vec4::new(1.0 + 2.0 * custom_epsilon, 1.0, 1.0, 1.0);
        approx::assert_abs_diff_ne!(v1, v3, epsilon = custom_epsilon);
    }

    #[test]
    fn test_partial_ord() {
        let v1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let v2 = Vec4::new(1.0, 2.0, 3.0, 5.0);
        assert!(v1 < v2);
    }
}
