use crate::Vec3;
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use derive_more::{Add, Constructor, Div, Sub};
use std::ops::Mul;

#[repr(C)]
#[derive(Constructor, Copy, Clone, Debug, PartialEq, Add, Sub, PartialOrd, Div)]
pub struct Mat3 {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
    pub g: f64,
    pub h: f64,
    pub i: f64,
}

impl Mat3 {
    pub const IDENTITY: Mat3 = Mat3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    pub const ZERO: Mat3 = Mat3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    #[inline]
    pub fn determinant(&self) -> f64 {
        self.a * (self.e * self.i - self.f * self.h) - self.b * (self.d * self.i - self.f * self.g)
            + self.c * (self.d * self.h - self.e * self.g)
    }

    #[inline]
    pub fn inverse(&self) -> Option<Mat3> {
        let det = self.determinant();
        if det.abs_diff_eq(&0.0, 1e-9) {
            None
        } else {
            let inv_det = 1.0 / det;
            Some(Mat3::new(
                inv_det * (self.e * self.i - self.f * self.h),
                inv_det * (self.c * self.h - self.b * self.i),
                inv_det * (self.b * self.f - self.c * self.e),
                inv_det * (self.f * self.g - self.d * self.i),
                inv_det * (self.a * self.i - self.c * self.g),
                inv_det * (self.c * self.d - self.a * self.f),
                inv_det * (self.d * self.h - self.e * self.g),
                inv_det * (self.b * self.g - self.a * self.h),
                inv_det * (self.a * self.e - self.b * self.d),
            ))
        }
    }

    #[inline]
    pub fn transpose(&self) -> Mat3 {
        Mat3::new(
            self.a, self.d, self.g, self.b, self.e, self.h, self.c, self.f, self.i,
        )
    }

    #[inline]
    pub fn from_rotation_x(angle: f64) -> Mat3 {
        let (sin, cos) = angle.sin_cos();
        Mat3::new(1.0, 0.0, 0.0, 0.0, cos, -sin, 0.0, sin, cos)
    }

    #[inline]
    pub fn from_rotation_y(angle: f64) -> Mat3 {
        let (sin, cos) = angle.sin_cos();
        Mat3::new(cos, 0.0, sin, 0.0, 1.0, 0.0, -sin, 0.0, cos)
    }

    #[inline]
    pub fn from_rotation_z(angle: f64) -> Mat3 {
        let (sin, cos) = angle.sin_cos();
        Mat3::new(cos, -sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0)
    }

    #[inline]
    pub fn from_scale(sx: f64, sy: f64, sz: f64) -> Mat3 {
        Mat3::new(sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, sz)
    }
}

impl AbsDiffEq for Mat3 {
    type Epsilon = f64;
    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }
    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.to_array()
            .iter()
            .zip(other.to_array().iter())
            .all(|(a, b)| a.abs_diff_eq(b, epsilon))
    }
}

impl RelativeEq for Mat3 {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }
    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_rel: Self::Epsilon) -> bool {
        self.to_array()
            .iter()
            .zip(other.to_array().iter())
            .all(|(a, b)| a.relative_eq(b, epsilon, max_rel))
    }
}

impl UlpsEq for Mat3 {
    #[inline]
    fn default_max_ulps() -> u32 {
        f64::default_max_ulps()
    }
    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.to_array()
            .iter()
            .zip(other.to_array().iter())
            .all(|(a, b)| a.ulps_eq(b, epsilon, max_ulps))
    }
}

impl Default for Mat3 {
    #[inline]
    fn default() -> Self {
        Mat3::IDENTITY
    }
}

impl std::fmt::Display for Mat3 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "[{}, {}, {}]\n[{}, {}, {}]\n[{}, {}, {}]",
            self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h, self.i
        )
    }
}

impl Mat3 {
    #[inline]
    pub fn to_array(&self) -> [f64; 9] {
        unsafe { *(self as *const Mat3 as *const [f64; 9]) }
    }
}

impl Mul<f64> for Mat3 {
    type Output = Mat3;
    #[inline]
    fn mul(self, scalar: f64) -> Self::Output {
        Mat3::new(
            self.a * scalar,
            self.b * scalar,
            self.c * scalar,
            self.d * scalar,
            self.e * scalar,
            self.f * scalar,
            self.g * scalar,
            self.h * scalar,
            self.i * scalar,
        )
    }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Self::Output {
        Vec3::new(
            self.a * v.x + self.b * v.y + self.c * v.z,
            self.d * v.x + self.e * v.y + self.f * v.z,
            self.g * v.x + self.h * v.y + self.i * v.z,
        )
    }
}

impl Mul<Mat3> for Mat3 {
    type Output = Mat3;
    #[inline]
    fn mul(self, m: Mat3) -> Self::Output {
        Mat3::new(
            self.a * m.a + self.b * m.d + self.c * m.g,
            self.a * m.b + self.b * m.e + self.c * m.h,
            self.a * m.c + self.b * m.f + self.c * m.i,
            self.d * m.a + self.e * m.d + self.f * m.g,
            self.d * m.b + self.e * m.e + self.f * m.h,
            self.d * m.c + self.e * m.f + self.f * m.i,
            self.g * m.a + self.h * m.d + self.i * m.g,
            self.g * m.b + self.h * m.e + self.i * m.h,
            self.g * m.c + self.h * m.f + self.i * m.i,
        )
    }
}

impl AsRef<[f64; 9]> for Mat3 {
    #[inline]
    fn as_ref(&self) -> &[f64; 9] {
        unsafe { &*(self as *const Mat3 as *const [f64; 9]) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    // === 常量与基础 trait ===
    #[test]
    fn test_constants() {
        let id = Mat3::IDENTITY;
        assert_eq!(id.a, 1.0);
        assert_eq!(id.e, 1.0);
        assert_eq!(id.i, 1.0);
        assert_eq!(id.b, 0.0);
        assert_eq!(id.c, 0.0);
        assert_eq!(id.d, 0.0);
        assert_eq!(id.f, 0.0);
        assert_eq!(id.g, 0.0);
        assert_eq!(id.h, 0.0);

        let zero = Mat3::ZERO;
        assert_eq!(zero.a, 0.0);
        assert_eq!(zero.i, 0.0);
    }

    #[test]
    fn test_default() {
        assert_eq!(Mat3::default(), Mat3::IDENTITY);
    }

    #[test]
    fn test_display() {
        let m = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        let s = format!("{}", m);
        assert_eq!(s, "[1, 2, 3]\n[4, 5, 6]\n[7, 8, 9]");
    }

    #[test]
    fn test_matrix3_approx_tolerance() {
        // 定义公差（替代之前的 EPSILON）
        let epsilon = 1e-9;
        let m1 = Mat3::IDENTITY;

        let m2 = Mat3::new(1.0 + epsilon / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        // 这会自动遍历 9 个元素进行比较
        approx::assert_abs_diff_eq!(m1, m2, epsilon = epsilon);

        // 2. 在公差范围外：使用 assert_abs_diff_ne!
        let m3 = Mat3::new(1.0 + 2.0 * epsilon, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        approx::assert_abs_diff_ne!(m1, m3, epsilon = epsilon);
    }

    #[test]
    fn test_determinant() {
        // Diagonal matrix
        let diag = Mat3::new(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0);
        assert_eq!(diag.determinant(), 24.0);

        // Identity
        assert_eq!(Mat3::IDENTITY.determinant(), 1.0);

        // Singular matrix (linearly dependent rows)
        let singular = Mat3::new(1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0);
        approx::assert_relative_eq!(singular.determinant(), 0.0);
    }

    #[test]
    fn test_determinant_correctness() {
        // From: https://www.mathsisfun.com/algebra/matrix-determinant.html
        // Matrix:
        // [6 1 1]
        // [4 -2 5]
        // [2 8 7]
        // Determinant = 6*(-2*7 - 5*8) - 1*(4*7 - 5*2) + 1*(4*8 - (-2)*2)
        //             = 6*(-14 - 40) - 1*(28 - 10) + 1*(32 + 4)
        //             = 6*(-54) - 18 + 36 = -324 -18 + 36 = -306
        let m = Mat3::new(6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0);
        approx::assert_relative_eq!(m.determinant(), -306.0);
    }

    #[test]
    fn test_inverse() {
        // Invertible matrix
        let m = Mat3::new(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0);
        let inv = m.inverse().expect("Should be invertible");
        let product = m * inv;
        assert_eq!(product, Mat3::IDENTITY);

        // Non-invertible matrix
        let singular = Mat3::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert!(singular.inverse().is_none());

        // Verify inverse formula with known case
        let m = Mat3::new(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0);
        let inv = m.inverse().unwrap();
        let reconstructed = m * inv;
        assert_eq!(reconstructed, Mat3::IDENTITY);
    }

    #[test]
    fn test_transpose() {
        let m = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        let t = m.transpose();
        let expected = Mat3::new(1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0);
        assert_eq!(t, expected);

        // Transpose of symmetric matrix is itself
        let sym = Mat3::new(1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0);
        assert_eq!(sym.transpose(), sym);
    }

    // === 标量运算 ===
    #[test]
    fn test_mul_scalar() {
        let m = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        let result = m * 2.0;
        let expected = Mat3::new(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_div_scalar() {
        let m = Mat3::new(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0);
        let result = m / 2.0;
        let expected = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        assert_eq!(result, expected);

        // Division by negative scalar
        let result = Mat3::IDENTITY / -1.0;
        let expected = Mat3::new(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);
        assert_eq!(result, expected);
    }

    // === 向量变换 ===
    #[test]
    fn test_mul_vec3() {
        // Scale transformation
        let scale = Mat3::from_scale(2.0, 3.0, 4.0);
        let v = Vec3::new(1.0, 1.0, 1.0);
        let result = scale * v;
        assert_eq!(result, Vec3::new(2.0, 3.0, 4.0));

        // Zero vector
        let zero_v = Vec3::ZERO;
        assert_eq!(scale * zero_v, Vec3::ZERO);

        // Identity transformation
        assert_eq!(Mat3::IDENTITY * v, v);
    }

    // === 矩阵乘法 ===
    #[test]
    fn test_mul_mat3() {
        // Diagonal matrices
        let a = Mat3::new(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0);
        let b = Mat3::new(5.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 7.0);
        let result = a * b;
        let expected = Mat3::new(10.0, 0.0, 0.0, 0.0, 18.0, 0.0, 0.0, 0.0, 28.0);
        assert_eq!(result, expected);

        // Non-diagonal
        let a = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        let b = Mat3::IDENTITY;
        assert_eq!(a * b, a);
        assert_eq!(b * a, a);
    }

    #[test]
    fn test_matrix_multiplication_associativity() {
        let a = Mat3::from_rotation_x(FRAC_PI_2);
        let b = Mat3::from_rotation_y(FRAC_PI_2);
        let c = Mat3::from_scale(2.0, 2.0, 2.0);

        let left = (a * b) * c;
        let right = a * (b * c);
        assert_eq!(left, right);
    }

    // === 构造函数 ===
    #[test]
    fn test_from_scale() {
        let m = Mat3::from_scale(2.0, 3.0, 4.0);
        let expected = Mat3::new(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0);
        assert_eq!(m, expected);
    }

    #[test]
    fn test_from_rotation_x() {
        // 旋转 2PI 理论上应该回到原位（单位矩阵）
        let m = Mat3::from_rotation_x(2.0 * PI);
        let identity = Mat3::IDENTITY;

        approx::assert_relative_eq!(m, identity, epsilon = 1e-15);
    }

    #[test]
    fn test_from_rotation_y() {
        let rot = Mat3::from_rotation_y(FRAC_PI_2); // 90° around Y
        let v = Vec3::new(1.0, 0.0, 0.0); // Point along X
        let result = rot * v;
        // Should become (0, 0, -1)
        assert!((result - Vec3::new(0.0, 0.0, -1.0)).mag() < 1e-10);
    }

    #[test]
    fn test_from_rotation_z() {
        let rot = Mat3::from_rotation_z(FRAC_PI_2); // 90° around Z
        let v = Vec3::new(1.0, 0.0, 0.0); // Point along X
        let result = rot * v;
        // Should become (0, 1, 0)
        assert!((result - Vec3::UNIT_Y).mag() < 1e-10);
    }

    // === 边界与错误处理 ===
    #[test]
    fn test_inverse_zero_matrix() {
        assert!(Mat3::ZERO.inverse().is_none());
    }

    #[test]
    fn test_div_by_zero() {
        let m = Mat3::IDENTITY;
        let result = m / 0.0; // f64 allows inf, but we test behavior
        assert!(result.a.is_infinite());
    }

    // === 数值稳定性 ===
    #[test]
    fn test_inverse_numerical_stability() {
        // Nearly singular matrix
        let eps = 1e-8;
        let nearly_singular = Mat3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, eps);
        let inv = nearly_singular.inverse().unwrap();
        let reconstructed = nearly_singular * inv;
        // Should be close to identity
        assert!((reconstructed.a - 1.0).abs() < 1e-6);
        assert!((reconstructed.e - 1.0).abs() < 1e-6);
        assert!((reconstructed.i - 1.0).abs() < 1e-6);
    }
}
