use crate::EPSILON;
use crate::Vec2;
use derive_more::{Add, Constructor, Div, Sub};
use std::ops::Mul;

#[derive(Constructor, Copy, Clone, Debug, PartialOrd, Add, Sub, Div)]
pub struct Mat2 {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl Mat2 {
    pub const IDENTITY: Mat2 = Mat2 {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
    };

    pub const ZERO: Mat2 = Mat2 {
        a: 0.0,
        b: 0.0,
        c: 0.0,
        d: 0.0,
    };

    pub fn determinant(&self) -> f64 {
        self.a * self.d - self.b * self.c
    }

    pub fn inverse(&self) -> Option<Mat2> {
        let det = self.determinant();
        if det.abs() < EPSILON {
            None
        } else {
            Some(Mat2::new(self.d, -self.b, -self.c, self.a) / det)
        }
    }

    pub fn transpose(&self) -> Mat2 {
        Mat2::new(self.a, self.c, self.b, self.d)
    }

    pub fn from_angle(angle: f64) -> Mat2 {
        let (sin, cos) = angle.sin_cos();
        Mat2::new(cos, -sin, sin, cos)
    }

    pub fn from_scale(sx: f64, sy: f64) -> Mat2 {
        Mat2::new(sx, 0.0, 0.0, sy)
    }
}

impl Default for Mat2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl std::fmt::Display for Mat2 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "[{}, {}]\n[{}, {}]", self.a, self.b, self.c, self.d)
    }
}

impl PartialEq for Mat2 {
    fn eq(&self, other: &Self) -> bool {
        (self.a - other.a).abs() < EPSILON
            && (self.b - other.b).abs() < EPSILON
            && (self.c - other.c).abs() < EPSILON
            && (self.d - other.d).abs() < EPSILON
    }
}

impl Mul<f64> for Mat2 {
    type Output = Mat2;

    fn mul(self, rhs: f64) -> Self::Output {
        Mat2::new(self.a * rhs, self.b * rhs, self.c * rhs, self.d * rhs)
    }
}

impl Mul<Vec2> for Mat2 {
    type Output = Vec2;

    fn mul(self, v: Vec2) -> Self::Output {
        Vec2::new(self.a * v.x + self.b * v.y, self.c * v.x + self.d * v.y)
    }
}

impl Mul<Mat2> for Mat2 {
    type Output = Mat2;

    fn mul(self, rhs: Mat2) -> Self::Output {
        Mat2::new(
            self.a * rhs.a + self.b * rhs.c,
            self.a * rhs.b + self.b * rhs.d,
            self.c * rhs.a + self.d * rhs.c,
            self.c * rhs.b + self.d * rhs.d,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn test_constants() {
        assert_eq!(Mat2::IDENTITY, Mat2::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(Mat2::ZERO, Mat2::new(0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_default() {
        assert_eq!(Mat2::default(), Mat2::IDENTITY);
    }

    #[test]
    fn test_determinant() {
        let m = Mat2::new(2.0, 0.0, 0.0, 3.0); // diag(2,3)
        assert_eq!(m.determinant(), 6.0);

        let m2 = Mat2::IDENTITY;
        assert_eq!(m2.determinant(), 1.0);

        let m3 = Mat2::ZERO;
        assert_eq!(m3.determinant(), 0.0);
    }

    #[test]
    fn test_inverse() {
        let m = Mat2::new(2.0, 0.0, 0.0, 4.0);
        let inv = m.inverse().unwrap();
        let identity = m * inv;
        assert_eq!(identity, Mat2::IDENTITY);

        // Singular matrix
        let singular = Mat2::new(1.0, 2.0, 2.0, 4.0); // det = 0
        assert!(singular.inverse().is_none());
    }

    #[test]
    fn test_transpose() {
        let m = Mat2::new(1.0, 2.0, 3.0, 4.0);
        let t = m.transpose();
        assert_eq!(t, Mat2::new(1.0, 3.0, 2.0, 4.0));
    }

    #[test]
    fn test_mul_scalar() {
        let m = Mat2::new(1.0, 2.0, 3.0, 4.0);
        let result = m * 2.0;
        assert_eq!(result, Mat2::new(2.0, 4.0, 6.0, 8.0));
    }

    #[test]
    fn test_div_scalar() {
        let m = Mat2::new(2.0, 4.0, 6.0, 8.0);
        let result = m / 2.0;
        assert_eq!(result, Mat2::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_mul_vec2() {
        let m = Mat2::new(2.0, 0.0, 0.0, 3.0); // scale(2,3)
        let v = Vec2::new(1.0, 1.0);
        let result = m * v;
        assert_eq!(result, Vec2::new(2.0, 3.0));
    }

    #[test]
    fn test_mul_mat2() {
        let a = Mat2::new(1.0, 2.0, 3.0, 4.0);
        let b = Mat2::new(5.0, 6.0, 7.0, 8.0);
        let result = a * b;
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(result, Mat2::new(19.0, 22.0, 43.0, 50.0));
    }

    #[test]
    fn test_identity_multiplication() {
        let m = Mat2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(Mat2::IDENTITY * m, m);
        assert_eq!(m * Mat2::IDENTITY, m);
    }

    #[test]
    fn test_rotation_matrix() {
        // 90-degree rotation: (1,0) -> (0,1)
        let rot = Mat2::from_angle(FRAC_PI_2);
        let v = Vec2::UNIT_X;
        let result = rot * v;
        assert!((result - Vec2::UNIT_Y).mag() < 1e-10);
    }

    #[test]
    fn test_scale_matrix() {
        let scale = Mat2::from_scale(2.0, 3.0);
        let v = Vec2::new(1.0, 1.0);
        let result = scale * v;
        assert_eq!(result, Vec2::new(2.0, 3.0));
    }

    #[test]
    fn test_partial_eq_with_epsilon() {
        let m1 = Mat2::new(1.0, 0.0, 0.0, 1.0);
        let m2 = Mat2::new(1.0 + EPSILON / 2.0, 0.0, 0.0, 1.0);
        assert_eq!(m1, m2);

        let m3 = Mat2::new(1.0 + 2.0 * EPSILON, 0.0, 0.0, 1.0);
        assert_ne!(m1, m3);
    }
}
