use crate::{Mat3, Vec3, Vec4};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use derive_more::{Add, Constructor, Div, Sub};
use std::ops::Mul;

#[repr(C)]
#[derive(Constructor, Copy, Clone, Debug, PartialEq, Add, Sub, PartialOrd, Div)]
pub struct Mat4 {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
    pub g: f64,
    pub h: f64,
    pub i: f64,
    pub j: f64,
    pub k: f64,
    pub l: f64,
    pub m: f64,
    pub n: f64,
    pub o: f64,
    pub p: f64,
}

impl Mat4 {
    pub const IDENTITY: Mat4 = Mat4::new(
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    );
    pub const ZERO: Mat4 = Mat4::new(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    );

    #[inline]
    fn get(&self, row: usize, col: usize) -> f64 {
        match (row, col) {
            (0, 0) => self.a,
            (0, 1) => self.b,
            (0, 2) => self.c,
            (0, 3) => self.d,
            (1, 0) => self.e,
            (1, 1) => self.f,
            (1, 2) => self.g,
            (1, 3) => self.h,
            (2, 0) => self.i,
            (2, 1) => self.j,
            (2, 2) => self.k,
            (2, 3) => self.l,
            (3, 0) => self.m,
            (3, 1) => self.n,
            (3, 2) => self.o,
            (3, 3) => self.p,
            _ => 0.0,
        }
    }

    #[inline]
    fn set(&mut self, row: usize, col: usize, value: f64) {
        match (row, col) {
            (0, 0) => self.a = value,
            (0, 1) => self.b = value,
            (0, 2) => self.c = value,
            (0, 3) => self.d = value,
            (1, 0) => self.e = value,
            (1, 1) => self.f = value,
            (1, 2) => self.g = value,
            (1, 3) => self.h = value,
            (2, 0) => self.i = value,
            (2, 1) => self.j = value,
            (2, 2) => self.k = value,
            (2, 3) => self.l = value,
            (3, 0) => self.m = value,
            (3, 1) => self.n = value,
            (3, 2) => self.o = value,
            (3, 3) => self.p = value,
            _ => {}
        }
    }

    #[inline]
    fn minor(&self, row: usize, col: usize) -> f64 {
        let mut values: [f64; 9] = [0.0; 9];
        let mut idx = 0;
        for r in 0..4 {
            for c in 0..4 {
                if r != row && c != col {
                    values[idx] = self.get(r, c);
                    idx += 1;
                }
            }
        }
        let m = Mat3::new(
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
            values[8],
        );
        m.determinant()
    }

    #[inline]
    pub fn determinant(&self) -> f64 {
        self.a * self.minor(0, 0) - self.b * self.minor(0, 1) + self.c * self.minor(0, 2)
            - self.d * self.minor(0, 3)
    }

    #[inline]
    pub fn inverse(&self) -> Option<Mat4> {
        let det = self.determinant();
        if det.abs_diff_eq(&0.0, 1e-9) {
            return None;
        }

        let inv_det = 1.0 / det;
        let mut result = Mat4::ZERO;

        for row in 0..4 {
            for col in 0..4 {
                let sign = if (row + col) % 2 == 0 { 1.0 } else { -1.0 };
                let cofactor = sign * self.minor(row, col);
                result.set(col, row, inv_det * cofactor);
            }
        }

        Some(result)
    }

    #[inline]
    pub fn transpose(&self) -> Mat4 {
        Mat4::new(
            self.a, self.e, self.i, self.m, self.b, self.f, self.j, self.n, self.c, self.g, self.k,
            self.o, self.d, self.h, self.l, self.p,
        )
    }

    #[inline]
    pub fn from_scale(sx: f64, sy: f64, sz: f64, sw: f64) -> Mat4 {
        Mat4::new(
            sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, sz, 0.0, 0.0, 0.0, 0.0, sw,
        )
    }

    #[inline]
    pub fn from_translation(tx: f64, ty: f64, tz: f64) -> Mat4 {
        Mat4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, tx, ty, tz, 1.0,
        )
    }

    #[inline]
    pub fn from_rotation_x(angle: f64) -> Mat4 {
        let (sin, cos) = angle.sin_cos();
        Mat4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, cos, -sin, 0.0, 0.0, sin, cos, 0.0, 0.0, 0.0, 0.0, 1.0,
        )
    }

    #[inline]
    pub fn from_rotation_y(angle: f64) -> Mat4 {
        let (sin, cos) = angle.sin_cos();
        Mat4::new(
            cos, 0.0, sin, 0.0, 0.0, 1.0, 0.0, 0.0, -sin, 0.0, cos, 0.0, 0.0, 0.0, 0.0, 1.0,
        )
    }

    #[inline]
    pub fn from_rotation_z(angle: f64) -> Mat4 {
        let (sin, cos) = angle.sin_cos();
        Mat4::new(
            cos, -sin, 0.0, 0.0, sin, cos, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        )
    }

    #[inline]
    pub fn from_rotation_axis(axis: &Vec3, angle: f64) -> Mat4 {
        let (sin, cos) = angle.sin_cos();
        let one_minus_cos = 1.0 - cos;
        let x = axis.x;
        let y = axis.y;
        let z = axis.z;

        Mat4::new(
            one_minus_cos * x * x + cos,
            one_minus_cos * x * y - sin * z,
            one_minus_cos * x * z + sin * y,
            0.0,
            one_minus_cos * x * y + sin * z,
            one_minus_cos * y * y + cos,
            one_minus_cos * y * z - sin * x,
            0.0,
            one_minus_cos * x * z - sin * y,
            one_minus_cos * y * z + sin * x,
            one_minus_cos * z * z + cos,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    }

    #[inline]
    pub fn from_look_at(eye: &Vec3, target: &Vec3, up: &Vec3) -> Mat4 {
        let z_axis = (*eye - *target).normalize();
        let x_axis = up.cross(&z_axis).normalize();
        let y_axis = z_axis.cross(&x_axis).normalize();

        Mat4::new(
            x_axis.x,
            y_axis.x,
            z_axis.x,
            0.0,
            x_axis.y,
            y_axis.y,
            z_axis.y,
            0.0,
            x_axis.z,
            y_axis.z,
            z_axis.z,
            0.0,
            -x_axis.dot(eye),
            -y_axis.dot(eye),
            -z_axis.dot(eye),
            1.0,
        )
    }

    #[inline]
    pub fn from_perspective(fov_y: f64, aspect: f64, near: f64, far: f64) -> Mat4 {
        let f = 1.0 / (fov_y / 2.0).tan();
        let range_inv = 1.0 / (near - far);

        Mat4::new(
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            (near + far) * range_inv,
            -1.0,
            0.0,
            0.0,
            near * far * range_inv * 2.0,
            0.0,
        )
    }

    #[inline]
    pub fn from_perspective_inf(fov_y: f64, aspect: f64, near: f64) -> Mat4 {
        let f = 1.0 / (fov_y / 2.0).tan();
        Mat4::new(
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            -2.0 * near,
            0.0,
        )
    }

    #[inline]
    pub fn from_orthographic(
        left: f64,
        right: f64,
        bottom: f64,
        top: f64,
        near: f64,
        far: f64,
    ) -> Mat4 {
        let sx = 2.0 / (right - left);
        let sy = 2.0 / (top - bottom);
        let sz = 2.0 / (far - near);
        let tx = -(right + left) / (right - left);
        let ty = -(top + bottom) / (top - bottom);
        let tz = -(far + near) / (far - near);

        Mat4::new(
            sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, sz, 0.0, tx, ty, tz, 1.0,
        )
    }

    #[inline]
    pub fn to_array(&self) -> [f64; 16] {
        unsafe { *(self as *const Mat4 as *const [f64; 16]) }
    }

    #[inline]
    pub fn extract_3x3(&self) -> Mat3 {
        Mat3::new(
            self.a, self.b, self.c, self.e, self.f, self.g, self.i, self.j, self.k,
        )
    }
}

impl AbsDiffEq for Mat4 {
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

impl RelativeEq for Mat4 {
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

impl UlpsEq for Mat4 {
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

impl Default for Mat4 {
    #[inline]
    fn default() -> Self {
        Mat4::IDENTITY
    }
}

impl std::fmt::Display for Mat4 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "[[{}, {}, {}, {}],\n [{}, {}, {}, {}],\n [{}, {}, {}, {}],\n [{}, {}, {}, {}]]",
            self.a,
            self.b,
            self.c,
            self.d,
            self.e,
            self.f,
            self.g,
            self.h,
            self.i,
            self.j,
            self.k,
            self.l,
            self.m,
            self.n,
            self.o,
            self.p
        )
    }
}

impl Mul<f64> for Mat4 {
    type Output = Mat4;
    #[inline]
    fn mul(self, scalar: f64) -> Self::Output {
        Mat4::new(
            self.a * scalar,
            self.b * scalar,
            self.c * scalar,
            self.d * scalar,
            self.e * scalar,
            self.f * scalar,
            self.g * scalar,
            self.h * scalar,
            self.i * scalar,
            self.j * scalar,
            self.k * scalar,
            self.l * scalar,
            self.m * scalar,
            self.n * scalar,
            self.o * scalar,
            self.p * scalar,
        )
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, v: Vec4) -> Self::Output {
        Vec4::new(
            self.a * v.x + self.b * v.y + self.c * v.z + self.d * v.w,
            self.e * v.x + self.f * v.y + self.g * v.z + self.h * v.w,
            self.i * v.x + self.j * v.y + self.k * v.z + self.l * v.w,
            self.m * v.x + self.n * v.y + self.o * v.z + self.p * v.w,
        )
    }
}

impl Mul<Mat4> for Mat4 {
    type Output = Mat4;
    #[inline]
    fn mul(self, m: Mat4) -> Self::Output {
        Mat4::new(
            self.a * m.a + self.b * m.e + self.c * m.i + self.d * m.m,
            self.a * m.b + self.b * m.f + self.c * m.j + self.d * m.n,
            self.a * m.c + self.b * m.g + self.c * m.k + self.d * m.o,
            self.a * m.d + self.b * m.h + self.c * m.l + self.d * m.p,
            self.e * m.a + self.f * m.e + self.g * m.i + self.h * m.m,
            self.e * m.b + self.f * m.f + self.g * m.j + self.h * m.n,
            self.e * m.c + self.f * m.g + self.g * m.k + self.h * m.o,
            self.e * m.d + self.f * m.h + self.g * m.l + self.h * m.p,
            self.i * m.a + self.j * m.e + self.k * m.i + self.l * m.m,
            self.i * m.b + self.j * m.f + self.k * m.j + self.l * m.n,
            self.i * m.c + self.j * m.g + self.k * m.k + self.l * m.o,
            self.i * m.d + self.j * m.h + self.k * m.l + self.l * m.p,
            self.m * m.a + self.n * m.e + self.o * m.i + self.p * m.m,
            self.m * m.b + self.n * m.f + self.o * m.j + self.p * m.n,
            self.m * m.c + self.n * m.g + self.o * m.k + self.p * m.o,
            self.m * m.d + self.n * m.h + self.o * m.l + self.p * m.p,
        )
    }
}

impl AsRef<[f64; 16]> for Mat4 {
    #[inline]
    fn as_ref(&self) -> &[f64; 16] {
        unsafe { &*(self as *const Mat4 as *const [f64; 16]) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn test_constants() {
        let id = Mat4::IDENTITY;
        assert_eq!(id.a, 1.0);
        assert_eq!(id.f, 1.0);
        assert_eq!(id.k, 1.0);
        assert_eq!(id.p, 1.0);
        assert_eq!(id.b, 0.0);
        assert_eq!(id.c, 0.0);
        assert_eq!(id.d, 0.0);
        assert_eq!(id.e, 0.0);
        assert_eq!(id.g, 0.0);
        assert_eq!(id.h, 0.0);
        assert_eq!(id.i, 0.0);
        assert_eq!(id.j, 0.0);
        assert_eq!(id.l, 0.0);
        assert_eq!(id.m, 0.0);
        assert_eq!(id.n, 0.0);
        assert_eq!(id.o, 0.0);

        let zero = Mat4::ZERO;
        assert_eq!(zero.a, 0.0);
        assert_eq!(zero.p, 0.0);
    }

    #[test]
    fn test_default() {
        assert_eq!(Mat4::default(), Mat4::IDENTITY);
    }

    #[test]
    fn test_display() {
        let m = Mat4::IDENTITY;
        let s = format!("{}", m);
        assert!(s.contains("1"));
    }

    #[test]
    fn test_determinant_identity() {
        assert_eq!(Mat4::IDENTITY.determinant(), 1.0);
    }

    #[test]
    fn test_determinant_diagonal() {
        let diag = Mat4::from_scale(2.0, 3.0, 4.0, 5.0);
        assert_eq!(diag.determinant(), 120.0);
    }

    #[test]
    fn test_determinant_singular() {
        let singular = Mat4::new(
            1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0,
        );
        approx::assert_relative_eq!(singular.determinant(), 0.0);
    }

    #[test]
    fn test_inverse_identity() {
        let inv = Mat4::IDENTITY.inverse().unwrap();
        approx::assert_relative_eq!(inv, Mat4::IDENTITY, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_diagonal() {
        let m = Mat4::from_scale(2.0, 3.0, 4.0, 5.0);
        let inv = m.inverse().unwrap();
        let product = m * inv;
        approx::assert_relative_eq!(product, Mat4::IDENTITY, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_singular() {
        assert!(Mat4::ZERO.inverse().is_none());
    }

    #[test]
    fn test_transpose_identity() {
        assert_eq!(Mat4::IDENTITY.transpose(), Mat4::IDENTITY);
    }

    #[test]
    fn test_transpose() {
        let m = Mat4::new(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let t = m.transpose();
        assert_eq!(t.a, 1.0);
        assert_eq!(t.b, 5.0);
        assert_eq!(t.c, 9.0);
        assert_eq!(t.d, 13.0);
        assert_eq!(t.e, 2.0);
        assert_eq!(t.f, 6.0);
        assert_eq!(t.g, 10.0);
        assert_eq!(t.h, 14.0);
    }

    #[test]
    fn test_mul_scalar() {
        let m = Mat4::IDENTITY;
        assert_eq!(m * 2.0, Mat4::from_scale(2.0, 2.0, 2.0, 2.0));
    }

    #[test]
    fn test_mul_vec4() {
        let m = Mat4::from_scale(2.0, 3.0, 4.0, 5.0);
        let v = Vec4::new(1.0, 1.0, 1.0, 1.0);
        let result = m * v;
        assert_eq!(result, Vec4::new(2.0, 3.0, 4.0, 5.0));
    }

    #[test]
    fn test_mul_mat4() {
        let a = Mat4::from_scale(2.0, 3.0, 4.0, 1.0);
        let b = Mat4::from_scale(5.0, 6.0, 7.0, 1.0);
        let result = a * b;
        let expected = Mat4::from_scale(10.0, 18.0, 28.0, 1.0);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_scale() {
        let m = Mat4::from_scale(2.0, 3.0, 4.0, 5.0);
        assert_eq!(m.a, 2.0);
        assert_eq!(m.f, 3.0);
        assert_eq!(m.k, 4.0);
        assert_eq!(m.p, 5.0);
    }

    #[test]
    fn test_from_translation() {
        let m = Mat4::from_translation(1.0, 2.0, 3.0);
        assert_eq!(m.m, 1.0);
        assert_eq!(m.n, 2.0);
        assert_eq!(m.o, 3.0);
        assert_eq!(m.p, 1.0);
    }

    #[test]
    fn test_from_rotation_x() {
        let m = Mat4::from_rotation_x(FRAC_PI_2);
        let v = Vec4::new(0.0, 1.0, 0.0, 0.0);
        let result = m * v;
        approx::assert_relative_eq!(result.y, 0.0, epsilon = 1e-10);
        approx::assert_relative_eq!(result.z, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_from_rotation_y() {
        let m = Mat4::from_rotation_y(FRAC_PI_2);
        let v = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let result = m * v;
        approx::assert_relative_eq!(result.x, 0.0, epsilon = 1e-10);
        approx::assert_relative_eq!(result.z, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_from_rotation_z() {
        let m = Mat4::from_rotation_z(FRAC_PI_2);
        let v = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let result = m * v;
        approx::assert_relative_eq!(result.x, 0.0, epsilon = 1e-10);
        approx::assert_relative_eq!(result.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_from_rotation_axis() {
        let axis = Vec3::UNIT_X.normalize();
        let m = Mat4::from_rotation_axis(&axis, FRAC_PI_2);
        let v = Vec4::new(0.0, 1.0, 0.0, 0.0);
        let result = m * v;
        approx::assert_relative_eq!(result.y, 0.0, epsilon = 1e-10);
        approx::assert_relative_eq!(result.z, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_from_look_at() {
        let eye = Vec3::new(0.0, 0.0, 5.0);
        let target = Vec3::ZERO;
        let up = Vec3::UNIT_Y;
        let m = Mat4::from_look_at(&eye, &target, &up);
        assert_eq!(m.p, 1.0);
    }

    #[test]
    fn test_from_perspective() {
        let m = Mat4::from_perspective(FRAC_PI_2, 1.0, 0.1, 100.0);
        approx::assert_relative_eq!(m.a, 1.0);
        approx::assert_relative_eq!(m.f, 1.0);
        assert!(m.l.abs() > 0.0);
    }

    #[test]
    fn test_from_orthographic() {
        let m = Mat4::from_orthographic(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
        assert_eq!(m.a, 1.0);
        assert_eq!(m.f, 1.0);
        assert_eq!(m.k, 1.0);
    }

    #[test]
    fn test_extract_3x3() {
        let m = Mat4::from_scale(2.0, 3.0, 4.0, 5.0);
        let m3 = m.extract_3x3();
        assert_eq!(m3.a, 2.0);
        assert_eq!(m3.e, 3.0);
        assert_eq!(m3.i, 4.0);
    }

    #[test]
    fn test_inverse_translation() {
        let m = Mat4::from_translation(1.0, 2.0, 3.0);
        let inv = m.inverse().unwrap();
        let combined = m * inv;
        approx::assert_relative_eq!(combined, Mat4::IDENTITY, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_rotation() {
        let m = Mat4::from_rotation_y(FRAC_PI_2);
        let inv = m.inverse().unwrap();
        let combined = m * inv;
        approx::assert_relative_eq!(combined, Mat4::IDENTITY, epsilon = 1e-10);
    }

    #[test]
    fn test_perspective_normalized_device_coordinates() {
        let m = Mat4::from_perspective(FRAC_PI_2, 1.0, 0.1, 100.0);
        let v = Vec4::new(0.0, 0.0, -1.0, 1.0);
        let result = m * v;
        approx::assert_relative_eq!(result.x, 0.0, epsilon = 1e-10);
        approx::assert_relative_eq!(result.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_associativity() {
        let a = Mat4::from_rotation_x(FRAC_PI_2);
        let b = Mat4::from_rotation_y(FRAC_PI_2);
        let c = Mat4::from_scale(2.0, 2.0, 2.0, 1.0);
        let left = (a * b) * c;
        let right = a * (b * c);
        assert_eq!(left, right);
    }
}
