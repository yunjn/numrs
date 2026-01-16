pub const EPSILON: f64 = 1e-9;

mod matrix;
mod vector;

pub use matrix::{Mat2, Mat3};
pub use vector::{Point2D, Point3D};
pub use vector::{Vec2, Vec3};
