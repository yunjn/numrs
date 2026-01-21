## numrs

A high-performance, precision-focused 2D/3D linear algebra library for learning.

## Features

- **2D Types**: `Vec2`, `Mat2` using `f64` precision.
- **3D Types**: `Vec3`, `Mat3` using `f64` precision.
- **4D Types**: `Vec4`, `Mat4` with homogeneous coordinates support.
- **Precision Focus**: All calculations use `f64` with `approx` crate for floating-point comparisons.

## Usage

### 2D/3D Vector Operations

```rust
use numrs::{Mat3, Vec3};

fn main() {
    let m1 = Mat3::IDENTITY;
    let v1 = Vec3::new(1.0, 2.0, 3.0);

    let result = m1 * v1;
    println!("{}", result);
}
```

### 4D Transforms with Perspective Projection

```rust
use numrs::{Mat4, Vec4};

fn main() {
    // Create a perspective projection matrix
    let projection = Mat4::from_perspective(std::f64::consts::PI / 4.0, 16.0 / 9.0, 0.1, 100.0);

    // Transform a point in homogeneous coordinates
    let v = Vec4::new(0.0, 0.0, -1.0, 1.0);
    let transformed = projection * v;

    // Perform perspective divide
    let ndc = transformed.to_homogeneous();
    println!("NDC coordinates: ({}, {}, {})", ndc.x, ndc.y, ndc.z);
}
```

### Floating-point Comparison

```rust
use approx::assert_relative_eq;

let m = Mat3::from_rotation_x(std::f64::consts::PI);
// Use relative equality for floating-point calculations
assert_relative_eq!(m, expected_matrix, epsilon = 1e-15);
```

## Roadmap

- [x] Add `Vec4` & `Mat4` support for 3D projections.
- [ ] Implement `Quaternion` for rotation handling.
- [ ] Explore SIMD acceleration

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/yunjn/numrs/blob/main/LICENSE) file for the full license text.