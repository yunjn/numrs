## numrs

A high-performance, precision-focused 2D/3D linear algebra library written in Rust.

## Features

- **Standard Math Types**: `Vec2`, `Vec3`, `Mat2`, `Mat3` using `f64` precision.


## Usage

### Matrix Multiplication

```rust
use numrs::{Mat3, Vec3};

fn main() {
    let m1 = Mat3::IDENTITY;
    let v1 = Vec3::new(1.0, 2.0, 3.0);

    let result = m1 * v1;
    println!("{}", result);
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

- [ ] Add `Mat4` support for 3D projections.
- [ ] Implement `Quaternion` for rotation handling.
- [ ] Explore SIMD acceleration

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/yunjn/numrs/blob/main/LICENSE) file for the full license text.