//! Exact geometric predicates and algorithms using integer arithmetic.
//!
//! Exactum provides robust computational geometry primitives that use exact
//! integer arithmetic to avoid floating-point precision issues.
//!
//! # Core Types
//!
//! - [`Point2`] and [`Point3`] - 2D and 3D points
//! - [`Vector2`] and [`Vector3`] - 2D and 3D displacement vectors

mod point;
mod vector;

pub use point::{Point2, Point3};
pub use vector::{Vector2, Vector3};
