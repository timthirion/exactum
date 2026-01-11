//! Exact geometric predicates and algorithms using integer arithmetic.
//!
//! Exactum provides robust computational geometry primitives that use exact
//! integer arithmetic to avoid floating-point precision issues.
//!
//! # Core Types
//!
//! - [`Point2`] and [`Point3`] - 2D and 3D points
//! - [`Vector2`] and [`Vector3`] - 2D and 3D displacement vectors
//!
//! # Predicates
//!
//! - [`predicates::orient2d`] - Orientation test for three 2D points
//! - [`predicates::incircle`] - In-circle test for four 2D points
//! - [`predicates::collinear`] - Collinearity test
//!
//! # Algorithms
//!
//! - [`algo::graham_scan`] - Convex hull via Graham scan
//! - [`algo::delaunay`] - Delaunay triangulation via Bowyer-Watson
//! - [`algo::voronoi`] - Voronoi diagram (dual of Delaunay)
//!
//! # Traits
//!
//! - [`Widen`] - Integer widening for overflow-safe multiplication

mod point;
mod rational;
mod vector;
mod widen;

pub mod algo;
pub mod ops;
pub mod predicates;

pub use point::{Point2, Point3};
pub use rational::Rational;
pub use vector::{Vector2, Vector3};
pub use widen::{Wide, Widen};
