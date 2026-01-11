//! Computational geometry algorithms.

mod convex_hull;
mod delaunay;

pub use convex_hull::graham_scan;
pub use delaunay::{delaunay, Triangle, Triangulation};
