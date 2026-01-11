//! Computational geometry algorithms.

pub mod boolean;
mod convex_hull;
mod delaunay;
mod kdtree;
mod quadtree;
pub mod sweep;
mod voronoi;

pub use convex_hull::graham_scan;
pub use delaunay::{delaunay, Triangle, Triangulation};
pub use kdtree::{KdTree2, KdTree3, NearestResult};
pub use quadtree::{Bounds, Quadtree};
pub use voronoi::{voronoi, voronoi_from_delaunay, VoronoiDiagram, VoronoiEdge, VoronoiVertex};
