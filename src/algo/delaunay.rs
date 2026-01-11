//! Delaunay triangulation using the Bowyer-Watson algorithm.

use std::cmp::Ordering;

use crate::predicates::{incircle, orient2d};
use crate::Point2;

/// A triangle represented by indices into a point array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Triangle {
    /// Indices of the three vertices in counter-clockwise order.
    pub vertices: [usize; 3],
}

impl Triangle {
    /// Creates a new triangle from three vertex indices.
    ///
    /// The vertices should be provided in counter-clockwise order.
    #[inline]
    pub fn new(a: usize, b: usize, c: usize) -> Self {
        Self {
            vertices: [a, b, c],
        }
    }

    /// Returns the three edges as pairs of vertex indices.
    pub fn edges(&self) -> [(usize, usize); 3] {
        let [a, b, c] = self.vertices;
        [(a, b), (b, c), (c, a)]
    }

    /// Checks if the triangle contains a vertex index.
    pub fn contains_vertex(&self, v: usize) -> bool {
        self.vertices.contains(&v)
    }
}

/// Result of Delaunay triangulation.
#[derive(Debug, Clone)]
pub struct Triangulation {
    /// The input points.
    pub points: Vec<Point2<i64>>,
    /// The triangles, each storing indices into `points`.
    pub triangles: Vec<Triangle>,
}

impl Triangulation {
    /// Returns an iterator over triangles as point triples.
    pub fn triangle_points(&self) -> impl Iterator<Item = [Point2<i64>; 3]> + '_ {
        self.triangles.iter().map(|t| {
            [
                self.points[t.vertices[0]],
                self.points[t.vertices[1]],
                self.points[t.vertices[2]],
            ]
        })
    }
}

/// Computes the Delaunay triangulation of a set of 2D points.
///
/// Uses the Bowyer-Watson incremental algorithm. Returns `None` if fewer
/// than 3 points are provided or all points are collinear.
///
/// # Time Complexity
///
/// O(n²) worst case, O(n log n) expected for random point distributions.
///
/// # Example
///
/// ```
/// use exactum::{Point2, algo::delaunay};
///
/// let points = vec![
///     Point2::new(0_i64, 0),
///     Point2::new(10, 0),
///     Point2::new(5, 10),
///     Point2::new(5, 5),
/// ];
///
/// let triangulation = delaunay(&points).unwrap();
/// assert_eq!(triangulation.triangles.len(), 3); // Interior point creates 3 triangles
/// ```
#[must_use]
pub fn delaunay(points: &[Point2<i64>]) -> Option<Triangulation> {
    if points.len() < 3 {
        return None;
    }

    // Create a working copy of points; we'll add super-triangle vertices at the end
    let mut all_points: Vec<Point2<i64>> = points.to_vec();
    let n = points.len();

    // Compute bounding box
    let (min_x, max_x, min_y, max_y) = bounding_box(points);

    // Create super-triangle that contains all points
    // We make it large enough to contain the bounding box with margin
    let dx = (max_x - min_x).max(1);
    let dy = (max_y - min_y).max(1);
    let delta = dx.max(dy);
    let mid_x = min_x + dx / 2;
    let mid_y = min_y + dy / 2;

    // Super-triangle vertices (indices n, n+1, n+2)
    let p0 = Point2::new(mid_x - 20 * delta, mid_y - delta);
    let p1 = Point2::new(mid_x, mid_y + 20 * delta);
    let p2 = Point2::new(mid_x + 20 * delta, mid_y - delta);

    all_points.push(p0);
    all_points.push(p1);
    all_points.push(p2);

    // Ensure super-triangle is CCW
    let super_tri = if orient2d(p0, p1, p2) == Ordering::Greater {
        Triangle::new(n, n + 1, n + 2)
    } else {
        Triangle::new(n, n + 2, n + 1)
    };

    let mut triangles: Vec<Triangle> = vec![super_tri];

    // Insert points one at a time
    for i in 0..n {
        let p = all_points[i];

        // Find all triangles whose circumcircle contains p
        let mut bad_triangles: Vec<usize> = Vec::new();
        for (ti, tri) in triangles.iter().enumerate() {
            let [ai, bi, ci] = tri.vertices;
            let a = all_points[ai];
            let b = all_points[bi];
            let c = all_points[ci];

            // incircle returns Greater if p is inside circumcircle of (a,b,c)
            // (a,b,c) must be CCW for correct sign
            if incircle(a, b, c, p) == Ordering::Greater {
                bad_triangles.push(ti);
            }
        }

        if bad_triangles.is_empty() {
            // Point might be outside all circumcircles (shouldn't happen with super-tri)
            continue;
        }

        // Find the boundary of the polygonal hole
        let polygon = find_cavity_boundary(&triangles, &bad_triangles);

        // Remove bad triangles (in reverse order to preserve indices)
        for &ti in bad_triangles.iter().rev() {
            triangles.swap_remove(ti);
        }

        // Retriangulate: connect new point to each edge of the cavity
        for (a, b) in polygon {
            // Ensure CCW orientation
            let pa = all_points[a];
            let pb = all_points[b];
            if orient2d(pa, pb, p) == Ordering::Greater {
                triangles.push(Triangle::new(a, b, i));
            } else {
                triangles.push(Triangle::new(b, a, i));
            }
        }
    }

    // Remove triangles that share a vertex with the super-triangle
    triangles.retain(|tri| {
        !tri.contains_vertex(n) && !tri.contains_vertex(n + 1) && !tri.contains_vertex(n + 2)
    });

    if triangles.is_empty() {
        return None; // All points were collinear
    }

    Some(Triangulation {
        points: points.to_vec(),
        triangles,
    })
}

/// Finds the boundary edges of the cavity formed by bad triangles.
fn find_cavity_boundary(triangles: &[Triangle], bad_indices: &[usize]) -> Vec<(usize, usize)> {
    let mut edge_count: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();

    for &ti in bad_indices {
        for (a, b) in triangles[ti].edges() {
            // Normalize edge direction for counting
            let edge = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(edge).or_insert(0) += 1;
        }
    }

    // Boundary edges appear exactly once among bad triangles
    let mut boundary = Vec::new();
    for &ti in bad_indices {
        for (a, b) in triangles[ti].edges() {
            let edge = if a < b { (a, b) } else { (b, a) };
            if edge_count[&edge] == 1 {
                // Keep original orientation from the bad triangle
                boundary.push((a, b));
            }
        }
    }

    boundary
}

fn bounding_box(points: &[Point2<i64>]) -> (i64, i64, i64, i64) {
    let mut min_x = i64::MAX;
    let mut max_x = i64::MIN;
    let mut min_y = i64::MAX;
    let mut max_y = i64::MIN;

    for p in points {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }

    (min_x, max_x, min_y, max_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delaunay_triangle() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 0),
            Point2::new(5, 10),
        ];
        let tri = delaunay(&points).unwrap();
        assert_eq!(tri.triangles.len(), 1);
    }

    #[test]
    fn delaunay_square() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 0),
            Point2::new(10, 10),
            Point2::new(0, 10),
        ];
        let tri = delaunay(&points).unwrap();
        // A square should produce 2 triangles
        assert_eq!(tri.triangles.len(), 2);
    }

    #[test]
    fn delaunay_with_interior_point() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 0),
            Point2::new(10, 10),
            Point2::new(0, 10),
            Point2::new(5, 5),
        ];
        let tri = delaunay(&points).unwrap();
        // 4 corners + 1 interior = 4 triangles
        assert_eq!(tri.triangles.len(), 4);
    }

    #[test]
    fn delaunay_empty() {
        let points: Vec<Point2<i64>> = vec![];
        assert!(delaunay(&points).is_none());
    }

    #[test]
    fn delaunay_two_points() {
        let points = vec![Point2::new(0_i64, 0), Point2::new(10, 0)];
        assert!(delaunay(&points).is_none());
    }

    #[test]
    fn delaunay_collinear() {
        let points = vec![Point2::new(0_i64, 0), Point2::new(5, 0), Point2::new(10, 0)];
        // Collinear points can't form a proper triangulation
        assert!(delaunay(&points).is_none());
    }

    #[test]
    fn delaunay_property_check() {
        // Verify the Delaunay property: no point is inside any circumcircle
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(100, 0),
            Point2::new(100, 100),
            Point2::new(0, 100),
            Point2::new(50, 50),
            Point2::new(25, 75),
            Point2::new(75, 25),
        ];

        let tri = delaunay(&points).unwrap();

        for triangle in &tri.triangles {
            let [ai, bi, ci] = triangle.vertices;
            let a = tri.points[ai];
            let b = tri.points[bi];
            let c = tri.points[ci];

            for (i, p) in tri.points.iter().enumerate() {
                if i == ai || i == bi || i == ci {
                    continue;
                }
                // No point should be strictly inside any circumcircle
                let result = incircle(a, b, c, *p);
                assert_ne!(
                    result,
                    Ordering::Greater,
                    "Point {} is inside circumcircle of triangle {:?}",
                    i,
                    triangle
                );
            }
        }
    }
}
