//! Voronoi diagram as the dual of Delaunay triangulation.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::algo::delaunay::{delaunay, Triangulation};
use crate::Point2;
use crate::Rational;

/// A Voronoi vertex with exact rational coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoronoiVertex {
    pub x: Rational,
    pub y: Rational,
}

impl VoronoiVertex {
    /// Converts to floating-point coordinates for visualization.
    pub fn to_f64(self) -> (f64, f64) {
        (self.x.to_f64(), self.y.to_f64())
    }
}

/// An edge in the Voronoi diagram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoronoiEdge {
    /// Index of the first Voronoi vertex, or `None` if the edge extends to infinity.
    pub start: Option<usize>,
    /// Index of the second Voronoi vertex, or `None` if the edge extends to infinity.
    pub end: Option<usize>,
    /// The two input points (indices) that this edge separates.
    pub sites: (usize, usize),
}

/// A Voronoi diagram.
#[derive(Debug, Clone)]
pub struct VoronoiDiagram {
    /// The original input points (sites).
    pub sites: Vec<Point2<i64>>,
    /// The Voronoi vertices (circumcenters of Delaunay triangles).
    pub vertices: Vec<VoronoiVertex>,
    /// The Voronoi edges.
    pub edges: Vec<VoronoiEdge>,
    /// For each site, the indices of adjacent Voronoi vertices forming its cell.
    /// Vertices are in counter-clockwise order.
    pub cells: Vec<Vec<usize>>,
}

/// Computes the Voronoi diagram from a set of 2D points.
///
/// Returns `None` if the triangulation fails (fewer than 3 points or all collinear).
///
/// # Example
///
/// ```
/// use exactum::{Point2, algo::voronoi};
///
/// let points = vec![
///     Point2::new(0_i64, 0),
///     Point2::new(10, 0),
///     Point2::new(5, 10),
///     Point2::new(5, 5),
/// ];
///
/// let diagram = voronoi(&points).unwrap();
/// println!("Voronoi has {} vertices", diagram.vertices.len());
/// ```
#[must_use]
pub fn voronoi(points: &[Point2<i64>]) -> Option<VoronoiDiagram> {
    let triangulation = delaunay(points)?;
    Some(voronoi_from_delaunay(&triangulation))
}

/// Computes the Voronoi diagram from an existing Delaunay triangulation.
#[must_use]
pub fn voronoi_from_delaunay(triangulation: &Triangulation) -> VoronoiDiagram {
    let n_sites = triangulation.points.len();

    // Compute circumcenter for each triangle -> Voronoi vertices
    let vertices: Vec<VoronoiVertex> = triangulation
        .triangles
        .iter()
        .map(|tri| {
            let a = triangulation.points[tri.vertices[0]];
            let b = triangulation.points[tri.vertices[1]];
            let c = triangulation.points[tri.vertices[2]];
            circumcenter(a, b, c)
        })
        .collect();

    // Build adjacency: for each edge, which triangles share it?
    let mut edge_to_triangles: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

    for (ti, tri) in triangulation.triangles.iter().enumerate() {
        for (a, b) in tri.edges() {
            let edge = if a < b { (a, b) } else { (b, a) };
            edge_to_triangles.entry(edge).or_default().push(ti);
        }
    }

    // Build Voronoi edges: connect circumcenters of adjacent triangles
    let mut edges: Vec<VoronoiEdge> = Vec::new();
    let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();

    for (edge, tris) in &edge_to_triangles {
        let (site_a, site_b) = *edge;

        if tris.len() == 2 {
            let (t1, t2) = (tris[0], tris[1]);
            let voronoi_edge_key = if t1 < t2 { (t1, t2) } else { (t2, t1) };

            if seen_edges.insert(voronoi_edge_key) {
                edges.push(VoronoiEdge {
                    start: Some(t1),
                    end: Some(t2),
                    sites: (site_a, site_b),
                });
            }
        } else if tris.len() == 1 {
            edges.push(VoronoiEdge {
                start: Some(tris[0]),
                end: None,
                sites: (site_a, site_b),
            });
        }
    }

    // Build cells: for each site, collect adjacent Voronoi vertices
    let mut cells: Vec<Vec<usize>> = vec![Vec::new(); n_sites];

    for (ti, tri) in triangulation.triangles.iter().enumerate() {
        for &site in &tri.vertices {
            if site < n_sites {
                cells[site].push(ti);
            }
        }
    }

    // Sort each cell's vertices in counter-clockwise order around the site
    for (site_idx, cell) in cells.iter_mut().enumerate() {
        let site = triangulation.points[site_idx];
        let site_x = Rational::from_int(site.x);
        let site_y = Rational::from_int(site.y);

        cell.sort_by(|&a, &b| {
            let va = &vertices[a];
            let vb = &vertices[b];

            // Vector from site to va
            let ax = va.x - site_x;
            let ay = va.y - site_y;

            // Vector from site to vb
            let bx = vb.x - site_x;
            let by = vb.y - site_y;

            compare_angles(ax, ay, bx, by)
        });
    }

    VoronoiDiagram {
        sites: triangulation.points.clone(),
        vertices,
        edges,
        cells,
    }
}

/// Compares two vectors by their angle from the positive x-axis.
///
/// Returns `Ordering::Less` if (ax, ay) has a smaller angle than (bx, by),
/// i.e., (ax, ay) comes before (bx, by) in counter-clockwise order starting from +x.
fn compare_angles(ax: Rational, ay: Rational, bx: Rational, by: Rational) -> Ordering {
    // Determine quadrant for each vector (0 = +x+y, 1 = -x+y, 2 = -x-y, 3 = +x-y)
    let quad_a = quadrant(ax, ay);
    let quad_b = quadrant(bx, by);

    if quad_a != quad_b {
        return quad_a.cmp(&quad_b);
    }

    // Same quadrant: use cross product
    // cross = ax * by - ay * bx
    // If cross > 0, a is CCW from b, meaning a has smaller angle
    let cross = ax * by - ay * bx;

    // Compare cross product to 0
    // cross.num and cross.denom have same sign relationship as the value
    // Since denom is always positive (product of positive denoms), just check num
    match cross.num.cmp(&0) {
        Ordering::Greater => Ordering::Less, // a before b
        Ordering::Less => Ordering::Greater, // b before a
        Ordering::Equal => Ordering::Equal,  // same angle
    }
}

/// Returns the quadrant of a vector (0-3), counter-clockwise from +x axis.
/// - 0: +x, +y (or +x axis)
/// - 1: -x, +y (or +y axis)
/// - 2: -x, -y (or -x axis)
/// - 3: +x, -y (or -y axis)
fn quadrant(x: Rational, y: Rational) -> u8 {
    if x.is_non_negative() {
        if y.is_negative() {
            3
        } else {
            0
        }
    } else if y.is_negative() {
        2
    } else {
        1
    }
}

/// Computes the circumcenter of a triangle with exact rational coordinates.
fn circumcenter(a: Point2<i64>, b: Point2<i64>, c: Point2<i64>) -> VoronoiVertex {
    let ax = a.x as i128;
    let ay = a.y as i128;
    let bx = b.x as i128;
    let by = b.y as i128;
    let cx = c.x as i128;
    let cy = c.y as i128;

    // d = 2 * (ax(by - cy) + bx(cy - ay) + cx(ay - by))
    let d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));

    let a_sq = ax * ax + ay * ay;
    let b_sq = bx * bx + by * by;
    let c_sq = cx * cx + cy * cy;

    let ux = a_sq * (by - cy) + b_sq * (cy - ay) + c_sq * (ay - by);
    let uy = a_sq * (cx - bx) + b_sq * (ax - cx) + c_sq * (bx - ax);

    // Handle degenerate case (collinear points)
    let denom = if d == 0 { 1 } else { d.abs() };
    let sign = if d < 0 { -1 } else { 1 };

    VoronoiVertex {
        x: Rational::new(sign * ux, denom),
        y: Rational::new(sign * uy, denom),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voronoi_triangle() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 0),
            Point2::new(5, 10),
        ];
        let diagram = voronoi(&points).unwrap();
        assert_eq!(diagram.vertices.len(), 1);
    }

    #[test]
    fn voronoi_square() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 0),
            Point2::new(10, 10),
            Point2::new(0, 10),
        ];
        let diagram = voronoi(&points).unwrap();
        assert_eq!(diagram.vertices.len(), 2);
    }

    #[test]
    fn voronoi_with_interior() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 0),
            Point2::new(10, 10),
            Point2::new(0, 10),
            Point2::new(5, 5),
        ];
        let diagram = voronoi(&points).unwrap();
        assert_eq!(diagram.vertices.len(), 4);
        assert_eq!(diagram.sites.len(), 5);
    }

    #[test]
    fn circumcenter_right_triangle() {
        let a = Point2::new(0_i64, 0);
        let b = Point2::new(10, 0);
        let c = Point2::new(0, 10);

        let cc = circumcenter(a, b, c);

        // Circumcenter should be at (5, 5) - midpoint of hypotenuse
        // Check exactly: x = 5/1, y = 5/1
        assert_eq!(cc.x.num, 5 * cc.x.denom);
        assert_eq!(cc.y.num, 5 * cc.y.denom);
    }

    #[test]
    fn circumcenter_obtuse_triangle() {
        // Obtuse triangle: (0,0), (100,0), (50,10)
        // This is a very flat triangle - the circumcenter should be OUTSIDE
        let a = Point2::new(0_i64, 0);
        let b = Point2::new(100, 0);
        let c = Point2::new(50, 10);

        let cc = circumcenter(a, b, c);
        let (x, y) = cc.to_f64();

        // The circumcenter x should be around 50 (center of base)
        assert!((x - 50.0).abs() < 1.0, "x = {}", x);
        // The circumcenter y should be NEGATIVE (outside the triangle, below the base)
        // For this obtuse triangle, y ≈ -117.5
        assert!(y < 0.0, "y = {} should be negative (outside triangle)", y);
    }

    #[test]
    fn circumcenter_equilateral() {
        // Equilateral-ish triangle - circumcenter should be inside
        let a = Point2::new(0_i64, 0);
        let b = Point2::new(100, 0);
        let c = Point2::new(50, 87); // ~= 50 + 50*sqrt(3) ≈ 86.6

        let cc = circumcenter(a, b, c);
        let (x, y) = cc.to_f64();

        // Circumcenter should be roughly at the centroid for equilateral
        assert!((x - 50.0).abs() < 1.0, "x = {}", x);
        // y should be positive (inside the triangle)
        assert!(y > 0.0 && y < 87.0, "y = {} should be inside triangle", y);
    }

    #[test]
    fn angle_comparison() {
        // (1, 0) vs (0, 1): (1,0) is at angle 0, (0,1) is at angle 90
        let ax = Rational::new(1, 1);
        let ay = Rational::new(0, 1);
        let bx = Rational::new(0, 1);
        let by = Rational::new(1, 1);

        assert_eq!(compare_angles(ax, ay, bx, by), Ordering::Less);
    }

    #[test]
    fn angle_comparison_same_quadrant() {
        // (2, 1) vs (1, 2): both in Q0, (2,1) has smaller angle
        let ax = Rational::new(2, 1);
        let ay = Rational::new(1, 1);
        let bx = Rational::new(1, 1);
        let by = Rational::new(2, 1);

        assert_eq!(compare_angles(ax, ay, bx, by), Ordering::Less);
    }

    #[test]
    fn angle_comparison_opposite_quadrants() {
        // (1, 1) in Q0 vs (-1, -1) in Q2
        let ax = Rational::new(1, 1);
        let ay = Rational::new(1, 1);
        let bx = Rational::new(-1, 1);
        let by = Rational::new(-1, 1);

        assert_eq!(compare_angles(ax, ay, bx, by), Ordering::Less);
    }
}
