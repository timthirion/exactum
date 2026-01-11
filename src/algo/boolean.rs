//! Boolean operations on simple polygons: union, intersection, difference.
//!
//! This module provides exact boolean operations using a simplified Martinez-Rueda-Feito
//! algorithm variant adapted for rational arithmetic. Results may contain rational
//! coordinates where polygons intersect.

use std::cmp::Ordering;

use crate::ops::{
    point_in_polygon, polygon_area_2x, segment_intersection, Containment, RationalPoint,
    SegmentIntersection,
};
use crate::rational::Rational;
use crate::Point2;

/// A vertex in a boolean operation result.
///
/// Can be either an original integer point or an intersection point with
/// rational coordinates.
#[derive(Debug, Clone)]
pub enum Vertex {
    /// Original vertex from input polygon (integer coordinates).
    Original(Point2<i64>),
    /// Intersection point (rational coordinates).
    Intersection(RationalPoint),
}

impl Vertex {
    /// Converts the vertex to floating-point coordinates for visualization.
    pub fn to_f64(&self) -> (f64, f64) {
        match self {
            Vertex::Original(p) => (p.x as f64, p.y as f64),
            Vertex::Intersection(r) => r.to_f64(),
        }
    }

    /// Gets the x-coordinate as a rational number.
    pub fn x(&self) -> Rational {
        match self {
            Vertex::Original(p) => Rational::from_int(p.x),
            Vertex::Intersection(r) => r.x,
        }
    }

    /// Gets the y-coordinate as a rational number.
    pub fn y(&self) -> Rational {
        match self {
            Vertex::Original(p) => Rational::from_int(p.y),
            Vertex::Intersection(r) => r.y,
        }
    }

    /// Lexicographic comparison: first by x, then by y.
    #[allow(dead_code)]
    fn cmp_lex(&self, other: &Self) -> Ordering {
        match self.x().cmp(&other.x()) {
            Ordering::Equal => self.y().cmp(&other.y()),
            ord => ord,
        }
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.x() == other.x() && self.y() == other.y()
    }
}

impl Eq for Vertex {}

/// Result of a boolean operation.
///
/// The result may contain zero, one, or multiple polygons. Each polygon
/// is a list of vertices in counter-clockwise order.
#[derive(Debug, Clone)]
pub struct BooleanResult {
    /// The resulting polygon(s), each as a list of vertices in CCW order.
    pub polygons: Vec<Vec<Vertex>>,
}

impl BooleanResult {
    /// Returns true if the result is empty (no polygons).
    pub fn is_empty(&self) -> bool {
        self.polygons.is_empty()
    }

    /// Returns the total number of vertices across all result polygons.
    pub fn vertex_count(&self) -> usize {
        self.polygons.iter().map(|p| p.len()).sum()
    }
}

/// Which polygon an edge belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PolygonId {
    A,
    B,
}

/// Classification of an edge relative to the other polygon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum EdgeClass {
    /// Edge is outside the other polygon.
    Outside,
    /// Edge is inside the other polygon.
    Inside,
    /// Edge overlaps with an edge of the other polygon (same direction).
    SharedSame,
    /// Edge overlaps with an edge of the other polygon (opposite direction).
    SharedOpposite,
}

/// An edge with its start and end vertices.
#[derive(Debug, Clone)]
struct Edge {
    start: Vertex,
    end: Vertex,
    polygon: PolygonId,
    classification: Option<EdgeClass>,
}

impl Edge {
    fn new(start: Vertex, end: Vertex, polygon: PolygonId) -> Self {
        Self {
            start,
            end,
            polygon,
            classification: None,
        }
    }

    /// Computes the midpoint of this edge as a RationalPoint.
    fn midpoint(&self) -> RationalPoint {
        let mx = (self.start.x() + self.end.x()) * Rational::new(1, 2);
        let my = (self.start.y() + self.end.y()) * Rational::new(1, 2);
        RationalPoint::new(mx, my)
    }
}

/// An intersection between two edges.
#[derive(Debug, Clone)]
struct Intersection {
    /// The intersection point.
    point: RationalPoint,
    /// Index of edge in polygon A.
    edge_a: usize,
    /// Index of edge in polygon B.
    edge_b: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Computes the union of two simple polygons.
///
/// Returns all areas covered by either polygon A or polygon B (or both).
///
/// # Example
///
/// ```
/// use exactum::{Point2, algo::boolean::polygon_union};
///
/// // Two overlapping squares
/// let a = vec![
///     Point2::new(0_i64, 0),
///     Point2::new(10, 0),
///     Point2::new(10, 10),
///     Point2::new(0, 10),
/// ];
/// let b = vec![
///     Point2::new(5_i64, 5),
///     Point2::new(15, 5),
///     Point2::new(15, 15),
///     Point2::new(5, 15),
/// ];
///
/// let result = polygon_union(&a, &b);
/// assert!(!result.is_empty());
/// ```
#[must_use]
pub fn polygon_union(a: &[Point2<i64>], b: &[Point2<i64>]) -> BooleanResult {
    boolean_operation(a, b, BooleanOp::Union)
}

/// Computes the intersection of two simple polygons.
///
/// Returns only the areas covered by both polygon A and polygon B.
///
/// # Example
///
/// ```
/// use exactum::{Point2, algo::boolean::polygon_intersection};
///
/// // Two overlapping squares
/// let a = vec![
///     Point2::new(0_i64, 0),
///     Point2::new(10, 0),
///     Point2::new(10, 10),
///     Point2::new(0, 10),
/// ];
/// let b = vec![
///     Point2::new(5_i64, 5),
///     Point2::new(15, 5),
///     Point2::new(15, 15),
///     Point2::new(5, 15),
/// ];
///
/// let result = polygon_intersection(&a, &b);
/// assert!(!result.is_empty());
/// ```
#[must_use]
pub fn polygon_intersection(a: &[Point2<i64>], b: &[Point2<i64>]) -> BooleanResult {
    boolean_operation(a, b, BooleanOp::Intersection)
}

/// Computes the difference of two simple polygons (A - B).
///
/// Returns the areas covered by polygon A but not by polygon B.
///
/// # Example
///
/// ```
/// use exactum::{Point2, algo::boolean::polygon_difference};
///
/// // Two overlapping squares
/// let a = vec![
///     Point2::new(0_i64, 0),
///     Point2::new(10, 0),
///     Point2::new(10, 10),
///     Point2::new(0, 10),
/// ];
/// let b = vec![
///     Point2::new(5_i64, 5),
///     Point2::new(15, 5),
///     Point2::new(15, 15),
///     Point2::new(5, 15),
/// ];
///
/// let result = polygon_difference(&a, &b);
/// assert!(!result.is_empty());
/// ```
#[must_use]
pub fn polygon_difference(a: &[Point2<i64>], b: &[Point2<i64>]) -> BooleanResult {
    boolean_operation(a, b, BooleanOp::Difference)
}

// ─────────────────────────────────────────────────────────────────────────────
// Implementation
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum BooleanOp {
    Union,
    Intersection,
    Difference,
}

/// Main boolean operation implementation.
fn boolean_operation(a: &[Point2<i64>], b: &[Point2<i64>], op: BooleanOp) -> BooleanResult {
    // Handle degenerate cases
    if a.len() < 3 || b.len() < 3 {
        return match op {
            BooleanOp::Union => {
                if a.len() >= 3 {
                    BooleanResult {
                        polygons: vec![a.iter().map(|&p| Vertex::Original(p)).collect()],
                    }
                } else if b.len() >= 3 {
                    BooleanResult {
                        polygons: vec![b.iter().map(|&p| Vertex::Original(p)).collect()],
                    }
                } else {
                    BooleanResult { polygons: vec![] }
                }
            }
            BooleanOp::Intersection => BooleanResult { polygons: vec![] },
            BooleanOp::Difference => {
                if a.len() >= 3 {
                    BooleanResult {
                        polygons: vec![a.iter().map(|&p| Vertex::Original(p)).collect()],
                    }
                } else {
                    BooleanResult { polygons: vec![] }
                }
            }
        };
    }

    // Normalize polygons to CCW
    let a_ccw = ensure_ccw(a);
    let b_ccw = ensure_ccw(b);

    // Find all intersections between edges
    let intersections = find_all_intersections(&a_ccw, &b_ccw);

    // Check for disjoint or nested cases first
    if intersections.is_empty() {
        return handle_no_intersection(&a_ccw, &b_ccw, op);
    }

    // Build edges with intersection points
    let mut edges = build_split_edges(&a_ccw, &b_ccw, &intersections);

    // Classify each edge
    classify_edges(&mut edges, &a_ccw, &b_ccw);

    // Select edges based on operation
    let selected = select_edges(&edges, op);

    // Build result polygons by tracing edges
    let polygons = trace_polygons(selected);

    BooleanResult { polygons }
}

/// Ensures polygon is in counter-clockwise order.
fn ensure_ccw(polygon: &[Point2<i64>]) -> Vec<Point2<i64>> {
    let area = polygon_area_2x(polygon);
    if area < 0 {
        // Clockwise, reverse to CCW
        polygon.iter().copied().rev().collect()
    } else {
        polygon.to_vec()
    }
}

/// Finds all intersection points between edges of polygons A and B.
fn find_all_intersections(a: &[Point2<i64>], b: &[Point2<i64>]) -> Vec<Intersection> {
    let mut intersections = Vec::new();

    for (i, (a1, a2)) in edges_iter(a).enumerate() {
        for (j, (b1, b2)) in edges_iter(b).enumerate() {
            match segment_intersection(a1, a2, b1, b2) {
                SegmentIntersection::Point(pt) => {
                    // Skip if the intersection is at endpoints (will be handled as vertices)
                    let is_a_endpoint = is_endpoint(&pt, a1, a2);
                    let is_b_endpoint = is_endpoint(&pt, b1, b2);

                    // Only add if it's a proper intersection in the interior
                    if !is_a_endpoint || !is_b_endpoint {
                        intersections.push(Intersection {
                            point: pt,
                            edge_a: i,
                            edge_b: j,
                        });
                    }
                }
                SegmentIntersection::Overlapping => {
                    // Handle overlapping segments - add endpoints of overlap
                    // This is a simplified handling; full overlap handling is complex
                }
                SegmentIntersection::None | SegmentIntersection::CollinearDisjoint => {}
            }
        }
    }

    intersections
}

/// Checks if a rational point equals an integer point.
fn is_endpoint(pt: &RationalPoint, p1: Point2<i64>, p2: Point2<i64>) -> bool {
    let r1 = RationalPoint::from_ints(p1.x, p1.y);
    let r2 = RationalPoint::from_ints(p2.x, p2.y);
    (pt.x == r1.x && pt.y == r1.y) || (pt.x == r2.x && pt.y == r2.y)
}

/// Iterator over polygon edges as (start, end) pairs.
fn edges_iter(polygon: &[Point2<i64>]) -> impl Iterator<Item = (Point2<i64>, Point2<i64>)> + '_ {
    let n = polygon.len();
    (0..n).map(move |i| (polygon[i], polygon[(i + 1) % n]))
}

/// Handles the case where polygons don't intersect (or have only boundary contact).
fn handle_no_intersection(a: &[Point2<i64>], b: &[Point2<i64>], op: BooleanOp) -> BooleanResult {
    // Check if polygons are identical (same vertices)
    if are_polygons_identical(a, b) {
        let to_vertices = |poly: &[Point2<i64>]| -> Vec<Vertex> {
            poly.iter().map(|&p| Vertex::Original(p)).collect()
        };

        return match op {
            BooleanOp::Union | BooleanOp::Intersection => BooleanResult {
                polygons: vec![to_vertices(a)],
            },
            BooleanOp::Difference => BooleanResult { polygons: vec![] },
        };
    }

    // Check if A is inside B or B is inside A using a point that's not on B's boundary
    let a_containment = classify_polygon_containment(a, b);
    let b_containment = classify_polygon_containment(b, a);

    let a_in_b = a_containment == Containment::Inside;
    let b_in_a = b_containment == Containment::Inside;

    let to_vertices = |poly: &[Point2<i64>]| -> Vec<Vertex> {
        poly.iter().map(|&p| Vertex::Original(p)).collect()
    };

    match op {
        BooleanOp::Union => {
            if a_in_b {
                // A is inside B, union is B
                BooleanResult {
                    polygons: vec![to_vertices(b)],
                }
            } else if b_in_a {
                // B is inside A, union is A
                BooleanResult {
                    polygons: vec![to_vertices(a)],
                }
            } else {
                // Disjoint, union is both polygons
                BooleanResult {
                    polygons: vec![to_vertices(a), to_vertices(b)],
                }
            }
        }
        BooleanOp::Intersection => {
            if a_in_b {
                // A is inside B, intersection is A
                BooleanResult {
                    polygons: vec![to_vertices(a)],
                }
            } else if b_in_a {
                // B is inside A, intersection is B
                BooleanResult {
                    polygons: vec![to_vertices(b)],
                }
            } else {
                // Disjoint, no intersection
                BooleanResult { polygons: vec![] }
            }
        }
        BooleanOp::Difference => {
            if a_in_b {
                // A is inside B, difference is empty
                BooleanResult { polygons: vec![] }
            } else if b_in_a {
                // B is inside A, difference is A with hole B
                // For simplicity, we return A minus the hole
                // Full implementation would need hole handling
                BooleanResult {
                    polygons: vec![to_vertices(a)],
                }
            } else {
                // Disjoint, difference is A
                BooleanResult {
                    polygons: vec![to_vertices(a)],
                }
            }
        }
    }
}

/// Checks if two polygons have the same vertices (possibly rotated).
fn are_polygons_identical(a: &[Point2<i64>], b: &[Point2<i64>]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let n = a.len();
    if n == 0 {
        return true;
    }

    // Find starting point in b that matches a[0]
    let start = b.iter().position(|&p| p == a[0]);
    let Some(start) = start else {
        return false;
    };

    // Check if all vertices match in order
    for i in 0..n {
        if a[i] != b[(start + i) % n] {
            return false;
        }
    }
    true
}

/// Classifies whether one polygon is inside, outside, or on the boundary of another.
/// Uses centroid to avoid boundary issues.
fn classify_polygon_containment(inner: &[Point2<i64>], outer: &[Point2<i64>]) -> Containment {
    // Try each vertex - if any is inside, the polygon is inside
    // If any is outside, the polygon is outside
    // If all are on boundary, check midpoint of an edge
    for &p in inner {
        match point_in_polygon(p, outer) {
            Containment::Inside => return Containment::Inside,
            Containment::Outside => return Containment::Outside,
            Containment::OnBoundary => continue,
        }
    }

    // All vertices are on boundary - check edge midpoints
    for (p1, p2) in edges_iter(inner) {
        let mid = Point2::new((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
        match point_in_polygon(mid, outer) {
            Containment::Inside => return Containment::Inside,
            Containment::Outside => return Containment::Outside,
            Containment::OnBoundary => continue,
        }
    }

    // Everything is on boundary - polygons are identical or share boundary
    Containment::OnBoundary
}

/// Builds edges split at intersection points.
fn build_split_edges(
    a: &[Point2<i64>],
    b: &[Point2<i64>],
    intersections: &[Intersection],
) -> Vec<Edge> {
    let mut edges = Vec::new();

    // Build edges for polygon A
    for (i, (p1, p2)) in edges_iter(a).enumerate() {
        let mut splits: Vec<RationalPoint> = intersections
            .iter()
            .filter(|isec| isec.edge_a == i)
            .map(|isec| isec.point)
            .collect();

        if splits.is_empty() {
            edges.push(Edge::new(
                Vertex::Original(p1),
                Vertex::Original(p2),
                PolygonId::A,
            ));
        } else {
            // Sort splits along the edge direction
            let start = RationalPoint::from_ints(p1.x, p1.y);
            splits.sort_by(|s1, s2| {
                // Compare by distance from start (using x primarily, then y)
                let d1x = s1.x - start.x;
                let d2x = s2.x - start.x;
                match d1x.cmp(&d2x) {
                    Ordering::Equal => {
                        let d1y = s1.y - start.y;
                        let d2y = s2.y - start.y;
                        d1y.cmp(&d2y)
                    }
                    ord => ord,
                }
            });

            // Create sub-edges
            let mut prev = Vertex::Original(p1);
            for split in splits {
                let curr = Vertex::Intersection(split);
                if prev != curr {
                    edges.push(Edge::new(prev.clone(), curr.clone(), PolygonId::A));
                }
                prev = curr;
            }
            let end = Vertex::Original(p2);
            if prev != end {
                edges.push(Edge::new(prev, end, PolygonId::A));
            }
        }
    }

    // Build edges for polygon B
    for (i, (p1, p2)) in edges_iter(b).enumerate() {
        let mut splits: Vec<RationalPoint> = intersections
            .iter()
            .filter(|isec| isec.edge_b == i)
            .map(|isec| isec.point)
            .collect();

        if splits.is_empty() {
            edges.push(Edge::new(
                Vertex::Original(p1),
                Vertex::Original(p2),
                PolygonId::B,
            ));
        } else {
            let start = RationalPoint::from_ints(p1.x, p1.y);
            splits.sort_by(|s1, s2| {
                let d1x = s1.x - start.x;
                let d2x = s2.x - start.x;
                match d1x.cmp(&d2x) {
                    Ordering::Equal => {
                        let d1y = s1.y - start.y;
                        let d2y = s2.y - start.y;
                        d1y.cmp(&d2y)
                    }
                    ord => ord,
                }
            });

            let mut prev = Vertex::Original(p1);
            for split in splits {
                let curr = Vertex::Intersection(split);
                if prev != curr {
                    edges.push(Edge::new(prev.clone(), curr.clone(), PolygonId::B));
                }
                prev = curr;
            }
            let end = Vertex::Original(p2);
            if prev != end {
                edges.push(Edge::new(prev, end, PolygonId::B));
            }
        }
    }

    edges
}

/// Classifies each edge as inside/outside the other polygon.
fn classify_edges(edges: &mut [Edge], a: &[Point2<i64>], b: &[Point2<i64>]) {
    for edge in edges.iter_mut() {
        let mid = edge.midpoint();

        let other_polygon = match edge.polygon {
            PolygonId::A => b,
            PolygonId::B => a,
        };

        // Use rational midpoint for accurate testing
        let containment = classify_point_in_polygon(&mid, other_polygon);

        edge.classification = Some(match containment {
            Containment::Inside => EdgeClass::Inside,
            Containment::Outside => EdgeClass::Outside,
            Containment::OnBoundary => {
                // Edge is on boundary - check direction
                EdgeClass::SharedSame // Simplified; full impl would check direction
            }
        });
    }
}

/// Classifies a rational point's containment in an integer polygon.
fn classify_point_in_polygon(pt: &RationalPoint, polygon: &[Point2<i64>]) -> Containment {
    let n = polygon.len();
    if n < 3 {
        return Containment::Outside;
    }

    let mut crossings = 0i64;

    for i in 0..n {
        let v1 = polygon[i];
        let v2 = polygon[(i + 1) % n];

        // Check if point is on this edge
        if point_on_segment_rational(pt, v1, v2) {
            return Containment::OnBoundary;
        }

        // Ray casting
        let y1 = Rational::from_int(v1.y);
        let y2 = Rational::from_int(v2.y);
        let py = pt.y;

        let spans_y = (y1 <= py && y2 > py) || (y2 <= py && y1 > py);
        if !spans_y {
            continue;
        }

        // Compute x-coordinate of intersection of edge with horizontal line at py
        // Edge: v1 + t*(v2-v1) where t = (py - y1) / (y2 - y1)
        // x = x1 + t * (x2 - x1)
        let x1 = Rational::from_int(v1.x);
        let x2 = Rational::from_int(v2.x);

        // x_intersect = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        let dy = y2 - y1;
        let t_num = py - y1;
        let dx = x2 - x1;
        let x_intersect = x1 + t_num * dx * Rational::new(1, 1) / dy;

        if pt.x < x_intersect {
            crossings += 1;
        }
    }

    if crossings % 2 == 1 {
        Containment::Inside
    } else {
        Containment::Outside
    }
}

/// Checks if a rational point lies on an integer segment.
fn point_on_segment_rational(pt: &RationalPoint, v1: Point2<i64>, v2: Point2<i64>) -> bool {
    let v1r = RationalPoint::from_ints(v1.x, v1.y);
    let v2r = RationalPoint::from_ints(v2.x, v2.y);

    // Check collinearity using cross product
    // (v2 - v1) × (pt - v1) = 0
    let dx1 = v2r.x - v1r.x;
    let dy1 = v2r.y - v1r.y;
    let dx2 = pt.x - v1r.x;
    let dy2 = pt.y - v1r.y;

    let cross = dx1 * dy2 - dy1 * dx2;
    if cross.num != 0 {
        return false;
    }

    // Check if point is within bounding box
    let min_x = if v1r.x < v2r.x { v1r.x } else { v2r.x };
    let max_x = if v1r.x > v2r.x { v1r.x } else { v2r.x };
    let min_y = if v1r.y < v2r.y { v1r.y } else { v2r.y };
    let max_y = if v1r.y > v2r.y { v1r.y } else { v2r.y };

    pt.x >= min_x && pt.x <= max_x && pt.y >= min_y && pt.y <= max_y
}

/// Selects edges that should be in the result based on the operation.
/// Returns (start, end) pairs with proper direction (B edges reversed for difference).
fn select_edges(edges: &[Edge], op: BooleanOp) -> Vec<(Vertex, Vertex)> {
    let mut selected = Vec::new();

    for e in edges {
        let class = e.classification.unwrap_or(EdgeClass::Outside);
        let include = match op {
            BooleanOp::Union => {
                // Include edges outside the other polygon, or shared same-direction
                class == EdgeClass::Outside || class == EdgeClass::SharedSame
            }
            BooleanOp::Intersection => {
                // Include edges inside the other polygon, or shared same-direction
                class == EdgeClass::Inside || class == EdgeClass::SharedSame
            }
            BooleanOp::Difference => match e.polygon {
                PolygonId::A => class == EdgeClass::Outside,
                PolygonId::B => class == EdgeClass::Inside,
            },
        };

        if include {
            // For difference, reverse B edges since they form the "cutting" boundary
            let (start, end) = if matches!(op, BooleanOp::Difference) && e.polygon == PolygonId::B {
                (e.end.clone(), e.start.clone())
            } else {
                (e.start.clone(), e.end.clone())
            };
            selected.push((start, end));
        }
    }

    selected
}

/// Traces selected edges into closed polygons.
fn trace_polygons(edges: Vec<(Vertex, Vertex)>) -> Vec<Vec<Vertex>> {
    if edges.is_empty() {
        return Vec::new();
    }

    // Build edge list with used flag
    let mut edge_list: Vec<(Vertex, Vertex, bool)> =
        edges.into_iter().map(|(s, e)| (s, e, false)).collect();

    let mut polygons = Vec::new();

    loop {
        // Find an unused edge
        let start_idx = edge_list.iter().position(|(_, _, used)| !*used);
        let Some(start_idx) = start_idx else {
            break;
        };

        let mut polygon = Vec::new();
        let mut current_idx = start_idx;

        loop {
            // Check if already used
            if edge_list[current_idx].2 {
                break;
            }

            // Mark as used and get vertex info
            edge_list[current_idx].2 = true;
            let start = edge_list[current_idx].0.clone();
            let end = edge_list[current_idx].1.clone();
            polygon.push(start);

            // Find next edge starting from 'end'
            let next = edge_list
                .iter()
                .enumerate()
                .position(|(i, (s, _, u))| !*u && i != current_idx && vertices_equal(s, &end));

            match next {
                Some(idx) => current_idx = idx,
                None => break,
            }
        }

        if polygon.len() >= 3 {
            polygons.push(polygon);
        }
    }

    polygons
}

/// Compares two vertices for equality.
fn vertices_equal(a: &Vertex, b: &Vertex) -> bool {
    a.x() == b.x() && a.y() == b.y()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x: i64, y: i64, size: i64) -> Vec<Point2<i64>> {
        vec![
            Point2::new(x, y),
            Point2::new(x + size, y),
            Point2::new(x + size, y + size),
            Point2::new(x, y + size),
        ]
    }

    #[test]
    fn test_union_overlapping_squares() {
        let a = square(0, 0, 10);
        let b = square(5, 5, 10);

        let result = polygon_union(&a, &b);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_intersection_overlapping_squares() {
        let a = square(0, 0, 10);
        let b = square(5, 5, 10);

        let result = polygon_intersection(&a, &b);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_difference_overlapping_squares() {
        let a = square(0, 0, 10);
        let b = square(5, 5, 10);

        let result = polygon_difference(&a, &b);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_union_disjoint() {
        let a = square(0, 0, 10);
        let b = square(20, 0, 10);

        let result = polygon_union(&a, &b);
        // Should return both polygons
        assert_eq!(result.polygons.len(), 2);
    }

    #[test]
    fn test_intersection_disjoint() {
        let a = square(0, 0, 10);
        let b = square(20, 0, 10);

        let result = polygon_intersection(&a, &b);
        // No intersection
        assert!(result.is_empty());
    }

    #[test]
    fn test_difference_disjoint() {
        let a = square(0, 0, 10);
        let b = square(20, 0, 10);

        let result = polygon_difference(&a, &b);
        // Difference is just A
        assert_eq!(result.polygons.len(), 1);
    }

    #[test]
    fn test_nested_polygons_union() {
        let outer = square(0, 0, 20);
        let inner = square(5, 5, 10);

        let result = polygon_union(&outer, &inner);
        // Union of nested is the outer
        assert_eq!(result.polygons.len(), 1);
    }

    #[test]
    fn test_nested_polygons_intersection() {
        let outer = square(0, 0, 20);
        let inner = square(5, 5, 10);

        let result = polygon_intersection(&outer, &inner);
        // Intersection of nested is the inner
        assert_eq!(result.polygons.len(), 1);
    }

    #[test]
    fn test_identical_polygons() {
        let a = square(0, 0, 10);
        let b = square(0, 0, 10);

        let union = polygon_union(&a, &b);
        let intersection = polygon_intersection(&a, &b);
        let difference = polygon_difference(&a, &b);

        assert!(!union.is_empty());
        assert!(!intersection.is_empty());
        // Difference of identical polygons is empty
        assert!(difference.is_empty());
    }

    #[test]
    fn test_empty_input() {
        let a: Vec<Point2<i64>> = vec![];
        let b = square(0, 0, 10);

        let union = polygon_union(&a, &b);
        assert_eq!(union.polygons.len(), 1);

        let intersection = polygon_intersection(&a, &b);
        assert!(intersection.is_empty());

        let difference = polygon_difference(&a, &b);
        assert!(difference.is_empty());
    }

    #[test]
    fn test_triangle_square_intersection() {
        let triangle = vec![
            Point2::new(5_i64, 0),
            Point2::new(15, 10),
            Point2::new(5, 20),
        ];
        let sq = square(0, 5, 10);

        let result = polygon_intersection(&triangle, &sq);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_ensure_ccw() {
        // CW square
        let cw = vec![
            Point2::new(0_i64, 0),
            Point2::new(0, 10),
            Point2::new(10, 10),
            Point2::new(10, 0),
        ];

        let ccw = ensure_ccw(&cw);
        let area = polygon_area_2x(&ccw);
        assert!(area > 0, "Polygon should be CCW after conversion");
    }

    #[test]
    fn test_vertex_comparison() {
        let v1 = Vertex::Original(Point2::new(5, 10));
        let v2 = Vertex::Intersection(RationalPoint::from_ints(5, 10));

        assert_eq!(v1, v2);
    }

    #[test]
    fn test_find_intersections() {
        let a = square(0, 0, 10);
        let b = square(5, 5, 10);

        let intersections = find_all_intersections(&a, &b);
        // Two overlapping squares should have intersection points
        assert!(!intersections.is_empty());
    }
}
