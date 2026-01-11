//! Primitive geometric operations: intersections, distances, and containment.
//!
//! This module provides exact geometric operations using integer and rational
//! arithmetic. Results are either integers (for distances) or rationals
//! (for intersection points).

use std::cmp::Ordering;

use crate::predicates::orient2d;
use crate::rational::Rational;
use crate::widen::{Wide, Widen};
use crate::{Point2, Vector2};

/// A point with exact rational coordinates.
///
/// Used for intersection points where integer coordinates are insufficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RationalPoint {
    pub x: Rational,
    pub y: Rational,
}

impl RationalPoint {
    /// Creates a new rational point.
    pub fn new(x: Rational, y: Rational) -> Self {
        Self { x, y }
    }

    /// Creates a rational point from integer coordinates.
    pub fn from_ints(x: i64, y: i64) -> Self {
        Self {
            x: Rational::from_int(x),
            y: Rational::from_int(y),
        }
    }

    /// Converts to floating-point coordinates for visualization.
    pub fn to_f64(self) -> (f64, f64) {
        (self.x.to_f64(), self.y.to_f64())
    }
}

/// Result of a containment test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Containment {
    /// Point is strictly inside the region.
    Inside,
    /// Point is strictly outside the region.
    Outside,
    /// Point is exactly on the boundary.
    OnBoundary,
}

/// Result of segment-segment intersection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentIntersection {
    /// Segments do not intersect.
    None,
    /// Segments intersect at a single point.
    Point(RationalPoint),
    /// Segments are collinear and overlap.
    Overlapping,
    /// Segments are collinear but do not overlap.
    CollinearDisjoint,
}

/// Result of line-segment intersection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineIntersection {
    /// Line and segment are parallel (no intersection).
    None,
    /// Line and segment intersect at a single point.
    Point(RationalPoint),
    /// Segment lies entirely on the line.
    Collinear,
}

/// Result of ray-segment intersection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RayIntersection {
    /// Ray and segment do not intersect.
    None,
    /// Ray and segment intersect at a single point.
    Point(RationalPoint),
    /// Ray and segment are collinear with overlap.
    CollinearOverlap,
    /// Ray and segment are collinear but do not overlap.
    CollinearDisjoint,
}

// ─────────────────────────────────────────────────────────────────────────────
// Distance
// ─────────────────────────────────────────────────────────────────────────────

/// Computes the squared distance between two points.
///
/// Returns the result in the widened type to avoid overflow.
/// Use squared distance to avoid the need for floating-point square roots.
///
/// # Example
///
/// ```
/// use exactum::{Point2, ops::distance_squared};
///
/// let a = Point2::new(0_i64, 0);
/// let b = Point2::new(3, 4);
/// assert_eq!(distance_squared(a, b), 25_i128); // 3² + 4² = 25
/// ```
#[must_use]
pub fn distance_squared<T: Widen>(a: Point2<T>, b: Point2<T>) -> T::Wide
where
    T::Wide: Wide<Narrow = T>,
{
    let dx = b.x.to_wide() - a.x.to_wide();
    let dy = b.y.to_wide() - a.y.to_wide();
    dx.clone() * dx + dy.clone() * dy
}

// ─────────────────────────────────────────────────────────────────────────────
// Point-in-Triangle
// ─────────────────────────────────────────────────────────────────────────────

/// Tests if a point lies inside, outside, or on the boundary of a triangle.
///
/// The triangle vertices can be in either clockwise or counter-clockwise order.
///
/// # Example
///
/// ```
/// use exactum::{Point2, ops::{point_in_triangle, Containment}};
///
/// let a = Point2::new(0_i64, 0);
/// let b = Point2::new(10, 0);
/// let c = Point2::new(5, 10);
///
/// assert_eq!(point_in_triangle(Point2::new(5, 5), a, b, c), Containment::Inside);
/// assert_eq!(point_in_triangle(Point2::new(0, 0), a, b, c), Containment::OnBoundary);
/// assert_eq!(point_in_triangle(Point2::new(20, 20), a, b, c), Containment::Outside);
/// ```
#[must_use]
pub fn point_in_triangle<T: Widen>(
    p: Point2<T>,
    a: Point2<T>,
    b: Point2<T>,
    c: Point2<T>,
) -> Containment
where
    T::Wide: Wide<Narrow = T>,
{
    let o1 = orient2d(a, b, p);
    let o2 = orient2d(b, c, p);
    let o3 = orient2d(c, a, p);

    // If any orientation is zero, point is on an edge
    if o1 == Ordering::Equal || o2 == Ordering::Equal || o3 == Ordering::Equal {
        // Check if the point is on the boundary
        // A point on a vertex or edge will have at least one zero orientation
        // and the other orientations must be consistent (same sign or zero)
        let non_zero: Vec<Ordering> = [o1, o2, o3]
            .into_iter()
            .filter(|&o| o != Ordering::Equal)
            .collect();

        if non_zero.is_empty() {
            // All three are Equal - point coincides with a vertex
            return Containment::OnBoundary;
        }

        // All non-zero orientations must have the same sign for the point to be on boundary
        let first = non_zero[0];
        if non_zero.iter().all(|&o| o == first) {
            return Containment::OnBoundary;
        }

        // Mixed signs means outside (degenerate case)
        return Containment::Outside;
    }

    // All orientations are non-zero
    // Point is inside if all have the same sign
    if o1 == o2 && o2 == o3 {
        Containment::Inside
    } else {
        Containment::Outside
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Segment-Segment Intersection
// ─────────────────────────────────────────────────────────────────────────────

/// Computes the intersection of two line segments.
///
/// Segments are defined by their endpoints: `a1`-`a2` and `b1`-`b2`.
///
/// # Example
///
/// ```
/// use exactum::{Point2, ops::{segment_intersection, SegmentIntersection}};
///
/// // Two crossing segments
/// let a1 = Point2::new(0_i64, 0);
/// let a2 = Point2::new(10, 10);
/// let b1 = Point2::new(0, 10);
/// let b2 = Point2::new(10, 0);
///
/// match segment_intersection(a1, a2, b1, b2) {
///     SegmentIntersection::Point(p) => {
///         let (x, y) = p.to_f64();
///         assert!((x - 5.0).abs() < 0.001);
///         assert!((y - 5.0).abs() < 0.001);
///     }
///     _ => panic!("Expected intersection point"),
/// }
/// ```
#[must_use]
pub fn segment_intersection(
    a1: Point2<i64>,
    a2: Point2<i64>,
    b1: Point2<i64>,
    b2: Point2<i64>,
) -> SegmentIntersection {
    // Compute orientations
    let o1 = orient2d(a1, a2, b1);
    let o2 = orient2d(a1, a2, b2);
    let o3 = orient2d(b1, b2, a1);
    let o4 = orient2d(b1, b2, a2);

    // General case: segments properly cross
    if o1 != o2 && o3 != o4 {
        // Check for endpoint touch (one orientation is zero)
        if o1 == Ordering::Equal {
            return SegmentIntersection::Point(RationalPoint::from_ints(b1.x, b1.y));
        }
        if o2 == Ordering::Equal {
            return SegmentIntersection::Point(RationalPoint::from_ints(b2.x, b2.y));
        }
        if o3 == Ordering::Equal {
            return SegmentIntersection::Point(RationalPoint::from_ints(a1.x, a1.y));
        }
        if o4 == Ordering::Equal {
            return SegmentIntersection::Point(RationalPoint::from_ints(a2.x, a2.y));
        }

        // Compute exact intersection point using parametric form
        // Line a: a1 + t*(a2-a1), t in [0,1]
        // Line b: b1 + s*(b2-b1), s in [0,1]
        //
        // t = ((b1-a1) × (b2-b1)) / ((a2-a1) × (b2-b1))
        let a1x = a1.x as i128;
        let a1y = a1.y as i128;
        let a2x = a2.x as i128;
        let a2y = a2.y as i128;
        let b1x = b1.x as i128;
        let b1y = b1.y as i128;
        let b2x = b2.x as i128;
        let b2y = b2.y as i128;

        // (a2 - a1) × (b2 - b1) = (a2x-a1x)*(b2y-b1y) - (a2y-a1y)*(b2x-b1x)
        let denom = (a2x - a1x) * (b2y - b1y) - (a2y - a1y) * (b2x - b1x);

        // (b1 - a1) × (b2 - b1) = (b1x-a1x)*(b2y-b1y) - (b1y-a1y)*(b2x-b1x)
        let numer = (b1x - a1x) * (b2y - b1y) - (b1y - a1y) * (b2x - b1x);

        // intersection = a1 + (numer/denom) * (a2 - a1)
        // x = a1x + numer*(a2x-a1x)/denom = (a1x*denom + numer*(a2x-a1x)) / denom
        // y = a1y + numer*(a2y-a1y)/denom = (a1y*denom + numer*(a2y-a1y)) / denom

        let abs_denom = denom.abs();
        let sign = if denom < 0 { -1 } else { 1 };

        let x_num = sign * (a1x * denom + numer * (a2x - a1x));
        let y_num = sign * (a1y * denom + numer * (a2y - a1y));

        return SegmentIntersection::Point(RationalPoint::new(
            Rational::new(x_num, abs_denom),
            Rational::new(y_num, abs_denom),
        ));
    }

    // Collinear case: all four points are collinear
    if o1 == Ordering::Equal && o2 == Ordering::Equal {
        // Check if segments overlap on the line
        if segments_overlap_1d(a1.x, a2.x, b1.x, b2.x)
            && segments_overlap_1d(a1.y, a2.y, b1.y, b2.y)
        {
            return SegmentIntersection::Overlapping;
        }
        return SegmentIntersection::CollinearDisjoint;
    }

    // No intersection
    SegmentIntersection::None
}

/// Helper: check if 1D intervals [min(a1,a2), max(a1,a2)] and [min(b1,b2), max(b1,b2)] overlap.
fn segments_overlap_1d(a1: i64, a2: i64, b1: i64, b2: i64) -> bool {
    let (a_min, a_max) = if a1 <= a2 { (a1, a2) } else { (a2, a1) };
    let (b_min, b_max) = if b1 <= b2 { (b1, b2) } else { (b2, b1) };
    a_max >= b_min && b_max >= a_min
}

// ─────────────────────────────────────────────────────────────────────────────
// Line-Segment Intersection
// ─────────────────────────────────────────────────────────────────────────────

/// Computes the intersection of a line and a segment.
///
/// The line is defined by a point `line_point` and a direction vector `line_dir`.
/// The segment is defined by its endpoints `seg_start` and `seg_end`.
///
/// # Example
///
/// ```
/// use exactum::{Point2, Vector2, ops::{line_segment_intersection, LineIntersection}};
///
/// // Horizontal line y=5 intersecting segment from (0,0) to (10,10)
/// let line_point = Point2::new(0_i64, 5);
/// let line_dir = Vector2::new(1_i64, 0);
/// let seg_start = Point2::new(0_i64, 0);
/// let seg_end = Point2::new(10, 10);
///
/// match line_segment_intersection(line_point, line_dir, seg_start, seg_end) {
///     LineIntersection::Point(p) => {
///         let (x, y) = p.to_f64();
///         assert!((x - 5.0).abs() < 0.001);
///         assert!((y - 5.0).abs() < 0.001);
///     }
///     _ => panic!("Expected intersection point"),
/// }
/// ```
#[must_use]
pub fn line_segment_intersection(
    line_point: Point2<i64>,
    line_dir: Vector2<i64>,
    seg_start: Point2<i64>,
    seg_end: Point2<i64>,
) -> LineIntersection {
    // Create a second point on the line
    let line_point2 = Point2::new(line_point.x + line_dir.x, line_point.y + line_dir.y);

    // Check orientations of segment endpoints relative to line
    let o1 = orient2d(line_point, line_point2, seg_start);
    let o2 = orient2d(line_point, line_point2, seg_end);

    // Both endpoints on same side of line -> no intersection
    if (o1 == Ordering::Greater && o2 == Ordering::Greater)
        || (o1 == Ordering::Less && o2 == Ordering::Less)
    {
        return LineIntersection::None;
    }

    // Both endpoints on the line -> segment is collinear
    if o1 == Ordering::Equal && o2 == Ordering::Equal {
        return LineIntersection::Collinear;
    }

    // One endpoint on line
    if o1 == Ordering::Equal {
        return LineIntersection::Point(RationalPoint::from_ints(seg_start.x, seg_start.y));
    }
    if o2 == Ordering::Equal {
        return LineIntersection::Point(RationalPoint::from_ints(seg_end.x, seg_end.y));
    }

    // Proper crossing - compute intersection
    // Line: line_point + t * line_dir
    // Segment: seg_start + s * (seg_end - seg_start), s in [0,1]
    //
    // Solve: line_dir × (seg_end - seg_start) for denominator
    //        (seg_start - line_point) × (seg_end - seg_start) for s numerator

    let lpx = line_point.x as i128;
    let lpy = line_point.y as i128;
    let ldx = line_dir.x as i128;
    let ldy = line_dir.y as i128;
    let s1x = seg_start.x as i128;
    let s1y = seg_start.y as i128;
    let s2x = seg_end.x as i128;
    let s2y = seg_end.y as i128;

    // segment direction
    let sdx = s2x - s1x;
    let sdy = s2y - s1y;

    // denom = line_dir × seg_dir = ldx * sdy - ldy * sdx
    let denom = ldx * sdy - ldy * sdx;

    if denom == 0 {
        // Parallel (already handled by collinear check above, but safety)
        return LineIntersection::None;
    }

    // s = ((seg_start - line_point) × line_dir) / denom
    // where (seg_start - line_point) × line_dir = (s1x - lpx) * ldy - (s1y - lpy) * ldx
    let s_numer = (s1x - lpx) * ldy - (s1y - lpy) * ldx;

    // intersection = seg_start + s * (seg_end - seg_start)
    // x = s1x + s_numer * sdx / denom
    // y = s1y + s_numer * sdy / denom

    let abs_denom = denom.abs();
    let sign = if denom < 0 { -1 } else { 1 };

    let x_num = sign * (s1x * denom + s_numer * sdx);
    let y_num = sign * (s1y * denom + s_numer * sdy);

    LineIntersection::Point(RationalPoint::new(
        Rational::new(x_num, abs_denom),
        Rational::new(y_num, abs_denom),
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Ray-Segment Intersection
// ─────────────────────────────────────────────────────────────────────────────

/// Computes the intersection of a ray and a segment.
///
/// The ray starts at `ray_origin` and extends infinitely in direction `ray_dir`.
/// The segment is defined by its endpoints `seg_start` and `seg_end`.
///
/// # Example
///
/// ```
/// use exactum::{Point2, Vector2, ops::{ray_segment_intersection, RayIntersection}};
///
/// // Ray from origin pointing right, segment crossing y-axis
/// let ray_origin = Point2::new(0_i64, 5);
/// let ray_dir = Vector2::new(1_i64, 0);
/// let seg_start = Point2::new(5, 0);
/// let seg_end = Point2::new(5, 10);
///
/// match ray_segment_intersection(ray_origin, ray_dir, seg_start, seg_end) {
///     RayIntersection::Point(p) => {
///         let (x, y) = p.to_f64();
///         assert!((x - 5.0).abs() < 0.001);
///         assert!((y - 5.0).abs() < 0.001);
///     }
///     _ => panic!("Expected intersection point"),
/// }
/// ```
#[must_use]
pub fn ray_segment_intersection(
    ray_origin: Point2<i64>,
    ray_dir: Vector2<i64>,
    seg_start: Point2<i64>,
    seg_end: Point2<i64>,
) -> RayIntersection {
    // Create a second point on the ray
    let ray_point2 = Point2::new(ray_origin.x + ray_dir.x, ray_origin.y + ray_dir.y);

    // Check orientations of segment endpoints relative to ray line
    let o1 = orient2d(ray_origin, ray_point2, seg_start);
    let o2 = orient2d(ray_origin, ray_point2, seg_end);

    // Both endpoints on same side -> no intersection with line
    if (o1 == Ordering::Greater && o2 == Ordering::Greater)
        || (o1 == Ordering::Less && o2 == Ordering::Less)
    {
        return RayIntersection::None;
    }

    // Both collinear with ray
    if o1 == Ordering::Equal && o2 == Ordering::Equal {
        // Check if segment overlaps with ray (in direction of ray_dir from origin)
        return check_collinear_ray_segment(ray_origin, ray_dir, seg_start, seg_end);
    }

    // Compute intersection point
    let rox = ray_origin.x as i128;
    let roy = ray_origin.y as i128;
    let rdx = ray_dir.x as i128;
    let rdy = ray_dir.y as i128;
    let s1x = seg_start.x as i128;
    let s1y = seg_start.y as i128;
    let s2x = seg_end.x as i128;
    let s2y = seg_end.y as i128;

    let sdx = s2x - s1x;
    let sdy = s2y - s1y;

    // denom = ray_dir × seg_dir
    let denom = rdx * sdy - rdy * sdx;

    if denom == 0 {
        // Parallel but not collinear (already handled above)
        return RayIntersection::None;
    }

    // t = ((seg_start - ray_origin) × seg_dir) / denom
    // t must be >= 0 for intersection to be on the ray
    let t_numer = (s1x - rox) * sdy - (s1y - roy) * sdx;

    // Check if t >= 0
    let t_positive = (t_numer >= 0 && denom > 0) || (t_numer <= 0 && denom < 0);
    if !t_positive {
        return RayIntersection::None;
    }

    // One endpoint on ray line
    if o1 == Ordering::Equal {
        // Check if seg_start is in ray direction
        if point_in_ray_direction(ray_origin, ray_dir, seg_start) {
            return RayIntersection::Point(RationalPoint::from_ints(seg_start.x, seg_start.y));
        }
        return RayIntersection::None;
    }
    if o2 == Ordering::Equal {
        if point_in_ray_direction(ray_origin, ray_dir, seg_end) {
            return RayIntersection::Point(RationalPoint::from_ints(seg_end.x, seg_end.y));
        }
        return RayIntersection::None;
    }

    // Compute intersection point
    // intersection = ray_origin + t * ray_dir
    // = (rox + t_numer * rdx / denom, roy + t_numer * rdy / denom)

    let abs_denom = denom.abs();
    let sign = if denom < 0 { -1 } else { 1 };

    let x_num = sign * (rox * denom + t_numer * rdx);
    let y_num = sign * (roy * denom + t_numer * rdy);

    RayIntersection::Point(RationalPoint::new(
        Rational::new(x_num, abs_denom),
        Rational::new(y_num, abs_denom),
    ))
}

/// Check if a point is in the direction of the ray from its origin.
fn point_in_ray_direction(ray_origin: Point2<i64>, ray_dir: Vector2<i64>, p: Point2<i64>) -> bool {
    let dx = p.x - ray_origin.x;
    let dy = p.y - ray_origin.y;

    // Check if (dx, dy) has same direction as ray_dir
    // This is true if their dot product is non-negative and they're parallel
    let dot = (dx as i128) * (ray_dir.x as i128) + (dy as i128) * (ray_dir.y as i128);
    dot >= 0
}

/// Handle collinear ray-segment case.
fn check_collinear_ray_segment(
    ray_origin: Point2<i64>,
    ray_dir: Vector2<i64>,
    seg_start: Point2<i64>,
    seg_end: Point2<i64>,
) -> RayIntersection {
    // Project points onto ray direction
    let t_start = project_onto_ray(ray_origin, ray_dir, seg_start);
    let t_end = project_onto_ray(ray_origin, ray_dir, seg_end);

    let (t_min, t_max) = if t_start <= t_end {
        (t_start, t_end)
    } else {
        (t_end, t_start)
    };

    // Ray starts at t=0 and goes to +infinity
    // Segment spans [t_min, t_max]
    if t_max < 0 {
        // Segment entirely behind ray
        RayIntersection::CollinearDisjoint
    } else if t_min > 0 || t_max > 0 {
        // Some overlap
        RayIntersection::CollinearOverlap
    } else {
        // t_max == 0 and t_min <= 0: segment ends at ray origin
        RayIntersection::CollinearOverlap
    }
}

/// Project a point onto the ray's parametric line (unnormalized).
/// Returns a value proportional to t in ray_origin + t * ray_dir.
fn project_onto_ray(ray_origin: Point2<i64>, ray_dir: Vector2<i64>, p: Point2<i64>) -> i128 {
    let dx = (p.x - ray_origin.x) as i128;
    let dy = (p.y - ray_origin.y) as i128;
    let rdx = ray_dir.x as i128;
    let rdy = ray_dir.y as i128;
    // Dot product (dx, dy) . (rdx, rdy)
    dx * rdx + dy * rdy
}

// ─────────────────────────────────────────────────────────────────────────────
// Point-in-Polygon
// ─────────────────────────────────────────────────────────────────────────────

/// Tests if a point lies inside, outside, or on the boundary of a simple polygon.
///
/// The polygon is given as a slice of vertices in order (either CW or CCW).
/// Uses the ray casting algorithm with exact predicates.
///
/// # Example
///
/// ```
/// use exactum::{Point2, ops::{point_in_polygon, Containment}};
///
/// let square = vec![
///     Point2::new(0_i64, 0),
///     Point2::new(10, 0),
///     Point2::new(10, 10),
///     Point2::new(0, 10),
/// ];
///
/// assert_eq!(point_in_polygon(Point2::new(5, 5), &square), Containment::Inside);
/// assert_eq!(point_in_polygon(Point2::new(0, 5), &square), Containment::OnBoundary);
/// assert_eq!(point_in_polygon(Point2::new(-5, 5), &square), Containment::Outside);
/// ```
#[must_use]
pub fn point_in_polygon<T: Widen>(p: Point2<T>, polygon: &[Point2<T>]) -> Containment
where
    T::Wide: Wide<Narrow = T>,
{
    let n = polygon.len();
    if n < 3 {
        return Containment::Outside;
    }

    let mut crossings = 0;

    for i in 0..n {
        let v1 = polygon[i];
        let v2 = polygon[(i + 1) % n];

        // Check if point is on this edge
        if point_on_segment(p, v1, v2) {
            return Containment::OnBoundary;
        }

        // Ray casting: count crossings of horizontal ray to the right
        // We use orient2d to determine which side of the edge the point is on

        // Skip if edge is entirely above or below p
        let (y1, y2) = (v1.y, v2.y);
        let py = p.y;

        // Edge must span the y coordinate (one strictly above, one at or below, or vice versa)
        let spans_y = (y1 <= py && y2 > py) || (y2 <= py && y1 > py);
        if !spans_y {
            continue;
        }

        // Check if ray from p going right crosses this edge
        // The ray crosses if p is to the left of the directed edge
        let orientation = orient2d(v1, v2, p);

        // For upward edge (y1 < y2), a left turn means p is to the left -> crossing
        // For downward edge (y1 > y2), a right turn means p is to the left -> crossing
        if y1 < y2 {
            if orientation == Ordering::Greater {
                crossings += 1;
            }
        } else if orientation == Ordering::Less {
            crossings += 1;
        }
    }

    if crossings % 2 == 1 {
        Containment::Inside
    } else {
        Containment::Outside
    }
}

/// Check if point p lies on segment v1-v2.
fn point_on_segment<T: Widen>(p: Point2<T>, v1: Point2<T>, v2: Point2<T>) -> bool
where
    T::Wide: Wide<Narrow = T>,
{
    // Point must be collinear with segment endpoints
    if orient2d(v1, v2, p) != Ordering::Equal {
        return false;
    }

    // Point must be within bounding box of segment
    let (min_x, max_x) = if v1.x <= v2.x {
        (v1.x, v2.x)
    } else {
        (v2.x, v1.x)
    };
    let (min_y, max_y) = if v1.y <= v2.y {
        (v1.y, v2.y)
    } else {
        (v2.y, v1.y)
    };

    p.x >= min_x && p.x <= max_x && p.y >= min_y && p.y <= max_y
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // distance_squared tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn distance_squared_basic() {
        let a = Point2::new(0_i64, 0);
        let b = Point2::new(3, 4);
        assert_eq!(distance_squared(a, b), 25_i128);
    }

    #[test]
    fn distance_squared_same_point() {
        let a = Point2::new(5_i64, 5);
        assert_eq!(distance_squared(a, a), 0_i128);
    }

    #[test]
    fn distance_squared_i32() {
        let a = Point2::new(0_i32, 0);
        let b = Point2::new(3, 4);
        assert_eq!(distance_squared(a, b), 25_i64);
    }

    #[test]
    fn distance_squared_negative() {
        let a = Point2::new(-3_i64, -4);
        let b = Point2::new(0, 0);
        assert_eq!(distance_squared(a, b), 25_i128);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // point_in_triangle tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn point_in_triangle_inside() {
        let a = Point2::new(0_i64, 0);
        let b = Point2::new(10, 0);
        let c = Point2::new(5, 10);
        assert_eq!(
            point_in_triangle(Point2::new(5, 3), a, b, c),
            Containment::Inside
        );
    }

    #[test]
    fn point_in_triangle_outside() {
        let a = Point2::new(0_i64, 0);
        let b = Point2::new(10, 0);
        let c = Point2::new(5, 10);
        assert_eq!(
            point_in_triangle(Point2::new(20, 20), a, b, c),
            Containment::Outside
        );
    }

    #[test]
    fn point_in_triangle_on_vertex() {
        let a = Point2::new(0_i64, 0);
        let b = Point2::new(10, 0);
        let c = Point2::new(5, 10);
        assert_eq!(point_in_triangle(a, a, b, c), Containment::OnBoundary);
    }

    #[test]
    fn point_in_triangle_on_edge() {
        let a = Point2::new(0_i64, 0);
        let b = Point2::new(10, 0);
        let c = Point2::new(5, 10);
        assert_eq!(
            point_in_triangle(Point2::new(5, 0), a, b, c),
            Containment::OnBoundary
        );
    }

    #[test]
    fn point_in_triangle_cw_order() {
        // Triangle in clockwise order should still work
        let a = Point2::new(0_i64, 0);
        let b = Point2::new(5, 10);
        let c = Point2::new(10, 0);
        assert_eq!(
            point_in_triangle(Point2::new(5, 3), a, b, c),
            Containment::Inside
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // segment_intersection tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn segment_intersection_crossing() {
        let a1 = Point2::new(0_i64, 0);
        let a2 = Point2::new(10, 10);
        let b1 = Point2::new(0, 10);
        let b2 = Point2::new(10, 0);

        match segment_intersection(a1, a2, b1, b2) {
            SegmentIntersection::Point(p) => {
                let (x, y) = p.to_f64();
                assert!((x - 5.0).abs() < 0.001, "x = {}", x);
                assert!((y - 5.0).abs() < 0.001, "y = {}", y);
            }
            other => panic!("Expected Point, got {:?}", other),
        }
    }

    #[test]
    fn segment_intersection_no_intersect() {
        let a1 = Point2::new(0_i64, 0);
        let a2 = Point2::new(1, 1);
        let b1 = Point2::new(2, 2);
        let b2 = Point2::new(3, 3);
        assert!(matches!(
            segment_intersection(a1, a2, b1, b2),
            SegmentIntersection::CollinearDisjoint
        ));
    }

    #[test]
    fn segment_intersection_parallel_no_intersect() {
        let a1 = Point2::new(0_i64, 0);
        let a2 = Point2::new(10, 0);
        let b1 = Point2::new(0, 5);
        let b2 = Point2::new(10, 5);
        assert!(matches!(
            segment_intersection(a1, a2, b1, b2),
            SegmentIntersection::None
        ));
    }

    #[test]
    fn segment_intersection_collinear_overlap() {
        let a1 = Point2::new(0_i64, 0);
        let a2 = Point2::new(5, 0);
        let b1 = Point2::new(3, 0);
        let b2 = Point2::new(10, 0);
        assert!(matches!(
            segment_intersection(a1, a2, b1, b2),
            SegmentIntersection::Overlapping
        ));
    }

    #[test]
    fn segment_intersection_t_junction() {
        let a1 = Point2::new(0_i64, 5);
        let a2 = Point2::new(10, 5);
        let b1 = Point2::new(5, 0);
        let b2 = Point2::new(5, 5);

        match segment_intersection(a1, a2, b1, b2) {
            SegmentIntersection::Point(p) => {
                let (x, y) = p.to_f64();
                assert!((x - 5.0).abs() < 0.001);
                assert!((y - 5.0).abs() < 0.001);
            }
            other => panic!("Expected Point, got {:?}", other),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // line_segment_intersection tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn line_segment_intersection_crossing() {
        let line_point = Point2::new(0_i64, 5);
        let line_dir = Vector2::new(1, 0);
        let seg_start = Point2::new(5, 0);
        let seg_end = Point2::new(5, 10);

        match line_segment_intersection(line_point, line_dir, seg_start, seg_end) {
            LineIntersection::Point(p) => {
                let (x, y) = p.to_f64();
                assert!((x - 5.0).abs() < 0.001);
                assert!((y - 5.0).abs() < 0.001);
            }
            other => panic!("Expected Point, got {:?}", other),
        }
    }

    #[test]
    fn line_segment_intersection_parallel() {
        let line_point = Point2::new(0_i64, 0);
        let line_dir = Vector2::new(1, 0);
        let seg_start = Point2::new(0, 5);
        let seg_end = Point2::new(10, 5);
        assert!(matches!(
            line_segment_intersection(line_point, line_dir, seg_start, seg_end),
            LineIntersection::None
        ));
    }

    #[test]
    fn line_segment_intersection_collinear() {
        let line_point = Point2::new(0_i64, 0);
        let line_dir = Vector2::new(1, 0);
        let seg_start = Point2::new(5, 0);
        let seg_end = Point2::new(10, 0);
        assert!(matches!(
            line_segment_intersection(line_point, line_dir, seg_start, seg_end),
            LineIntersection::Collinear
        ));
    }

    #[test]
    fn line_segment_intersection_endpoint() {
        let line_point = Point2::new(0_i64, 0);
        let line_dir = Vector2::new(1, 1);
        let seg_start = Point2::new(5, 5);
        let seg_end = Point2::new(5, 10);

        match line_segment_intersection(line_point, line_dir, seg_start, seg_end) {
            LineIntersection::Point(p) => {
                let (x, y) = p.to_f64();
                assert!((x - 5.0).abs() < 0.001);
                assert!((y - 5.0).abs() < 0.001);
            }
            other => panic!("Expected Point, got {:?}", other),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ray_segment_intersection tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn ray_segment_intersection_crossing() {
        let ray_origin = Point2::new(0_i64, 5);
        let ray_dir = Vector2::new(1, 0);
        let seg_start = Point2::new(5, 0);
        let seg_end = Point2::new(5, 10);

        match ray_segment_intersection(ray_origin, ray_dir, seg_start, seg_end) {
            RayIntersection::Point(p) => {
                let (x, y) = p.to_f64();
                assert!((x - 5.0).abs() < 0.001);
                assert!((y - 5.0).abs() < 0.001);
            }
            other => panic!("Expected Point, got {:?}", other),
        }
    }

    #[test]
    fn ray_segment_intersection_behind() {
        // Ray pointing right, segment to the left
        let ray_origin = Point2::new(10_i64, 5);
        let ray_dir = Vector2::new(1, 0);
        let seg_start = Point2::new(5, 0);
        let seg_end = Point2::new(5, 10);

        assert!(matches!(
            ray_segment_intersection(ray_origin, ray_dir, seg_start, seg_end),
            RayIntersection::None
        ));
    }

    #[test]
    fn ray_segment_intersection_parallel() {
        let ray_origin = Point2::new(0_i64, 0);
        let ray_dir = Vector2::new(1, 0);
        let seg_start = Point2::new(0, 5);
        let seg_end = Point2::new(10, 5);
        assert!(matches!(
            ray_segment_intersection(ray_origin, ray_dir, seg_start, seg_end),
            RayIntersection::None
        ));
    }

    #[test]
    fn ray_segment_intersection_collinear_overlap() {
        let ray_origin = Point2::new(0_i64, 0);
        let ray_dir = Vector2::new(1, 0);
        let seg_start = Point2::new(5, 0);
        let seg_end = Point2::new(10, 0);
        assert!(matches!(
            ray_segment_intersection(ray_origin, ray_dir, seg_start, seg_end),
            RayIntersection::CollinearOverlap
        ));
    }

    #[test]
    fn ray_segment_intersection_collinear_behind() {
        let ray_origin = Point2::new(10_i64, 0);
        let ray_dir = Vector2::new(1, 0);
        let seg_start = Point2::new(0, 0);
        let seg_end = Point2::new(5, 0);
        assert!(matches!(
            ray_segment_intersection(ray_origin, ray_dir, seg_start, seg_end),
            RayIntersection::CollinearDisjoint
        ));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // point_in_polygon tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn point_in_polygon_square_inside() {
        let square = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 0),
            Point2::new(10, 10),
            Point2::new(0, 10),
        ];
        assert_eq!(
            point_in_polygon(Point2::new(5, 5), &square),
            Containment::Inside
        );
    }

    #[test]
    fn point_in_polygon_square_outside() {
        let square = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 0),
            Point2::new(10, 10),
            Point2::new(0, 10),
        ];
        assert_eq!(
            point_in_polygon(Point2::new(-5, 5), &square),
            Containment::Outside
        );
    }

    #[test]
    fn point_in_polygon_on_edge() {
        let square = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 0),
            Point2::new(10, 10),
            Point2::new(0, 10),
        ];
        assert_eq!(
            point_in_polygon(Point2::new(0, 5), &square),
            Containment::OnBoundary
        );
    }

    #[test]
    fn point_in_polygon_on_vertex() {
        let square = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 0),
            Point2::new(10, 10),
            Point2::new(0, 10),
        ];
        assert_eq!(
            point_in_polygon(Point2::new(0, 0), &square),
            Containment::OnBoundary
        );
    }

    #[test]
    fn point_in_polygon_concave() {
        // L-shaped polygon
        let l_shape = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 0),
            Point2::new(10, 5),
            Point2::new(5, 5),
            Point2::new(5, 10),
            Point2::new(0, 10),
        ];
        assert_eq!(
            point_in_polygon(Point2::new(2, 2), &l_shape),
            Containment::Inside
        );
        assert_eq!(
            point_in_polygon(Point2::new(7, 7), &l_shape),
            Containment::Outside
        );
    }

    #[test]
    fn point_in_polygon_triangle() {
        let triangle = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 0),
            Point2::new(5, 10),
        ];
        assert_eq!(
            point_in_polygon(Point2::new(5, 3), &triangle),
            Containment::Inside
        );
        assert_eq!(
            point_in_polygon(Point2::new(0, 5), &triangle),
            Containment::Outside
        );
    }
}
