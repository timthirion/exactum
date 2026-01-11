//! Exact geometric predicates.
//!
//! All predicates return [`std::cmp::Ordering`] to indicate the geometric
//! relationship. They use widening multiplication to avoid overflow.

use std::cmp::Ordering;

use crate::widen::{Wide, Widen};
use crate::Point2;

/// Computes the orientation of three 2D points.
///
/// Returns:
/// - `Ordering::Greater` if `a → b → c` is counter-clockwise (left turn)
/// - `Ordering::Less` if clockwise (right turn)
/// - `Ordering::Equal` if collinear
///
/// The result is the sign of the cross product `(b - a) × (c - a)`,
/// equivalently the sign of the 2×2 determinant:
///
/// ```text
/// | b.x - a.x   c.x - a.x |
/// | b.y - a.y   c.y - a.y |
/// ```
///
/// # Example
///
/// ```
/// use std::cmp::Ordering;
/// use exactum::{Point2, predicates::orient2d};
///
/// let a = Point2::new(0_i64, 0);
/// let b = Point2::new(1, 0);
/// let c = Point2::new(0, 1);
///
/// assert_eq!(orient2d(a, b, c), Ordering::Greater); // counter-clockwise
/// assert_eq!(orient2d(a, c, b), Ordering::Less);    // clockwise
/// ```
#[must_use]
pub fn orient2d<T: Widen>(a: Point2<T>, b: Point2<T>, c: Point2<T>) -> Ordering
where
    T::Wide: Wide<Narrow = T>,
{
    let ax = a.x.to_wide();
    let ay = a.y.to_wide();
    let bx = b.x.to_wide();
    let by = b.y.to_wide();
    let cx = c.x.to_wide();
    let cy = c.y.to_wide();

    let abx = bx - ax.clone();
    let aby = by - ay.clone();
    let acx = cx - ax;
    let acy = cy - ay;

    let det = abx * acy - aby * acx;
    det.sign()
}

/// Tests if point `d` lies inside the circumcircle of triangle `abc`.
///
/// Assumes `a`, `b`, `c` are in counter-clockwise order. Returns:
/// - `Ordering::Greater` if `d` is inside the circumcircle
/// - `Ordering::Less` if outside
/// - `Ordering::Equal` if exactly on the circle (cocircular)
///
/// This computes the sign of a 3×3 determinant using the lifted parabola method:
///
/// ```text
/// | ax-dx  ay-dy  (ax-dx)² + (ay-dy)² |
/// | bx-dx  by-dy  (bx-dx)² + (by-dy)² |
/// | cx-dx  cy-dy  (cx-dx)² + (cy-dy)² |
/// ```
///
/// # Note
///
/// This function is limited to `i64` coordinates. The determinant computation
/// involves products of products which can exceed `i128` for large coordinates.
/// For coordinates with absolute value less than 2^30, the computation is exact.
#[must_use]
pub fn incircle(a: Point2<i64>, b: Point2<i64>, c: Point2<i64>, d: Point2<i64>) -> Ordering {
    // Translate so that d is at origin
    let ax = (a.x - d.x) as i128;
    let ay = (a.y - d.y) as i128;
    let bx = (b.x - d.x) as i128;
    let by = (b.y - d.y) as i128;
    let cx = (c.x - d.x) as i128;
    let cy = (c.y - d.y) as i128;

    // Compute the lifted coordinates (x² + y²)
    let az = ax * ax + ay * ay;
    let bz = bx * bx + by * by;
    let cz = cx * cx + cy * cy;

    // 3×3 determinant:
    // | ax  ay  az |
    // | bx  by  bz |
    // | cx  cy  cz |
    let det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);

    det.cmp(&0)
}

/// Tests if three points are collinear.
#[must_use]
#[inline]
pub fn collinear<T: Widen>(a: Point2<T>, b: Point2<T>, c: Point2<T>) -> bool
where
    T::Wide: Wide<Narrow = T>,
{
    orient2d(a, b, c) == Ordering::Equal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orient2d_ccw_i64() {
        let a = Point2::new(0_i64, 0);
        let b = Point2::new(1, 0);
        let c = Point2::new(0, 1);
        assert_eq!(orient2d(a, b, c), Ordering::Greater);
    }

    #[test]
    fn orient2d_ccw_i32() {
        let a = Point2::new(0_i32, 0);
        let b = Point2::new(1, 0);
        let c = Point2::new(0, 1);
        assert_eq!(orient2d(a, b, c), Ordering::Greater);
    }

    #[test]
    fn orient2d_cw() {
        let a = Point2::new(0_i64, 0);
        let b = Point2::new(0, 1);
        let c = Point2::new(1, 0);
        assert_eq!(orient2d(a, b, c), Ordering::Less);
    }

    #[test]
    fn orient2d_collinear() {
        let a = Point2::new(0_i64, 0);
        let b = Point2::new(1, 1);
        let c = Point2::new(2, 2);
        assert_eq!(orient2d(a, b, c), Ordering::Equal);
    }

    #[test]
    fn incircle_inside() {
        // Triangle with vertices at (0,0), (4,0), (0,4)
        // Point (1,1) should be inside the circumcircle
        let a = Point2::new(0, 0);
        let b = Point2::new(4, 0);
        let c = Point2::new(0, 4);
        let d = Point2::new(1, 1);
        assert_eq!(incircle(a, b, c, d), Ordering::Greater);
    }

    #[test]
    fn incircle_outside() {
        let a = Point2::new(0, 0);
        let b = Point2::new(4, 0);
        let c = Point2::new(0, 4);
        let d = Point2::new(10, 10);
        assert_eq!(incircle(a, b, c, d), Ordering::Less);
    }

    #[test]
    fn incircle_on_circle() {
        // Unit circle: points at (1,0), (0,1), (-1,0), (0,-1) are cocircular
        let a = Point2::new(1, 0);
        let b = Point2::new(0, 1);
        let c = Point2::new(-1, 0);
        let d = Point2::new(0, -1);
        assert_eq!(incircle(a, b, c, d), Ordering::Equal);
    }

    #[test]
    fn collinear_true() {
        let a = Point2::new(0_i64, 0);
        let b = Point2::new(5, 5);
        let c = Point2::new(10, 10);
        assert!(collinear(a, b, c));
    }

    #[test]
    fn collinear_false() {
        let a = Point2::new(0_i64, 0);
        let b = Point2::new(1, 0);
        let c = Point2::new(0, 1);
        assert!(!collinear(a, b, c));
    }
}
