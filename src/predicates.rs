//! Exact geometric predicates.
//!
//! All predicates return [`std::cmp::Ordering`] to indicate the geometric
//! relationship. They use widening multiplication to avoid overflow.

use std::cmp::Ordering;

use crate::widen::{Wide, Widen};
use crate::{Point2, Point3};

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

// 3D Predicates

/// Computes the orientation of four 3D points.
///
/// Returns:
/// - `Ordering::Greater` if `d` is above the plane through `a`, `b`, `c`
///   (in the positive half-space defined by the right-hand rule normal)
/// - `Ordering::Less` if below (negative half-space)
/// - `Ordering::Equal` if coplanar
///
/// The result is the sign of the 3×3 determinant:
///
/// ```text
/// | b.x - a.x   c.x - a.x   d.x - a.x |
/// | b.y - a.y   c.y - a.y   d.y - a.y |
/// | b.z - a.z   c.z - a.z   d.z - a.z |
/// ```
///
/// This is equivalent to 6× the signed volume of tetrahedron `abcd`.
///
/// # Example
///
/// ```
/// use std::cmp::Ordering;
/// use exactum::{Point3, predicates::orient3d};
///
/// let a = Point3::new(0_i64, 0, 0);
/// let b = Point3::new(1, 0, 0);
/// let c = Point3::new(0, 1, 0);
/// let d = Point3::new(0, 0, 1);
///
/// assert_eq!(orient3d(a, b, c, d), Ordering::Greater); // d above plane abc
/// assert_eq!(orient3d(a, b, c, Point3::new(0, 0, -1)), Ordering::Less); // below
/// ```
#[must_use]
pub fn orient3d(a: Point3<i64>, b: Point3<i64>, c: Point3<i64>, d: Point3<i64>) -> Ordering {
    // Translate so a is at origin
    let bx = (b.x - a.x) as i128;
    let by = (b.y - a.y) as i128;
    let bz = (b.z - a.z) as i128;
    let cx = (c.x - a.x) as i128;
    let cy = (c.y - a.y) as i128;
    let cz = (c.z - a.z) as i128;
    let dx = (d.x - a.x) as i128;
    let dy = (d.y - a.y) as i128;
    let dz = (d.z - a.z) as i128;

    // 3×3 determinant via cofactor expansion along first row:
    // det = bx * (cy*dz - cz*dy) - cx * (by*dz - bz*dy) + dx * (by*cz - bz*cy)
    let det = bx * (cy * dz - cz * dy) - cx * (by * dz - bz * dy) + dx * (by * cz - bz * cy);

    det.cmp(&0)
}

/// Tests if point `e` lies inside the circumsphere of tetrahedron `abcd`.
///
/// Assumes `a`, `b`, `c`, `d` form a positively-oriented tetrahedron
/// (i.e., `orient3d(a, b, c, d) > 0`). Returns:
/// - `Ordering::Greater` if `e` is inside the circumsphere
/// - `Ordering::Less` if outside
/// - `Ordering::Equal` if exactly on the sphere (cospherical)
///
/// This computes the sign of a 4×4 determinant using the lifted paraboloid method:
///
/// ```text
/// | ax-ex  ay-ey  az-ez  (ax-ex)² + (ay-ey)² + (az-ez)² |
/// | bx-ex  by-ey  bz-ez  (bx-ex)² + (by-ey)² + (bz-ez)² |
/// | cx-ex  cy-ey  cz-ez  (cx-ex)² + (cy-ey)² + (cz-ez)² |
/// | dx-ex  dy-ey  dz-ez  (dx-ex)² + (dy-ey)² + (dz-ez)² |
/// ```
///
/// # Note
///
/// This function is limited to `i64` coordinates. The determinant computation
/// involves products of products which can overflow for large coordinates.
/// For coordinates with absolute value less than 2^20, the computation is exact.
///
/// # Example
///
/// ```
/// use std::cmp::Ordering;
/// use exactum::{Point3, predicates::insphere};
///
/// // Regular tetrahedron vertices (approximately)
/// let a = Point3::new(1_i64, 1, 1);
/// let b = Point3::new(-1, -1, 1);
/// let c = Point3::new(-1, 1, -1);
/// let d = Point3::new(1, -1, -1);
///
/// // Origin is inside this tetrahedron's circumsphere
/// let e = Point3::new(0, 0, 0);
/// assert_eq!(insphere(a, b, c, d, e), Ordering::Greater);
/// ```
#[must_use]
pub fn insphere(
    a: Point3<i64>,
    b: Point3<i64>,
    c: Point3<i64>,
    d: Point3<i64>,
    e: Point3<i64>,
) -> Ordering {
    // Translate so e is at origin
    let ax = (a.x - e.x) as i128;
    let ay = (a.y - e.y) as i128;
    let az = (a.z - e.z) as i128;
    let bx = (b.x - e.x) as i128;
    let by = (b.y - e.y) as i128;
    let bz = (b.z - e.z) as i128;
    let cx = (c.x - e.x) as i128;
    let cy = (c.y - e.y) as i128;
    let cz = (c.z - e.z) as i128;
    let dx = (d.x - e.x) as i128;
    let dy = (d.y - e.y) as i128;
    let dz = (d.z - e.z) as i128;

    // Lifted coordinates
    let aw = ax * ax + ay * ay + az * az;
    let bw = bx * bx + by * by + bz * bz;
    let cw = cx * cx + cy * cy + cz * cz;
    let dw = dx * dx + dy * dy + dz * dz;

    // 4×4 determinant via cofactor expansion
    // We expand along the last column (w coordinates)
    //
    // det = aw * M_a - bw * M_b + cw * M_c - dw * M_d
    //
    // where M_i is the 3×3 minor obtained by deleting row i and column 4

    // M_a = det of rows b,c,d, columns x,y,z
    let m_a = det3x3(bx, by, bz, cx, cy, cz, dx, dy, dz);

    // M_b = det of rows a,c,d, columns x,y,z
    let m_b = det3x3(ax, ay, az, cx, cy, cz, dx, dy, dz);

    // M_c = det of rows a,b,d, columns x,y,z
    let m_c = det3x3(ax, ay, az, bx, by, bz, dx, dy, dz);

    // M_d = det of rows a,b,c, columns x,y,z
    let m_d = det3x3(ax, ay, az, bx, by, bz, cx, cy, cz);

    let det = aw * m_a - bw * m_b + cw * m_c - dw * m_d;

    det.cmp(&0)
}

/// Computes a 3×3 determinant.
#[inline]
#[allow(clippy::too_many_arguments)]
fn det3x3(a: i128, b: i128, c: i128, d: i128, e: i128, f: i128, g: i128, h: i128, i: i128) -> i128 {
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

/// Tests if four 3D points are coplanar.
#[must_use]
#[inline]
pub fn coplanar(a: Point3<i64>, b: Point3<i64>, c: Point3<i64>, d: Point3<i64>) -> bool {
    orient3d(a, b, c, d) == Ordering::Equal
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

    // 3D predicate tests

    #[test]
    fn orient3d_above() {
        let a = Point3::new(0_i64, 0, 0);
        let b = Point3::new(1, 0, 0);
        let c = Point3::new(0, 1, 0);
        let d = Point3::new(0, 0, 1);
        assert_eq!(orient3d(a, b, c, d), Ordering::Greater);
    }

    #[test]
    fn orient3d_below() {
        let a = Point3::new(0_i64, 0, 0);
        let b = Point3::new(1, 0, 0);
        let c = Point3::new(0, 1, 0);
        let d = Point3::new(0, 0, -1);
        assert_eq!(orient3d(a, b, c, d), Ordering::Less);
    }

    #[test]
    fn orient3d_coplanar() {
        let a = Point3::new(0_i64, 0, 0);
        let b = Point3::new(1, 0, 0);
        let c = Point3::new(0, 1, 0);
        let d = Point3::new(1, 1, 0); // On the xy plane
        assert_eq!(orient3d(a, b, c, d), Ordering::Equal);
    }

    #[test]
    fn coplanar_true() {
        let a = Point3::new(0_i64, 0, 0);
        let b = Point3::new(1, 0, 0);
        let c = Point3::new(0, 1, 0);
        let d = Point3::new(2, 3, 0);
        assert!(coplanar(a, b, c, d));
    }

    #[test]
    fn coplanar_false() {
        let a = Point3::new(0_i64, 0, 0);
        let b = Point3::new(1, 0, 0);
        let c = Point3::new(0, 1, 0);
        let d = Point3::new(0, 0, 1);
        assert!(!coplanar(a, b, c, d));
    }

    #[test]
    fn insphere_inside() {
        // Tetrahedron with vertices at unit positions
        let a = Point3::new(1_i64, 1, 1);
        let b = Point3::new(-1, -1, 1);
        let c = Point3::new(-1, 1, -1);
        let d = Point3::new(1, -1, -1);
        // Origin should be inside the circumsphere
        let e = Point3::new(0, 0, 0);
        assert_eq!(insphere(a, b, c, d, e), Ordering::Greater);
    }

    #[test]
    fn insphere_outside() {
        let a = Point3::new(1_i64, 1, 1);
        let b = Point3::new(-1, -1, 1);
        let c = Point3::new(-1, 1, -1);
        let d = Point3::new(1, -1, -1);
        // Far point should be outside
        let e = Point3::new(10, 10, 10);
        assert_eq!(insphere(a, b, c, d, e), Ordering::Less);
    }

    #[test]
    fn insphere_on_sphere() {
        // Vertices of a regular tetrahedron inscribed in a sphere
        // All vertices are equidistant from origin, so any vertex
        // should be on the circumsphere of the others
        let a = Point3::new(1_i64, 0, 0);
        let b = Point3::new(0, 1, 0);
        let c = Point3::new(0, 0, 1);
        // d on the sphere: point at distance 1 from origin
        let d = Point3::new(-1, 0, 0);
        // e also on sphere
        let e = Point3::new(0, -1, 0);
        // This should be cospherical (on the unit sphere)
        assert_eq!(insphere(a, b, c, d, e), Ordering::Equal);
    }
}
