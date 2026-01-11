//! Convex hull algorithms.

use std::cmp::Ordering;

use crate::predicates::orient2d;
use crate::widen::{Wide, Widen};
use crate::Point2;

/// Computes the convex hull of a set of 2D points using Graham scan.
///
/// Returns the vertices of the convex hull in counter-clockwise order,
/// starting from the lowest point (and leftmost if tied).
///
/// # Time Complexity
///
/// O(n log n) where n is the number of points.
///
/// # Example
///
/// ```
/// use exactum::{Point2, algo::graham_scan};
///
/// let points = vec![
///     Point2::new(0_i64, 0),
///     Point2::new(4, 0),
///     Point2::new(2, 2),  // interior point
///     Point2::new(0, 4),
///     Point2::new(4, 4),
/// ];
///
/// let hull = graham_scan(&points);
/// assert_eq!(hull.len(), 4); // square corners, interior point excluded
/// ```
#[must_use]
pub fn graham_scan<T>(points: &[Point2<T>]) -> Vec<Point2<T>>
where
    T: Widen,
    T::Wide: Wide<Narrow = T>,
{
    if points.len() < 3 {
        return points.to_vec();
    }

    // Find the lowest point (and leftmost if tied)
    let mut sorted: Vec<_> = points.to_vec();
    let pivot_idx = sorted
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.y.cmp(&b.y).then_with(|| a.x.cmp(&b.x)))
        .map(|(i, _)| i)
        .unwrap();

    sorted.swap(0, pivot_idx);
    let pivot = sorted[0];

    // Remove duplicates of the pivot (they cause degenerate orient2d calls)
    let mut write = 1;
    for read in 1..sorted.len() {
        if sorted[read] != pivot {
            sorted[write] = sorted[read];
            write += 1;
        }
    }
    sorted.truncate(write);

    if sorted.len() < 3 {
        return sorted;
    }

    // Sort remaining points by polar angle with respect to pivot
    sorted[1..].sort_by(|a, b| {
        let o = orient2d(pivot, *a, *b);
        match o {
            Ordering::Equal => {
                // Collinear: keep the farther point, but we need to sort
                // by distance so the closer ones come first (they'll be
                // eliminated by the hull-building step)
                let da = distance_squared_to_pivot(pivot, *a);
                let db = distance_squared_to_pivot(pivot, *b);
                da.cmp(&db)
            }
            Ordering::Greater => Ordering::Less, // a has smaller angle (comes first)
            Ordering::Less => Ordering::Greater, // b has smaller angle (comes first)
        }
    });

    // Remove collinear points, keeping only the farthest
    // We do this by deduplicating runs of collinear points
    let mut unique = vec![sorted[0]];
    let mut i = 1;
    while i < sorted.len() {
        // Find the end of the collinear run
        let mut j = i;
        while j + 1 < sorted.len() && orient2d(pivot, sorted[i], sorted[j + 1]) == Ordering::Equal {
            j += 1;
        }
        // Keep only the farthest point in this collinear run
        unique.push(sorted[j]);
        i = j + 1;
    }

    if unique.len() < 3 {
        return unique;
    }

    // Build hull using a stack
    let mut hull: Vec<Point2<T>> = Vec::with_capacity(unique.len());

    for p in unique {
        // Pop points that would make a right turn (or be collinear)
        while hull.len() >= 2 {
            let top = hull[hull.len() - 1];
            let below = hull[hull.len() - 2];
            if orient2d(below, top, p) != Ordering::Greater {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }

    hull
}

/// Squared distance from pivot to point (for tie-breaking collinear points).
fn distance_squared_to_pivot<T: Widen>(pivot: Point2<T>, p: Point2<T>) -> T::Wide
where
    T::Wide: Wide<Narrow = T>,
{
    let dx = p.x.to_wide() - pivot.x.to_wide();
    let dy = p.y.to_wide() - pivot.y.to_wide();
    dx.clone() * dx + dy.clone() * dy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graham_scan_square_with_interior() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(4, 0),
            Point2::new(4, 4),
            Point2::new(0, 4),
            Point2::new(2, 2), // interior point
        ];
        let hull = graham_scan(&points);
        assert_eq!(hull.len(), 4);
        // Verify all corners are present
        assert!(hull.contains(&Point2::new(0, 0)));
        assert!(hull.contains(&Point2::new(4, 0)));
        assert!(hull.contains(&Point2::new(4, 4)));
        assert!(hull.contains(&Point2::new(0, 4)));
    }

    #[test]
    fn graham_scan_triangle() {
        let points = vec![Point2::new(0_i64, 0), Point2::new(4, 0), Point2::new(2, 3)];
        let hull = graham_scan(&points);
        assert_eq!(hull.len(), 3);
    }

    #[test]
    fn graham_scan_collinear() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(1, 0),
            Point2::new(2, 0),
            Point2::new(3, 0),
        ];
        let hull = graham_scan(&points);
        // Collinear points form a degenerate hull (line segment)
        assert_eq!(hull.len(), 2);
        assert!(hull.contains(&Point2::new(0, 0)));
        assert!(hull.contains(&Point2::new(3, 0)));
    }

    #[test]
    fn graham_scan_single_point() {
        let points = vec![Point2::new(5_i64, 5)];
        let hull = graham_scan(&points);
        assert_eq!(hull.len(), 1);
    }

    #[test]
    fn graham_scan_two_points() {
        let points = vec![Point2::new(0_i64, 0), Point2::new(1, 1)];
        let hull = graham_scan(&points);
        assert_eq!(hull.len(), 2);
    }

    #[test]
    fn graham_scan_duplicate_points() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(0, 0),
            Point2::new(4, 0),
            Point2::new(4, 4),
            Point2::new(0, 4),
        ];
        let hull = graham_scan(&points);
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn graham_scan_ccw_order() {
        // Verify the hull is in counter-clockwise order
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(4, 0),
            Point2::new(4, 4),
            Point2::new(0, 4),
        ];
        let hull = graham_scan(&points);

        // Check each consecutive triple is CCW
        for i in 0..hull.len() {
            let a = hull[i];
            let b = hull[(i + 1) % hull.len()];
            let c = hull[(i + 2) % hull.len()];
            assert_eq!(orient2d(a, b, c), Ordering::Greater);
        }
    }

    #[test]
    fn graham_scan_i32() {
        // Test with i32 coordinates
        let points = vec![
            Point2::new(0_i32, 0),
            Point2::new(10, 0),
            Point2::new(10, 10),
            Point2::new(0, 10),
            Point2::new(5, 5),
        ];
        let hull = graham_scan(&points);
        assert_eq!(hull.len(), 4);
    }
}
