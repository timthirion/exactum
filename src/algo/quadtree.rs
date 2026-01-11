//! Point quadtree for efficient 2D spatial queries.
//!
//! Quadtrees recursively subdivide 2D space into four quadrants, making them
//! effective for spatial indexing with non-uniform point distributions.

use std::collections::BinaryHeap;

use crate::ops::distance_squared;
use crate::Point2;

use super::kdtree::NearestResult;

/// Default number of points per leaf node before splitting.
const DEFAULT_BUCKET_CAPACITY: usize = 8;

/// Axis-aligned bounding box for quadtree regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds {
    /// Minimum corner (inclusive).
    pub min: Point2<i64>,
    /// Maximum corner (inclusive).
    pub max: Point2<i64>,
}

impl Bounds {
    /// Creates a new bounding box.
    #[must_use]
    pub fn new(min: Point2<i64>, max: Point2<i64>) -> Self {
        Self { min, max }
    }

    /// Returns true if the point is inside or on the boundary.
    #[must_use]
    pub fn contains(&self, point: Point2<i64>) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }

    /// Returns true if this box intersects another box.
    #[must_use]
    pub fn intersects(&self, other: &Bounds) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    /// Returns the center point of the bounds.
    #[must_use]
    pub fn center(&self) -> Point2<i64> {
        Point2::new((self.min.x + self.max.x) / 2, (self.min.y + self.max.y) / 2)
    }

    /// Returns the bounds for a specific quadrant.
    /// Quadrants: 0=NE, 1=NW, 2=SW, 3=SE
    #[must_use]
    pub fn quadrant_bounds(&self, quadrant: usize, center: Point2<i64>) -> Bounds {
        match quadrant {
            0 => Bounds::new(center, self.max), // NE
            1 => Bounds::new(
                Point2::new(self.min.x, center.y),
                Point2::new(center.x, self.max.y),
            ), // NW
            2 => Bounds::new(self.min, center), // SW
            3 => Bounds::new(
                Point2::new(center.x, self.min.y),
                Point2::new(self.max.x, center.y),
            ), // SE
            _ => panic!("Invalid quadrant: {}", quadrant),
        }
    }

    /// Returns the squared distance from a point to the nearest point in the bounds.
    /// Returns 0 if the point is inside the bounds.
    #[must_use]
    pub fn distance_squared_to(&self, point: Point2<i64>) -> i128 {
        let dx = if point.x < self.min.x {
            (self.min.x - point.x) as i128
        } else if point.x > self.max.x {
            (point.x - self.max.x) as i128
        } else {
            0
        };

        let dy = if point.y < self.min.y {
            (self.min.y - point.y) as i128
        } else if point.y > self.max.y {
            (point.y - self.max.y) as i128
        } else {
            0
        };

        dx * dx + dy * dy
    }
}

/// Internal node structure for the quadtree.
#[derive(Debug, Clone)]
enum QuadNode {
    /// Leaf node containing point indices.
    Leaf { point_indices: Vec<usize> },
    /// Internal node with four children.
    Internal {
        center: Point2<i64>,
        /// Children in order: NE, NW, SW, SE
        children: [Option<Box<QuadNode>>; 4],
    },
}

/// A point quadtree for efficient 2D spatial queries.
///
/// # Example
///
/// ```
/// use exactum::{Point2, algo::Quadtree};
///
/// let points = vec![
///     Point2::new(0_i64, 0),
///     Point2::new(10, 10),
///     Point2::new(5, 5),
///     Point2::new(3, 7),
/// ];
///
/// let tree = Quadtree::new(&points);
///
/// // Find nearest point to (4, 4)
/// let result = tree.nearest(Point2::new(4, 4)).unwrap();
/// assert_eq!(result.point_idx, 2); // Point (5, 5) is closest
/// ```
#[derive(Debug, Clone)]
pub struct Quadtree {
    root: Option<QuadNode>,
    points: Vec<Point2<i64>>,
    bounds: Bounds,
}

impl Quadtree {
    /// Builds a quadtree from a set of points with default bucket capacity.
    ///
    /// Construction takes O(n log n) time on average.
    #[must_use]
    pub fn new(points: &[Point2<i64>]) -> Self {
        Self::with_capacity(points, DEFAULT_BUCKET_CAPACITY)
    }

    /// Builds a quadtree with a custom bucket capacity.
    ///
    /// Larger bucket capacity means shallower trees but slower leaf searches.
    #[must_use]
    pub fn with_capacity(points: &[Point2<i64>], bucket_capacity: usize) -> Self {
        if points.is_empty() {
            return Self {
                root: None,
                points: Vec::new(),
                bounds: Bounds::new(Point2::new(0, 0), Point2::new(0, 0)),
            };
        }

        let points_vec = points.to_vec();

        // Compute bounding box
        let mut min_x = i64::MAX;
        let mut min_y = i64::MAX;
        let mut max_x = i64::MIN;
        let mut max_y = i64::MIN;

        for p in &points_vec {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }

        let bounds = Bounds::new(Point2::new(min_x, min_y), Point2::new(max_x, max_y));

        // Build tree by inserting all points
        let indices: Vec<usize> = (0..points_vec.len()).collect();
        let root = Self::build_node(&points_vec, &indices, &bounds, bucket_capacity);

        Self {
            root: Some(root),
            points: points_vec,
            bounds,
        }
    }

    fn build_node(
        points: &[Point2<i64>],
        indices: &[usize],
        bounds: &Bounds,
        bucket_capacity: usize,
    ) -> QuadNode {
        if indices.len() <= bucket_capacity {
            return QuadNode::Leaf {
                point_indices: indices.to_vec(),
            };
        }

        let center = bounds.center();

        // Partition points into quadrants
        let mut quadrant_indices: [Vec<usize>; 4] = Default::default();

        for &idx in indices {
            let q = Self::get_quadrant(center, points[idx]);
            quadrant_indices[q].push(idx);
        }

        // Build children recursively
        let mut children: [Option<Box<QuadNode>>; 4] = Default::default();

        for (i, child_indices) in quadrant_indices.iter().enumerate() {
            if !child_indices.is_empty() {
                let child_bounds = bounds.quadrant_bounds(i, center);
                children[i] = Some(Box::new(Self::build_node(
                    points,
                    child_indices,
                    &child_bounds,
                    bucket_capacity,
                )));
            }
        }

        QuadNode::Internal { center, children }
    }

    /// Determines which quadrant a point belongs to relative to a center.
    /// Returns: 0=NE, 1=NW, 2=SW, 3=SE
    fn get_quadrant(center: Point2<i64>, point: Point2<i64>) -> usize {
        let right = point.x >= center.x;
        let top = point.y >= center.y;
        match (right, top) {
            (true, true) => 0,   // NE
            (false, true) => 1,  // NW
            (false, false) => 2, // SW
            (true, false) => 3,  // SE
        }
    }

    /// Finds the nearest point to the query.
    ///
    /// Returns `None` if the tree is empty.
    #[must_use]
    pub fn nearest(&self, query: Point2<i64>) -> Option<NearestResult> {
        self.root.as_ref().map(|root| {
            self.nearest_recursive(
                root,
                &self.bounds,
                query,
                NearestResult {
                    point_idx: usize::MAX,
                    distance_squared: i128::MAX,
                },
            )
        })
    }

    fn nearest_recursive(
        &self,
        node: &QuadNode,
        bounds: &Bounds,
        query: Point2<i64>,
        mut best: NearestResult,
    ) -> NearestResult {
        match node {
            QuadNode::Leaf { point_indices } => {
                for &idx in point_indices {
                    let dist = distance_squared(self.points[idx], query);
                    if dist < best.distance_squared {
                        best = NearestResult {
                            point_idx: idx,
                            distance_squared: dist,
                        };
                    }
                }
                best
            }
            QuadNode::Internal { center, children } => {
                // Order children by distance to query point
                let mut child_order: Vec<(usize, i128)> = children
                    .iter()
                    .enumerate()
                    .filter_map(|(i, child)| {
                        child.as_ref().map(|_| {
                            let child_bounds = bounds.quadrant_bounds(i, *center);
                            (i, child_bounds.distance_squared_to(query))
                        })
                    })
                    .collect();

                child_order.sort_by_key(|&(_, dist)| dist);

                for (i, dist_to_bounds) in child_order {
                    // Prune if this quadrant can't contain a closer point
                    if dist_to_bounds >= best.distance_squared {
                        break;
                    }

                    if let Some(child) = &children[i] {
                        let child_bounds = bounds.quadrant_bounds(i, *center);
                        best = self.nearest_recursive(child, &child_bounds, query, best);
                    }
                }

                best
            }
        }
    }

    /// Finds the k nearest points to the query.
    ///
    /// Returns up to k results, sorted by distance (closest first).
    #[must_use]
    pub fn k_nearest(&self, query: Point2<i64>, k: usize) -> Vec<NearestResult> {
        if k == 0 || self.root.is_none() {
            return Vec::new();
        }

        // Use max-heap to track k nearest
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

        self.k_nearest_recursive(
            self.root.as_ref().unwrap(),
            &self.bounds,
            query,
            k,
            &mut heap,
        );

        // Convert heap to sorted vector
        let mut results: Vec<NearestResult> = heap
            .into_iter()
            .map(|e| NearestResult {
                point_idx: e.point_idx,
                distance_squared: e.distance_squared,
            })
            .collect();

        results.sort_by_key(|r| r.distance_squared);
        results
    }

    fn k_nearest_recursive(
        &self,
        node: &QuadNode,
        bounds: &Bounds,
        query: Point2<i64>,
        k: usize,
        heap: &mut BinaryHeap<HeapEntry>,
    ) {
        match node {
            QuadNode::Leaf { point_indices } => {
                for &idx in point_indices {
                    let dist = distance_squared(self.points[idx], query);

                    if heap.len() < k {
                        heap.push(HeapEntry {
                            point_idx: idx,
                            distance_squared: dist,
                        });
                    } else if let Some(worst) = heap.peek() {
                        if dist < worst.distance_squared {
                            heap.pop();
                            heap.push(HeapEntry {
                                point_idx: idx,
                                distance_squared: dist,
                            });
                        }
                    }
                }
            }
            QuadNode::Internal { center, children } => {
                // Order children by distance
                let mut child_order: Vec<(usize, i128)> = children
                    .iter()
                    .enumerate()
                    .filter_map(|(i, child)| {
                        child.as_ref().map(|_| {
                            let child_bounds = bounds.quadrant_bounds(i, *center);
                            (i, child_bounds.distance_squared_to(query))
                        })
                    })
                    .collect();

                child_order.sort_by_key(|&(_, dist)| dist);

                for (i, dist_to_bounds) in child_order {
                    // Prune if heap is full and this quadrant can't improve
                    if heap.len() >= k {
                        if let Some(worst) = heap.peek() {
                            if dist_to_bounds >= worst.distance_squared {
                                continue;
                            }
                        }
                    }

                    if let Some(child) = &children[i] {
                        let child_bounds = bounds.quadrant_bounds(i, *center);
                        self.k_nearest_recursive(child, &child_bounds, query, k, heap);
                    }
                }
            }
        }
    }

    /// Finds all points within the given bounding box.
    ///
    /// Returns indices of points where `min <= point <= max` (inclusive).
    #[must_use]
    pub fn range_query(&self, min: Point2<i64>, max: Point2<i64>) -> Vec<usize> {
        let mut results = Vec::new();
        let query_bounds = Bounds::new(min, max);

        if let Some(root) = &self.root {
            self.range_recursive(root, &self.bounds, &query_bounds, &mut results);
        }

        results
    }

    fn range_recursive(
        &self,
        node: &QuadNode,
        bounds: &Bounds,
        query_bounds: &Bounds,
        results: &mut Vec<usize>,
    ) {
        match node {
            QuadNode::Leaf { point_indices } => {
                for &idx in point_indices {
                    if query_bounds.contains(self.points[idx]) {
                        results.push(idx);
                    }
                }
            }
            QuadNode::Internal { center, children } => {
                for (i, child) in children.iter().enumerate() {
                    if let Some(child_node) = child {
                        let child_bounds = bounds.quadrant_bounds(i, *center);
                        if query_bounds.intersects(&child_bounds) {
                            self.range_recursive(child_node, &child_bounds, query_bounds, results);
                        }
                    }
                }
            }
        }
    }

    /// Returns the number of points in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns true if the tree is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Returns the bounding box of all points in the tree.
    #[must_use]
    pub fn bounds(&self) -> Bounds {
        self.bounds
    }

    /// Returns a reference to the point at the given index.
    #[must_use]
    pub fn get_point(&self, idx: usize) -> Option<&Point2<i64>> {
        self.points.get(idx)
    }
}

/// Entry for the k-nearest max-heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HeapEntry {
    point_idx: usize,
    distance_squared: i128,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for max-heap behavior
        self.distance_squared
            .cmp(&other.distance_squared)
            .then_with(|| self.point_idx.cmp(&other.point_idx))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounds_contains() {
        let bounds = Bounds::new(Point2::new(0, 0), Point2::new(10, 10));

        assert!(bounds.contains(Point2::new(5, 5)));
        assert!(bounds.contains(Point2::new(0, 0)));
        assert!(bounds.contains(Point2::new(10, 10)));
        assert!(!bounds.contains(Point2::new(-1, 5)));
        assert!(!bounds.contains(Point2::new(11, 5)));
    }

    #[test]
    fn test_bounds_intersects() {
        let b1 = Bounds::new(Point2::new(0, 0), Point2::new(10, 10));
        let b2 = Bounds::new(Point2::new(5, 5), Point2::new(15, 15));
        let b3 = Bounds::new(Point2::new(20, 20), Point2::new(30, 30));

        assert!(b1.intersects(&b2));
        assert!(b2.intersects(&b1));
        assert!(!b1.intersects(&b3));
    }

    #[test]
    fn test_bounds_distance_squared() {
        let bounds = Bounds::new(Point2::new(0, 0), Point2::new(10, 10));

        // Inside
        assert_eq!(bounds.distance_squared_to(Point2::new(5, 5)), 0);

        // On boundary
        assert_eq!(bounds.distance_squared_to(Point2::new(0, 0)), 0);

        // Outside
        assert_eq!(bounds.distance_squared_to(Point2::new(-3, 5)), 9); // 3^2
        assert_eq!(bounds.distance_squared_to(Point2::new(-3, -4)), 25); // 3^2 + 4^2
    }

    #[test]
    fn test_build_empty() {
        let tree = Quadtree::new(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.nearest(Point2::new(0, 0)).is_none());
    }

    #[test]
    fn test_build_single_point() {
        let points = vec![Point2::new(5_i64, 5)];
        let tree = Quadtree::new(&points);

        assert_eq!(tree.len(), 1);
        let result = tree.nearest(Point2::new(0, 0)).unwrap();
        assert_eq!(result.point_idx, 0);
    }

    #[test]
    fn test_build_many_points() {
        let points: Vec<Point2<i64>> = (0..100).map(|i| Point2::new(i % 10, i / 10)).collect();
        let tree = Quadtree::new(&points);

        assert_eq!(tree.len(), 100);
    }

    #[test]
    fn test_custom_capacity() {
        let points: Vec<Point2<i64>> = (0..20).map(|i| Point2::new(i, i)).collect();
        let tree = Quadtree::with_capacity(&points, 4);

        assert_eq!(tree.len(), 20);
        // Should still work correctly
        let result = tree.nearest(Point2::new(10, 10)).unwrap();
        assert_eq!(result.point_idx, 10);
    }

    #[test]
    fn test_nearest_exact_match() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 10),
            Point2::new(5, 5),
        ];
        let tree = Quadtree::new(&points);

        let result = tree.nearest(Point2::new(5, 5)).unwrap();
        assert_eq!(result.point_idx, 2);
        assert_eq!(result.distance_squared, 0);
    }

    #[test]
    fn test_nearest_simple() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 10),
            Point2::new(5, 5),
            Point2::new(3, 7),
        ];
        let tree = Quadtree::new(&points);

        let result = tree.nearest(Point2::new(4, 4)).unwrap();
        assert_eq!(result.point_idx, 2); // (5,5) is closest to (4,4)
    }

    #[test]
    fn test_nearest_different_quadrants() {
        let points = vec![
            Point2::new(-10_i64, -10), // SW
            Point2::new(10, -10),      // SE
            Point2::new(-10, 10),      // NW
            Point2::new(10, 10),       // NE
        ];
        let tree = Quadtree::new(&points);

        // Query in NE quadrant
        let result = tree.nearest(Point2::new(8, 8)).unwrap();
        assert_eq!(result.point_idx, 3);

        // Query in SW quadrant
        let result = tree.nearest(Point2::new(-8, -8)).unwrap();
        assert_eq!(result.point_idx, 0);
    }

    #[test]
    fn test_k_nearest_basic() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(1, 1),
            Point2::new(2, 2),
            Point2::new(10, 10),
        ];
        let tree = Quadtree::new(&points);

        let results = tree.k_nearest(Point2::new(0, 0), 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].point_idx, 0); // (0,0)
        assert_eq!(results[1].point_idx, 1); // (1,1)
        assert_eq!(results[2].point_idx, 2); // (2,2)
    }

    #[test]
    fn test_k_nearest_less_than_k() {
        let points = vec![Point2::new(0_i64, 0), Point2::new(1, 1)];
        let tree = Quadtree::new(&points);

        let results = tree.k_nearest(Point2::new(0, 0), 5);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_k_nearest_zero() {
        let points = vec![Point2::new(0_i64, 0)];
        let tree = Quadtree::new(&points);

        let results = tree.k_nearest(Point2::new(0, 0), 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_range_query_all() {
        let points = vec![Point2::new(1_i64, 1), Point2::new(2, 2), Point2::new(3, 3)];
        let tree = Quadtree::new(&points);

        let results = tree.range_query(Point2::new(0, 0), Point2::new(10, 10));
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_range_query_none() {
        let points = vec![Point2::new(0_i64, 0), Point2::new(1, 1), Point2::new(2, 2)];
        let tree = Quadtree::new(&points);

        let results = tree.range_query(Point2::new(10, 10), Point2::new(20, 20));
        assert!(results.is_empty());
    }

    #[test]
    fn test_range_query_partial() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(5, 5),
            Point2::new(10, 10),
            Point2::new(15, 15),
        ];
        let tree = Quadtree::new(&points);

        let results = tree.range_query(Point2::new(4, 4), Point2::new(11, 11));
        assert_eq!(results.len(), 2);
        assert!(results.contains(&1)); // (5,5)
        assert!(results.contains(&2)); // (10,10)
    }

    #[test]
    fn test_get_point() {
        let points = vec![Point2::new(5_i64, 5), Point2::new(10, 10)];
        let tree = Quadtree::new(&points);

        assert_eq!(tree.get_point(0), Some(&Point2::new(5, 5)));
        assert_eq!(tree.get_point(1), Some(&Point2::new(10, 10)));
        assert_eq!(tree.get_point(2), None);
    }

    #[test]
    fn test_bounds() {
        let points = vec![
            Point2::new(-5_i64, 0),
            Point2::new(10, 20),
            Point2::new(0, -3),
        ];
        let tree = Quadtree::new(&points);

        let bounds = tree.bounds();
        assert_eq!(bounds.min, Point2::new(-5, -3));
        assert_eq!(bounds.max, Point2::new(10, 20));
    }
}
