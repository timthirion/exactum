//! Point octree for efficient 3D spatial queries.
//!
//! Octrees recursively subdivide 3D space into 8 octants, making them
//! the natural 3D extension of quadtrees.

use std::collections::BinaryHeap;

use crate::ops::distance_squared_3d;
use crate::Point3;

use super::kdtree::NearestResult;

/// Default number of points per leaf node before splitting.
const DEFAULT_BUCKET_CAPACITY: usize = 8;

/// 3D axis-aligned bounding box for octree regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds3 {
    /// Minimum corner (inclusive).
    pub min: Point3<i64>,
    /// Maximum corner (inclusive).
    pub max: Point3<i64>,
}

impl Bounds3 {
    /// Creates a new 3D bounding box.
    #[must_use]
    pub fn new(min: Point3<i64>, max: Point3<i64>) -> Self {
        Self { min, max }
    }

    /// Returns true if the point is inside or on the boundary.
    #[must_use]
    pub fn contains(&self, point: Point3<i64>) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Returns true if this box intersects another box.
    #[must_use]
    pub fn intersects(&self, other: &Bounds3) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Returns the center point of the bounds.
    #[must_use]
    pub fn center(&self) -> Point3<i64> {
        Point3::new(
            (self.min.x + self.max.x) / 2,
            (self.min.y + self.max.y) / 2,
            (self.min.z + self.max.z) / 2,
        )
    }

    /// Returns the bounds for a specific octant.
    ///
    /// Octants are numbered based on (x >= center, y >= center, z >= center):
    /// - 0: (+, +, +)
    /// - 1: (-, +, +)
    /// - 2: (-, -, +)
    /// - 3: (+, -, +)
    /// - 4: (+, +, -)
    /// - 5: (-, +, -)
    /// - 6: (-, -, -)
    /// - 7: (+, -, -)
    #[must_use]
    pub fn octant_bounds(&self, octant: usize, center: Point3<i64>) -> Bounds3 {
        let (x_min, x_max) = if octant & 1 == 0 {
            (center.x, self.max.x)
        } else {
            (self.min.x, center.x)
        };

        let (y_min, y_max) = if octant & 2 == 0 {
            (center.y, self.max.y)
        } else {
            (self.min.y, center.y)
        };

        let (z_min, z_max) = if octant & 4 == 0 {
            (center.z, self.max.z)
        } else {
            (self.min.z, center.z)
        };

        Bounds3::new(
            Point3::new(x_min, y_min, z_min),
            Point3::new(x_max, y_max, z_max),
        )
    }

    /// Returns the squared distance from a point to the nearest point in the bounds.
    /// Returns 0 if the point is inside the bounds.
    #[must_use]
    pub fn distance_squared_to(&self, point: Point3<i64>) -> i128 {
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

        let dz = if point.z < self.min.z {
            (self.min.z - point.z) as i128
        } else if point.z > self.max.z {
            (point.z - self.max.z) as i128
        } else {
            0
        };

        dx * dx + dy * dy + dz * dz
    }
}

/// Internal node structure for the octree.
#[derive(Debug, Clone)]
enum OctNode {
    /// Leaf node containing point indices.
    Leaf { point_indices: Vec<usize> },
    /// Internal node with eight children.
    Internal {
        center: Point3<i64>,
        /// Children indexed by octant number (0-7).
        children: [Option<Box<OctNode>>; 8],
    },
}

/// A point octree for efficient 3D spatial queries.
///
/// # Example
///
/// ```
/// use exactum::{Point3, algo::Octree};
///
/// let points = vec![
///     Point3::new(0_i64, 0, 0),
///     Point3::new(10, 10, 10),
///     Point3::new(5, 5, 5),
///     Point3::new(3, 7, 2),
/// ];
///
/// let tree = Octree::new(&points);
///
/// // Find nearest point to (4, 4, 4)
/// let result = tree.nearest(Point3::new(4, 4, 4)).unwrap();
/// assert_eq!(result.point_idx, 2); // Point (5, 5, 5) is closest
/// ```
#[derive(Debug, Clone)]
pub struct Octree {
    root: Option<OctNode>,
    points: Vec<Point3<i64>>,
    bounds: Bounds3,
}

impl Octree {
    /// Builds an octree from a set of points with default bucket capacity.
    ///
    /// Construction takes O(n log n) time on average.
    #[must_use]
    pub fn new(points: &[Point3<i64>]) -> Self {
        Self::with_capacity(points, DEFAULT_BUCKET_CAPACITY)
    }

    /// Builds an octree with a custom bucket capacity.
    ///
    /// Larger bucket capacity means shallower trees but slower leaf searches.
    #[must_use]
    pub fn with_capacity(points: &[Point3<i64>], bucket_capacity: usize) -> Self {
        if points.is_empty() {
            return Self {
                root: None,
                points: Vec::new(),
                bounds: Bounds3::new(Point3::new(0, 0, 0), Point3::new(0, 0, 0)),
            };
        }

        let points_vec = points.to_vec();

        // Compute bounding box
        let mut min_x = i64::MAX;
        let mut min_y = i64::MAX;
        let mut min_z = i64::MAX;
        let mut max_x = i64::MIN;
        let mut max_y = i64::MIN;
        let mut max_z = i64::MIN;

        for p in &points_vec {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            min_z = min_z.min(p.z);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
            max_z = max_z.max(p.z);
        }

        let bounds = Bounds3::new(
            Point3::new(min_x, min_y, min_z),
            Point3::new(max_x, max_y, max_z),
        );

        // Build tree
        let indices: Vec<usize> = (0..points_vec.len()).collect();
        let root = Self::build_node(&points_vec, &indices, &bounds, bucket_capacity);

        Self {
            root: Some(root),
            points: points_vec,
            bounds,
        }
    }

    fn build_node(
        points: &[Point3<i64>],
        indices: &[usize],
        bounds: &Bounds3,
        bucket_capacity: usize,
    ) -> OctNode {
        if indices.len() <= bucket_capacity {
            return OctNode::Leaf {
                point_indices: indices.to_vec(),
            };
        }

        let center = bounds.center();

        // Partition points into octants
        let mut octant_indices: [Vec<usize>; 8] = Default::default();

        for &idx in indices {
            let o = Self::get_octant(center, points[idx]);
            octant_indices[o].push(idx);
        }

        // Build children recursively
        let mut children: [Option<Box<OctNode>>; 8] = Default::default();

        for (i, child_indices) in octant_indices.iter().enumerate() {
            if !child_indices.is_empty() {
                let child_bounds = bounds.octant_bounds(i, center);
                children[i] = Some(Box::new(Self::build_node(
                    points,
                    child_indices,
                    &child_bounds,
                    bucket_capacity,
                )));
            }
        }

        OctNode::Internal { center, children }
    }

    /// Determines which octant a point belongs to relative to a center.
    fn get_octant(center: Point3<i64>, point: Point3<i64>) -> usize {
        let x_bit = if point.x >= center.x { 0 } else { 1 };
        let y_bit = if point.y >= center.y { 0 } else { 2 };
        let z_bit = if point.z >= center.z { 0 } else { 4 };
        x_bit | y_bit | z_bit
    }

    /// Finds the nearest point to the query.
    ///
    /// Returns `None` if the tree is empty.
    #[must_use]
    pub fn nearest(&self, query: Point3<i64>) -> Option<NearestResult> {
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
        node: &OctNode,
        bounds: &Bounds3,
        query: Point3<i64>,
        mut best: NearestResult,
    ) -> NearestResult {
        match node {
            OctNode::Leaf { point_indices } => {
                for &idx in point_indices {
                    let dist = distance_squared_3d(self.points[idx], query);
                    if dist < best.distance_squared {
                        best = NearestResult {
                            point_idx: idx,
                            distance_squared: dist,
                        };
                    }
                }
                best
            }
            OctNode::Internal { center, children } => {
                // Order children by distance to query point
                let mut child_order: Vec<(usize, i128)> = children
                    .iter()
                    .enumerate()
                    .filter_map(|(i, child)| {
                        child.as_ref().map(|_| {
                            let child_bounds = bounds.octant_bounds(i, *center);
                            (i, child_bounds.distance_squared_to(query))
                        })
                    })
                    .collect();

                child_order.sort_by_key(|&(_, dist)| dist);

                for (i, dist_to_bounds) in child_order {
                    // Prune if this octant can't contain a closer point
                    if dist_to_bounds >= best.distance_squared {
                        break;
                    }

                    if let Some(child) = &children[i] {
                        let child_bounds = bounds.octant_bounds(i, *center);
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
    pub fn k_nearest(&self, query: Point3<i64>, k: usize) -> Vec<NearestResult> {
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
        node: &OctNode,
        bounds: &Bounds3,
        query: Point3<i64>,
        k: usize,
        heap: &mut BinaryHeap<HeapEntry>,
    ) {
        match node {
            OctNode::Leaf { point_indices } => {
                for &idx in point_indices {
                    let dist = distance_squared_3d(self.points[idx], query);

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
            OctNode::Internal { center, children } => {
                // Order children by distance
                let mut child_order: Vec<(usize, i128)> = children
                    .iter()
                    .enumerate()
                    .filter_map(|(i, child)| {
                        child.as_ref().map(|_| {
                            let child_bounds = bounds.octant_bounds(i, *center);
                            (i, child_bounds.distance_squared_to(query))
                        })
                    })
                    .collect();

                child_order.sort_by_key(|&(_, dist)| dist);

                for (i, dist_to_bounds) in child_order {
                    // Prune if heap is full and this octant can't improve
                    if heap.len() >= k {
                        if let Some(worst) = heap.peek() {
                            if dist_to_bounds >= worst.distance_squared {
                                continue;
                            }
                        }
                    }

                    if let Some(child) = &children[i] {
                        let child_bounds = bounds.octant_bounds(i, *center);
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
    pub fn range_query(&self, min: Point3<i64>, max: Point3<i64>) -> Vec<usize> {
        let mut results = Vec::new();
        let query_bounds = Bounds3::new(min, max);

        if let Some(root) = &self.root {
            self.range_recursive(root, &self.bounds, &query_bounds, &mut results);
        }

        results
    }

    fn range_recursive(
        &self,
        node: &OctNode,
        bounds: &Bounds3,
        query_bounds: &Bounds3,
        results: &mut Vec<usize>,
    ) {
        match node {
            OctNode::Leaf { point_indices } => {
                for &idx in point_indices {
                    if query_bounds.contains(self.points[idx]) {
                        results.push(idx);
                    }
                }
            }
            OctNode::Internal { center, children } => {
                for (i, child) in children.iter().enumerate() {
                    if let Some(child_node) = child {
                        let child_bounds = bounds.octant_bounds(i, *center);
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
    pub fn bounds(&self) -> Bounds3 {
        self.bounds
    }

    /// Returns a reference to the point at the given index.
    #[must_use]
    pub fn get_point(&self, idx: usize) -> Option<&Point3<i64>> {
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
    fn test_bounds3_contains() {
        let bounds = Bounds3::new(Point3::new(0, 0, 0), Point3::new(10, 10, 10));

        assert!(bounds.contains(Point3::new(5, 5, 5)));
        assert!(bounds.contains(Point3::new(0, 0, 0)));
        assert!(bounds.contains(Point3::new(10, 10, 10)));
        assert!(!bounds.contains(Point3::new(-1, 5, 5)));
        assert!(!bounds.contains(Point3::new(5, 5, 11)));
    }

    #[test]
    fn test_bounds3_intersects() {
        let b1 = Bounds3::new(Point3::new(0, 0, 0), Point3::new(10, 10, 10));
        let b2 = Bounds3::new(Point3::new(5, 5, 5), Point3::new(15, 15, 15));
        let b3 = Bounds3::new(Point3::new(20, 20, 20), Point3::new(30, 30, 30));

        assert!(b1.intersects(&b2));
        assert!(b2.intersects(&b1));
        assert!(!b1.intersects(&b3));
    }

    #[test]
    fn test_bounds3_distance_squared() {
        let bounds = Bounds3::new(Point3::new(0, 0, 0), Point3::new(10, 10, 10));

        // Inside
        assert_eq!(bounds.distance_squared_to(Point3::new(5, 5, 5)), 0);

        // On boundary
        assert_eq!(bounds.distance_squared_to(Point3::new(0, 0, 0)), 0);

        // Outside on one axis
        assert_eq!(bounds.distance_squared_to(Point3::new(-3, 5, 5)), 9); // 3^2

        // Outside on two axes
        assert_eq!(bounds.distance_squared_to(Point3::new(-3, -4, 5)), 25); // 3^2 + 4^2

        // Outside on three axes
        assert_eq!(bounds.distance_squared_to(Point3::new(-3, -4, -5)), 50); // 3^2 + 4^2 + 5^2
    }

    #[test]
    fn test_build_empty() {
        let tree = Octree::new(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.nearest(Point3::new(0, 0, 0)).is_none());
    }

    #[test]
    fn test_build_single_point() {
        let points = vec![Point3::new(5_i64, 5, 5)];
        let tree = Octree::new(&points);

        assert_eq!(tree.len(), 1);
        let result = tree.nearest(Point3::new(0, 0, 0)).unwrap();
        assert_eq!(result.point_idx, 0);
    }

    #[test]
    fn test_build_many_points() {
        let points: Vec<Point3<i64>> = (0..100)
            .map(|i| Point3::new(i % 10, (i / 10) % 10, i / 100))
            .collect();
        let tree = Octree::new(&points);

        assert_eq!(tree.len(), 100);
    }

    #[test]
    fn test_custom_capacity() {
        let points: Vec<Point3<i64>> = (0..20).map(|i| Point3::new(i, i, i)).collect();
        let tree = Octree::with_capacity(&points, 4);

        assert_eq!(tree.len(), 20);
        let result = tree.nearest(Point3::new(10, 10, 10)).unwrap();
        assert_eq!(result.point_idx, 10);
    }

    #[test]
    fn test_nearest_exact_match() {
        let points = vec![
            Point3::new(0_i64, 0, 0),
            Point3::new(10, 10, 10),
            Point3::new(5, 5, 5),
        ];
        let tree = Octree::new(&points);

        let result = tree.nearest(Point3::new(5, 5, 5)).unwrap();
        assert_eq!(result.point_idx, 2);
        assert_eq!(result.distance_squared, 0);
    }

    #[test]
    fn test_nearest_simple() {
        let points = vec![
            Point3::new(0_i64, 0, 0),
            Point3::new(10, 10, 10),
            Point3::new(5, 5, 5),
            Point3::new(3, 7, 2),
        ];
        let tree = Octree::new(&points);

        let result = tree.nearest(Point3::new(4, 4, 4)).unwrap();
        assert_eq!(result.point_idx, 2); // (5,5,5) is closest to (4,4,4)
    }

    #[test]
    fn test_nearest_different_octants() {
        let points = vec![
            Point3::new(-10_i64, -10, -10), // octant 6 (-, -, -)
            Point3::new(10, -10, -10),      // octant 7 (+, -, -)
            Point3::new(-10, 10, -10),      // octant 5 (-, +, -)
            Point3::new(10, 10, -10),       // octant 4 (+, +, -)
            Point3::new(-10, -10, 10),      // octant 2 (-, -, +)
            Point3::new(10, -10, 10),       // octant 3 (+, -, +)
            Point3::new(-10, 10, 10),       // octant 1 (-, +, +)
            Point3::new(10, 10, 10),        // octant 0 (+, +, +)
        ];
        let tree = Octree::new(&points);

        // Query in octant 0
        let result = tree.nearest(Point3::new(8, 8, 8)).unwrap();
        assert_eq!(result.point_idx, 7);

        // Query in octant 6
        let result = tree.nearest(Point3::new(-8, -8, -8)).unwrap();
        assert_eq!(result.point_idx, 0);
    }

    #[test]
    fn test_k_nearest_basic() {
        let points = vec![
            Point3::new(0_i64, 0, 0),
            Point3::new(1, 1, 1),
            Point3::new(2, 2, 2),
            Point3::new(10, 10, 10),
        ];
        let tree = Octree::new(&points);

        let results = tree.k_nearest(Point3::new(0, 0, 0), 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].point_idx, 0); // (0,0,0)
        assert_eq!(results[1].point_idx, 1); // (1,1,1)
        assert_eq!(results[2].point_idx, 2); // (2,2,2)
    }

    #[test]
    fn test_k_nearest_less_than_k() {
        let points = vec![Point3::new(0_i64, 0, 0), Point3::new(1, 1, 1)];
        let tree = Octree::new(&points);

        let results = tree.k_nearest(Point3::new(0, 0, 0), 5);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_k_nearest_zero() {
        let points = vec![Point3::new(0_i64, 0, 0)];
        let tree = Octree::new(&points);

        let results = tree.k_nearest(Point3::new(0, 0, 0), 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_range_query_all() {
        let points = vec![
            Point3::new(1_i64, 1, 1),
            Point3::new(2, 2, 2),
            Point3::new(3, 3, 3),
        ];
        let tree = Octree::new(&points);

        let results = tree.range_query(Point3::new(0, 0, 0), Point3::new(10, 10, 10));
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_range_query_none() {
        let points = vec![
            Point3::new(0_i64, 0, 0),
            Point3::new(1, 1, 1),
            Point3::new(2, 2, 2),
        ];
        let tree = Octree::new(&points);

        let results = tree.range_query(Point3::new(10, 10, 10), Point3::new(20, 20, 20));
        assert!(results.is_empty());
    }

    #[test]
    fn test_range_query_partial() {
        let points = vec![
            Point3::new(0_i64, 0, 0),
            Point3::new(5, 5, 5),
            Point3::new(10, 10, 10),
            Point3::new(15, 15, 15),
        ];
        let tree = Octree::new(&points);

        let results = tree.range_query(Point3::new(4, 4, 4), Point3::new(11, 11, 11));
        assert_eq!(results.len(), 2);
        assert!(results.contains(&1)); // (5,5,5)
        assert!(results.contains(&2)); // (10,10,10)
    }

    #[test]
    fn test_get_point() {
        let points = vec![Point3::new(5_i64, 5, 5), Point3::new(10, 10, 10)];
        let tree = Octree::new(&points);

        assert_eq!(tree.get_point(0), Some(&Point3::new(5, 5, 5)));
        assert_eq!(tree.get_point(1), Some(&Point3::new(10, 10, 10)));
        assert_eq!(tree.get_point(2), None);
    }

    #[test]
    fn test_bounds() {
        let points = vec![
            Point3::new(-5_i64, 0, 1),
            Point3::new(10, 20, 30),
            Point3::new(0, -3, -7),
        ];
        let tree = Octree::new(&points);

        let bounds = tree.bounds();
        assert_eq!(bounds.min, Point3::new(-5, -3, -7));
        assert_eq!(bounds.max, Point3::new(10, 20, 30));
    }
}
