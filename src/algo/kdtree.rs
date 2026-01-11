//! KD-tree for efficient spatial queries on point sets.
//!
//! Provides nearest-neighbor, k-nearest, and range queries in O(log n) average time.
//! Supports both 2D and 3D point sets.

use std::collections::BinaryHeap;

use crate::ops::{distance_squared, distance_squared_3d};
use crate::{Point2, Point3};

/// Result of a nearest-neighbor query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NearestResult {
    /// Index of the point in the original point array.
    pub point_idx: usize,
    /// Squared distance to the query point.
    pub distance_squared: i128,
}

/// Internal node structure for 2D KD-tree.
#[derive(Debug, Clone, Copy)]
struct KdNode2 {
    point_idx: usize,
    left: Option<usize>,
    right: Option<usize>,
    split_dim: u8, // 0 = x, 1 = y
}

/// A 2D KD-tree for efficient spatial queries.
///
/// # Example
///
/// ```
/// use exactum::{Point2, algo::KdTree2};
///
/// let points = vec![
///     Point2::new(0_i64, 0),
///     Point2::new(10, 10),
///     Point2::new(5, 5),
///     Point2::new(3, 7),
/// ];
///
/// let tree = KdTree2::new(&points);
///
/// // Find nearest point to (4, 4)
/// let result = tree.nearest(Point2::new(4, 4)).unwrap();
/// assert_eq!(result.point_idx, 2); // Point (5, 5) is closest
/// ```
#[derive(Debug, Clone)]
pub struct KdTree2 {
    nodes: Vec<KdNode2>,
    points: Vec<Point2<i64>>,
    root: Option<usize>,
}

impl KdTree2 {
    /// Builds a KD-tree from a set of points.
    ///
    /// Construction takes O(n log n) time.
    #[must_use]
    pub fn new(points: &[Point2<i64>]) -> Self {
        if points.is_empty() {
            return Self {
                nodes: Vec::new(),
                points: Vec::new(),
                root: None,
            };
        }

        let points_vec = points.to_vec();
        let mut indices: Vec<usize> = (0..points.len()).collect();
        let mut nodes = Vec::with_capacity(points.len());

        let root = Self::build_recursive(&points_vec, &mut indices, &mut nodes, 0);

        Self {
            nodes,
            points: points_vec,
            root: Some(root),
        }
    }

    fn build_recursive(
        points: &[Point2<i64>],
        indices: &mut [usize],
        nodes: &mut Vec<KdNode2>,
        depth: usize,
    ) -> usize {
        let split_dim = (depth % 2) as u8;

        // Find median using nth_element partitioning
        let mid = indices.len() / 2;
        indices.select_nth_unstable_by(mid, |&a, &b| {
            let pa = &points[a];
            let pb = &points[b];
            if split_dim == 0 {
                pa.x.cmp(&pb.x)
            } else {
                pa.y.cmp(&pb.y)
            }
        });

        let point_idx = indices[mid];
        let node_idx = nodes.len();

        // Placeholder node
        nodes.push(KdNode2 {
            point_idx,
            left: None,
            right: None,
            split_dim,
        });

        // Build left subtree
        let left = if mid > 0 {
            Some(Self::build_recursive(
                points,
                &mut indices[..mid],
                nodes,
                depth + 1,
            ))
        } else {
            None
        };

        // Build right subtree
        let right = if mid + 1 < indices.len() {
            Some(Self::build_recursive(
                points,
                &mut indices[mid + 1..],
                nodes,
                depth + 1,
            ))
        } else {
            None
        };

        // Update node with children
        nodes[node_idx].left = left;
        nodes[node_idx].right = right;

        node_idx
    }

    /// Finds the nearest point to the query.
    ///
    /// Returns `None` if the tree is empty.
    #[must_use]
    pub fn nearest(&self, query: Point2<i64>) -> Option<NearestResult> {
        self.root.map(|root| {
            self.nearest_recursive(
                root,
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
        node_idx: usize,
        query: Point2<i64>,
        mut best: NearestResult,
    ) -> NearestResult {
        let node = &self.nodes[node_idx];
        let point = self.points[node.point_idx];

        // Check distance to this point
        let dist = distance_squared(query, point);
        if dist < best.distance_squared {
            best = NearestResult {
                point_idx: node.point_idx,
                distance_squared: dist,
            };
        }

        // Determine which side to search first
        let query_val = if node.split_dim == 0 {
            query.x
        } else {
            query.y
        };
        let point_val = if node.split_dim == 0 {
            point.x
        } else {
            point.y
        };

        let (first, second) = if query_val < point_val {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        // Search the closer side first
        if let Some(first_idx) = first {
            best = self.nearest_recursive(first_idx, query, best);
        }

        // Check if we need to search the other side
        let split_dist = (query_val as i128 - point_val as i128).pow(2);
        if split_dist < best.distance_squared {
            if let Some(second_idx) = second {
                best = self.nearest_recursive(second_idx, query, best);
            }
        }

        best
    }

    /// Finds the k nearest points to the query.
    ///
    /// Returns at most k results, sorted by distance (closest first).
    #[must_use]
    pub fn k_nearest(&self, query: Point2<i64>, k: usize) -> Vec<NearestResult> {
        if k == 0 || self.root.is_none() {
            return Vec::new();
        }

        // Use max-heap to track k nearest (we want to remove the farthest)
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

        self.k_nearest_recursive(self.root.unwrap(), query, k, &mut heap);

        // Convert to sorted vec (closest first)
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
        node_idx: usize,
        query: Point2<i64>,
        k: usize,
        heap: &mut BinaryHeap<HeapEntry>,
    ) {
        let node = &self.nodes[node_idx];
        let point = self.points[node.point_idx];

        let dist = distance_squared(query, point);

        // Add to heap if we have room or this is closer than the farthest
        if heap.len() < k {
            heap.push(HeapEntry {
                distance_squared: dist,
                point_idx: node.point_idx,
            });
        } else if let Some(farthest) = heap.peek() {
            if dist < farthest.distance_squared {
                heap.pop();
                heap.push(HeapEntry {
                    distance_squared: dist,
                    point_idx: node.point_idx,
                });
            }
        }

        let query_val = if node.split_dim == 0 {
            query.x
        } else {
            query.y
        };
        let point_val = if node.split_dim == 0 {
            point.x
        } else {
            point.y
        };

        let (first, second) = if query_val < point_val {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if let Some(first_idx) = first {
            self.k_nearest_recursive(first_idx, query, k, heap);
        }

        // Check if we need to search the other side
        let split_dist = (query_val as i128 - point_val as i128).pow(2);
        let should_search = heap.len() < k
            || heap
                .peek()
                .map_or(true, |f| split_dist < f.distance_squared);

        if should_search {
            if let Some(second_idx) = second {
                self.k_nearest_recursive(second_idx, query, k, heap);
            }
        }
    }

    /// Finds all points within the given axis-aligned bounding box.
    ///
    /// Returns indices of points where `min.x <= p.x <= max.x` and `min.y <= p.y <= max.y`.
    #[must_use]
    pub fn range_query(&self, min: Point2<i64>, max: Point2<i64>) -> Vec<usize> {
        let mut results = Vec::new();
        if let Some(root) = self.root {
            self.range_query_recursive(root, min, max, &mut results);
        }
        results
    }

    fn range_query_recursive(
        &self,
        node_idx: usize,
        min: Point2<i64>,
        max: Point2<i64>,
        results: &mut Vec<usize>,
    ) {
        let node = &self.nodes[node_idx];
        let point = self.points[node.point_idx];

        // Check if this point is in range
        if point.x >= min.x && point.x <= max.x && point.y >= min.y && point.y <= max.y {
            results.push(node.point_idx);
        }

        let point_val = if node.split_dim == 0 {
            point.x
        } else {
            point.y
        };
        let min_val = if node.split_dim == 0 { min.x } else { min.y };
        let max_val = if node.split_dim == 0 { max.x } else { max.y };

        // Search left if range extends left of split
        if min_val <= point_val {
            if let Some(left) = node.left {
                self.range_query_recursive(left, min, max, results);
            }
        }

        // Search right if range extends right of split
        if max_val >= point_val {
            if let Some(right) = node.right {
                self.range_query_recursive(right, min, max, results);
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

    /// Returns a reference to the point at the given index.
    #[must_use]
    pub fn get_point(&self, idx: usize) -> Option<&Point2<i64>> {
        self.points.get(idx)
    }
}

/// Internal node structure for 3D KD-tree.
#[derive(Debug, Clone, Copy)]
struct KdNode3 {
    point_idx: usize,
    left: Option<usize>,
    right: Option<usize>,
    split_dim: u8, // 0 = x, 1 = y, 2 = z
}

/// A 3D KD-tree for efficient spatial queries.
///
/// # Example
///
/// ```
/// use exactum::{Point3, algo::KdTree3};
///
/// let points = vec![
///     Point3::new(0_i64, 0, 0),
///     Point3::new(10, 10, 10),
///     Point3::new(5, 5, 5),
/// ];
///
/// let tree = KdTree3::new(&points);
/// let result = tree.nearest(Point3::new(4, 4, 4)).unwrap();
/// assert_eq!(result.point_idx, 2); // Point (5, 5, 5) is closest
/// ```
#[derive(Debug, Clone)]
pub struct KdTree3 {
    nodes: Vec<KdNode3>,
    points: Vec<Point3<i64>>,
    root: Option<usize>,
}

impl KdTree3 {
    /// Builds a KD-tree from a set of 3D points.
    #[must_use]
    pub fn new(points: &[Point3<i64>]) -> Self {
        if points.is_empty() {
            return Self {
                nodes: Vec::new(),
                points: Vec::new(),
                root: None,
            };
        }

        let points_vec = points.to_vec();
        let mut indices: Vec<usize> = (0..points.len()).collect();
        let mut nodes = Vec::with_capacity(points.len());

        let root = Self::build_recursive(&points_vec, &mut indices, &mut nodes, 0);

        Self {
            nodes,
            points: points_vec,
            root: Some(root),
        }
    }

    fn build_recursive(
        points: &[Point3<i64>],
        indices: &mut [usize],
        nodes: &mut Vec<KdNode3>,
        depth: usize,
    ) -> usize {
        let split_dim = (depth % 3) as u8;

        let mid = indices.len() / 2;
        indices.select_nth_unstable_by(mid, |&a, &b| {
            let pa = &points[a];
            let pb = &points[b];
            match split_dim {
                0 => pa.x.cmp(&pb.x),
                1 => pa.y.cmp(&pb.y),
                _ => pa.z.cmp(&pb.z),
            }
        });

        let point_idx = indices[mid];
        let node_idx = nodes.len();

        nodes.push(KdNode3 {
            point_idx,
            left: None,
            right: None,
            split_dim,
        });

        let left = if mid > 0 {
            Some(Self::build_recursive(
                points,
                &mut indices[..mid],
                nodes,
                depth + 1,
            ))
        } else {
            None
        };

        let right = if mid + 1 < indices.len() {
            Some(Self::build_recursive(
                points,
                &mut indices[mid + 1..],
                nodes,
                depth + 1,
            ))
        } else {
            None
        };

        nodes[node_idx].left = left;
        nodes[node_idx].right = right;

        node_idx
    }

    /// Finds the nearest point to the query.
    #[must_use]
    pub fn nearest(&self, query: Point3<i64>) -> Option<NearestResult> {
        self.root.map(|root| {
            self.nearest_recursive(
                root,
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
        node_idx: usize,
        query: Point3<i64>,
        mut best: NearestResult,
    ) -> NearestResult {
        let node = &self.nodes[node_idx];
        let point = self.points[node.point_idx];

        let dist = distance_squared_3d(query, point);
        if dist < best.distance_squared {
            best = NearestResult {
                point_idx: node.point_idx,
                distance_squared: dist,
            };
        }

        let query_val = match node.split_dim {
            0 => query.x,
            1 => query.y,
            _ => query.z,
        };
        let point_val = match node.split_dim {
            0 => point.x,
            1 => point.y,
            _ => point.z,
        };

        let (first, second) = if query_val < point_val {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if let Some(first_idx) = first {
            best = self.nearest_recursive(first_idx, query, best);
        }

        let split_dist = (query_val as i128 - point_val as i128).pow(2);
        if split_dist < best.distance_squared {
            if let Some(second_idx) = second {
                best = self.nearest_recursive(second_idx, query, best);
            }
        }

        best
    }

    /// Finds the k nearest points to the query.
    #[must_use]
    pub fn k_nearest(&self, query: Point3<i64>, k: usize) -> Vec<NearestResult> {
        if k == 0 || self.root.is_none() {
            return Vec::new();
        }

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        self.k_nearest_recursive(self.root.unwrap(), query, k, &mut heap);

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
        node_idx: usize,
        query: Point3<i64>,
        k: usize,
        heap: &mut BinaryHeap<HeapEntry>,
    ) {
        let node = &self.nodes[node_idx];
        let point = self.points[node.point_idx];

        let dist = distance_squared_3d(query, point);

        if heap.len() < k {
            heap.push(HeapEntry {
                distance_squared: dist,
                point_idx: node.point_idx,
            });
        } else if let Some(farthest) = heap.peek() {
            if dist < farthest.distance_squared {
                heap.pop();
                heap.push(HeapEntry {
                    distance_squared: dist,
                    point_idx: node.point_idx,
                });
            }
        }

        let query_val = match node.split_dim {
            0 => query.x,
            1 => query.y,
            _ => query.z,
        };
        let point_val = match node.split_dim {
            0 => point.x,
            1 => point.y,
            _ => point.z,
        };

        let (first, second) = if query_val < point_val {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if let Some(first_idx) = first {
            self.k_nearest_recursive(first_idx, query, k, heap);
        }

        let split_dist = (query_val as i128 - point_val as i128).pow(2);
        let should_search = heap.len() < k
            || heap
                .peek()
                .map_or(true, |f| split_dist < f.distance_squared);

        if should_search {
            if let Some(second_idx) = second {
                self.k_nearest_recursive(second_idx, query, k, heap);
            }
        }
    }

    /// Finds all points within the given axis-aligned bounding box.
    #[must_use]
    pub fn range_query(&self, min: Point3<i64>, max: Point3<i64>) -> Vec<usize> {
        let mut results = Vec::new();
        if let Some(root) = self.root {
            self.range_query_recursive(root, min, max, &mut results);
        }
        results
    }

    fn range_query_recursive(
        &self,
        node_idx: usize,
        min: Point3<i64>,
        max: Point3<i64>,
        results: &mut Vec<usize>,
    ) {
        let node = &self.nodes[node_idx];
        let point = self.points[node.point_idx];

        if point.x >= min.x
            && point.x <= max.x
            && point.y >= min.y
            && point.y <= max.y
            && point.z >= min.z
            && point.z <= max.z
        {
            results.push(node.point_idx);
        }

        let point_val = match node.split_dim {
            0 => point.x,
            1 => point.y,
            _ => point.z,
        };
        let min_val = match node.split_dim {
            0 => min.x,
            1 => min.y,
            _ => min.z,
        };
        let max_val = match node.split_dim {
            0 => max.x,
            1 => max.y,
            _ => max.z,
        };

        if min_val <= point_val {
            if let Some(left) = node.left {
                self.range_query_recursive(left, min, max, results);
            }
        }

        if max_val >= point_val {
            if let Some(right) = node.right {
                self.range_query_recursive(right, min, max, results);
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

    /// Returns a reference to the point at the given index.
    #[must_use]
    pub fn get_point(&self, idx: usize) -> Option<&Point3<i64>> {
        self.points.get(idx)
    }
}

/// Helper struct for k-nearest heap (max-heap by distance).
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct HeapEntry {
    distance_squared: i128,
    point_idx: usize,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: larger distances come first
        self.distance_squared.cmp(&other.distance_squared)
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
    fn test_build_empty() {
        let tree = KdTree2::new(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.nearest(Point2::new(0, 0)).is_none());
    }

    #[test]
    fn test_build_single_point() {
        let points = vec![Point2::new(5_i64, 5)];
        let tree = KdTree2::new(&points);

        assert_eq!(tree.len(), 1);
        let result = tree.nearest(Point2::new(0, 0)).unwrap();
        assert_eq!(result.point_idx, 0);
    }

    #[test]
    fn test_build_multiple_points() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 10),
            Point2::new(5, 5),
            Point2::new(3, 7),
            Point2::new(8, 2),
        ];
        let tree = KdTree2::new(&points);
        assert_eq!(tree.len(), 5);
    }

    #[test]
    fn test_nearest_exact_match() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 10),
            Point2::new(5, 5),
        ];
        let tree = KdTree2::new(&points);

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
        let tree = KdTree2::new(&points);

        // Query (4, 4) - closest to (5, 5)
        let result = tree.nearest(Point2::new(4, 4)).unwrap();
        assert_eq!(result.point_idx, 2);
        assert_eq!(result.distance_squared, 2); // (5-4)² + (5-4)² = 2
    }

    #[test]
    fn test_nearest_corner_case() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(100, 0),
            Point2::new(0, 100),
            Point2::new(100, 100),
        ];
        let tree = KdTree2::new(&points);

        // Query near each corner
        let result = tree.nearest(Point2::new(1, 1)).unwrap();
        assert_eq!(result.point_idx, 0);

        let result = tree.nearest(Point2::new(99, 1)).unwrap();
        assert_eq!(result.point_idx, 1);

        let result = tree.nearest(Point2::new(1, 99)).unwrap();
        assert_eq!(result.point_idx, 2);

        let result = tree.nearest(Point2::new(99, 99)).unwrap();
        assert_eq!(result.point_idx, 3);
    }

    #[test]
    fn test_k_nearest_basic() {
        let points = vec![
            Point2::new(0_i64, 0),
            Point2::new(10, 10),
            Point2::new(5, 5),
            Point2::new(3, 3),
            Point2::new(7, 7),
        ];
        let tree = KdTree2::new(&points);

        let results = tree.k_nearest(Point2::new(4, 4), 3);
        assert_eq!(results.len(), 3);

        // Should be sorted by distance
        assert!(results[0].distance_squared <= results[1].distance_squared);
        assert!(results[1].distance_squared <= results[2].distance_squared);
    }

    #[test]
    fn test_k_nearest_less_than_k() {
        let points = vec![Point2::new(0_i64, 0), Point2::new(10, 10)];
        let tree = KdTree2::new(&points);

        let results = tree.k_nearest(Point2::new(5, 5), 5);
        assert_eq!(results.len(), 2); // Only 2 points exist
    }

    #[test]
    fn test_k_nearest_zero() {
        let points = vec![Point2::new(0_i64, 0)];
        let tree = KdTree2::new(&points);

        let results = tree.k_nearest(Point2::new(0, 0), 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_range_query_all() {
        let points = vec![Point2::new(1_i64, 1), Point2::new(2, 2), Point2::new(3, 3)];
        let tree = KdTree2::new(&points);

        let results = tree.range_query(Point2::new(0, 0), Point2::new(10, 10));
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_range_query_none() {
        let points = vec![Point2::new(0_i64, 0), Point2::new(1, 1), Point2::new(2, 2)];
        let tree = KdTree2::new(&points);

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
        let tree = KdTree2::new(&points);

        let results = tree.range_query(Point2::new(3, 3), Point2::new(12, 12));
        assert_eq!(results.len(), 2); // (5,5) and (10,10)
    }

    #[test]
    fn test_3d_nearest() {
        let points = vec![
            Point3::new(0_i64, 0, 0),
            Point3::new(10, 10, 10),
            Point3::new(5, 5, 5),
        ];
        let tree = KdTree3::new(&points);

        let result = tree.nearest(Point3::new(4, 4, 4)).unwrap();
        assert_eq!(result.point_idx, 2);
        assert_eq!(result.distance_squared, 3); // 1² + 1² + 1² = 3
    }

    #[test]
    fn test_3d_k_nearest() {
        let points = vec![
            Point3::new(0_i64, 0, 0),
            Point3::new(10, 10, 10),
            Point3::new(5, 5, 5),
            Point3::new(3, 3, 3),
        ];
        let tree = KdTree3::new(&points);

        let results = tree.k_nearest(Point3::new(4, 4, 4), 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_3d_range() {
        let points = vec![
            Point3::new(0_i64, 0, 0),
            Point3::new(5, 5, 5),
            Point3::new(10, 10, 10),
        ];
        let tree = KdTree3::new(&points);

        let results = tree.range_query(Point3::new(1, 1, 1), Point3::new(9, 9, 9));
        assert_eq!(results.len(), 1); // Only (5,5,5)
    }

    #[test]
    fn test_get_point() {
        let points = vec![Point2::new(5_i64, 10), Point2::new(15, 20)];
        let tree = KdTree2::new(&points);

        assert_eq!(tree.get_point(0), Some(&Point2::new(5, 10)));
        assert_eq!(tree.get_point(1), Some(&Point2::new(15, 20)));
        assert_eq!(tree.get_point(2), None);
    }
}
