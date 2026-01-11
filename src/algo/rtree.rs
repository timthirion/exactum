//! R-tree for spatial indexing of axis-aligned bounding boxes.
//!
//! R-trees efficiently index rectangles (bounding boxes), supporting range queries
//! and nearest-neighbor searches. Uses STR (Sort-Tile-Recursive) bulk loading for
//! optimal tree quality.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::Point2;

use super::quadtree::Bounds;

/// Default maximum entries per node.
const DEFAULT_NODE_CAPACITY: usize = 16;

/// An entry in the R-tree (bounding box + data index).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RTreeEntry {
    /// The bounding box of this entry.
    pub bounds: Bounds,
    /// Index into the original data array.
    pub data_idx: usize,
}

impl RTreeEntry {
    /// Creates a new R-tree entry.
    #[must_use]
    pub fn new(bounds: Bounds, data_idx: usize) -> Self {
        Self { bounds, data_idx }
    }
}

/// Result of a nearest-neighbor query on the R-tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RTreeNearestResult {
    /// Index of the entry in the original data array.
    pub data_idx: usize,
    /// Bounding box of the nearest entry.
    pub bounds: Bounds,
    /// Squared distance from the query point to the nearest point on the bounds.
    pub distance_squared: i128,
}

/// A child reference in an internal node.
#[derive(Debug, Clone)]
struct RTreeChild {
    /// Minimum bounding rectangle enclosing all entries in this subtree.
    bounds: Bounds,
    /// The child node.
    node: Box<RTreeNode>,
}

/// An R-tree node.
#[derive(Debug, Clone)]
enum RTreeNode {
    /// Leaf node containing entries.
    Leaf { entries: Vec<RTreeEntry> },
    /// Internal node containing children.
    Internal { children: Vec<RTreeChild> },
}

impl RTreeNode {
    /// Returns the MBR of this node.
    fn bounds(&self) -> Bounds {
        match self {
            RTreeNode::Leaf { entries } => {
                let mut bounds = entries[0].bounds;
                for entry in entries.iter().skip(1) {
                    bounds = mbr_union(&bounds, &entry.bounds);
                }
                bounds
            }
            RTreeNode::Internal { children } => {
                let mut bounds = children[0].bounds;
                for child in children.iter().skip(1) {
                    bounds = mbr_union(&bounds, &child.bounds);
                }
                bounds
            }
        }
    }
}

/// Computes the minimum bounding rectangle containing both inputs.
fn mbr_union(a: &Bounds, b: &Bounds) -> Bounds {
    Bounds::new(
        Point2::new(a.min.x.min(b.min.x), a.min.y.min(b.min.y)),
        Point2::new(a.max.x.max(b.max.x), a.max.y.max(b.max.y)),
    )
}

/// An R-tree for spatial indexing of bounding boxes.
///
/// # Example
///
/// ```
/// use exactum::{Point2, algo::{Bounds, RTree, RTreeEntry}};
///
/// // Create entries (bounding boxes with data indices)
/// let entries = vec![
///     RTreeEntry::new(Bounds::new(Point2::new(0_i64, 0), Point2::new(10, 10)), 0),
///     RTreeEntry::new(Bounds::new(Point2::new(20, 20), Point2::new(30, 30)), 1),
///     RTreeEntry::new(Bounds::new(Point2::new(5, 5), Point2::new(25, 25)), 2),
/// ];
///
/// let tree = RTree::new(&entries);
///
/// // Find entries intersecting a query box
/// let query = Bounds::new(Point2::new(8, 8), Point2::new(12, 12));
/// let results = tree.query(&query);
/// assert!(results.contains(&0)); // Box (0,0)-(10,10) intersects
/// assert!(results.contains(&2)); // Box (5,5)-(25,25) intersects
/// ```
#[derive(Debug, Clone)]
pub struct RTree {
    root: Option<RTreeNode>,
    size: usize,
}

impl RTree {
    /// Builds an R-tree from entries using STR bulk loading.
    ///
    /// Construction takes O(n log n) time.
    #[must_use]
    pub fn new(entries: &[RTreeEntry]) -> Self {
        Self::with_capacity(entries, DEFAULT_NODE_CAPACITY)
    }

    /// Builds an R-tree with a custom node capacity.
    #[must_use]
    pub fn with_capacity(entries: &[RTreeEntry], capacity: usize) -> Self {
        if entries.is_empty() {
            return Self {
                root: None,
                size: 0,
            };
        }

        let capacity = capacity.max(2); // Minimum capacity of 2
        let mut sorted_entries: Vec<RTreeEntry> = entries.to_vec();

        // STR: Sort by x-center, then build leaves
        sorted_entries.sort_by_key(|e| {
            let center = e.bounds.center();
            (center.x, center.y)
        });

        let root = Self::str_build_level(&sorted_entries, capacity, true);

        Self {
            root: Some(root),
            size: entries.len(),
        }
    }

    /// STR bulk loading: builds one level of the tree.
    fn str_build_level(entries: &[RTreeEntry], capacity: usize, _is_leaf: bool) -> RTreeNode {
        if entries.len() <= capacity {
            return RTreeNode::Leaf {
                entries: entries.to_vec(),
            };
        }

        // Calculate number of slices (vertical strips)
        let n = entries.len();
        let leaves_needed = (n + capacity - 1) / capacity;
        let slices = (leaves_needed as f64).sqrt().ceil() as usize;
        let entries_per_slice = (n + slices - 1) / slices;

        let mut leaves: Vec<RTreeNode> = Vec::new();

        // Process each vertical slice
        for slice_start in (0..n).step_by(entries_per_slice) {
            let slice_end = (slice_start + entries_per_slice).min(n);
            let mut slice: Vec<RTreeEntry> = entries[slice_start..slice_end].to_vec();

            // Sort slice by y-center
            slice.sort_by_key(|e| e.bounds.center().y);

            // Create leaf nodes from this slice
            for chunk in slice.chunks(capacity) {
                leaves.push(RTreeNode::Leaf {
                    entries: chunk.to_vec(),
                });
            }
        }

        // If we only have one leaf, return it
        if leaves.len() == 1 {
            return leaves.remove(0);
        }

        // Build internal nodes recursively
        Self::build_internal_levels(leaves, capacity)
    }

    /// Recursively builds internal levels from a set of child nodes.
    fn build_internal_levels(nodes: Vec<RTreeNode>, capacity: usize) -> RTreeNode {
        if nodes.len() <= capacity {
            let children: Vec<RTreeChild> = nodes
                .into_iter()
                .map(|node| {
                    let bounds = node.bounds();
                    RTreeChild {
                        bounds,
                        node: Box::new(node),
                    }
                })
                .collect();

            return RTreeNode::Internal { children };
        }

        // Sort nodes by x-center of their MBR
        let mut nodes_with_bounds: Vec<(RTreeNode, Bounds)> = nodes
            .into_iter()
            .map(|n| {
                let b = n.bounds();
                (n, b)
            })
            .collect();

        nodes_with_bounds.sort_by_key(|(_, b)| b.center().x);

        // Calculate slices
        let n = nodes_with_bounds.len();
        let groups_needed = (n + capacity - 1) / capacity;
        let slices = (groups_needed as f64).sqrt().ceil() as usize;
        let nodes_per_slice = (n + slices - 1) / slices;

        let mut internal_nodes: Vec<RTreeNode> = Vec::new();

        for slice_start in (0..n).step_by(nodes_per_slice) {
            let slice_end = (slice_start + nodes_per_slice).min(n);
            let mut slice: Vec<(RTreeNode, Bounds)> =
                nodes_with_bounds[slice_start..slice_end].to_vec();

            // Sort by y-center
            slice.sort_by_key(|(_, b)| b.center().y);

            // Create internal nodes from this slice
            for chunk in slice.chunks(capacity) {
                let children: Vec<RTreeChild> = chunk
                    .iter()
                    .cloned()
                    .map(|(node, bounds)| RTreeChild {
                        bounds,
                        node: Box::new(node),
                    })
                    .collect();

                internal_nodes.push(RTreeNode::Internal { children });
            }
        }

        if internal_nodes.len() == 1 {
            return internal_nodes.remove(0);
        }

        Self::build_internal_levels(internal_nodes, capacity)
    }

    /// Finds all entries whose bounds intersect the query box.
    #[must_use]
    pub fn query(&self, query: &Bounds) -> Vec<usize> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            self.query_recursive(root, query, &mut results);
        }
        results
    }

    fn query_recursive(&self, node: &RTreeNode, query: &Bounds, results: &mut Vec<usize>) {
        match node {
            RTreeNode::Leaf { entries } => {
                for entry in entries {
                    if entry.bounds.intersects(query) {
                        results.push(entry.data_idx);
                    }
                }
            }
            RTreeNode::Internal { children } => {
                for child in children {
                    if child.bounds.intersects(query) {
                        self.query_recursive(&child.node, query, results);
                    }
                }
            }
        }
    }

    /// Finds all entries whose bounds contain the query point.
    #[must_use]
    pub fn contains_point(&self, point: Point2<i64>) -> Vec<usize> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            self.contains_point_recursive(root, point, &mut results);
        }
        results
    }

    fn contains_point_recursive(
        &self,
        node: &RTreeNode,
        point: Point2<i64>,
        results: &mut Vec<usize>,
    ) {
        match node {
            RTreeNode::Leaf { entries } => {
                for entry in entries {
                    if entry.bounds.contains(point) {
                        results.push(entry.data_idx);
                    }
                }
            }
            RTreeNode::Internal { children } => {
                for child in children {
                    if child.bounds.contains(point) {
                        self.contains_point_recursive(&child.node, point, results);
                    }
                }
            }
        }
    }

    /// Finds the entry whose bounds are closest to the query point.
    ///
    /// Distance is measured as the minimum distance from the query point to any
    /// point on the entry's bounding box.
    #[must_use]
    pub fn nearest(&self, query: Point2<i64>) -> Option<RTreeNearestResult> {
        let root = self.root.as_ref()?;

        // Priority queue: (distance, node_ref)
        // Use Reverse for min-heap behavior
        let mut queue: BinaryHeap<Reverse<(i128, QueueItem)>> = BinaryHeap::new();

        let root_bounds = root.bounds();
        let root_dist = root_bounds.distance_squared_to(query);
        queue.push(Reverse((root_dist, QueueItem::Node(root))));

        let mut best: Option<RTreeNearestResult> = None;

        while let Some(Reverse((dist, item))) = queue.pop() {
            // Prune if this item can't improve the best
            if let Some(ref b) = best {
                if dist >= b.distance_squared {
                    break;
                }
            }

            match item {
                QueueItem::Node(node) => match node {
                    RTreeNode::Leaf { entries } => {
                        for entry in entries {
                            let entry_dist = entry.bounds.distance_squared_to(query);
                            let dominated = best
                                .as_ref()
                                .is_some_and(|b| entry_dist >= b.distance_squared);
                            if !dominated {
                                best = Some(RTreeNearestResult {
                                    data_idx: entry.data_idx,
                                    bounds: entry.bounds,
                                    distance_squared: entry_dist,
                                });
                            }
                        }
                    }
                    RTreeNode::Internal { children } => {
                        for child in children {
                            let child_dist = child.bounds.distance_squared_to(query);
                            let dominated = best
                                .as_ref()
                                .is_some_and(|b| child_dist >= b.distance_squared);
                            if !dominated {
                                queue.push(Reverse((child_dist, QueueItem::Node(&child.node))));
                            }
                        }
                    }
                },
            }
        }

        best
    }

    /// Returns the number of entries in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns true if the tree is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns the bounding box containing all entries, or None if empty.
    #[must_use]
    pub fn bounds(&self) -> Option<Bounds> {
        self.root.as_ref().map(|r| r.bounds())
    }
}

/// Queue item for nearest-neighbor search.
enum QueueItem<'a> {
    Node(&'a RTreeNode),
}

impl PartialEq for QueueItem<'_> {
    fn eq(&self, _other: &Self) -> bool {
        false // Nodes are never equal for queue purposes
    }
}

impl Eq for QueueItem<'_> {}

impl PartialOrd for QueueItem<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueItem<'_> {
    fn cmp(&self, _other: &Self) -> std::cmp::Ordering {
        std::cmp::Ordering::Equal // Distance determines order, not the item itself
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(x1: i64, y1: i64, x2: i64, y2: i64, idx: usize) -> RTreeEntry {
        RTreeEntry::new(Bounds::new(Point2::new(x1, y1), Point2::new(x2, y2)), idx)
    }

    #[test]
    fn test_build_empty() {
        let tree = RTree::new(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.bounds().is_none());
    }

    #[test]
    fn test_build_single() {
        let entries = vec![make_entry(0, 0, 10, 10, 0)];
        let tree = RTree::new(&entries);

        assert_eq!(tree.len(), 1);
        assert!(!tree.is_empty());
        assert_eq!(
            tree.bounds(),
            Some(Bounds::new(Point2::new(0, 0), Point2::new(10, 10)))
        );
    }

    #[test]
    fn test_build_many() {
        let entries: Vec<RTreeEntry> = (0..100)
            .map(|i| {
                let x = (i % 10) * 10;
                let y = (i / 10) * 10;
                make_entry(x, y, x + 5, y + 5, i as usize)
            })
            .collect();

        let tree = RTree::new(&entries);
        assert_eq!(tree.len(), 100);
    }

    #[test]
    fn test_query_all_match() {
        let entries = vec![
            make_entry(0, 0, 10, 10, 0),
            make_entry(5, 5, 15, 15, 1),
            make_entry(10, 10, 20, 20, 2),
        ];
        let tree = RTree::new(&entries);

        let query = Bounds::new(Point2::new(0, 0), Point2::new(100, 100));
        let results = tree.query(&query);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_query_none_match() {
        let entries = vec![make_entry(0, 0, 10, 10, 0), make_entry(5, 5, 15, 15, 1)];
        let tree = RTree::new(&entries);

        let query = Bounds::new(Point2::new(100, 100), Point2::new(200, 200));
        let results = tree.query(&query);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_partial_match() {
        let entries = vec![
            make_entry(0, 0, 10, 10, 0),
            make_entry(20, 20, 30, 30, 1),
            make_entry(5, 5, 25, 25, 2),
        ];
        let tree = RTree::new(&entries);

        // Query box that hits entries 0 and 2
        let query = Bounds::new(Point2::new(8, 8), Point2::new(12, 12));
        let results = tree.query(&query);
        assert_eq!(results.len(), 2);
        assert!(results.contains(&0));
        assert!(results.contains(&2));
    }

    #[test]
    fn test_contains_point_inside() {
        let entries = vec![
            make_entry(0, 0, 10, 10, 0),
            make_entry(5, 5, 15, 15, 1),
            make_entry(20, 20, 30, 30, 2),
        ];
        let tree = RTree::new(&entries);

        let results = tree.contains_point(Point2::new(7, 7));
        assert_eq!(results.len(), 2);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
    }

    #[test]
    fn test_contains_point_outside() {
        let entries = vec![make_entry(0, 0, 10, 10, 0), make_entry(20, 20, 30, 30, 1)];
        let tree = RTree::new(&entries);

        let results = tree.contains_point(Point2::new(15, 15));
        assert!(results.is_empty());
    }

    #[test]
    fn test_contains_point_boundary() {
        let entries = vec![make_entry(0, 0, 10, 10, 0)];
        let tree = RTree::new(&entries);

        // On boundary should be contained
        assert_eq!(tree.contains_point(Point2::new(0, 0)).len(), 1);
        assert_eq!(tree.contains_point(Point2::new(10, 10)).len(), 1);
        assert_eq!(tree.contains_point(Point2::new(5, 0)).len(), 1);
    }

    #[test]
    fn test_nearest_exact() {
        let entries = vec![make_entry(0, 0, 10, 10, 0), make_entry(20, 20, 30, 30, 1)];
        let tree = RTree::new(&entries);

        let result = tree.nearest(Point2::new(5, 5)).unwrap();
        assert_eq!(result.data_idx, 0);
        assert_eq!(result.distance_squared, 0); // Point inside box
    }

    #[test]
    fn test_nearest_closest() {
        let entries = vec![
            make_entry(0, 0, 10, 10, 0),
            make_entry(100, 100, 110, 110, 1),
        ];
        let tree = RTree::new(&entries);

        // Point closer to first box
        let result = tree.nearest(Point2::new(15, 15)).unwrap();
        assert_eq!(result.data_idx, 0);
        // Distance from (15,15) to (10,10) corner: sqrt(50) -> 50 squared
        assert_eq!(result.distance_squared, 50);
    }

    #[test]
    fn test_nearest_empty() {
        let tree = RTree::new(&[]);
        assert!(tree.nearest(Point2::new(0, 0)).is_none());
    }

    #[test]
    fn test_overlapping_boxes() {
        // Multiple overlapping boxes
        let entries = vec![
            make_entry(0, 0, 20, 20, 0),
            make_entry(5, 5, 25, 25, 1),
            make_entry(10, 10, 30, 30, 2),
        ];
        let tree = RTree::new(&entries);

        // Point in all three
        let results = tree.contains_point(Point2::new(15, 15));
        assert_eq!(results.len(), 3);

        // Point only in first two
        let results = tree.contains_point(Point2::new(8, 8));
        assert_eq!(results.len(), 2);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
    }

    #[test]
    fn test_custom_capacity() {
        let entries: Vec<RTreeEntry> = (0..50)
            .map(|i| make_entry(i * 2, i * 2, i * 2 + 1, i * 2 + 1, i as usize))
            .collect();

        let tree = RTree::with_capacity(&entries, 4);
        assert_eq!(tree.len(), 50);

        // Should still query correctly
        let results = tree.query(&Bounds::new(Point2::new(0, 0), Point2::new(10, 10)));
        assert!(!results.is_empty());
    }

    #[test]
    fn test_large_dataset() {
        // Build tree with many entries
        let entries: Vec<RTreeEntry> = (0..1000)
            .map(|i| {
                let x = (i % 100) * 10;
                let y = (i / 100) * 10;
                make_entry(x, y, x + 5, y + 5, i as usize)
            })
            .collect();

        let tree = RTree::new(&entries);
        assert_eq!(tree.len(), 1000);

        // Query should work correctly
        let results = tree.query(&Bounds::new(Point2::new(50, 50), Point2::new(60, 60)));
        assert!(!results.is_empty());

        // Nearest should work
        let result = tree.nearest(Point2::new(55, 55)).unwrap();
        assert_eq!(result.distance_squared, 0); // Should find a box containing the point
    }
}
