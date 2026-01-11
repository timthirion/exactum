//! Bentley-Ottmann sweep line algorithm for finding segment intersections.
//!
//! This module implements the classic sweep line algorithm for finding all
//! intersections among n line segments in O((n + k) log n) time, where k is
//! the number of intersections.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::ops::{segment_intersection, RationalPoint, SegmentIntersection};
use crate::rational::Rational;
use crate::Point2;

/// A line segment defined by two endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    /// First endpoint (normalized to be the left endpoint).
    pub p1: Point2<i64>,
    /// Second endpoint (normalized to be the right endpoint).
    pub p2: Point2<i64>,
}

impl Segment {
    /// Creates a new segment, normalizing so p1 is the left endpoint.
    pub fn new(a: Point2<i64>, b: Point2<i64>) -> Self {
        if a.x < b.x || (a.x == b.x && a.y <= b.y) {
            Self { p1: a, p2: b }
        } else {
            Self { p1: b, p2: a }
        }
    }

    /// Returns true if this segment is vertical (same x-coordinate).
    pub fn is_vertical(&self) -> bool {
        self.p1.x == self.p2.x
    }

    /// Computes the y-coordinate of this segment at the given x position.
    /// Returns None if x is outside the segment's x-range.
    fn y_at_x(&self, x: Rational) -> Option<Rational> {
        let x1 = Rational::from_int(self.p1.x);
        let x2 = Rational::from_int(self.p2.x);

        if x < x1 || x > x2 {
            return None;
        }

        if self.is_vertical() {
            // For vertical segments, return the lower y
            return Some(Rational::from_int(self.p1.y));
        }

        let y1 = Rational::from_int(self.p1.y);
        let y2 = Rational::from_int(self.p2.y);

        // y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        let t = (x - x1) / (x2 - x1);
        Some(y1 + t * (y2 - y1))
    }
}

/// An intersection found by the sweep line algorithm.
#[derive(Debug, Clone)]
pub struct Intersection {
    /// The exact intersection point.
    pub point: RationalPoint,
    /// Indices of the two intersecting segments.
    pub segments: (usize, usize),
}

/// Event types for the sweep line.
#[derive(Debug, Clone)]
enum Event {
    /// Left endpoint of a segment (segment starts).
    LeftEndpoint { segment_idx: usize },
    /// Right endpoint of a segment (segment ends).
    RightEndpoint { segment_idx: usize },
    /// Intersection point between two segments.
    Crossing {
        seg_a: usize,
        seg_b: usize,
        point: RationalPoint,
    },
}

/// An event with its position, used for priority queue ordering.
#[derive(Debug, Clone)]
struct QueuedEvent {
    x: Rational,
    y: Rational,
    event: Event,
}

impl PartialEq for QueuedEvent {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl Eq for QueuedEvent {}

impl PartialOrd for QueuedEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap, so reverse order for min-heap behavior
        // We want smallest x first, then smallest y
        match other.x.cmp(&self.x) {
            Ordering::Equal => other.y.cmp(&self.y),
            ord => ord,
        }
    }
}

/// Sweep line status: maintains active segments ordered by y-coordinate.
struct SweepStatus {
    /// Active segment indices, maintained in sorted order by y at current x.
    active: Vec<usize>,
    /// Current sweep line x-position.
    sweep_x: Rational,
    /// Reference to all segments.
    segments: Vec<Segment>,
}

impl SweepStatus {
    fn new(segments: Vec<Segment>) -> Self {
        Self {
            active: Vec::new(),
            sweep_x: Rational::from_int(i64::MIN),
            segments,
        }
    }

    fn set_sweep_x(&mut self, x: Rational) {
        self.sweep_x = x;
    }

    /// Compares two segments by y-coordinate at current sweep position.
    fn compare_segments(&self, a: usize, b: usize) -> Ordering {
        let seg_a = &self.segments[a];
        let seg_b = &self.segments[b];

        let y_a = seg_a.y_at_x(self.sweep_x);
        let y_b = seg_b.y_at_x(self.sweep_x);

        match (y_a, y_b) {
            (Some(ya), Some(yb)) => ya.cmp(&yb),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => a.cmp(&b), // Fallback to index comparison
        }
    }

    /// Finds the position where a segment should be inserted.
    fn find_insert_position(&self, segment_idx: usize) -> usize {
        self.active
            .binary_search_by(|&idx| self.compare_segments(idx, segment_idx))
            .unwrap_or_else(|pos| pos)
    }

    /// Inserts a segment into the active set, returns its position.
    fn insert(&mut self, segment_idx: usize) -> usize {
        let pos = self.find_insert_position(segment_idx);
        self.active.insert(pos, segment_idx);
        pos
    }

    /// Removes a segment from the active set, returns its former position.
    fn remove(&mut self, segment_idx: usize) -> Option<usize> {
        if let Some(pos) = self.active.iter().position(|&idx| idx == segment_idx) {
            self.active.remove(pos);
            Some(pos)
        } else {
            None
        }
    }

    /// Gets the neighbor above (higher y) the segment at position pos.
    fn neighbor_above(&self, pos: usize) -> Option<usize> {
        if pos + 1 < self.active.len() {
            Some(self.active[pos + 1])
        } else {
            None
        }
    }

    /// Gets the neighbor below (lower y) the segment at position pos.
    fn neighbor_below(&self, pos: usize) -> Option<usize> {
        if pos > 0 {
            Some(self.active[pos - 1])
        } else {
            None
        }
    }

    /// Swaps two adjacent segments at positions pos and pos+1.
    fn swap_adjacent(&mut self, pos: usize) {
        if pos + 1 < self.active.len() {
            self.active.swap(pos, pos + 1);
        }
    }

    /// Finds the position of a segment in the active set.
    fn find_position(&self, segment_idx: usize) -> Option<usize> {
        self.active.iter().position(|&idx| idx == segment_idx)
    }
}

/// Finds all intersections among a set of line segments using the
/// Bentley-Ottmann sweep line algorithm.
///
/// Returns a list of intersection points, each with the indices of
/// the two segments that intersect there.
///
/// # Example
///
/// ```
/// use exactum::{Point2, algo::sweep::{Segment, find_intersections}};
///
/// let segments = vec![
///     Segment::new(Point2::new(0_i64, 0), Point2::new(10, 10)),
///     Segment::new(Point2::new(0_i64, 10), Point2::new(10, 0)),
/// ];
///
/// let intersections = find_intersections(&segments);
/// assert_eq!(intersections.len(), 1);
///
/// let (x, y) = intersections[0].point.to_f64();
/// assert!((x - 5.0).abs() < 0.001);
/// assert!((y - 5.0).abs() < 0.001);
/// ```
#[must_use]
pub fn find_intersections(segments: &[Segment]) -> Vec<Intersection> {
    if segments.len() < 2 {
        return Vec::new();
    }

    let mut event_queue: BinaryHeap<QueuedEvent> = BinaryHeap::new();
    let mut status = SweepStatus::new(segments.to_vec());
    let mut result: Vec<Intersection> = Vec::new();

    // Track which intersection pairs we've already found to avoid duplicates
    let mut found_intersections: HashSet<(usize, usize)> = HashSet::new();

    // Initialize event queue with all segment endpoints
    for (idx, seg) in segments.iter().enumerate() {
        // Left endpoint
        event_queue.push(QueuedEvent {
            x: Rational::from_int(seg.p1.x),
            y: Rational::from_int(seg.p1.y),
            event: Event::LeftEndpoint { segment_idx: idx },
        });

        // Right endpoint
        event_queue.push(QueuedEvent {
            x: Rational::from_int(seg.p2.x),
            y: Rational::from_int(seg.p2.y),
            event: Event::RightEndpoint { segment_idx: idx },
        });
    }

    // Helper to check and add intersection event
    let check_intersection = |a: usize,
                              b: usize,
                              sweep_x: Rational,
                              found: &mut HashSet<(usize, usize)>,
                              queue: &mut BinaryHeap<QueuedEvent>,
                              segs: &[Segment]| {
        let key = if a < b { (a, b) } else { (b, a) };
        if found.contains(&key) {
            return;
        }

        let seg_a = segs[a];
        let seg_b = segs[b];

        if let SegmentIntersection::Point(pt) =
            segment_intersection(seg_a.p1, seg_a.p2, seg_b.p1, seg_b.p2)
        {
            // Only add if intersection is at or ahead of sweep line
            // The found_intersections set prevents duplicates
            if pt.x >= sweep_x {
                found.insert(key);
                queue.push(QueuedEvent {
                    x: pt.x,
                    y: pt.y,
                    event: Event::Crossing {
                        seg_a: a,
                        seg_b: b,
                        point: pt,
                    },
                });
            }
        }
    };

    // Process events
    while let Some(event) = event_queue.pop() {
        status.set_sweep_x(event.x);

        match event.event {
            Event::LeftEndpoint { segment_idx } => {
                let pos = status.insert(segment_idx);

                // Check for intersections with neighbors
                if let Some(above) = status.neighbor_above(pos) {
                    check_intersection(
                        segment_idx,
                        above,
                        event.x,
                        &mut found_intersections,
                        &mut event_queue,
                        segments,
                    );
                }
                if let Some(below) = status.neighbor_below(pos) {
                    check_intersection(
                        segment_idx,
                        below,
                        event.x,
                        &mut found_intersections,
                        &mut event_queue,
                        segments,
                    );
                }
            }

            Event::RightEndpoint { segment_idx } => {
                if let Some(pos) = status.find_position(segment_idx) {
                    let above = status.neighbor_above(pos);
                    let below = status.neighbor_below(pos);

                    status.remove(segment_idx);

                    // Check if former neighbors now intersect
                    if let (Some(a), Some(b)) = (above, below) {
                        check_intersection(
                            a,
                            b,
                            event.x,
                            &mut found_intersections,
                            &mut event_queue,
                            segments,
                        );
                    }
                }
            }

            Event::Crossing {
                seg_a,
                seg_b,
                point,
            } => {
                // Record the intersection
                let key = if seg_a < seg_b {
                    (seg_a, seg_b)
                } else {
                    (seg_b, seg_a)
                };
                result.push(Intersection {
                    point,
                    segments: key,
                });

                // Swap the segments in the status structure
                if let (Some(pos_a), Some(pos_b)) =
                    (status.find_position(seg_a), status.find_position(seg_b))
                {
                    let (lower_pos, upper_pos) = if pos_a < pos_b {
                        (pos_a, pos_b)
                    } else {
                        (pos_b, pos_a)
                    };

                    // Only swap if adjacent
                    if upper_pos == lower_pos + 1 {
                        status.swap_adjacent(lower_pos);

                        // After swap, check for new intersections
                        let lower_seg = status.active[lower_pos];
                        let upper_seg = status.active[upper_pos];

                        // Lower segment now might intersect with new neighbor below
                        if let Some(below) = status.neighbor_below(lower_pos) {
                            check_intersection(
                                lower_seg,
                                below,
                                event.x,
                                &mut found_intersections,
                                &mut event_queue,
                                segments,
                            );
                        }

                        // Upper segment now might intersect with new neighbor above
                        if let Some(above) = status.neighbor_above(upper_pos) {
                            check_intersection(
                                upper_seg,
                                above,
                                event.x,
                                &mut found_intersections,
                                &mut event_queue,
                                segments,
                            );
                        }
                    }
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_crossing_segments() {
        let segments = vec![
            Segment::new(Point2::new(0_i64, 0), Point2::new(10, 10)),
            Segment::new(Point2::new(0_i64, 10), Point2::new(10, 0)),
        ];

        let intersections = find_intersections(&segments);
        assert_eq!(intersections.len(), 1);

        let (x, y) = intersections[0].point.to_f64();
        assert!((x - 5.0).abs() < 0.001);
        assert!((y - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_parallel_segments() {
        let segments = vec![
            Segment::new(Point2::new(0_i64, 0), Point2::new(10, 0)),
            Segment::new(Point2::new(0_i64, 5), Point2::new(10, 5)),
        ];

        let intersections = find_intersections(&segments);
        assert!(intersections.is_empty());
    }

    #[test]
    fn test_disjoint_segments() {
        let segments = vec![
            Segment::new(Point2::new(0_i64, 0), Point2::new(5, 0)),
            Segment::new(Point2::new(10_i64, 0), Point2::new(15, 0)),
        ];

        let intersections = find_intersections(&segments);
        assert!(intersections.is_empty());
    }

    #[test]
    fn test_multiple_intersections() {
        // Three segments forming a triangle-like pattern
        let segments = vec![
            Segment::new(Point2::new(0_i64, 5), Point2::new(10, 5)),
            Segment::new(Point2::new(0_i64, 0), Point2::new(10, 10)),
            Segment::new(Point2::new(0_i64, 10), Point2::new(10, 0)),
        ];

        let intersections = find_intersections(&segments);
        // Segments 1 and 2 intersect at (5, 5)
        // Segment 0 intersects segment 1 at (5, 5)
        // Segment 0 intersects segment 2 at (5, 5)
        // All three meet at (5, 5), so we get 3 pairwise intersections
        assert_eq!(intersections.len(), 3);
    }

    #[test]
    fn test_grid_pattern() {
        // Two horizontal and two vertical segments
        let segments = vec![
            Segment::new(Point2::new(0_i64, 0), Point2::new(10, 0)),
            Segment::new(Point2::new(0_i64, 10), Point2::new(10, 10)),
            Segment::new(Point2::new(0_i64, 0), Point2::new(0, 10)),
            Segment::new(Point2::new(10_i64, 0), Point2::new(10, 10)),
        ];

        let intersections = find_intersections(&segments);
        // Corners: (0,0), (10,0), (0,10), (10,10)
        // These are at endpoints, which may or may not be counted depending on implementation
        // For segments that share endpoints, segment_intersection returns Point
        assert_eq!(intersections.len(), 4);
    }

    #[test]
    fn test_shared_endpoint() {
        // Two segments sharing an endpoint
        let segments = vec![
            Segment::new(Point2::new(0_i64, 0), Point2::new(5, 5)),
            Segment::new(Point2::new(5_i64, 5), Point2::new(10, 0)),
        ];

        let intersections = find_intersections(&segments);
        // Shared endpoint at (5, 5)
        assert_eq!(intersections.len(), 1);
    }

    #[test]
    fn test_t_junction() {
        // One segment ending on another
        let segments = vec![
            Segment::new(Point2::new(0_i64, 5), Point2::new(10, 5)),
            Segment::new(Point2::new(5_i64, 0), Point2::new(5, 5)),
        ];

        let intersections = find_intersections(&segments);
        assert_eq!(intersections.len(), 1);

        let (x, y) = intersections[0].point.to_f64();
        assert!((x - 5.0).abs() < 0.001);
        assert!((y - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_single_segment() {
        let segments = vec![Segment::new(Point2::new(0_i64, 0), Point2::new(10, 10))];

        let intersections = find_intersections(&segments);
        assert!(intersections.is_empty());
    }

    #[test]
    fn test_empty_input() {
        let segments: Vec<Segment> = vec![];
        let intersections = find_intersections(&segments);
        assert!(intersections.is_empty());
    }

    #[test]
    fn test_segment_normalization() {
        let s1 = Segment::new(Point2::new(10_i64, 10), Point2::new(0, 0));
        let s2 = Segment::new(Point2::new(0_i64, 0), Point2::new(10, 10));

        // Both should have p1 as the left endpoint
        assert_eq!(s1.p1, s2.p1);
        assert_eq!(s1.p2, s2.p2);
    }

    #[test]
    fn test_vertical_segment() {
        let segments = vec![
            Segment::new(Point2::new(5_i64, 0), Point2::new(5, 10)),
            Segment::new(Point2::new(0_i64, 5), Point2::new(10, 5)),
        ];

        let intersections = find_intersections(&segments);
        assert_eq!(intersections.len(), 1);

        let (x, y) = intersections[0].point.to_f64();
        assert!((x - 5.0).abs() < 0.001);
        assert!((y - 5.0).abs() < 0.001);
    }
}
