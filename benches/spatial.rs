//! Benchmarks for spatial data structures.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use exactum::algo::{
    delaunay, Bounds, KdTree2, KdTree3, Octree, Quadtree, RTree, RTreeEntry,
};
use exactum::{Point2, Point3};

const SIZES: [usize; 4] = [100, 1_000, 10_000, 100_000];
const SEED: u64 = 42;

fn generate_points_2d(n: usize, seed: u64) -> Vec<Point2<i64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| Point2::new(rng.gen_range(0..10_000), rng.gen_range(0..10_000)))
        .collect()
}

fn generate_points_3d(n: usize, seed: u64) -> Vec<Point3<i64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            Point3::new(
                rng.gen_range(0..10_000),
                rng.gen_range(0..10_000),
                rng.gen_range(0..10_000),
            )
        })
        .collect()
}

fn generate_rtree_entries(n: usize, seed: u64) -> Vec<RTreeEntry> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|i| {
            let x = rng.gen_range(0..10_000);
            let y = rng.gen_range(0..10_000);
            let w = rng.gen_range(10..100);
            let h = rng.gen_range(10..100);
            RTreeEntry::new(
                Bounds::new(Point2::new(x, y), Point2::new(x + w, y + h)),
                i,
            )
        })
        .collect()
}

fn bench_kdtree2_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("kdtree2_construction");

    for &size in &SIZES {
        let points = generate_points_2d(size, SEED);

        group.bench_with_input(BenchmarkId::from_parameter(size), &points, |b, points| {
            b.iter(|| KdTree2::new(black_box(points)))
        });
    }
    group.finish();
}

fn bench_kdtree2_nearest(c: &mut Criterion) {
    let mut group = c.benchmark_group("kdtree2_nearest");

    for &size in &SIZES {
        let points = generate_points_2d(size, SEED);
        let tree = KdTree2::new(&points);
        let query = Point2::new(5000, 5000);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| tree.nearest(black_box(query)))
        });
    }
    group.finish();
}

fn bench_kdtree2_k_nearest(c: &mut Criterion) {
    let mut group = c.benchmark_group("kdtree2_k_nearest");

    for &size in &SIZES {
        let points = generate_points_2d(size, SEED);
        let tree = KdTree2::new(&points);
        let query = Point2::new(5000, 5000);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| tree.k_nearest(black_box(query), 10))
        });
    }
    group.finish();
}

fn bench_kdtree2_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("kdtree2_range");

    for &size in &SIZES {
        let points = generate_points_2d(size, SEED);
        let tree = KdTree2::new(&points);
        let min = Point2::new(4500, 4500);
        let max = Point2::new(5500, 5500);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| tree.range_query(black_box(min), black_box(max)))
        });
    }
    group.finish();
}

fn bench_kdtree3_nearest(c: &mut Criterion) {
    let mut group = c.benchmark_group("kdtree3_nearest");

    for &size in &SIZES {
        let points = generate_points_3d(size, SEED);
        let tree = KdTree3::new(&points);
        let query = Point3::new(5000, 5000, 5000);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| tree.nearest(black_box(query)))
        });
    }
    group.finish();
}

fn bench_quadtree_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("quadtree_construction");

    for &size in &SIZES {
        let points = generate_points_2d(size, SEED);

        group.bench_with_input(BenchmarkId::from_parameter(size), &points, |b, points| {
            b.iter(|| Quadtree::new(black_box(points)))
        });
    }
    group.finish();
}

fn bench_quadtree_nearest(c: &mut Criterion) {
    let mut group = c.benchmark_group("quadtree_nearest");

    for &size in &SIZES {
        let points = generate_points_2d(size, SEED);
        let tree = Quadtree::new(&points);
        let query = Point2::new(5000, 5000);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| tree.nearest(black_box(query)))
        });
    }
    group.finish();
}

fn bench_quadtree_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("quadtree_range");

    for &size in &SIZES {
        let points = generate_points_2d(size, SEED);
        let tree = Quadtree::new(&points);
        let min = Point2::new(4500, 4500);
        let max = Point2::new(5500, 5500);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| tree.range_query(black_box(min), black_box(max)))
        });
    }
    group.finish();
}

fn bench_octree_nearest(c: &mut Criterion) {
    let mut group = c.benchmark_group("octree_nearest");

    for &size in &SIZES {
        let points = generate_points_3d(size, SEED);
        let tree = Octree::new(&points);
        let query = Point3::new(5000, 5000, 5000);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| tree.nearest(black_box(query)))
        });
    }
    group.finish();
}

fn bench_octree_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("octree_range");

    for &size in &SIZES {
        let points = generate_points_3d(size, SEED);
        let tree = Octree::new(&points);
        let min = Point3::new(4500, 4500, 4500);
        let max = Point3::new(5500, 5500, 5500);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| tree.range_query(black_box(min), black_box(max)))
        });
    }
    group.finish();
}

fn bench_rtree_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("rtree_construction");

    for &size in &SIZES {
        let entries = generate_rtree_entries(size, SEED);

        group.bench_with_input(BenchmarkId::from_parameter(size), &entries, |b, entries| {
            b.iter(|| RTree::new(black_box(entries)))
        });
    }
    group.finish();
}

fn bench_rtree_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("rtree_query");

    for &size in &SIZES {
        let entries = generate_rtree_entries(size, SEED);
        let tree = RTree::new(&entries);
        let query = Bounds::new(Point2::new(4500, 4500), Point2::new(5500, 5500));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| tree.query(black_box(&query)))
        });
    }
    group.finish();
}

fn bench_rtree_nearest(c: &mut Criterion) {
    let mut group = c.benchmark_group("rtree_nearest");

    for &size in &SIZES {
        let entries = generate_rtree_entries(size, SEED);
        let tree = RTree::new(&entries);
        let query = Point2::new(5000, 5000);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| tree.nearest(black_box(query)))
        });
    }
    group.finish();
}

fn bench_delaunay_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("delaunay_construction");

    // Smaller sizes for Delaunay as it's O(n²) worst case
    for &size in &[100, 500, 1_000, 5_000] {
        let points = generate_points_2d(size, SEED);

        group.bench_with_input(BenchmarkId::from_parameter(size), &points, |b, points| {
            b.iter(|| delaunay(black_box(points)))
        });
    }
    group.finish();
}

fn bench_triangulation_locate(c: &mut Criterion) {
    let mut group = c.benchmark_group("triangulation_locate");

    for &size in &[100, 500, 1_000, 5_000] {
        let points = generate_points_2d(size, SEED);
        let tri = delaunay(&points).unwrap();
        let query = Point2::new(5000, 5000);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| tri.locate(black_box(query)))
        });
    }
    group.finish();
}

criterion_group!(
    construction,
    bench_kdtree2_construction,
    bench_quadtree_construction,
    bench_rtree_construction,
    bench_delaunay_construction,
);

criterion_group!(
    kdtree_queries,
    bench_kdtree2_nearest,
    bench_kdtree2_k_nearest,
    bench_kdtree2_range,
    bench_kdtree3_nearest,
);

criterion_group!(
    quadtree_queries,
    bench_quadtree_nearest,
    bench_quadtree_range,
);

criterion_group!(octree_queries, bench_octree_nearest, bench_octree_range,);

criterion_group!(
    rtree_queries,
    bench_rtree_query,
    bench_rtree_nearest,
);

criterion_group!(triangulation_queries, bench_triangulation_locate,);

criterion_main!(
    construction,
    kdtree_queries,
    quadtree_queries,
    octree_queries,
    rtree_queries,
    triangulation_queries
);
