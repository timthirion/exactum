# Exactum

Integer-only computational geometry library in Rust.

## Project Goals

- **Exact arithmetic**: No floating-point. All computations use integers with overflow detection.
- **Layered architecture**: Build complex algorithms from verified primitives.
- **Generic over integer types**: Support `i32`, `i64`, `i128`.
- **Minimal dependencies**: Zero deps for core functionality; optional deps for benchmarks/testing.

## Code Style

Use official Rust style conventions:
- 4-space indentation
- Run `cargo fmt` and `cargo clippy` before committing
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

### Comments

Follow standard Rust documentation conventions:
- Use `///` for doc comments on public items (functions, structs, enums, etc.)
- Use `//!` for module-level documentation at the top of files
- Use `//` for implementation notes and inline comments
- Doc comments should explain *what* something does, not *how*
- Keep comments concise; let the code speak for itself where possible

**Do not use:**
- Segmentation comments or visual dividers (e.g., `// ─────────` or `// =====`)
- Banner-style comment blocks to separate sections

## Module Structure

```
exactum/
├── Cargo.toml
├── benches/
│   └── spatial.rs         # Criterion benchmarks
└── src/
    ├── lib.rs             # Crate root, re-exports
    ├── point.rs           # Point2, Point3
    ├── vector.rs          # Vector2, Vector3
    ├── rational.rs        # Rational numbers for exact intersection
    ├── widen.rs           # Integer widening traits
    ├── predicates.rs      # orient2d, orient3d, incircle, insphere
    ├── ops.rs             # Intersections, containment, distance
    └── algo/
        ├── mod.rs         # Algorithm re-exports
        ├── convex_hull.rs # Graham scan
        ├── delaunay.rs    # Delaunay triangulation + point location
        ├── voronoi.rs     # Voronoi diagrams
        ├── boolean.rs     # Polygon union/intersection/difference
        ├── sweep.rs       # Bentley-Ottmann sweep line
        ├── kdtree.rs      # KD-tree (2D and 3D)
        ├── quadtree.rs    # Quadtree + Bounds
        ├── octree.rs      # Octree + Bounds3
        └── rtree.rs       # R-tree for bounding boxes
```

## Implemented Features

### Core Primitives
- `Point2<T>`, `Point3<T>` - 2D/3D points generic over integer types
- `Vector2<T>`, `Vector3<T>` - Displacement vectors
- `Rational` - Exact rational numbers for intersection points
- Basic arithmetic operations with operator overloading

### Geometric Predicates
Exact geometric predicates returning `Ordering`:
- `orient2d(a, b, c)` - CCW/CW/collinear test
- `orient3d(a, b, c, d)` - Above/below/coplanar test
- `incircle(a, b, c, d)` - Point inside circumcircle test
- `insphere(a, b, c, d, e)` - Point inside circumsphere test
- `collinear(a, b, c)` - Collinearity test

### Primitive Operations
- Segment-segment, line-segment, ray-segment intersection
- Distance squared (stays integer)
- Point-in-triangle, point-in-polygon, point-in-tetrahedron
- Polygon area, centroid, convexity test

### Algorithms
- Convex hull: Graham scan
- Delaunay triangulation: Bowyer-Watson algorithm
- Voronoi diagrams: Dual of Delaunay
- Boolean polygon operations: Union, intersection, difference
- Bentley-Ottmann sweep line for segment intersections

### Spatial Data Structures
| Structure | Dimension | Operations |
|-----------|-----------|------------|
| `KdTree2` | 2D | nearest, k_nearest, range_query |
| `KdTree3` | 3D | nearest, k_nearest, range_query |
| `Quadtree` | 2D | nearest, k_nearest, range_query |
| `Octree` | 3D | nearest, k_nearest, range_query |
| `RTree` | 2D | query, contains_point, nearest |

### Point Location
- `Triangulation::locate(point)` - O(log n) point location in triangulations

## Technical Notes

1. **Overflow handling**: 2x2 determinant of `i64` uses `i128`. Distance squared uses `i128`.
2. **Exact intersection points**: Returned as `Rational` coordinates.
3. **Degenerate cases**: Collinear points, cocircular points handled explicitly.
4. **Lazy indexing**: Point location index built on first query.

## Running Tests and Benchmarks

```bash
cargo test              # Run all tests
cargo clippy            # Check for lints
cargo fmt --check       # Check formatting
cargo bench             # Run benchmarks
```
