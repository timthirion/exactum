# Exactum

Integer-only computational geometry library in Rust.

## Project Goals

- **Exact arithmetic**: No floating-point. All computations use integers with overflow detection or arbitrary-precision fallback.
- **Layered architecture**: Build complex algorithms from verified primitives.
- **Generic over integer types**: Support `i32`, `i64`, `i128`, and optionally arbitrary-precision integers.
- **Minimal dependencies**: Zero deps for core functionality; optional deps for benchmarks/testing.

## Code Style

Use official Rust style conventions:
- 4-space indentation
- Run `cargo fmt` and `cargo clippy` before committing
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

## Module Structure

```
exactum/
├── Cargo.toml
└── src/
    ├── lib.rs           # Crate root, re-exports
    ├── point.rs         # Point2, Point3
    ├── vector.rs        # Vector2, Vector3
    ├── predicates.rs    # orient2d, incircle, etc.
    ├── ops.rs           # Intersections, containment, distance
    └── algo/
        ├── mod.rs
        ├── convex_hull.rs
        └── triangulation.rs
```

## Implementation Phases

### Phase 1: Core Primitives (current)

- `Point2<T>`, `Point3<T>` - 2D/3D points generic over integer types
- `Vector2<T>`, `Vector3<T>` - Displacement vectors
- Basic arithmetic operations with operator overloading

### Phase 2: Geometric Predicates

Exact geometric predicates returning `Ordering`:
- `orient2d(a, b, c)` - CCW/CW/collinear test (2x2 determinant)
- `orient3d(a, b, c, d)` - Above/below/coplanar (3x3 determinant)
- `incircle(a, b, c, d)` - Point inside circumcircle (4x4 lifted determinant)
- `insphere(a, b, c, d, e)` - Point inside circumsphere (5x5 lifted determinant)

### Phase 3: Primitive Operations

- Segment-segment, line-segment, ray-segment intersection
- Distance squared (stays integer)
- Point-in-triangle, point-in-polygon

### Phase 4: Core Algorithms

- Convex hull: Graham scan, gift wrapping, quickhull
- Triangulation: Ear clipping, monotone polygon, Delaunay
- Polygon operations: Area, centroid, convexity test

### Phase 5: Advanced Algorithms

- Voronoi diagrams (dual of Delaunay)
- Bentley-Ottmann line segment intersection
- Boolean polygon operations (union/intersection/difference)
- Spatial data structures: R-tree, quadtree, KD-tree

## Technical Considerations

1. **Overflow in determinants**: 2x2 determinant of `i64` needs `i128`. For `incircle`, may need 256-bit or lazy evaluation.
2. **Exact intersection points**: Return rational coordinates (numerator/denominator) or defer materialization.
3. **Degenerate cases**: Handle collinear points, cocircular points explicitly.
4. **Performance**: Fast path for small coordinates, slow path (bigint) for edge cases.
