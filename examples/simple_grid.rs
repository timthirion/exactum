//! Generate a Voronoi diagram from a regular grid of points.
//!
//! Usage: cargo run --example simple_grid > simple.svg

use std::collections::HashSet;

use exactum::algo::{delaunay, voronoi_from_delaunay};
use exactum::Point2;

fn main() {
    // Regular 3x3 grid of points - easy to verify visually
    let points: Vec<Point2<i64>> = vec![
        // Row 1
        Point2::new(100, 100),
        Point2::new(250, 100),
        Point2::new(400, 100),
        // Row 2
        Point2::new(100, 250),
        Point2::new(250, 250),
        Point2::new(400, 250),
        // Row 3
        Point2::new(100, 400),
        Point2::new(250, 400),
        Point2::new(400, 400),
    ];

    let triangulation = delaunay(&points).expect("Failed to triangulate");
    let voronoi = voronoi_from_delaunay(&triangulation);

    // Generate SVG
    let size = 500;
    println!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {0} {0}" width="{0}" height="{0}">"##,
        size
    );
    println!(
        r##"  <rect width="{}" height="{}" fill="white"/>"##,
        size, size
    );

    // Draw circumcircles to prove circumcenters are correct (deduplicated)
    println!(r##"  <g fill="none" stroke="#ddd" stroke-width="1">"##);
    let mut seen_circles: HashSet<(i32, i32)> = HashSet::new();
    for (i, tri) in triangulation.triangles.iter().enumerate() {
        let v = &voronoi.vertices[i];
        let (cx, cy) = v.to_f64();
        let key = (cx as i32, cy as i32);
        if seen_circles.contains(&key) {
            continue;
        }
        seen_circles.insert(key);

        // Compute radius (distance to first vertex)
        let p = triangulation.points[tri.vertices[0]];
        let dx = cx - p.x as f64;
        let dy = cy - p.y as f64;
        let r = (dx * dx + dy * dy).sqrt();

        // Only draw if center is reasonably within bounds
        if cx > -200.0 && cx < 700.0 && cy > -200.0 && cy < 700.0 {
            println!(
                r##"    <circle cx="{:.1}" cy="{:.1}" r="{:.1}"/>"##,
                cx, cy, r
            );
        }
    }
    println!("  </g>");

    // Draw Voronoi edges (skip zero-length edges from shared circumcenters)
    println!(r##"  <g stroke="#2196F3" stroke-width="1.5" fill="none">"##);
    for edge in &voronoi.edges {
        if let (Some(start), Some(end)) = (edge.start, edge.end) {
            if start == end {
                continue;
            }
            let (x1, y1) = voronoi.vertices[start].to_f64();
            let (x2, y2) = voronoi.vertices[end].to_f64();
            // Skip if same position (shared circumcenter)
            if (x1 - x2).abs() < 0.1 && (y1 - y2).abs() < 0.1 {
                continue;
            }
            println!(
                r##"    <line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}"/>"##,
                x1, y1, x2, y2
            );
        }
    }
    println!("  </g>");

    // Draw Delaunay triangles
    println!(r##"  <g stroke="#333" stroke-width="1" fill="none">"##);
    for tri in &triangulation.triangles {
        let [a, b, c] = tri.vertices;
        let pa = triangulation.points[a];
        let pb = triangulation.points[b];
        let pc = triangulation.points[c];
        println!(
            r##"    <polygon points="{},{} {},{} {},{}"/>"##,
            pa.x, pa.y, pb.x, pb.y, pc.x, pc.y
        );
    }
    println!("  </g>");

    // Draw input points (sites)
    println!(r##"  <g fill="#E91E63">"##);
    for p in &points {
        println!(r##"    <circle cx="{}" cy="{}" r="6"/>"##, p.x, p.y);
    }
    println!("  </g>");

    // Draw Voronoi vertices (circumcenters) - deduplicated
    println!(r##"  <g fill="#2196F3">"##);
    let mut seen_vertices: HashSet<(i32, i32)> = HashSet::new();
    for v in &voronoi.vertices {
        let (x, y) = v.to_f64();
        let key = (x as i32, y as i32);
        if seen_vertices.contains(&key) {
            continue;
        }
        seen_vertices.insert(key);
        if x >= 0.0 && x <= size as f64 && y >= 0.0 && y <= size as f64 {
            println!(r##"    <circle cx="{:.1}" cy="{:.1}" r="4"/>"##, x, y);
        }
    }
    println!("  </g>");

    println!("</svg>");
}
