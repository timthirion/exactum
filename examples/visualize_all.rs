//! Generate showcase SVG visualizations for exactum.
//!
//! Usage:
//!   cargo run --example visualize_all
//!
//! This generates four SVG files in the screenshots/ directory:
//!   - convex_hull.svg
//!   - quadtree.svg
//!   - boolean_ops.svg
//!   - sweep_line.svg

use std::fs;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use exactum::algo::{
    boolean::{polygon_difference, polygon_intersection, polygon_union, BooleanResult},
    graham_scan,
    sweep::{find_intersections, Segment},
};
use exactum::Point2;

const SIZE: i64 = 500;
const SEED: u64 = 42;

fn main() {
    fs::create_dir_all("screenshots").expect("Failed to create screenshots directory");

    println!("Generating convex_hull.svg...");
    let svg = generate_convex_hull();
    fs::write("screenshots/convex_hull.svg", svg).expect("Failed to write convex_hull.svg");

    println!("Generating quadtree.svg...");
    let svg = generate_quadtree();
    fs::write("screenshots/quadtree.svg", svg).expect("Failed to write quadtree.svg");

    println!("Generating boolean_ops.svg...");
    let svg = generate_boolean_ops();
    fs::write("screenshots/boolean_ops.svg", svg).expect("Failed to write boolean_ops.svg");

    println!("Generating sweep_line.svg...");
    let svg = generate_sweep_line();
    fs::write("screenshots/sweep_line.svg", svg).expect("Failed to write sweep_line.svg");

    println!("Done! Generated 4 SVG files in screenshots/");
}

fn generate_convex_hull() -> String {
    let mut rng = StdRng::seed_from_u64(SEED);
    let margin = 40;

    // Generate random points
    let points: Vec<Point2<i64>> = (0..50)
        .map(|_| {
            Point2::new(
                rng.gen_range(margin..SIZE - margin),
                rng.gen_range(margin..SIZE - margin),
            )
        })
        .collect();

    // Compute convex hull
    let hull = graham_scan(&points);

    // Find the pivot (lowest y, then leftmost x) - this is the first point in hull
    let pivot = points
        .iter()
        .min_by(|a, b| a.y.cmp(&b.y).then(a.x.cmp(&b.x)))
        .unwrap();

    let mut svg = svg_header(SIZE, SIZE);

    // Draw radial lines from pivot to all points (light gray, behind everything)
    svg.push_str("  <g stroke=\"#e0e0e0\" stroke-width=\"1\" stroke-dasharray=\"4,4\">\n");
    for p in &points {
        if p != pivot {
            svg.push_str(&format!(
                "    <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"/>\n",
                pivot.x, pivot.y, p.x, p.y
            ));
        }
    }
    svg.push_str("  </g>\n");

    // Draw hull polygon (filled with transparent color)
    svg.push_str("  <polygon points=\"");
    for p in &hull {
        svg.push_str(&format!("{},{} ", p.x, p.y));
    }
    svg.push_str("\" fill=\"rgba(76, 175, 80, 0.2)\" stroke=\"#4CAF50\" stroke-width=\"2.5\"/>\n");

    // Draw all points
    svg.push_str("  <g fill=\"#333\">\n");
    for p in &points {
        let is_hull_point = hull.contains(p);
        let (color, r) = if is_hull_point {
            ("#4CAF50", 6)
        } else {
            ("#999", 4)
        };
        svg.push_str(&format!(
            "    <circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{}\"/>\n",
            p.x, p.y, r, color
        ));
    }
    svg.push_str("  </g>\n");

    // Highlight pivot point
    svg.push_str(&format!(
        "  <circle cx=\"{}\" cy=\"{}\" r=\"8\" fill=\"#E91E63\" stroke=\"white\" stroke-width=\"2\"/>\n",
        pivot.x, pivot.y
    ));

    svg.push_str("</svg>\n");
    svg
}

fn generate_quadtree() -> String {
    let mut rng = StdRng::seed_from_u64(SEED + 1);
    let margin = 20;

    // Generate clustered points for more interesting subdivision
    let mut points: Vec<Point2<i64>> = Vec::new();

    // Cluster 1: top-left
    for _ in 0..15 {
        points.push(Point2::new(rng.gen_range(50..150), rng.gen_range(50..150)));
    }

    // Cluster 2: center
    for _ in 0..25 {
        points.push(Point2::new(
            rng.gen_range(200..300),
            rng.gen_range(200..300),
        ));
    }

    // Cluster 3: bottom-right
    for _ in 0..20 {
        points.push(Point2::new(
            rng.gen_range(350..450),
            rng.gen_range(350..450),
        ));
    }

    // Scattered points
    for _ in 0..15 {
        points.push(Point2::new(
            rng.gen_range(margin..SIZE - margin),
            rng.gen_range(margin..SIZE - margin),
        ));
    }

    let mut svg = svg_header(SIZE, SIZE);

    // Compute bounding box
    let min_x = points.iter().map(|p| p.x).min().unwrap() - 10;
    let min_y = points.iter().map(|p| p.y).min().unwrap() - 10;
    let max_x = points.iter().map(|p| p.x).max().unwrap() + 10;
    let max_y = points.iter().map(|p| p.y).max().unwrap() + 10;

    // Draw quadtree cells by computing subdivisions
    svg.push_str("  <g stroke=\"#2196F3\" stroke-width=\"1\" fill=\"none\">\n");
    draw_quadtree_cells(
        &mut svg,
        &points,
        QuadBounds {
            min_x,
            min_y,
            max_x,
            max_y,
        },
        4,
        0,
    );
    svg.push_str("  </g>\n");

    // Draw points
    svg.push_str("  <g fill=\"#E91E63\">\n");
    for p in &points {
        svg.push_str(&format!(
            "    <circle cx=\"{}\" cy=\"{}\" r=\"4\"/>\n",
            p.x, p.y
        ));
    }
    svg.push_str("  </g>\n");

    svg.push_str("</svg>\n");
    svg
}

struct QuadBounds {
    min_x: i64,
    min_y: i64,
    max_x: i64,
    max_y: i64,
}

fn draw_quadtree_cells(
    svg: &mut String,
    points: &[Point2<i64>],
    bounds: QuadBounds,
    bucket_capacity: usize,
    depth: usize,
) {
    let QuadBounds {
        min_x,
        min_y,
        max_x,
        max_y,
    } = bounds;
    // Draw this cell's bounds
    let w = max_x - min_x;
    let h = max_y - min_y;

    // Color varies by depth
    let colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"];
    let color = colors[depth % colors.len()];
    let opacity = 0.4 + (depth as f64 * 0.1).min(0.6);

    svg.push_str(&format!(
        "    <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" stroke=\"{}\" stroke-opacity=\"{:.2}\" fill=\"{}\" fill-opacity=\"0.02\"/>\n",
        min_x, min_y, w, h, color, opacity, color
    ));

    // Count points in this cell
    let cell_points: Vec<_> = points
        .iter()
        .filter(|p| p.x >= min_x && p.x <= max_x && p.y >= min_y && p.y <= max_y)
        .collect();

    // If too many points, subdivide
    if cell_points.len() > bucket_capacity && depth < 6 {
        let mid_x = (min_x + max_x) / 2;
        let mid_y = (min_y + max_y) / 2;

        // NW quadrant
        draw_quadtree_cells(
            svg,
            points,
            QuadBounds {
                min_x,
                min_y,
                max_x: mid_x,
                max_y: mid_y,
            },
            bucket_capacity,
            depth + 1,
        );
        // NE quadrant
        draw_quadtree_cells(
            svg,
            points,
            QuadBounds {
                min_x: mid_x,
                min_y,
                max_x,
                max_y: mid_y,
            },
            bucket_capacity,
            depth + 1,
        );
        // SW quadrant
        draw_quadtree_cells(
            svg,
            points,
            QuadBounds {
                min_x,
                min_y: mid_y,
                max_x: mid_x,
                max_y,
            },
            bucket_capacity,
            depth + 1,
        );
        // SE quadrant
        draw_quadtree_cells(
            svg,
            points,
            QuadBounds {
                min_x: mid_x,
                min_y: mid_y,
                max_x,
                max_y,
            },
            bucket_capacity,
            depth + 1,
        );
    }
}

fn generate_boolean_ops() -> String {
    // Create two overlapping polygons
    let poly_a = create_star(Point2::new(180, 200), 120, 60, 5);
    let poly_b = create_regular_polygon(Point2::new(280, 260), 100, 6);

    // Compute boolean operations
    let union = polygon_union(&poly_a, &poly_b);
    let intersection = polygon_intersection(&poly_a, &poly_b);
    let difference = polygon_difference(&poly_a, &poly_b);

    // Create a wide SVG with three panels
    let panel_width = 320;
    let total_width = panel_width * 3;
    let height = 400;

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
        total_width, height, total_width, height
    ));
    svg.push('\n');
    svg.push_str(&format!(
        r#"  <rect width="{}" height="{}" fill="white"/>"#,
        total_width, height
    ));
    svg.push('\n');

    // Panel dividers
    svg.push_str(&format!(
        "  <line x1=\"{}\" y1=\"0\" x2=\"{}\" y2=\"{}\" stroke=\"#ddd\" stroke-width=\"1\"/>\n",
        panel_width, panel_width, height
    ));
    svg.push_str(&format!(
        "  <line x1=\"{}\" y1=\"0\" x2=\"{}\" y2=\"{}\" stroke=\"#ddd\" stroke-width=\"1\"/>\n",
        panel_width * 2,
        panel_width * 2,
        height
    ));

    // Labels
    svg.push_str("  <text x=\"160\" y=\"30\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"16\" font-weight=\"bold\" fill=\"#333\">Union</text>\n");
    svg.push_str(&format!("  <text x=\"{}\" y=\"30\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"16\" font-weight=\"bold\" fill=\"#333\">Intersection</text>\n", panel_width + 160));
    svg.push_str(&format!("  <text x=\"{}\" y=\"30\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"16\" font-weight=\"bold\" fill=\"#333\">Difference</text>\n", panel_width * 2 + 160));

    // Offset for centering
    let y_offset = 40;

    // Panel 1: Union
    draw_boolean_panel(&mut svg, 0, y_offset, &poly_a, &poly_b, &union, "#4CAF50");

    // Panel 2: Intersection
    draw_boolean_panel(
        &mut svg,
        panel_width,
        y_offset,
        &poly_a,
        &poly_b,
        &intersection,
        "#2196F3",
    );

    // Panel 3: Difference
    draw_boolean_panel(
        &mut svg,
        panel_width * 2,
        y_offset,
        &poly_a,
        &poly_b,
        &difference,
        "#FF9800",
    );

    svg.push_str("</svg>\n");
    svg
}

fn draw_boolean_panel(
    svg: &mut String,
    x_offset: i64,
    y_offset: i64,
    poly_a: &[Point2<i64>],
    poly_b: &[Point2<i64>],
    result: &BooleanResult,
    result_color: &str,
) {
    // Draw original polygons (faded)
    svg.push_str(&format!(
        "  <polygon points=\"{}\" fill=\"rgba(233, 30, 99, 0.1)\" stroke=\"#E91E63\" stroke-width=\"1\" stroke-dasharray=\"4,4\" transform=\"translate({}, {})\"/>\n",
        points_to_svg_string(poly_a),
        x_offset,
        y_offset
    ));
    svg.push_str(&format!(
        "  <polygon points=\"{}\" fill=\"rgba(156, 39, 176, 0.1)\" stroke=\"#9C27B0\" stroke-width=\"1\" stroke-dasharray=\"4,4\" transform=\"translate({}, {})\"/>\n",
        points_to_svg_string(poly_b),
        x_offset,
        y_offset
    ));

    // Draw result polygons
    for poly in &result.polygons {
        if poly.len() >= 3 {
            let points_str: String = poly
                .iter()
                .map(|v| {
                    let (x, y) = v.to_f64();
                    format!("{:.1},{:.1}", x, y)
                })
                .collect::<Vec<_>>()
                .join(" ");
            svg.push_str(&format!(
                "  <polygon points=\"{}\" fill=\"{}\" fill-opacity=\"0.4\" stroke=\"{}\" stroke-width=\"2\" transform=\"translate({}, {})\"/>\n",
                points_str,
                result_color,
                result_color,
                x_offset,
                y_offset
            ));
        }
    }
}

fn create_star(center: Point2<i64>, outer_r: i64, inner_r: i64, points: usize) -> Vec<Point2<i64>> {
    let mut result = Vec::new();
    for i in 0..(points * 2) {
        let angle = std::f64::consts::PI * 2.0 * (i as f64) / (points * 2) as f64
            - std::f64::consts::PI / 2.0;
        let r = if i % 2 == 0 { outer_r } else { inner_r };
        let x = center.x + (angle.cos() * r as f64) as i64;
        let y = center.y + (angle.sin() * r as f64) as i64;
        result.push(Point2::new(x, y));
    }
    result
}

fn create_regular_polygon(center: Point2<i64>, radius: i64, sides: usize) -> Vec<Point2<i64>> {
    let mut result = Vec::new();
    for i in 0..sides {
        let angle =
            std::f64::consts::PI * 2.0 * (i as f64) / sides as f64 - std::f64::consts::PI / 2.0;
        let x = center.x + (angle.cos() * radius as f64) as i64;
        let y = center.y + (angle.sin() * radius as f64) as i64;
        result.push(Point2::new(x, y));
    }
    result
}

fn points_to_svg_string(points: &[Point2<i64>]) -> String {
    points
        .iter()
        .map(|p| format!("{},{}", p.x, p.y))
        .collect::<Vec<_>>()
        .join(" ")
}

fn generate_sweep_line() -> String {
    let mut rng = StdRng::seed_from_u64(SEED + 2);
    let margin = 40;

    // Generate random line segments
    let segments: Vec<Segment> = (0..20)
        .map(|_| {
            let x1 = rng.gen_range(margin..SIZE - margin);
            let y1 = rng.gen_range(margin..SIZE - margin);
            // Create segments with reasonable length
            let len = rng.gen_range(80..200);
            let angle = rng.gen_range(0.0..std::f64::consts::PI * 2.0);
            let x2 = (x1 as f64 + angle.cos() * len as f64) as i64;
            let y2 = (y1 as f64 + angle.sin() * len as f64) as i64;
            Segment::new(
                Point2::new(
                    x1.clamp(margin, SIZE - margin),
                    y1.clamp(margin, SIZE - margin),
                ),
                Point2::new(
                    x2.clamp(margin, SIZE - margin),
                    y2.clamp(margin, SIZE - margin),
                ),
            )
        })
        .collect();

    // Find intersections using Bentley-Ottmann
    let intersections = find_intersections(&segments);

    let mut svg = svg_header(SIZE, SIZE);

    // Draw segments
    svg.push_str("  <g stroke=\"#333\" stroke-width=\"2\" stroke-linecap=\"round\">\n");
    for seg in &segments {
        svg.push_str(&format!(
            "    <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"/>\n",
            seg.p1.x, seg.p1.y, seg.p2.x, seg.p2.y
        ));
    }
    svg.push_str("  </g>\n");

    // Draw segment endpoints
    svg.push_str("  <g fill=\"#666\">\n");
    for seg in &segments {
        svg.push_str(&format!(
            "    <circle cx=\"{}\" cy=\"{}\" r=\"3\"/>\n",
            seg.p1.x, seg.p1.y
        ));
        svg.push_str(&format!(
            "    <circle cx=\"{}\" cy=\"{}\" r=\"3\"/>\n",
            seg.p2.x, seg.p2.y
        ));
    }
    svg.push_str("  </g>\n");

    // Draw intersections
    svg.push_str("  <g fill=\"#E91E63\" stroke=\"white\" stroke-width=\"2\">\n");
    for intersection in &intersections {
        // Convert rational coordinates to float for display
        let x = intersection.point.x.to_f64();
        let y = intersection.point.y.to_f64();
        svg.push_str(&format!(
            "    <circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"6\"/>\n",
            x, y
        ));
    }
    svg.push_str("  </g>\n");

    // Add count label
    svg.push_str(&format!(
        "  <text x=\"{}\" y=\"30\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"14\" fill=\"#666\">{} intersections found</text>\n",
        SIZE - 10,
        intersections.len()
    ));

    svg.push_str("</svg>\n");
    svg
}

fn svg_header(width: i64, height: i64) -> String {
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">
  <rect width="{}" height="{}" fill="white"/>
"#,
        width, height, width, height, width, height
    )
}
