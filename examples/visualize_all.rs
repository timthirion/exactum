//! Generate showcase SVG visualizations for exactum.
//!
//! Usage:
//!   cargo run --example visualize_all
//!
//! This generates SVG files in the screenshots/ directory (all 800px wide):
//!   - logo.svg
//!   - voronoi_delaunay.svg
//!   - convex_hull.svg
//!   - quadtree.svg
//!   - boolean_ops.svg
//!   - sweep_line.svg

use std::fs;

use fast_poisson::Poisson2D;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use exactum::algo::{
    boolean::{polygon_difference, polygon_intersection, polygon_union, BooleanResult},
    delaunay, graham_scan,
    sweep::{find_intersections, Segment},
    voronoi_from_delaunay,
};
use exactum::Point2;

const WIDTH: i64 = 800;
const SEED: u64 = 42;

fn main() {
    fs::create_dir_all("screenshots").expect("Failed to create screenshots directory");

    println!("Generating logo.svg...");
    let svg = generate_logo();
    fs::write("screenshots/logo.svg", svg).expect("Failed to write logo.svg");

    println!("Generating voronoi_delaunay.svg...");
    let svg = generate_voronoi_delaunay();
    fs::write("screenshots/voronoi_delaunay.svg", svg)
        .expect("Failed to write voronoi_delaunay.svg");

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

    println!("Done! Generated 6 SVG files in screenshots/");
}

/// Generate the "exactum" logo using Delaunay triangulation with blue noise
fn generate_logo() -> String {
    // Logo dimensions - 800px wide to match other screenshots
    let width: i64 = 800;
    let height: i64 = 140;
    let scale: i64 = 14; // Grid scale for letters

    // Define lowercase letters as grid-based block polygons
    // Each letter uses a 4-wide x 5-tall grid for lowercase x-height

    struct Letter {
        name: char,
        rects: Vec<(i64, i64, i64, i64)>, // (x, y, w, h) in grid units
        width: i64,                       // letter width in grid units
    }

    let letters = vec![
        Letter {
            name: 'e',
            rects: vec![
                (0, 1, 1, 3), // left stem
                (1, 0, 2, 1), // top bar
                (1, 2, 2, 1), // middle bar
                (1, 4, 2, 1), // bottom bar
                (3, 1, 1, 1), // top right corner
                (3, 3, 1, 1), // bottom right corner
            ],
            width: 4,
        },
        Letter {
            name: 'x',
            rects: vec![], // Diagonal handled specially
            width: 4,
        },
        Letter {
            name: 'a',
            rects: vec![
                (0, 2, 1, 3), // left stem (shorter, starts at middle)
                (3, 1, 1, 4), // right stem
                (1, 1, 2, 1), // top bar
                (1, 4, 2, 1), // bottom bar
                (1, 2, 2, 1), // middle bar
            ],
            width: 4,
        },
        Letter {
            name: 'c',
            rects: vec![
                (0, 1, 1, 3), // left stem
                (1, 0, 3, 1), // top bar
                (1, 4, 3, 1), // bottom bar
            ],
            width: 4,
        },
        Letter {
            name: 't',
            rects: vec![
                (1, 0, 1, 5), // center stem (full height for t)
                (0, 1, 3, 1), // crossbar
            ],
            width: 3,
        },
        Letter {
            name: 'u',
            rects: vec![
                (0, 0, 1, 4), // left stem
                (3, 0, 1, 4), // right stem
                (1, 4, 2, 1), // bottom bar
            ],
            width: 4,
        },
        Letter {
            name: 'm',
            rects: vec![
                (0, 0, 1, 5), // left stem
                (2, 1, 1, 4), // middle stem
                (4, 0, 1, 5), // right stem
                (1, 0, 1, 1), // top left hump
                (3, 0, 1, 1), // top right hump
            ],
            width: 5,
        },
    ];

    // Convert letter rectangles to polygons and collect points for triangulation
    let mut all_letter_polys: Vec<Vec<Point2<i64>>> = Vec::new();
    let mut tri_points: Vec<Point2<i64>> = Vec::new();
    let letter_gap: i64 = 8;
    let mut cursor_x: i64 = 150; // Center the text in 800px width
    let baseline: i64 = 45;

    for letter in &letters {
        if letter.name == 'x' {
            // Special handling for x - draw as two diagonal bars
            let x_left = cursor_x;
            let x_width = letter.width * scale;
            let x_height = 5 * scale;

            // Forward diagonal: top-left to bottom-right
            let diag1 = vec![
                Point2::new(x_left, baseline),
                Point2::new(x_left + scale, baseline),
                Point2::new(x_left + x_width, baseline + x_height),
                Point2::new(x_left + x_width - scale, baseline + x_height),
            ];
            for p in &diag1 {
                tri_points.push(*p);
            }
            all_letter_polys.push(diag1);

            // Backward diagonal: top-right to bottom-left
            let diag2 = vec![
                Point2::new(x_left + x_width - scale, baseline),
                Point2::new(x_left + x_width, baseline),
                Point2::new(x_left + scale, baseline + x_height),
                Point2::new(x_left, baseline + x_height),
            ];
            for p in &diag2 {
                tri_points.push(*p);
            }
            all_letter_polys.push(diag2);

            cursor_x += x_width + letter_gap;
        } else {
            for &(rx, ry, rw, rh) in &letter.rects {
                let x1 = cursor_x + rx * scale;
                let y1 = baseline + ry * scale;
                let x2 = x1 + rw * scale;
                let y2 = y1 + rh * scale;

                let poly = vec![
                    Point2::new(x1, y1),
                    Point2::new(x2, y1),
                    Point2::new(x2, y2),
                    Point2::new(x1, y2),
                ];

                for p in &poly {
                    tri_points.push(*p);
                }
                all_letter_polys.push(poly);
            }

            cursor_x += letter.width * scale + letter_gap;
        }
    }

    // Use blue noise (Poisson disk sampling) for background triangulation points
    let poisson = Poisson2D::new()
        .with_seed(42)
        .with_dimensions([width as f64 - 20.0, height as f64 - 20.0], 25.0);

    for [x, y] in poisson.iter() {
        tri_points.push(Point2::new((x + 10.0) as i64, (y + 10.0) as i64));
    }

    // Compute Delaunay triangulation
    let triangulation = delaunay(&tri_points);

    // Build SVG
    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
        width, height, width, height
    ));
    svg.push('\n');
    svg.push_str(&format!(
        "  <rect width=\"{}\" height=\"{}\" fill=\"#1a1a2e\"/>",
        width, height
    ));
    svg.push('\n');

    // Draw Delaunay triangulation in background (subtle)
    if let Some(tri) = &triangulation {
        svg.push_str("  <g stroke=\"#2d2d4a\" stroke-width=\"0.5\" fill=\"none\">\n");
        for triangle in &tri.triangles {
            let [a, b, c] = triangle.vertices;
            let pa = tri.points[a];
            let pb = tri.points[b];
            let pc = tri.points[c];
            svg.push_str(&format!(
                "    <polygon points=\"{},{} {},{} {},{}\" fill=\"none\"/>\n",
                pa.x, pa.y, pb.x, pb.y, pc.x, pc.y
            ));
        }
        svg.push_str("  </g>\n");
    }

    // Draw letter polygons with outline
    for poly in &all_letter_polys {
        let points_str: String = poly
            .iter()
            .map(|p| format!("{},{}", p.x, p.y))
            .collect::<Vec<_>>()
            .join(" ");
        svg.push_str(&format!(
            "  <polygon points=\"{}\" fill=\"#1a1a2e\" stroke=\"#ffffff\" stroke-width=\"2\" stroke-linejoin=\"round\" paint-order=\"stroke fill\"/>\n",
            points_str
        ));
    }

    svg.push_str("</svg>\n");
    svg
}

/// Generate Voronoi diagram and Delaunay triangulation using blue noise
fn generate_voronoi_delaunay() -> String {
    let width = 800;
    let height = 400;

    // Generate points using Poisson disk sampling (blue noise)
    let poisson = Poisson2D::new()
        .with_seed(SEED)
        .with_dimensions([(width - 40) as f64, (height - 40) as f64], 45.0);

    let points: Vec<Point2<i64>> = poisson
        .iter()
        .map(|[x, y]| Point2::new((x + 20.0) as i64, (y + 20.0) as i64))
        .collect();

    // Compute Delaunay triangulation
    let triangulation = match delaunay(&points) {
        Some(t) => t,
        None => return String::new(),
    };

    // Compute Voronoi diagram
    let voronoi = voronoi_from_delaunay(&triangulation);

    let mut svg = String::new();
    svg.push_str(&format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">
  <rect width="{}" height="{}" fill="#1a1a2e"/>
"##,
        width, height, width, height, width, height
    ));

    // Draw Delaunay triangles with transparent blue
    svg.push_str("  <g stroke=\"rgba(52,152,219,0.4)\" stroke-width=\"1\" fill=\"none\">\n");
    for tri in &triangulation.triangles {
        let [a, b, c] = tri.vertices;
        let pa = triangulation.points[a];
        let pb = triangulation.points[b];
        let pc = triangulation.points[c];
        svg.push_str(&format!(
            "    <polygon points=\"{},{} {},{} {},{}\" fill=\"none\"/>\n",
            pa.x, pa.y, pb.x, pb.y, pc.x, pc.y
        ));
    }
    svg.push_str("  </g>\n");

    // Draw Voronoi edges in red
    svg.push_str("  <g stroke=\"#e74c3c\" stroke-width=\"1.5\" fill=\"none\">\n");
    for edge in &voronoi.edges {
        let (x1, y1, x2, y2) = if let (Some(start), Some(end)) = (edge.start, edge.end) {
            let (x1, y1) = voronoi.vertices[start].to_f64();
            let (x2, y2) = voronoi.vertices[end].to_f64();
            (x1, y1, x2, y2)
        } else if let Some(start) = edge.start {
            // Infinite edge - extend to boundary
            let (x1, y1) = voronoi.vertices[start].to_f64();
            let (site_a, site_b) = edge.sites;
            let pa = points[site_a];
            let pb = points[site_b];

            // Perpendicular direction to the Delaunay edge
            let edge_dx = (pb.x - pa.x) as f64;
            let edge_dy = (pb.y - pa.y) as f64;
            let (perp_x, perp_y) = (-edge_dy, edge_dx);

            let len = (perp_x * perp_x + perp_y * perp_y).sqrt();
            if len < 1e-10 {
                continue;
            }
            let (nx, ny) = (perp_x / len, perp_y / len);

            // Determine correct direction: away from centroid
            let centroid_x: f64 =
                points.iter().map(|p| p.x as f64).sum::<f64>() / points.len() as f64;
            let centroid_y: f64 =
                points.iter().map(|p| p.y as f64).sum::<f64>() / points.len() as f64;
            let mid_x = (pa.x + pb.x) as f64 / 2.0;
            let mid_y = (pa.y + pb.y) as f64 / 2.0;
            let to_centroid_x = centroid_x - mid_x;
            let to_centroid_y = centroid_y - mid_y;
            let dot = nx * to_centroid_x + ny * to_centroid_y;
            let (nx, ny) = if dot > 0.0 { (-nx, -ny) } else { (nx, ny) };

            let ext = (width + height) as f64;
            (x1, y1, x1 + nx * ext, y1 + ny * ext)
        } else {
            continue;
        };

        // Clip to viewport
        if let Some((cx1, cy1, cx2, cy2)) = clip_line(x1, y1, x2, y2, width as f64, height as f64) {
            svg.push_str(&format!(
                "    <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\"/>\n",
                cx1, cy1, cx2, cy2
            ));
        }
    }
    svg.push_str("  </g>\n");

    // Draw points
    svg.push_str("  <g fill=\"#ecf0f1\">\n");
    for p in &points {
        svg.push_str(&format!(
            "    <circle cx=\"{}\" cy=\"{}\" r=\"3\"/>\n",
            p.x, p.y
        ));
    }
    svg.push_str("  </g>\n");

    svg.push_str("</svg>\n");
    svg
}

/// Clip a line segment to viewport [0, width] x [0, height]
fn clip_line(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    width: f64,
    height: f64,
) -> Option<(f64, f64, f64, f64)> {
    let margin = width.max(height) * 0.05;
    let min_x = -margin;
    let min_y = -margin;
    let max_x = width + margin;
    let max_y = height + margin;

    let dx = x2 - x1;
    let dy = y2 - y1;

    let mut t0 = 0.0_f64;
    let mut t1 = 1.0_f64;

    for (p, q) in [
        (-dx, x1 - min_x),
        (dx, max_x - x1),
        (-dy, y1 - min_y),
        (dy, max_y - y1),
    ] {
        if p.abs() < 1e-10 {
            if q < 0.0 {
                return None;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
        }
    }

    if t0 > t1 {
        return None;
    }

    Some((x1 + t0 * dx, y1 + t0 * dy, x1 + t1 * dx, y1 + t1 * dy))
}

fn generate_convex_hull() -> String {
    let mut rng = StdRng::seed_from_u64(SEED);
    let width = WIDTH;
    let height = 400;
    let margin = 40;

    // Generate random points
    let points: Vec<Point2<i64>> = (0..50)
        .map(|_| {
            Point2::new(
                rng.gen_range(margin..width - margin),
                rng.gen_range(margin..height - margin),
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

    let mut svg = svg_header(width, height);

    // Draw radial lines from pivot to all points (subtle, behind everything)
    svg.push_str("  <g stroke=\"#3d3d5c\" stroke-width=\"1\" stroke-dasharray=\"4,4\">\n");
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
    svg.push_str("  <g fill=\"#ecf0f1\">\n");
    for p in &points {
        let is_hull_point = hull.contains(p);
        let (color, r) = if is_hull_point {
            ("#4CAF50", 6)
        } else {
            ("#7f8c8d", 4)
        };
        svg.push_str(&format!(
            "    <circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{}\"/>\n",
            p.x, p.y, r, color
        ));
    }
    svg.push_str("  </g>\n");

    // Highlight pivot point
    svg.push_str(&format!(
        "  <circle cx=\"{}\" cy=\"{}\" r=\"8\" fill=\"#E91E63\" stroke=\"#1a1a2e\" stroke-width=\"2\"/>\n",
        pivot.x, pivot.y
    ));

    svg.push_str("</svg>\n");
    svg
}

fn generate_quadtree() -> String {
    let mut rng = StdRng::seed_from_u64(SEED + 1);
    let width = WIDTH;
    let height = 400;
    let margin = 20;

    // Generate clustered points for more interesting subdivision
    let mut points: Vec<Point2<i64>> = Vec::new();

    // Cluster 1: top-left
    for _ in 0..15 {
        points.push(Point2::new(rng.gen_range(50..200), rng.gen_range(50..150)));
    }

    // Cluster 2: center
    for _ in 0..25 {
        points.push(Point2::new(
            rng.gen_range(350..500),
            rng.gen_range(150..280),
        ));
    }

    // Cluster 3: bottom-right
    for _ in 0..20 {
        points.push(Point2::new(
            rng.gen_range(600..750),
            rng.gen_range(250..370),
        ));
    }

    // Scattered points
    for _ in 0..15 {
        points.push(Point2::new(
            rng.gen_range(margin..width - margin),
            rng.gen_range(margin..height - margin),
        ));
    }

    let mut svg = svg_header(width, height);

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
    // Create two overlapping polygons - centered in ~266px panels
    let poly_a = create_star(Point2::new(100, 140), 70, 35, 5);
    let poly_b = create_regular_polygon(Point2::new(160, 175), 60, 6);

    // Compute boolean operations
    let union = polygon_union(&poly_a, &poly_b);
    let intersection = polygon_intersection(&poly_a, &poly_b);
    let difference = polygon_difference(&poly_a, &poly_b);

    // Create a wide SVG with three panels (800px total)
    let total_width: i64 = 800;
    let panel_width = total_width / 3; // ~266px each
    let height: i64 = 350;

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
        total_width, height, total_width, height
    ));
    svg.push('\n');
    svg.push_str(&format!(
        "  <rect width=\"{}\" height=\"{}\" fill=\"#1a1a2e\"/>",
        total_width, height
    ));
    svg.push('\n');

    // Panel dividers
    svg.push_str(&format!(
        "  <line x1=\"{}\" y1=\"0\" x2=\"{}\" y2=\"{}\" stroke=\"#3d3d5c\" stroke-width=\"1\"/>\n",
        panel_width, panel_width, height
    ));
    svg.push_str(&format!(
        "  <line x1=\"{}\" y1=\"0\" x2=\"{}\" y2=\"{}\" stroke=\"#3d3d5c\" stroke-width=\"1\"/>\n",
        panel_width * 2,
        panel_width * 2,
        height
    ));

    // Labels - centered in each panel
    let label_x = panel_width / 2;
    svg.push_str(&format!("  <text x=\"{}\" y=\"30\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"16\" font-weight=\"bold\" fill=\"#ecf0f1\">Union</text>\n", label_x));
    svg.push_str(&format!("  <text x=\"{}\" y=\"30\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"16\" font-weight=\"bold\" fill=\"#ecf0f1\">Intersection</text>\n", panel_width + label_x));
    svg.push_str(&format!("  <text x=\"{}\" y=\"30\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"16\" font-weight=\"bold\" fill=\"#ecf0f1\">Difference</text>\n", panel_width * 2 + label_x));

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
    let width = WIDTH;
    let height = 400;
    let margin = 40;

    // Generate random line segments
    let segments: Vec<Segment> = (0..25)
        .map(|_| {
            let x1 = rng.gen_range(margin..width - margin);
            let y1 = rng.gen_range(margin..height - margin);
            // Create segments with reasonable length
            let len = rng.gen_range(100..250);
            let angle = rng.gen_range(0.0..std::f64::consts::PI * 2.0);
            let x2 = (x1 as f64 + angle.cos() * len as f64) as i64;
            let y2 = (y1 as f64 + angle.sin() * len as f64) as i64;
            Segment::new(
                Point2::new(
                    x1.clamp(margin, width - margin),
                    y1.clamp(margin, height - margin),
                ),
                Point2::new(
                    x2.clamp(margin, width - margin),
                    y2.clamp(margin, height - margin),
                ),
            )
        })
        .collect();

    // Find intersections using Bentley-Ottmann
    let intersections = find_intersections(&segments);

    let mut svg = svg_header(width, height);

    // Draw segments
    svg.push_str("  <g stroke=\"#7f8c8d\" stroke-width=\"2\" stroke-linecap=\"round\">\n");
    for seg in &segments {
        svg.push_str(&format!(
            "    <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"/>\n",
            seg.p1.x, seg.p1.y, seg.p2.x, seg.p2.y
        ));
    }
    svg.push_str("  </g>\n");

    // Draw segment endpoints
    svg.push_str("  <g fill=\"#95a5a6\">\n");
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
    svg.push_str("  <g fill=\"#E91E63\" stroke=\"#1a1a2e\" stroke-width=\"2\">\n");
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
        "  <text x=\"{}\" y=\"30\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"14\" fill=\"#95a5a6\">{} intersections found</text>\n",
        width - 10,
        intersections.len()
    ));

    svg.push_str("</svg>\n");
    svg
}

fn svg_header(width: i64, height: i64) -> String {
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">
  <rect width="{}" height="{}" fill="#1a1a2e"/>
"##,
        width, height, width, height, width, height
    )
}
