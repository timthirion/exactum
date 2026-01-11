//! SVG visualization for Delaunay triangulations and Voronoi diagrams.
//!
//! Usage:
//!   cargo run --example visualize -- [OPTIONS] > output.svg
//!
//! Options:
//!   --points N      Number of random points (default: 20)
//!   --size S        Domain size, points in [10, S-10] (default: 500)
//!   --seed S        Random seed (default: random)
//!   --delaunay      Show Delaunay triangulation (default: on)
//!   --voronoi       Show Voronoi diagram (default: off)
//!   --circumcircles Show circumcircles (default: off)
//!   --no-delaunay   Hide Delaunay triangulation
//!   --no-points     Hide points
//!   --help          Show this help

use std::env;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use exactum::algo::{delaunay, voronoi_from_delaunay, Triangulation, VoronoiDiagram};
use exactum::Point2;

struct Config {
    num_points: usize,
    size: i64,
    seed: Option<u64>,
    show_delaunay: bool,
    show_voronoi: bool,
    show_circumcircles: bool,
    show_points: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_points: 20,
            size: 500,
            seed: None,
            show_delaunay: true,
            show_voronoi: false,
            show_circumcircles: false,
            show_points: true,
        }
    }
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config::default();
    let args: Vec<String> = env::args().collect();
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--points" => {
                i += 1;
                config.num_points = args
                    .get(i)
                    .ok_or("--points requires a value")?
                    .parse()
                    .map_err(|_| "Invalid number for --points")?;
            }
            "--size" => {
                i += 1;
                config.size = args
                    .get(i)
                    .ok_or("--size requires a value")?
                    .parse()
                    .map_err(|_| "Invalid number for --size")?;
            }
            "--seed" => {
                i += 1;
                config.seed = Some(
                    args.get(i)
                        .ok_or("--seed requires a value")?
                        .parse()
                        .map_err(|_| "Invalid number for --seed")?,
                );
            }
            "--delaunay" => config.show_delaunay = true,
            "--voronoi" => config.show_voronoi = true,
            "--circumcircles" => config.show_circumcircles = true,
            "--no-delaunay" => config.show_delaunay = false,
            "--no-points" => config.show_points = false,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("Unknown option: {}", other)),
        }
        i += 1;
    }

    Ok(config)
}

fn print_help() {
    eprintln!(
        r#"SVG visualization for Delaunay triangulations and Voronoi diagrams.

Usage:
  cargo run --example visualize -- [OPTIONS] > output.svg

Options:
  --points N      Number of random points (default: 20)
  --size S        Domain size, points in [10, S-10] (default: 500)
  --seed S        Random seed (default: random)
  --delaunay      Show Delaunay triangulation (default: on)
  --voronoi       Show Voronoi diagram (default: off)
  --circumcircles Show circumcircles around triangles (default: off)
  --no-delaunay   Hide Delaunay triangulation
  --no-points     Hide points
  --help          Show this help

Examples:
  cargo run --example visualize -- --points 30 > tri.svg
  cargo run --example visualize -- --voronoi --no-delaunay > voronoi.svg
  cargo run --example visualize -- --voronoi --points 50 --seed 42 > both.svg
  cargo run --example visualize -- --circumcircles --seed 42 > circles.svg"#
    );
}

fn generate_points(config: &Config) -> Vec<Point2<i64>> {
    let mut rng: StdRng = match config.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let margin = 10;
    let min = margin;
    let max = config.size - margin;

    (0..config.num_points)
        .map(|_| {
            let x = rng.gen_range(min..max);
            let y = rng.gen_range(min..max);
            Point2::new(x, y)
        })
        .collect()
}

/// Clip a line segment to the viewport [0, size] x [0, size].
/// Returns None if the segment is entirely outside.
fn clip_line(x1: f64, y1: f64, x2: f64, y2: f64, size: f64) -> Option<(f64, f64, f64, f64)> {
    // Cohen-Sutherland-like clipping
    let margin = size * 0.1; // Small margin to avoid edge artifacts
    let min = -margin;
    let max = size + margin;

    // Simple parametric clipping
    let dx = x2 - x1;
    let dy = y2 - y1;

    let mut t0 = 0.0_f64;
    let mut t1 = 1.0_f64;

    for (p, q) in [
        (-dx, x1 - min), // left
        (dx, max - x1),  // right
        (-dy, y1 - min), // top
        (dy, max - y1),  // bottom
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

    let nx1 = x1 + t0 * dx;
    let ny1 = y1 + t0 * dy;
    let nx2 = x1 + t1 * dx;
    let ny2 = y1 + t1 * dy;

    Some((nx1, ny1, nx2, ny2))
}

fn render_svg(
    config: &Config,
    points: &[Point2<i64>],
    triangulation: &Triangulation,
    voronoi: Option<&VoronoiDiagram>,
) -> String {
    let mut svg = String::new();
    let size = config.size as f64;

    // SVG header
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
        config.size, config.size, config.size, config.size
    ));
    svg.push('\n');

    // Background
    svg.push_str(&format!(
        r#"  <rect width="{}" height="{}" fill="white"/>"#,
        config.size, config.size
    ));
    svg.push('\n');

    // Voronoi edges (draw first so they're behind)
    if config.show_voronoi {
        if let Some(v) = voronoi {
            svg.push_str("  <g stroke=\"#2196F3\" stroke-width=\"1.5\" fill=\"none\">\n");
            for edge in &v.edges {
                let (x1, y1, x2, y2) = if let (Some(start), Some(end)) = (edge.start, edge.end) {
                    let (x1, y1) = v.vertices[start].to_f64();
                    let (x2, y2) = v.vertices[end].to_f64();
                    (x1, y1, x2, y2)
                } else if let Some(start) = edge.start {
                    // Infinite edge - extend to boundary
                    let (x1, y1) = v.vertices[start].to_f64();
                    let (site_a, site_b) = edge.sites;
                    let pa = points[site_a];
                    let pb = points[site_b];

                    // Perpendicular direction to the Delaunay edge
                    // Edge vector: (pb - pa), perpendicular: rotate 90° CCW = (-dy, dx)
                    let edge_dx = (pb.x - pa.x) as f64;
                    let edge_dy = (pb.y - pa.y) as f64;
                    let (perp_x, perp_y) = (-edge_dy, edge_dx);

                    // Normalize
                    let len = (perp_x * perp_x + perp_y * perp_y).sqrt();
                    if len < 1e-10 {
                        continue;
                    }
                    let (nx, ny) = (perp_x / len, perp_y / len);

                    // Determine correct direction: away from the centroid of all points
                    // (For convex hull edges, this gives the outward direction)
                    let centroid_x: f64 =
                        points.iter().map(|p| p.x as f64).sum::<f64>() / points.len() as f64;
                    let centroid_y: f64 =
                        points.iter().map(|p| p.y as f64).sum::<f64>() / points.len() as f64;

                    // Midpoint of the edge
                    let mid_x = (pa.x + pb.x) as f64 / 2.0;
                    let mid_y = (pa.y + pb.y) as f64 / 2.0;

                    // Vector from midpoint toward centroid
                    let to_centroid_x = centroid_x - mid_x;
                    let to_centroid_y = centroid_y - mid_y;

                    // If perpendicular points toward centroid, flip it
                    let dot = nx * to_centroid_x + ny * to_centroid_y;
                    let (nx, ny) = if dot > 0.0 { (-nx, -ny) } else { (nx, ny) };

                    let ext = size * 2.0;
                    (x1, y1, x1 + nx * ext, y1 + ny * ext)
                } else {
                    continue;
                };

                // Clip to viewport and draw
                if let Some((cx1, cy1, cx2, cy2)) = clip_line(x1, y1, x2, y2, size) {
                    svg.push_str(&format!(
                        "    <line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\"/>\n",
                        cx1, cy1, cx2, cy2
                    ));
                }
            }
            svg.push_str("  </g>\n");
        }
    }

    // Circumcircles (draw before triangles so they're behind)
    if config.show_circumcircles {
        if let Some(v) = voronoi {
            svg.push_str("  <g stroke=\"#ddd\" stroke-width=\"1\" fill=\"none\">\n");
            for (i, tri) in triangulation.triangles.iter().enumerate() {
                let (cx, cy) = v.vertices[i].to_f64();

                // Compute radius (distance to first vertex)
                let p = triangulation.points[tri.vertices[0]];
                let dx = cx - p.x as f64;
                let dy = cy - p.y as f64;
                let r = (dx * dx + dy * dy).sqrt();

                // Only draw if center is reasonably within extended bounds
                if cx > -size && cx < size * 2.0 && cy > -size && cy < size * 2.0 {
                    svg.push_str(&format!(
                        "    <circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{:.1}\"/>\n",
                        cx, cy, r
                    ));
                }
            }
            svg.push_str("  </g>\n");
        }
    }

    // Delaunay edges
    if config.show_delaunay {
        svg.push_str("  <g stroke=\"#333\" stroke-width=\"1\" fill=\"none\">\n");
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
    }

    // Points
    if config.show_points {
        svg.push_str("  <g fill=\"#E91E63\">\n");
        for p in points {
            svg.push_str(&format!(
                "    <circle cx=\"{}\" cy=\"{}\" r=\"4\"/>\n",
                p.x, p.y
            ));
        }
        svg.push_str("  </g>\n");
    }

    // Voronoi vertices (small dots)
    if config.show_voronoi {
        if let Some(v) = voronoi {
            svg.push_str("  <g fill=\"#2196F3\">\n");
            for vertex in &v.vertices {
                let (x, y) = vertex.to_f64();
                // Only draw if within bounds
                if x >= 0.0 && x <= config.size as f64 && y >= 0.0 && y <= config.size as f64 {
                    svg.push_str(&format!(
                        "    <circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"2.5\"/>\n",
                        x, y
                    ));
                }
            }
            svg.push_str("  </g>\n");
        }
    }

    svg.push_str("</svg>\n");
    svg
}

fn main() {
    let config = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("Use --help for usage information.");
            std::process::exit(1);
        }
    };

    // Generate random points
    let points = generate_points(&config);

    // Compute triangulation
    let triangulation = match delaunay(&points) {
        Some(t) => t,
        None => {
            eprintln!(
                "Error: Could not compute triangulation (need at least 3 non-collinear points)"
            );
            std::process::exit(1);
        }
    };

    // Compute Voronoi if needed (also needed for circumcircles)
    let voronoi = if config.show_voronoi || config.show_circumcircles {
        Some(voronoi_from_delaunay(&triangulation))
    } else {
        None
    };

    // Render and output
    let svg = render_svg(&config, &points, &triangulation, voronoi.as_ref());
    print!("{}", svg);
}
