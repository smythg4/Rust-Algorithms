//! Smallest enclosing circle of a 2-D point set, via Welzl's algorithm.
//!
//! Given a finite set of points `P ⊂ ℝ²`, the smallest enclosing circle (also
//! known as the minimum bounding circle or 1-centre) is the unique disk of
//! minimum radius that contains every point of `P`. Welzl (1991) showed that
//! a randomised recursive scheme — repeatedly try to enclose `P \ {p}` and,
//! whenever the freshly-drawn point `p` falls outside that disk, push it into
//! a "boundary" set of at most three points — runs in **expected `O(n)` time**
//! on a random permutation of the input.
//!
//! This module implements the move-to-front variant: starting from the empty
//! disk, points are processed in a deterministic random order (driven by a
//! seeded `XorShift64` PRNG, so no `rand` dependency) and any point found to be
//! uncovered is moved to the front and used to anchor a recomputed disk. The
//! base cases are
//!
//! - 0 points → no disk (`None`),
//! - 1 point  → degenerate disk centred at the point with radius 0,
//! - 2 points → disk on the segment, centred at the midpoint with radius
//!   half the distance,
//! - 3 points → the circumscribed circle of the triangle they form (or, if
//!   the points are collinear / two coincide, the smallest 2-point disk
//!   covering all three).
//!
//! Complexity: expected `O(n)` time, `O(n)` extra space for the shuffled
//! index buffer and the (bounded by three) boundary set on the recursion
//! stack.
//!
//! Numerical-stability caveat: the 3-point circumscribed circle is computed
//! from a `2 × 2` determinant of point differences. For nearly-collinear
//! triples this determinant is close to zero and the resulting centre /
//! radius are sensitive to floating-point round-off. Callers that need exact
//! results on degenerate input should use rational or interval arithmetic
//! externally; this implementation falls back to the smallest 2-point disk
//! whenever the determinant is below a small absolute threshold, which keeps
//! the routine robust on integer-valued or well-separated inputs but does
//! not guarantee bit-exact answers on adversarial floating-point input.

/// Squared Euclidean distance between two points.
fn dist_sq(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx.mul_add(dx, dy * dy)
}

/// Smallest disk through two points: midpoint centre, radius = ½ · |ab|.
fn disk_from_two(a: (f64, f64), b: (f64, f64)) -> ((f64, f64), f64) {
    let cx = 0.5 * (a.0 + b.0);
    let cy = 0.5 * (a.1 + b.1);
    let r = 0.5 * dist_sq(a, b).sqrt();
    ((cx, cy), r)
}

/// Circumscribed circle of three points, or `None` if they are (numerically)
/// collinear. The radius returned is the *exact* distance from the computed
/// centre to `a`; callers may verify all three points lie on the circle.
fn circumscribed_circle(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> Option<((f64, f64), f64)> {
    let ax = a.0;
    let ay = a.1;
    let bx = b.0 - ax;
    let by = b.1 - ay;
    let cx = c.0 - ax;
    let cy = c.1 - ay;
    let d = 2.0 * bx.mul_add(cy, -(by * cx));
    if d.abs() < 1e-20 {
        return None;
    }
    let b_sq = bx.mul_add(bx, by * by);
    let c_sq = cx.mul_add(cx, cy * cy);
    let ux = cy.mul_add(b_sq, -(by * c_sq)) / d;
    let uy = bx.mul_add(c_sq, -(cx * b_sq)) / d;
    let centre = (ax + ux, ay + uy);
    let r = ux.mul_add(ux, uy * uy).sqrt();
    Some((centre, r))
}

/// Smallest disk enclosing all of `a`, `b`, `c`. Tries the three two-point
/// disks first and falls back to the circumscribed circle.
fn disk_from_three(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> ((f64, f64), f64) {
    // Eps for "is `p` inside disk `(centre, r)`" tests inside the small
    // 3-point routine: scale to the magnitude of the input so that integer
    // coordinates and large coordinates are both handled.
    let scale = 1.0_f64
        .max(a.0.abs())
        .max(a.1.abs())
        .max(b.0.abs())
        .max(b.1.abs())
        .max(c.0.abs())
        .max(c.1.abs());
    let eps = 1e-10 * scale;

    let candidates = [
        disk_from_two(a, b),
        disk_from_two(a, c),
        disk_from_two(b, c),
    ];
    for &(centre, r) in &candidates {
        if dist_sq(centre, a).sqrt() <= r + eps
            && dist_sq(centre, b).sqrt() <= r + eps
            && dist_sq(centre, c).sqrt() <= r + eps
        {
            return (centre, r);
        }
    }
    circumscribed_circle(a, b, c).unwrap_or_else(|| {
        // Pure fallback for pathological collinear input: take the largest
        // pairwise distance.
        let mut best = candidates[0];
        for &cand in &candidates[1..] {
            if cand.1 > best.1 {
                best = cand;
            }
        }
        best
    })
}

/// Deterministic `XorShift64` PRNG; avoids pulling in the `rand` crate.
struct XorShift64(u64);

impl XorShift64 {
    const fn new(seed: u64) -> Self {
        // XorShift forbids state == 0; pick a non-zero default in that case.
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    const fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform integer in `0..=bound` (inclusive). `bound` < `u64::MAX`.
    const fn gen_range(&mut self, bound: usize) -> usize {
        // Lemire-style rejection-free mapping is unnecessary at this scale;
        // a plain modulo is fine for shuffling.
        (self.next_u64() % (bound as u64 + 1)) as usize
    }
}

/// Fisher–Yates shuffle driven by a seeded `XorShift64`.
fn shuffle<T>(slice: &mut [T], rng: &mut XorShift64) {
    for i in (1..slice.len()).rev() {
        let j = rng.gen_range(i);
        slice.swap(i, j);
    }
}

/// Returns true iff `p` lies inside the closed disk `(centre, r)`, with a
/// small tolerance scaled to the radius.
fn in_disk(p: (f64, f64), centre: (f64, f64), r: f64) -> bool {
    // Use squared distance to avoid a sqrt on the hot path; tolerance is
    // expressed as `(r + ε)²` with ε scaled to the working magnitude.
    let scale = 1.0_f64.max(r).max(centre.0.abs()).max(centre.1.abs());
    let eps = 1e-12 * scale;
    let r_tol = r + eps;
    dist_sq(p, centre) <= r_tol * r_tol
}

/// Smallest disk enclosing all points in `points` *and* passing through every
/// point in `boundary` (`|boundary| ≤ 3`). Iterative move-to-front Welzl.
fn welzl(points: &mut [(f64, f64)], boundary: &mut Vec<(f64, f64)>) -> Option<((f64, f64), f64)> {
    let mut disk: Option<((f64, f64), f64)> = match boundary.as_slice() {
        [] => None,
        [a] => Some(((a.0, a.1), 0.0)),
        [a, b] => Some(disk_from_two(*a, *b)),
        [a, b, c] => Some(disk_from_three(*a, *b, *c)),
        _ => unreachable!("boundary holds at most three points"),
    };

    if boundary.len() == 3 {
        return disk;
    }

    for i in 0..points.len() {
        let p = points[i];
        let inside = match disk {
            Some((c, r)) => in_disk(p, c, r),
            None => false,
        };
        if !inside {
            boundary.push(p);
            // Recurse on the prefix `points[..i]`, with `p` pinned on the
            // boundary. Because boundary grows monotonically here and is
            // capped at three, the total work is O(n) in expectation.
            let prefix_disk = welzl(&mut points[..i], boundary);
            boundary.pop();
            disk = prefix_disk;
            // Move-to-front: bring the offender to the head so future
            // top-level passes see it early.
            points[..=i].rotate_right(1);
        }
    }
    disk
}

/// Returns the smallest enclosing circle of `points` as
/// `Some((centre, radius))`, or `None` if `points` is empty.
///
/// The algorithm is Welzl's randomised move-to-front variant; the random
/// permutation that drives the expected `O(n)` running time is produced by
/// a deterministic `XorShift64` PRNG seeded with `seed`, so the result is
/// reproducible across runs and platforms.
///
/// # Examples
///
/// ```
/// use rust_algorithms::geometry::welzl_smallest_enclosing_circle::smallest_enclosing_circle;
///
/// let pts = [(0.0, 0.0), (2.0, 0.0), (1.0, 1.0)];
/// let ((cx, cy), r) = smallest_enclosing_circle(&pts, 0xC0FFEE).unwrap();
/// assert!((cx - 1.0).abs() < 1e-9);
/// assert!(cy.abs() < 1e-9 || (cy - 0.0).abs() < 1.0); // centre on x-axis side
/// assert!(r >= 1.0 - 1e-9);
/// ```
pub fn smallest_enclosing_circle(points: &[(f64, f64)], seed: u64) -> Option<((f64, f64), f64)> {
    if points.is_empty() {
        return None;
    }
    let mut buf: Vec<(f64, f64)> = points.to_vec();
    let mut rng = XorShift64::new(seed);
    shuffle(&mut buf, &mut rng);
    let mut boundary: Vec<(f64, f64)> = Vec::with_capacity(3);
    welzl(&mut buf, &mut boundary)
}

#[cfg(test)]
mod tests {
    use super::{circumscribed_circle, disk_from_three, disk_from_two, smallest_enclosing_circle};
    use quickcheck_macros::quickcheck;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    fn covers_all(points: &[(f64, f64)], centre: (f64, f64), r: f64, eps: f64) -> bool {
        points.iter().all(|&p| {
            let dx = p.0 - centre.0;
            let dy = p.1 - centre.1;
            dx.hypot(dy) <= r + eps
        })
    }

    #[test]
    fn empty_returns_none() {
        let pts: Vec<(f64, f64)> = Vec::new();
        assert!(smallest_enclosing_circle(&pts, 1).is_none());
    }

    #[test]
    fn single_point_zero_radius() {
        let pts = [(3.5, -2.0)];
        let ((cx, cy), r) = smallest_enclosing_circle(&pts, 1).unwrap();
        assert!(approx_eq(cx, 3.5, EPS));
        assert!(approx_eq(cy, -2.0, EPS));
        assert!(approx_eq(r, 0.0, EPS));
    }

    #[test]
    fn two_points_midpoint_circle() {
        let pts = [(0.0, 0.0), (4.0, 0.0)];
        let ((cx, cy), r) = smallest_enclosing_circle(&pts, 7).unwrap();
        assert!(approx_eq(cx, 2.0, EPS));
        assert!(approx_eq(cy, 0.0, EPS));
        assert!(approx_eq(r, 2.0, EPS));
    }

    #[test]
    fn two_coincident_points() {
        let pts = [(1.0, 1.0), (1.0, 1.0)];
        let ((cx, cy), r) = smallest_enclosing_circle(&pts, 11).unwrap();
        assert!(approx_eq(cx, 1.0, EPS));
        assert!(approx_eq(cy, 1.0, EPS));
        assert!(approx_eq(r, 0.0, EPS));
    }

    #[test]
    fn three_collinear_uses_endpoints() {
        // Smallest disk through (0,0), (1,0), (2,0) is the diameter [0,2].
        let pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)];
        let ((cx, cy), r) = smallest_enclosing_circle(&pts, 42).unwrap();
        assert!(approx_eq(cx, 1.0, EPS));
        assert!(approx_eq(cy, 0.0, EPS));
        assert!(approx_eq(r, 1.0, EPS));
        assert!(covers_all(&pts, (cx, cy), r, EPS));
    }

    #[test]
    fn three_points_form_circumscribed_circle() {
        // Right triangle with hypotenuse from (0,0) to (4,2): smallest
        // enclosing circle has the hypotenuse as diameter, centre (2,1),
        // radius √5.
        let pts = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0)];
        let ((cx, cy), r) = smallest_enclosing_circle(&pts, 99).unwrap();
        assert!(approx_eq(cx, 2.0, EPS));
        assert!(approx_eq(cy, 1.0, EPS));
        assert!(approx_eq(r, 5.0_f64.sqrt(), EPS));
    }

    #[test]
    fn equilateral_triangle_circumcircle() {
        // Equilateral triangle with side 1: circumradius = 1/√3.
        let s = 1.0_f64;
        let h = (3.0_f64).sqrt() / 2.0;
        let pts = [(-0.5, 0.0), (0.5, 0.0), (0.0, h)];
        let (_, r) = smallest_enclosing_circle(&pts, 5).unwrap();
        let expected = s / (3.0_f64).sqrt();
        assert!(approx_eq(r, expected, 1e-9));
    }

    #[test]
    fn unit_square_diagonal_disk() {
        // Smallest enclosing circle of the unit square is centred at the
        // centroid with radius √2 / 2.
        let pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let ((cx, cy), r) = smallest_enclosing_circle(&pts, 13).unwrap();
        assert!(approx_eq(cx, 0.5, EPS));
        assert!(approx_eq(cy, 0.5, EPS));
        assert!(approx_eq(r, 0.5_f64 * 2.0_f64.sqrt(), EPS));
    }

    #[test]
    fn redundant_interior_point_is_ignored() {
        // The centre of the unit square lies strictly inside the smallest
        // enclosing circle of its corners; adding it must not change the
        // answer.
        let with_interior = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)];
        let ((cx, cy), r) = smallest_enclosing_circle(&with_interior, 17).unwrap();
        assert!(approx_eq(cx, 0.5, EPS));
        assert!(approx_eq(cy, 0.5, EPS));
        assert!(approx_eq(r, 0.5_f64 * 2.0_f64.sqrt(), EPS));
    }

    #[test]
    fn duplicates_do_not_break_invariants() {
        let pts = [(0.0, 0.0), (2.0, 0.0), (0.0, 0.0), (1.0, 1.0), (2.0, 0.0)];
        let ((cx, cy), r) = smallest_enclosing_circle(&pts, 31).unwrap();
        assert!(covers_all(&pts, (cx, cy), r, 1e-9));
        // No smaller covering disk exists: the diameter is at least the
        // distance between (0,0) and (2,0), so r ≥ 1.
        assert!(r >= 1.0 - 1e-9);
    }

    #[test]
    fn classic_5_point_example() {
        // Small fixed example: smallest disk is the circle on the diameter
        // (-3, 0)–(3, 0), centre origin, radius 3.
        let pts = [(-3.0, 0.0), (3.0, 0.0), (0.0, 2.0), (0.0, -2.0), (1.0, 1.0)];
        let ((cx, cy), r) = smallest_enclosing_circle(&pts, 2024).unwrap();
        assert!(approx_eq(cx, 0.0, EPS));
        assert!(approx_eq(cy, 0.0, EPS));
        assert!(approx_eq(r, 3.0, EPS));
    }

    #[test]
    fn seed_independence_for_well_defined_input() {
        // Different seeds must produce the same disk on a deterministic
        // input (the answer is unique).
        let pts = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0), (2.0, 1.5)];
        let a = smallest_enclosing_circle(&pts, 1).unwrap();
        let b = smallest_enclosing_circle(&pts, 0xDEAD_BEEF).unwrap();
        let c = smallest_enclosing_circle(&pts, 12_345_678_901).unwrap();
        assert!(approx_eq(a.0 .0, b.0 .0, EPS));
        assert!(approx_eq(a.0 .1, b.0 .1, EPS));
        assert!(approx_eq(a.1, b.1, EPS));
        assert!(approx_eq(a.0 .0, c.0 .0, EPS));
        assert!(approx_eq(a.0 .1, c.0 .1, EPS));
        assert!(approx_eq(a.1, c.1, EPS));
    }

    #[test]
    fn helper_disk_from_two() {
        let ((cx, cy), r) = disk_from_two((0.0, 0.0), (6.0, 8.0));
        assert!(approx_eq(cx, 3.0, EPS));
        assert!(approx_eq(cy, 4.0, EPS));
        assert!(approx_eq(r, 5.0, EPS));
    }

    #[test]
    fn helper_disk_from_three_collinear() {
        // Collinear → the routine falls back to the longest 2-point disk.
        let ((cx, cy), r) = disk_from_three((0.0, 0.0), (1.0, 0.0), (2.0, 0.0));
        assert!(approx_eq(cx, 1.0, EPS));
        assert!(approx_eq(cy, 0.0, EPS));
        assert!(approx_eq(r, 1.0, EPS));
    }

    #[test]
    fn helper_circumscribed_circle_collinear_returns_none() {
        assert!(circumscribed_circle((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)).is_none());
    }

    // Property test: for any small (≤ 15) point set, every input point must
    // lie inside the returned disk (within an absolute + relative tolerance).
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_all_points_inside(coords: Vec<(i16, i16)>, seed: u64) -> bool {
        if coords.is_empty() || coords.len() > 15 {
            return true;
        }
        let pts: Vec<(f64, f64)> = coords
            .iter()
            .map(|&(x, y)| (f64::from(x), f64::from(y)))
            .collect();
        let Some(((cx, cy), r)) = smallest_enclosing_circle(&pts, seed) else {
            return false;
        };
        // Tolerance scaled to coordinate magnitude.
        let mag = pts
            .iter()
            .fold(1.0_f64, |m, &(x, y)| m.max(x.abs()).max(y.abs()));
        let eps = 1e-9 * mag.max(r);
        pts.iter().all(|&(x, y)| {
            let dx = x - cx;
            let dy = y - cy;
            dx.hypot(dy) <= r + eps
        })
    }

    // Property test: brute-force comparison. For ≤ 8 points, the smallest
    // enclosing circle is determined by either two diametrically-opposite
    // points or three boundary points; iterating those candidates and
    // picking the smallest covering disk yields the optimum, which Welzl
    // must match.
    #[allow(clippy::needless_pass_by_value)]
    #[quickcheck]
    fn quickcheck_matches_brute_force(coords: Vec<(i8, i8)>, seed: u64) -> bool {
        if coords.is_empty() || coords.len() > 8 {
            return true;
        }
        let pts: Vec<(f64, f64)> = coords
            .iter()
            .map(|&(x, y)| (f64::from(x), f64::from(y)))
            .collect();
        let Some((_, r_welzl)) = smallest_enclosing_circle(&pts, seed) else {
            return false;
        };

        // Brute force: minimise radius over all 2-point and 3-point disks
        // that cover every input point.
        let mag = pts
            .iter()
            .fold(1.0_f64, |m, &(x, y)| m.max(x.abs()).max(y.abs()));
        let eps = 1e-9 * mag.max(1.0);
        let covers = |c: (f64, f64), r: f64| -> bool {
            pts.iter().all(|&(x, y)| {
                let dx = x - c.0;
                let dy = y - c.1;
                dx.hypot(dy) <= r + eps
            })
        };

        let mut best = f64::INFINITY;
        // Single-point disks.
        for &p in &pts {
            if covers(p, 0.0) {
                best = best.min(0.0);
            }
        }
        // Two-point disks.
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let (c, r) = disk_from_two(pts[i], pts[j]);
                if covers(c, r) && r < best {
                    best = r;
                }
            }
        }
        // Three-point disks.
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                for k in (j + 1)..pts.len() {
                    if let Some((c, r)) = circumscribed_circle(pts[i], pts[j], pts[k]) {
                        if covers(c, r) && r < best {
                            best = r;
                        }
                    }
                }
            }
        }

        // Welzl's radius must be within tolerance of the brute-force optimum.
        (r_welzl - best).abs() <= 1e-7_f64.mul_add(mag.max(1.0), 1e-9)
    }
}
