//! Floyd–Warshall all-pairs shortest paths. Handles negative weights, no
//! negative cycles. O(V³).

/// Sentinel "infinity" used both in input and output.
pub const INF: i64 = i64::MAX / 4;

/// Returns the all-pairs shortest-path matrix. Input `dist` must be a square
/// matrix where `dist[i][j]` is the direct edge weight from `i` to `j`, or
/// [`INF`] if no edge, with `dist[i][i] == 0`.
///
/// `Err` is returned if a negative cycle is detected.
pub fn floyd_warshall(mut dist: Vec<Vec<i64>>) -> Result<Vec<Vec<i64>>, &'static str> {
    let n = dist.len();
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if dist[i][k] >= INF || dist[k][j] >= INF {
                    continue;
                }
                let candidate = dist[i][k] + dist[k][j];
                if candidate < dist[i][j] {
                    dist[i][j] = candidate;
                }
            }
        }
    }
    for i in 0..n {
        if dist[i][i] < 0 {
            return Err("negative cycle detected");
        }
    }
    Ok(dist)
}

#[cfg(test)]
mod tests {
    use super::{floyd_warshall, INF};

    #[test]
    fn small() {
        let m = vec![
            vec![0, 3, INF, 7],
            vec![8, 0, 2, INF],
            vec![5, INF, 0, 1],
            vec![2, INF, INF, 0],
        ];
        let r = floyd_warshall(m).unwrap();
        assert_eq!(r[0][1], 3);
        assert_eq!(r[0][2], 5);
        assert_eq!(r[0][3], 6);
        // 2 -> 3 -> 0 -> 1 = 1 + 2 + 3 = 6
        assert_eq!(r[2][1], 6);
    }

    #[test]
    fn negative_edge_no_cycle() {
        let m = vec![vec![0, 4, 5], vec![INF, 0, -3], vec![INF, INF, 0]];
        let r = floyd_warshall(m).unwrap();
        assert_eq!(r[0][2], 1);
    }

    #[test]
    fn detects_negative_cycle() {
        let m = vec![vec![0, 1, INF], vec![INF, 0, -1], vec![-1, INF, 0]];
        assert!(floyd_warshall(m).is_err());
    }

    #[test]
    fn empty() {
        let r = floyd_warshall(vec![]).unwrap();
        assert!(r.is_empty());
    }
}
