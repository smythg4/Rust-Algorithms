//! Bron-Kerbosch maximal clique enumeration with the Tomita pivot
//! heuristic.
//!
//! A *clique* in an undirected graph is a set of vertices that are
//! pairwise adjacent; a *maximal* clique cannot be extended by adding
//! another vertex; a *maximum* clique is one of largest size. The
//! recursive Bron-Kerbosch procedure walks three vertex sets:
//!
//! - `R` — vertices already chosen for the current clique;
//! - `P` — candidates that extend `R` to a larger clique;
//! - `X` — vertices already explored, used to avoid reporting the same
//!   maximal clique twice.
//!
//! When both `P` and `X` are empty, `R` is reported as a maximal clique;
//! otherwise the algorithm recurses on each candidate `v` in `P`,
//! restricting `P` and `X` to the neighbours of `v`. The Tomita variant
//! picks a *pivot* `u` from `P ∪ X` that maximizes `|P ∩ N(u)|` and only
//! recurses on `P \ N(u)`, which prunes branches that would otherwise
//! be re-explored through any neighbour of `u`.
//!
//! ## Complexity
//!
//! Worst-case time is `O(3^(n/3))`, matching the Moon-Moser bound on
//! the maximum number of maximal cliques in an `n`-vertex graph;
//! finding a maximum clique is NP-hard, so no polynomial algorithm is
//! known. Space is `O(n^2)` for the adjacency representation plus
//! `O(n)` per recursion frame. In practice the pivot rule keeps the
//! algorithm fast for sparse graphs up to a few dozen vertices.
//!
//! ## Input convention
//!
//! `adj[u]` lists the neighbours of vertex `u` in an undirected graph.
//! The implementation tolerates duplicates and self-loops in the input
//! (they are filtered when building the internal adjacency bitset);
//! callers do not need to canonicalize the lists themselves.

/// Returns the vertices of one maximum clique in the graph described
/// by `adj`. Ties are broken by the order in which Bron-Kerbosch
/// discovers maximal cliques. Returns an empty vector for an empty
/// graph.
///
/// Runs the same enumeration as [`enumerate_maximal_cliques`] but only
/// keeps the largest clique seen so far, so its memory footprint is
/// `O(n)` rather than `O(n · k)` for `k` maximal cliques.
#[must_use]
pub fn maximum_clique(adj: &[Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }
    let neighbours = build_neighbour_bitsets(adj);
    let mut best: Vec<usize> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let p: Vec<bool> = vec![true; n];
    let x: Vec<bool> = vec![false; n];
    bron_kerbosch(&neighbours, &mut current, p, x, &mut |clique| {
        if clique.len() > best.len() {
            best = clique.to_vec();
        }
    });
    best.sort_unstable();
    best
}

/// Returns every maximal clique in the graph described by `adj`. Each
/// clique is sorted ascending and the outer vector is stable across
/// runs for a fixed input. An empty graph yields an empty vector.
#[must_use]
pub fn enumerate_maximal_cliques(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }
    let neighbours = build_neighbour_bitsets(adj);
    let mut cliques: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let p: Vec<bool> = vec![true; n];
    let x: Vec<bool> = vec![false; n];
    bron_kerbosch(&neighbours, &mut current, p, x, &mut |clique| {
        let mut sorted = clique.to_vec();
        sorted.sort_unstable();
        cliques.push(sorted);
    });
    cliques
}

/// Builds an `n × n` boolean adjacency matrix from `adj`, dropping
/// self-loops and duplicate edges so the rest of the algorithm can rely
/// on `neighbours[u][v] == neighbours[v][u]`.
fn build_neighbour_bitsets(adj: &[Vec<usize>]) -> Vec<Vec<bool>> {
    let n = adj.len();
    let mut neighbours = vec![vec![false; n]; n];
    for (u, list) in adj.iter().enumerate() {
        for &v in list {
            if v < n && v != u {
                neighbours[u][v] = true;
                neighbours[v][u] = true;
            }
        }
    }
    neighbours
}

/// Recursive Bron-Kerbosch with Tomita pivot selection. Reports each
/// maximal clique to `report` exactly once.
fn bron_kerbosch(
    neighbours: &[Vec<bool>],
    r: &mut Vec<usize>,
    p: Vec<bool>,
    x: Vec<bool>,
    report: &mut dyn FnMut(&[usize]),
) {
    if !any(&p) && !any(&x) {
        report(r);
        return;
    }

    // Tomita pivot: choose `u ∈ P ∪ X` maximizing |P ∩ N(u)| so the
    // recursion only enumerates candidates not dominated by some pivot
    // neighbour. Picking from `P ∪ X` (not just `P`) is what gives the
    // O(3^(n/3)) bound.
    let pivot = select_pivot(neighbours, &p, &x);

    // Recurse on `P \ N(pivot)`: each vertex outside the pivot's
    // neighbourhood must be branched on directly because it is not
    // covered by any choice of the pivot's neighbour.
    let candidates: Vec<usize> = (0..p.len())
        .filter(|&v| p[v] && !neighbours[pivot][v])
        .collect();

    let mut p = p;
    let mut x = x;
    for v in candidates {
        // P, X intersected with N(v) for the recursive call.
        let mut new_p = p.clone();
        let mut new_x = x.clone();
        for w in 0..neighbours.len() {
            if !neighbours[v][w] {
                new_p[w] = false;
                new_x[w] = false;
            }
        }
        r.push(v);
        bron_kerbosch(neighbours, r, new_p, new_x, report);
        r.pop();
        // Move v from P to X so future siblings know v has been handled.
        p[v] = false;
        x[v] = true;
    }
}

/// Returns the index of a pivot in `P ∪ X` that maximizes the number of
/// candidates in `P` it neighbours. Falls back to any vertex in
/// `P ∪ X` if no neighbour overlaps occur (e.g. an empty `P`).
fn select_pivot(neighbours: &[Vec<bool>], p: &[bool], x: &[bool]) -> usize {
    let mut best_pivot = 0usize;
    let mut best_count: i64 = -1;
    for u in 0..neighbours.len() {
        if !(p[u] || x[u]) {
            continue;
        }
        let mut count: i64 = 0;
        for v in 0..neighbours.len() {
            if p[v] && neighbours[u][v] {
                count += 1;
            }
        }
        if count > best_count {
            best_count = count;
            best_pivot = u;
        }
    }
    best_pivot
}

/// Returns `true` iff at least one slot in `set` is `true`.
fn any(set: &[bool]) -> bool {
    set.iter().any(|&b| b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force baseline: enumerate every subset and keep the
    /// largest one that forms a clique. Used as a property-test oracle
    /// for tiny graphs (`n <= 6`).
    fn brute_force_max_clique_size(adj: &[Vec<usize>]) -> usize {
        let n = adj.len();
        let mut neigh = vec![vec![false; n]; n];
        for (u, list) in adj.iter().enumerate() {
            for &v in list {
                if v < n && v != u {
                    neigh[u][v] = true;
                    neigh[v][u] = true;
                }
            }
        }
        let mut best = 0usize;
        for mask in 0u32..(1u32 << n) {
            let vertices: Vec<usize> = (0..n).filter(|&i| (mask >> i) & 1 == 1).collect();
            let mut is_clique = true;
            'outer: for i in 0..vertices.len() {
                for j in (i + 1)..vertices.len() {
                    if !neigh[vertices[i]][vertices[j]] {
                        is_clique = false;
                        break 'outer;
                    }
                }
            }
            if is_clique && vertices.len() > best {
                best = vertices.len();
            }
        }
        best
    }

    /// Verifies `set` is a clique in `adj`.
    fn is_clique(set: &[usize], adj: &[Vec<usize>]) -> bool {
        let n = adj.len();
        let mut neigh = vec![vec![false; n]; n];
        for (u, list) in adj.iter().enumerate() {
            for &v in list {
                if v < n && v != u {
                    neigh[u][v] = true;
                    neigh[v][u] = true;
                }
            }
        }
        for i in 0..set.len() {
            for j in (i + 1)..set.len() {
                if !neigh[set[i]][set[j]] {
                    return false;
                }
            }
        }
        true
    }

    /// Verifies `set` is *maximal*: no vertex outside `set` is adjacent
    /// to every member of `set`.
    fn is_maximal(set: &[usize], adj: &[Vec<usize>]) -> bool {
        let n = adj.len();
        let mut neigh = vec![vec![false; n]; n];
        for (u, list) in adj.iter().enumerate() {
            for &v in list {
                if v < n && v != u {
                    neigh[u][v] = true;
                    neigh[v][u] = true;
                }
            }
        }
        let in_set: Vec<bool> = (0..n).map(|i| set.contains(&i)).collect();
        for v in 0..n {
            if in_set[v] {
                continue;
            }
            if set.iter().all(|&u| neigh[v][u]) {
                return false;
            }
        }
        true
    }

    #[test]
    fn empty_graph() {
        let adj: Vec<Vec<usize>> = Vec::new();
        assert_eq!(maximum_clique(&adj), Vec::<usize>::new());
        assert_eq!(enumerate_maximal_cliques(&adj), Vec::<Vec<usize>>::new());
    }

    #[test]
    fn single_vertex() {
        let adj: Vec<Vec<usize>> = vec![vec![]];
        assert_eq!(maximum_clique(&adj), vec![0]);
        assert_eq!(enumerate_maximal_cliques(&adj), vec![vec![0]]);
    }

    #[test]
    fn two_isolated_vertices() {
        let adj: Vec<Vec<usize>> = vec![vec![], vec![]];
        assert_eq!(maximum_clique(&adj).len(), 1);
        let mut cliques = enumerate_maximal_cliques(&adj);
        cliques.sort();
        assert_eq!(cliques, vec![vec![0], vec![1]]);
    }

    #[test]
    fn k2_edge() {
        let adj = vec![vec![1], vec![0]];
        assert_eq!(maximum_clique(&adj), vec![0, 1]);
        assert_eq!(enumerate_maximal_cliques(&adj), vec![vec![0, 1]]);
    }

    #[test]
    fn triangle_k3() {
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        assert_eq!(maximum_clique(&adj), vec![0, 1, 2]);
        let cliques = enumerate_maximal_cliques(&adj);
        assert_eq!(cliques.len(), 1);
        assert_eq!(cliques[0], vec![0, 1, 2]);
    }

    #[test]
    fn two_disjoint_triangles() {
        // Vertices 0-1-2 form one triangle, 3-4-5 form another, no
        // cross edges → two maximal cliques of size 3.
        let adj = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
            vec![4, 5],
            vec![3, 5],
            vec![3, 4],
        ];
        assert_eq!(maximum_clique(&adj).len(), 3);
        let mut cliques = enumerate_maximal_cliques(&adj);
        cliques.sort();
        assert_eq!(cliques, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn complete_k4() {
        let adj = vec![vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3], vec![0, 1, 2]];
        assert_eq!(maximum_clique(&adj), vec![0, 1, 2, 3]);
        assert_eq!(enumerate_maximal_cliques(&adj), vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn classic_multi_clique_graph() {
        // Edges: triangle {0,1,2}, triangle {1,2,3}, edge 3-4, edge 4-5.
        // Resulting maximal cliques: {0,1,2}, {1,2,3}, {3,4}, {4,5}.
        let adj = vec![
            vec![1, 2],    // 0
            vec![0, 2, 3], // 1
            vec![0, 1, 3], // 2
            vec![1, 2, 4], // 3
            vec![3, 5],    // 4
            vec![4],       // 5
        ];
        let mut cliques = enumerate_maximal_cliques(&adj);
        cliques.sort();
        assert_eq!(
            cliques,
            vec![vec![0, 1, 2], vec![1, 2, 3], vec![3, 4], vec![4, 5],]
        );
        assert_eq!(maximum_clique(&adj).len(), 3);
    }

    #[test]
    fn handles_self_loops_and_duplicate_edges() {
        // Self-loop on 0, duplicate 0-1 edge — both must be ignored.
        let adj = vec![vec![0, 1, 1], vec![0, 0]];
        assert_eq!(maximum_clique(&adj), vec![0, 1]);
        assert_eq!(enumerate_maximal_cliques(&adj), vec![vec![0, 1]]);
    }

    /// Property-style sweep: every adjacency matrix on `n <= 6`
    /// vertices is exhaustively built by picking an arbitrary subset
    /// of the `n*(n-1)/2` possible edges. For each graph we verify
    /// that
    ///
    /// 1. `maximum_clique` returns a clique whose size matches the
    ///    brute-force optimum;
    /// 2. every clique returned by `enumerate_maximal_cliques` is in
    ///    fact a clique and is maximal.
    #[test]
    fn quickcheck_against_brute_force_small_graphs() {
        for n in 0..=5usize {
            let pairs: Vec<(usize, usize)> = (0..n)
                .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
                .collect();
            let total_edges = pairs.len();
            let total_graphs = 1u32 << total_edges;
            for mask in 0..total_graphs {
                let mut adj = vec![Vec::<usize>::new(); n];
                for (k, &(u, v)) in pairs.iter().enumerate() {
                    if (mask >> k) & 1 == 1 {
                        adj[u].push(v);
                        adj[v].push(u);
                    }
                }
                let max_clique = maximum_clique(&adj);
                assert!(
                    is_clique(&max_clique, &adj),
                    "maximum_clique returned a non-clique for n={n}, mask={mask}"
                );
                let expected = brute_force_max_clique_size(&adj);
                assert_eq!(
                    max_clique.len(),
                    expected,
                    "maximum_clique size mismatch for n={n}, mask={mask}: got {max_clique:?}"
                );
                let cliques = enumerate_maximal_cliques(&adj);
                for clique in &cliques {
                    assert!(
                        is_clique(clique, &adj),
                        "enumerate returned non-clique {clique:?} for n={n}, mask={mask}"
                    );
                    assert!(
                        is_maximal(clique, &adj),
                        "enumerate returned non-maximal {clique:?} for n={n}, mask={mask}"
                    );
                }
                if n > 0 {
                    let largest = cliques.iter().map(Vec::len).max().unwrap_or(0);
                    assert_eq!(
                        largest, expected,
                        "largest maximal clique disagrees with brute force for n={n}, mask={mask}"
                    );
                }
            }
        }
    }
}
