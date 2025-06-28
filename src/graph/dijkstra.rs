//! Dijkstra's single-source shortest paths on a graph with non-negative edge
//! weights. O((V + E) log V) using a binary heap.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    cost: u64,
    node: usize,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse cost ordering, ties broken by node id.
        other
            .cost
            .cmp(&self.cost)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Returns the shortest distance from `start` to every node, or `u64::MAX`
/// if unreachable. `graph[u]` is a list of `(neighbour, weight)` pairs.
///
/// Panics if any weight overflows when summed; otherwise tolerates large
/// edge counts.
pub fn dijkstra(graph: &[Vec<(usize, u64)>], start: usize) -> Vec<u64> {
    let n = graph.len();
    let mut dist = vec![u64::MAX; n];
    if start >= n {
        return dist;
    }
    dist[start] = 0;
    let mut heap = BinaryHeap::new();
    heap.push(State {
        cost: 0,
        node: start,
    });
    while let Some(State { cost, node }) = heap.pop() {
        if cost > dist[node] {
            continue;
        }
        for &(v, w) in &graph[node] {
            let next = cost.saturating_add(w);
            if next < dist[v] {
                dist[v] = next;
                heap.push(State {
                    cost: next,
                    node: v,
                });
            }
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::dijkstra;

    #[test]
    fn classic() {
        // 0 --1--> 1 --2--> 2
        //  \--4--> 2
        let g = vec![vec![(1, 1), (2, 4)], vec![(2, 2)], vec![]];
        let d = dijkstra(&g, 0);
        assert_eq!(d, vec![0, 1, 3]);
    }

    #[test]
    fn unreachable() {
        let g = vec![vec![(1, 5)], vec![], vec![]];
        let d = dijkstra(&g, 0);
        assert_eq!(d, vec![0, 5, u64::MAX]);
    }

    #[test]
    fn self_loop_ignored() {
        let g = vec![vec![(0, 1), (1, 2)], vec![]];
        let d = dijkstra(&g, 0);
        assert_eq!(d, vec![0, 2]);
    }

    #[test]
    fn empty() {
        let g: Vec<Vec<(usize, u64)>> = vec![];
        assert!(dijkstra(&g, 0).is_empty());
    }
}
