//! Breadth-first search on an unweighted graph represented as adjacency lists.

use std::collections::VecDeque;

/// Returns the order in which nodes are visited starting from `start`.
/// `graph[i]` lists the neighbours of node `i`. Disconnected components are
/// not visited.
pub fn bfs(graph: &[Vec<usize>], start: usize) -> Vec<usize> {
    let n = graph.len();
    if start >= n {
        return Vec::new();
    }
    let mut visited = vec![false; n];
    let mut order = Vec::new();
    let mut queue = VecDeque::new();
    visited[start] = true;
    queue.push_back(start);
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &v in &graph[u] {
            if !visited[v] {
                visited[v] = true;
                queue.push_back(v);
            }
        }
    }
    order
}

/// Returns the shortest path length (in edges) from `start` to every node,
/// or `usize::MAX` if unreachable.
pub fn bfs_distances(graph: &[Vec<usize>], start: usize) -> Vec<usize> {
    let n = graph.len();
    let mut dist = vec![usize::MAX; n];
    if start >= n {
        return dist;
    }
    dist[start] = 0;
    let mut queue = VecDeque::from([start]);
    while let Some(u) = queue.pop_front() {
        for &v in &graph[u] {
            if dist[v] == usize::MAX {
                dist[v] = dist[u] + 1;
                queue.push_back(v);
            }
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::{bfs, bfs_distances};

    fn sample() -> Vec<Vec<usize>> {
        // 0 -- 1 -- 2
        // |    |
        // 3    4
        vec![vec![1, 3], vec![0, 2, 4], vec![1], vec![0], vec![1]]
    }

    #[test]
    fn order_starts_with_start() {
        let g = sample();
        let order = bfs(&g, 0);
        assert_eq!(order[0], 0);
        assert_eq!(order.len(), 5);
    }

    #[test]
    fn distances_are_minimal() {
        let g = sample();
        let d = bfs_distances(&g, 0);
        assert_eq!(d, vec![0, 1, 2, 1, 2]);
    }

    #[test]
    fn unreachable_marked_max() {
        let g = vec![vec![1], vec![0], vec![]];
        let d = bfs_distances(&g, 0);
        assert_eq!(d[2], usize::MAX);
    }

    #[test]
    fn empty_graph() {
        let g: Vec<Vec<usize>> = vec![];
        assert!(bfs(&g, 0).is_empty());
    }
}
