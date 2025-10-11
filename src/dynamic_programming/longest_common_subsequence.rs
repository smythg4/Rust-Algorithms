//! Longest common subsequence between two slices. O(n · m) time, O(n · m) space.

/// Returns the LCS as a `Vec<T>`.
pub fn longest_common_subsequence<T: Clone + Eq>(a: &[T], b: &[T]) -> Vec<T> {
    let (n, m) = (a.len(), b.len());
    let mut dp = vec![vec![0_usize; m + 1]; n + 1];
    for i in 0..n {
        for j in 0..m {
            dp[i + 1][j + 1] = if a[i] == b[j] {
                dp[i][j] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    let mut out = Vec::with_capacity(dp[n][m]);
    let (mut i, mut j) = (n, m);
    while i > 0 && j > 0 {
        if a[i - 1] == b[j - 1] {
            out.push(a[i - 1].clone());
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] >= dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    out.reverse();
    out
}

/// Returns just the length of the LCS (lighter alternative).
pub fn lcs_length<T: Eq>(a: &[T], b: &[T]) -> usize {
    let (n, m) = (a.len(), b.len());
    let mut prev = vec![0_usize; m + 1];
    let mut curr = vec![0_usize; m + 1];
    for i in 0..n {
        for j in 0..m {
            curr[j + 1] = if a[i] == b[j] {
                prev[j] + 1
            } else {
                curr[j].max(prev[j + 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

#[cfg(test)]
mod tests {
    use super::{lcs_length, longest_common_subsequence};

    #[test]
    fn classic() {
        let a: Vec<char> = "ABCBDAB".chars().collect();
        let b: Vec<char> = "BDCAB".chars().collect();
        let lcs = longest_common_subsequence(&a, &b);
        assert_eq!(lcs.len(), 4);
        // Valid answers: "BCAB" or "BDAB".
        let s: String = lcs.into_iter().collect();
        assert!(s == "BCAB" || s == "BDAB");
    }

    #[test]
    fn no_overlap() {
        let a: Vec<char> = "abc".chars().collect();
        let b: Vec<char> = "xyz".chars().collect();
        assert_eq!(lcs_length(&a, &b), 0);
        assert!(longest_common_subsequence(&a, &b).is_empty());
    }

    #[test]
    fn identical() {
        let a: Vec<i32> = vec![1, 2, 3];
        assert_eq!(longest_common_subsequence(&a, &a), a);
    }

    #[test]
    fn empty() {
        let a: Vec<i32> = vec![];
        let b: Vec<i32> = vec![1, 2];
        assert_eq!(lcs_length(&a, &b), 0);
    }
}
