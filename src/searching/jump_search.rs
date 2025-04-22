//! Jump search on a sorted slice. O(√n) comparisons.

/// Returns the index of an element equal to `target` in a sorted `slice`,
/// or `None`. Slice MUST be sorted in non-decreasing order.
pub fn jump_search<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    let n = slice.len();
    if n == 0 {
        return None;
    }
    let step = (n as f64).sqrt() as usize;
    let step = step.max(1);

    let mut prev = 0;
    let mut curr = step.min(n);
    while &slice[curr - 1] < target {
        prev = curr;
        curr = (curr + step).min(n);
        if prev >= n {
            return None;
        }
    }
    for i in prev..curr {
        if &slice[i] == target {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::jump_search;

    #[test]
    fn empty() {
        let v: [i32; 0] = [];
        assert_eq!(jump_search(&v, &0), None);
    }

    #[test]
    fn found_and_missing() {
        let v: Vec<i32> = (0..100).map(|x| x * 2).collect();
        assert_eq!(jump_search(&v, &50), Some(25));
        assert_eq!(jump_search(&v, &51), None);
        assert_eq!(jump_search(&v, &198), Some(99));
    }

    #[test]
    fn first_and_last() {
        let v = [1, 3, 5, 7, 9];
        assert_eq!(jump_search(&v, &1), Some(0));
        assert_eq!(jump_search(&v, &9), Some(4));
    }
}
