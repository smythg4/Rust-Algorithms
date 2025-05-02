//! Exponential search: doubles a bound until it brackets the target, then
//! falls back to binary search. Useful for unbounded or very large sorted
//! collections. O(log i) where `i` is the position of the target.

use super::binary_search::binary_search;

/// Returns the index of an element equal to `target` in a sorted `slice`,
/// or `None`. Slice MUST be sorted in non-decreasing order.
pub fn exponential_search<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    let n = slice.len();
    if n == 0 {
        return None;
    }
    if &slice[0] == target {
        return Some(0);
    }
    let mut bound = 1;
    while bound < n && &slice[bound] <= target {
        bound *= 2;
    }
    let lo = bound / 2;
    let hi = bound.min(n);
    binary_search(&slice[lo..hi], target).map(|i| lo + i)
}

#[cfg(test)]
mod tests {
    use super::exponential_search;

    #[test]
    fn finds_target() {
        let v: Vec<i32> = (0..1024).collect();
        for x in [0_i32, 1, 7, 100, 1023] {
            assert_eq!(exponential_search(&v, &x), Some(x as usize));
        }
    }

    #[test]
    fn missing() {
        let v = [2, 4, 6, 8, 10];
        assert_eq!(exponential_search(&v, &5), None);
        assert_eq!(exponential_search(&v, &11), None);
    }

    #[test]
    fn empty() {
        let v: [i32; 0] = [];
        assert_eq!(exponential_search(&v, &1), None);
    }
}
