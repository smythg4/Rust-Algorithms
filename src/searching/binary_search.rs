//! Binary search on a sorted slice. O(log n).

use std::cmp::Ordering;

/// Returns the index of an element equal to `target` in a sorted `slice`,
/// or `None`. Slice MUST be sorted in non-decreasing order.
pub fn binary_search<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    let (mut lo, mut hi) = (0_usize, slice.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match slice[mid].cmp(target) {
            Ordering::Equal => return Some(mid),
            Ordering::Less => lo = mid + 1,
            Ordering::Greater => hi = mid,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::binary_search;

    #[test]
    fn empty() {
        let v: [i32; 0] = [];
        assert_eq!(binary_search(&v, &0), None);
    }

    #[test]
    fn finds_each_element() {
        let v = [1, 3, 5, 7, 9, 11, 13];
        for (i, x) in v.iter().enumerate() {
            assert_eq!(binary_search(&v, x), Some(i));
        }
    }

    #[test]
    fn missing() {
        let v = [1, 3, 5, 7];
        assert_eq!(binary_search(&v, &4), None);
        assert_eq!(binary_search(&v, &0), None);
        assert_eq!(binary_search(&v, &8), None);
    }

    #[test]
    fn no_overflow_on_large_indices() {
        let v: Vec<usize> = (0..100_000).collect();
        assert_eq!(binary_search(&v, &99_999), Some(99_999));
    }
}
