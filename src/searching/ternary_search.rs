//! Ternary search on a sorted slice. O(log₃ n) comparisons; constant factor
//! larger than binary search but useful for finding the extremum of a
//! unimodal function.

/// Returns the index of an element equal to `target` in a sorted `slice`,
/// or `None`. Slice MUST be sorted in non-decreasing order.
pub fn ternary_search<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    let (mut lo, mut hi) = (0_usize, slice.len());
    while lo < hi {
        if hi - lo < 3 {
            return slice[lo..hi]
                .iter()
                .position(|x| x == target)
                .map(|i| lo + i);
        }
        let third = (hi - lo) / 3;
        let m1 = lo + third;
        let m2 = hi - third;
        if &slice[m1] == target {
            return Some(m1);
        }
        if &slice[m2 - 1] == target {
            return Some(m2 - 1);
        }
        if target < &slice[m1] {
            hi = m1;
        } else if target > &slice[m2 - 1] {
            lo = m2;
        } else {
            lo = m1 + 1;
            hi = m2 - 1;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::ternary_search;

    #[test]
    fn empty() {
        let v: [i32; 0] = [];
        assert_eq!(ternary_search(&v, &0), None);
    }

    #[test]
    fn finds() {
        let v: Vec<i32> = (0..30).collect();
        for x in &v {
            assert_eq!(ternary_search(&v, x), Some(*x as usize));
        }
    }

    #[test]
    fn missing() {
        let v = [1, 3, 5, 7, 9];
        assert_eq!(ternary_search(&v, &2), None);
        assert_eq!(ternary_search(&v, &10), None);
    }
}
