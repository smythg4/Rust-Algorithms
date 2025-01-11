//! Bubble sort. Stable, in-place, O(n²) worst/average, O(n) best (already sorted).

/// Sorts `slice` in non-decreasing order using bubble sort.
///
/// Uses an early-exit flag so an already-sorted input runs in linear time.
pub fn bubble_sort<T: Ord>(slice: &mut [T]) {
    let n = slice.len();
    if n < 2 {
        return;
    }
    for i in 0..n - 1 {
        let mut swapped = false;
        for j in 0..n - 1 - i {
            if slice[j] > slice[j + 1] {
                slice.swap(j, j + 1);
                swapped = true;
            }
        }
        if !swapped {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::bubble_sort;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty() {
        let mut v: Vec<i32> = vec![];
        bubble_sort(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn single() {
        let mut v = vec![42];
        bubble_sort(&mut v);
        assert_eq!(v, vec![42]);
    }

    #[test]
    fn already_sorted() {
        let mut v = vec![1, 2, 3, 4, 5];
        bubble_sort(&mut v);
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn reverse() {
        let mut v = vec![5, 4, 3, 2, 1];
        bubble_sort(&mut v);
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn duplicates() {
        let mut v = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        bubble_sort(&mut v);
        assert_eq!(v, vec![1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]);
    }

    #[test]
    fn strings() {
        let mut v = vec!["pear", "apple", "banana"];
        bubble_sort(&mut v);
        assert_eq!(v, vec!["apple", "banana", "pear"]);
    }

    #[quickcheck]
    fn matches_std_sort(mut input: Vec<i32>) -> bool {
        let mut expected = input.clone();
        expected.sort();
        bubble_sort(&mut input);
        input == expected
    }
}
