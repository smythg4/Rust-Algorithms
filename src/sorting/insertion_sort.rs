//! Insertion sort. Stable, in-place, O(n²) worst, O(n) best, efficient on small/nearly-sorted data.

/// Sorts `slice` in non-decreasing order using insertion sort.
pub fn insertion_sort<T: Ord>(slice: &mut [T]) {
    for i in 1..slice.len() {
        let mut j = i;
        while j > 0 && slice[j - 1] > slice[j] {
            slice.swap(j - 1, j);
            j -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::insertion_sort;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty() {
        let mut v: Vec<i32> = vec![];
        insertion_sort(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn nearly_sorted_runs_fast() {
        let mut v = vec![1, 2, 3, 5, 4, 6, 7];
        insertion_sort(&mut v);
        assert_eq!(v, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn duplicates_stable_value_order() {
        let mut v = vec![(2, 'a'), (1, 'b'), (2, 'c'), (1, 'd')];
        insertion_sort(&mut v);
        // Stable sort: equal first-element pairs keep original order.
        assert_eq!(v, vec![(1, 'b'), (1, 'd'), (2, 'a'), (2, 'c')]);
    }

    #[quickcheck]
    fn matches_std_sort(mut input: Vec<i32>) -> bool {
        let mut expected = input.clone();
        expected.sort();
        insertion_sort(&mut input);
        input == expected
    }
}
