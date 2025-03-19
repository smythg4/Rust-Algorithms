//! Shell sort with Ciura's gap sequence. In-place, O(n^1.3) typical.

const GAPS: &[usize] = &[701, 301, 132, 57, 23, 10, 4, 1];

/// Sorts `slice` in non-decreasing order using Shell sort.
pub fn shell_sort<T: Ord + Clone>(slice: &mut [T]) {
    let n = slice.len();
    for &gap in GAPS {
        if gap >= n {
            continue;
        }
        for i in gap..n {
            let temp = slice[i].clone();
            let mut j = i;
            while j >= gap && slice[j - gap] > temp {
                slice.swap(j, j - gap);
                j -= gap;
            }
            slice[j] = temp;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::shell_sort;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty() {
        let mut v: Vec<i32> = vec![];
        shell_sort(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn random() {
        let mut v = vec![23, 29, 15, 19, 31, 7, 9, 5, 2];
        shell_sort(&mut v);
        assert_eq!(v, vec![2, 5, 7, 9, 15, 19, 23, 29, 31]);
    }

    #[quickcheck]
    fn matches_std_sort(mut input: Vec<i32>) -> bool {
        let mut expected = input.clone();
        expected.sort();
        shell_sort(&mut input);
        input == expected
    }
}
