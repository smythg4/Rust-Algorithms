//! Linear search. O(n). Works on any slice with `PartialEq`.

/// Returns the index of the first element equal to `target`, or `None`.
pub fn linear_search<T: PartialEq>(slice: &[T], target: &T) -> Option<usize> {
    slice.iter().position(|x| x == target)
}

#[cfg(test)]
mod tests {
    use super::linear_search;

    #[test]
    fn found() {
        let v = [3, 1, 4, 1, 5, 9, 2];
        assert_eq!(linear_search(&v, &5), Some(4));
    }

    #[test]
    fn not_found() {
        let v = [3, 1, 4];
        assert_eq!(linear_search(&v, &7), None);
    }

    #[test]
    fn first_match() {
        let v = [1, 2, 1, 2];
        assert_eq!(linear_search(&v, &1), Some(0));
    }

    #[test]
    fn empty() {
        let v: [i32; 0] = [];
        assert_eq!(linear_search(&v, &1), None);
    }
}
