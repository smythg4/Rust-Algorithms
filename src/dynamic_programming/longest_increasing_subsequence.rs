//! Longest strictly increasing subsequence using patience-sort O(n log n).

/// Returns the length of the longest strictly increasing subsequence of `seq`.
pub fn lis_length<T: Ord + Clone>(seq: &[T]) -> usize {
    let mut tails: Vec<T> = Vec::new();
    for x in seq {
        match tails.binary_search(x) {
            Ok(_) => {
                // Strictly increasing: skip duplicates.
            }
            Err(idx) => {
                if idx == tails.len() {
                    tails.push(x.clone());
                } else {
                    tails[idx] = x.clone();
                }
            }
        }
    }
    tails.len()
}

#[cfg(test)]
mod tests {
    use super::lis_length;

    #[test]
    fn classic() {
        assert_eq!(lis_length(&[10, 9, 2, 5, 3, 7, 101, 18]), 4);
    }

    #[test]
    fn all_equal() {
        assert_eq!(lis_length(&[5, 5, 5, 5]), 1);
    }

    #[test]
    fn strictly_increasing() {
        assert_eq!(lis_length(&[1, 2, 3, 4, 5]), 5);
    }

    #[test]
    fn strictly_decreasing() {
        assert_eq!(lis_length(&[5, 4, 3, 2, 1]), 1);
    }

    #[test]
    fn empty() {
        let v: [i32; 0] = [];
        assert_eq!(lis_length(&v), 0);
    }
}
