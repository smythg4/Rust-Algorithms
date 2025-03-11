//! LSD radix sort for `u32`. Stable, O(d·(n + b)) where `d` = digits,
//! `b` = base (here 256). Uses counting sort per byte.

const BASE: usize = 256;

/// Sorts `slice` of `u32` values in non-decreasing order using LSD radix sort.
pub fn radix_sort(slice: &mut [u32]) {
    if slice.len() < 2 {
        return;
    }
    let mut buf = vec![0_u32; slice.len()];
    for shift in (0..32).step_by(8) {
        let mut counts = [0_usize; BASE];
        for &x in slice.iter() {
            counts[((x >> shift) & 0xFF) as usize] += 1;
        }
        // Prefix sums turn counts into starting positions.
        let mut sum = 0;
        for c in &mut counts {
            let s = sum;
            sum += *c;
            *c = s;
        }
        for &x in slice.iter() {
            let idx = ((x >> shift) & 0xFF) as usize;
            buf[counts[idx]] = x;
            counts[idx] += 1;
        }
        slice.copy_from_slice(&buf);
    }
}

#[cfg(test)]
mod tests {
    use super::radix_sort;
    use quickcheck_macros::quickcheck;

    #[test]
    fn empty() {
        let mut v: Vec<u32> = vec![];
        radix_sort(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn random() {
        let mut v = vec![170, 45, 75, 90, 802, 24, 2, 66];
        radix_sort(&mut v);
        assert_eq!(v, vec![2, 24, 45, 66, 75, 90, 170, 802]);
    }

    #[test]
    fn extremes() {
        let mut v = vec![u32::MAX, 0, u32::MAX - 1, 1];
        radix_sort(&mut v);
        assert_eq!(v, vec![0, 1, u32::MAX - 1, u32::MAX]);
    }

    #[quickcheck]
    fn matches_std_sort(mut input: Vec<u32>) -> bool {
        let mut expected = input.clone();
        expected.sort();
        radix_sort(&mut input);
        input == expected
    }
}
