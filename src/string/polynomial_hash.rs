//! Polynomial string hashing. Given a byte string `s` of length `n`, a base
//! `b` and a prime modulus `p`, the hash of `s` is
//!
//! ```text
//! H(s) = (s[0] * b^(n-1) + s[1] * b^(n-2) + ... + s[n-1]) mod p.
//! ```
//!
//! Precomputing prefix hashes `h[i+1] = (h[i] * b + s[i]) mod p` and powers
//! `pow[i] = b^i mod p` in O(n) time lets us answer the hash of any
//! substring `s[l..=r]` in O(1) via
//!
//! ```text
//! hash(l, r) = (h[r+1] - h[l] * pow[r - l + 1]) mod p.
//! ```
//!
//! Two substrings with the same hash *probably* have the same content; with
//! a single 61-bit Mersenne modulus the collision probability per query is
//! roughly `1 / p ≈ 4.3e-19`. For adversarial inputs (CTFs, hash-flooding)
//! pair this with a second independent (base, modulus) — "double hashing" —
//! to drive the probability to ~`1 / p²`.
//!
//! This module uses `(1 << 61) - 1` as the default modulus and `u128`
//! intermediates so `(value < p) * (other < p)` cannot overflow.
//!
//! Reference: CP-Algorithms / CSES Handbook §26.3.
//!
//! Complexity: `new` is O(n) time + O(n) space; `hash` and `equal` are O(1).
//!
//! # Example
//! ```
//! use rust_algorithms::string::polynomial_hash::{PolynomialHash, MERSENNE_61};
//! let s = b"abcabc";
//! let h = PolynomialHash::new(s, 257, MERSENNE_61);
//! // "abc" appears at [0, 2] and [3, 5].
//! assert!(h.equal(0, 2, 3, 5));
//! assert!(!h.equal(0, 2, 0, 1));
//! ```

/// Mersenne prime `2^61 - 1`, a popular modulus for polynomial hashing.
pub const MERSENNE_61: u64 = (1u64 << 61) - 1;

/// Polynomial rolling hash with O(1) substring-hash queries.
///
/// Stores prefix hashes and base powers modulo `modulus`. Use a prime
/// modulus and a base coprime with it (e.g. 31, 131, 257) for the standard
/// collision-probability bound of `~1 / modulus` per comparison.
pub struct PolynomialHash {
    prefix: Vec<u64>,
    pow: Vec<u64>,
    base: u64,
    modulus: u64,
}

impl PolynomialHash {
    /// Builds prefix hashes and base powers for `s` in O(n).
    ///
    /// `base` must be smaller than `modulus`; `modulus` must be prime and
    /// satisfy `modulus < 2^63` so the `u128` intermediate fits without
    /// overflow. The empty string yields `prefix = [0]`.
    #[must_use]
    pub fn new(s: &[u8], base: u64, modulus: u64) -> Self {
        let n = s.len();
        let mut prefix = vec![0_u64; n + 1];
        let mut pow = vec![0_u64; n + 1];
        pow[0] = 1 % modulus;
        let b = base % modulus;
        for i in 0..n {
            let h = u128::from(prefix[i]) * u128::from(b) + u128::from(s[i]);
            prefix[i + 1] = (h % u128::from(modulus)) as u64;
            let p = u128::from(pow[i]) * u128::from(b);
            pow[i + 1] = (p % u128::from(modulus)) as u64;
        }
        Self {
            prefix,
            pow,
            base: b,
            modulus,
        }
    }

    /// Returns the hash of the closed-range substring `s[l..=r]` in O(1).
    ///
    /// # Panics
    /// Panics if `l > r` or `r` is out of bounds for the source string.
    #[must_use]
    pub fn hash(&self, l: usize, r: usize) -> u64 {
        assert!(l <= r, "polynomial_hash::hash requires l <= r");
        assert!(
            r + 1 < self.prefix.len(),
            "polynomial_hash::hash index out of bounds"
        );
        let m = u128::from(self.modulus);
        let high = u128::from(self.prefix[r + 1]);
        let low = u128::from(self.prefix[l]) * u128::from(self.pow[r - l + 1]);
        // Add `m * m` (which fits in u128 because modulus < 2^63) so the
        // subtraction stays non-negative before taking the final modulus.
        let diff = (high + m * m - low % m) % m;
        diff as u64
    }

    /// Returns `true` if `s[l1..=r1]` and `s[l2..=r2]` hash to the same
    /// value. Same-length matching hashes mean equality with probability
    /// `~1 - 1/modulus`; different lengths are reported as unequal directly.
    #[must_use]
    pub fn equal(&self, l1: usize, r1: usize, l2: usize, r2: usize) -> bool {
        if r1 - l1 != r2 - l2 {
            return false;
        }
        self.hash(l1, r1) == self.hash(l2, r2)
    }

    /// Hash of the full source string (or `0` for the empty string).
    #[must_use]
    pub fn full(&self) -> u64 {
        *self.prefix.last().unwrap_or(&0)
    }

    /// Length of the source string.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.prefix.len() - 1
    }

    /// `true` if the source string is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.prefix.len() == 1
    }

    /// Returns the configured base (already reduced modulo `modulus`).
    #[must_use]
    pub const fn base(&self) -> u64 {
        self.base
    }

    /// Returns the configured modulus.
    #[must_use]
    pub const fn modulus(&self) -> u64 {
        self.modulus
    }
}

#[cfg(test)]
mod tests {
    use super::{PolynomialHash, MERSENNE_61};
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    const BASE: u64 = 257;

    fn manual_hash(s: &[u8], base: u64, modulus: u64) -> u64 {
        let m = u128::from(modulus);
        let mut h: u128 = 0;
        for &c in s {
            h = (h * u128::from(base) + u128::from(c)) % m;
        }
        h as u64
    }

    #[test]
    fn empty_string() {
        let h = PolynomialHash::new(b"", BASE, MERSENNE_61);
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        assert_eq!(h.full(), 0);
    }

    #[test]
    fn single_char() {
        let h = PolynomialHash::new(b"a", BASE, MERSENNE_61);
        assert_eq!(h.len(), 1);
        assert_eq!(h.hash(0, 0), u64::from(b'a'));
        assert_eq!(h.full(), u64::from(b'a'));
    }

    #[test]
    fn full_string_matches_manual() {
        let s = b"the quick brown fox jumps over the lazy dog";
        let h = PolynomialHash::new(s, BASE, MERSENNE_61);
        assert_eq!(h.full(), manual_hash(s, BASE, MERSENNE_61));
        assert_eq!(h.hash(0, s.len() - 1), manual_hash(s, BASE, MERSENNE_61));
    }

    #[test]
    fn identical_substrings_share_hash() {
        // "abc" appears at [0, 2] and [3, 5]; "ab" appears at [0, 1] and [3, 4].
        let h = PolynomialHash::new(b"abcabc", BASE, MERSENNE_61);
        assert_eq!(h.hash(0, 2), h.hash(3, 5));
        assert_eq!(h.hash(0, 1), h.hash(3, 4));
        assert!(h.equal(0, 2, 3, 5));
    }

    #[test]
    fn different_substrings_distinct_hash() {
        let h = PolynomialHash::new(b"abcdef", BASE, MERSENNE_61);
        assert_ne!(h.hash(0, 2), h.hash(1, 3)); // "abc" vs "bcd"
        assert_ne!(h.hash(0, 0), h.hash(1, 1)); // 'a' vs 'b'
        assert!(!h.equal(0, 2, 1, 3));
    }

    #[test]
    fn equal_rejects_different_lengths() {
        let h = PolynomialHash::new(b"aaaa", BASE, MERSENNE_61);
        // Even though all chars are the same, different lengths must not be equal.
        assert!(!h.equal(0, 1, 0, 2));
    }

    #[test]
    fn classic_palindrome_check() {
        // "abacaba" — check that the two "aba" substrings share a hash.
        let s = b"abacaba";
        let h = PolynomialHash::new(s, BASE, MERSENNE_61);
        assert_eq!(h.hash(0, 2), h.hash(4, 6));
        assert_ne!(h.hash(0, 2), h.hash(1, 3)); // "aba" vs "bac"
    }

    #[test]
    fn substring_invariant_across_positions() {
        // The hash of a substring depends only on its content, not its
        // location, so duplicating a fragment yields equal hashes.
        let s = b"xxhelloyyhellozz";
        let h = PolynomialHash::new(s, BASE, MERSENNE_61);
        assert_eq!(h.hash(2, 6), h.hash(9, 13));
    }

    #[test]
    fn matches_alternate_modulus() {
        // Smoke-check that a non-Mersenne prime also works.
        let s = b"polynomial-hash";
        let modulus = 1_000_000_007_u64;
        let h = PolynomialHash::new(s, BASE, modulus);
        assert_eq!(h.full(), manual_hash(s, BASE, modulus));
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_equal_substrings_match(s: Vec<u8>, a: u8, b: u8, len: u8) -> TestResult {
        if s.is_empty() || s.len() > 50 {
            return TestResult::discard();
        }
        let n = s.len();
        let li = (a as usize) % n;
        let lj = (b as usize) % n;
        let max_len = n - li.max(lj);
        if max_len == 0 {
            return TestResult::discard();
        }
        let length = (len as usize) % max_len + 1;
        let h = PolynomialHash::new(&s, BASE, MERSENNE_61);
        let r1 = li + length - 1;
        let r2 = lj + length - 1;
        let content_eq = s[li..=r1] == s[lj..=r2];
        let hash_eq = h.equal(li, r1, lj, r2);
        // Equal content must imply equal hash. The reverse direction can
        // false-positive once in roughly `modulus` queries, which is
        // negligible here.
        if content_eq && !hash_eq {
            return TestResult::failed();
        }
        if !content_eq && hash_eq {
            return TestResult::failed();
        }
        TestResult::passed()
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn prop_full_hash_matches_manual(s: Vec<u8>) -> TestResult {
        if s.len() > 50 {
            return TestResult::discard();
        }
        let h = PolynomialHash::new(&s, BASE, MERSENNE_61);
        if h.full() != manual_hash(&s, BASE, MERSENNE_61) {
            return TestResult::failed();
        }
        TestResult::passed()
    }
}
