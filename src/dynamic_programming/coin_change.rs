//! Coin change: minimum number of coins to make `amount` from unlimited
//! supplies of given denominations. O(amount · |coins|).

/// Returns the minimum number of coins needed to sum to `amount`, or `None`
/// if impossible.
pub fn coin_change(coins: &[u32], amount: u32) -> Option<u32> {
    let amount = amount as usize;
    let mut dp = vec![u32::MAX; amount + 1];
    dp[0] = 0;
    for cap in 1..=amount {
        for &c in coins {
            let c = c as usize;
            if c <= cap && dp[cap - c] != u32::MAX {
                let candidate = dp[cap - c] + 1;
                if candidate < dp[cap] {
                    dp[cap] = candidate;
                }
            }
        }
    }
    if dp[amount] == u32::MAX {
        None
    } else {
        Some(dp[amount])
    }
}

#[cfg(test)]
mod tests {
    use super::coin_change;

    #[test]
    fn classic() {
        assert_eq!(coin_change(&[1, 2, 5], 11), Some(3));
    }

    #[test]
    fn impossible() {
        assert_eq!(coin_change(&[2], 3), None);
    }

    #[test]
    fn zero_amount_zero_coins() {
        assert_eq!(coin_change(&[1, 2, 5], 0), Some(0));
    }

    #[test]
    fn single_coin_exact() {
        assert_eq!(coin_change(&[7], 14), Some(2));
    }
}
