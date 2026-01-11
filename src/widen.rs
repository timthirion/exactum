//! Traits for overflow-safe integer widening.

use std::cmp::Ordering;

/// Types that can widen to avoid overflow in multiplication.
///
/// When computing determinants, multiplying two `n`-bit integers produces
/// a `2n`-bit result. This trait maps each integer type to a wider type
/// that can hold the product without overflow.
///
/// # Implementations
///
/// - `i32` widens to `i64`
/// - `i64` widens to `i128`
pub trait Widen: Copy + Ord {
    /// A wider type that can hold products of `Self` values.
    type Wide: Wide<Narrow = Self>;

    /// Convert to the wide type.
    fn to_wide(self) -> Self::Wide;
}

/// The wide side of a widening relationship.
///
/// This trait is implemented by types that serve as the "wide" type
/// for some narrower integer type.
pub trait Wide:
    Clone
    + Ord
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
{
    /// The narrow type this is widened from.
    type Narrow: Widen<Wide = Self>;

    /// Returns the zero value.
    fn zero() -> Self;

    /// Compare to zero and return ordering.
    fn sign(&self) -> Ordering;
}

// i32 → i64

impl Widen for i32 {
    type Wide = i64;

    #[inline]
    fn to_wide(self) -> i64 {
        self as i64
    }
}

impl Wide for i64 {
    type Narrow = i32;

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn sign(&self) -> Ordering {
        self.cmp(&0)
    }
}

// i64 → i128

impl Widen for i64 {
    type Wide = i128;

    #[inline]
    fn to_wide(self) -> i128 {
        self as i128
    }
}

impl Wide for i128 {
    type Narrow = i64;

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn sign(&self) -> Ordering {
        self.cmp(&0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widen_i32() {
        let x: i32 = 100_000;
        let wide: i64 = x.to_wide();
        assert_eq!(wide, 100_000_i64);
    }

    #[test]
    fn widen_i64() {
        let x: i64 = 1_000_000_000_000;
        let wide: i128 = x.to_wide();
        assert_eq!(wide, 1_000_000_000_000_i128);
    }

    #[test]
    fn wide_sign() {
        assert_eq!((-5_i128).sign(), Ordering::Less);
        assert_eq!(0_i128.sign(), Ordering::Equal);
        assert_eq!(5_i128.sign(), Ordering::Greater);
    }
}
