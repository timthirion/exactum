//! Exact rational number type for computational geometry.
//!
//! Used for intersection points and circumcenters where integer coordinates
//! are not sufficient.

use std::cmp::Ordering;
use std::ops::{Add, Mul, Neg, Sub};

/// A rational number represented as numerator/denominator.
///
/// Used for exact coordinates (e.g., intersection points, circumcenters).
/// The denominator is always positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rational {
    pub num: i128,
    pub denom: i128,
}

impl Rational {
    /// Creates a new rational number.
    ///
    /// The denominator must be positive. No automatic reduction is performed.
    pub fn new(num: i128, denom: i128) -> Self {
        debug_assert!(denom > 0, "Denominator must be positive");
        Self { num, denom }
    }

    /// Creates a rational from an integer.
    pub fn from_int(n: i64) -> Self {
        Self {
            num: n as i128,
            denom: 1,
        }
    }

    /// Returns the sign of this rational: -1, 0, or 1.
    pub fn signum(self) -> i128 {
        self.num.signum()
    }

    /// Returns true if this rational is negative.
    pub fn is_negative(self) -> bool {
        self.num < 0
    }

    /// Returns true if this rational is non-negative (>= 0).
    pub fn is_non_negative(self) -> bool {
        self.num >= 0
    }

    /// Returns true if this rational is zero.
    pub fn is_zero(self) -> bool {
        self.num == 0
    }

    /// Converts to f64 for visualization purposes.
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.denom as f64
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        // a/b vs c/d => a*d vs c*b (since denominators are positive)
        let lhs = self.num * other.denom;
        let rhs = other.num * self.denom;
        lhs.cmp(&rhs)
    }
}

impl Neg for Rational {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            num: -self.num,
            denom: self.denom,
        }
    }
}

impl Add for Rational {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // a/b + c/d = (ad + bc) / bd
        Self {
            num: self.num * other.denom + other.num * self.denom,
            denom: self.denom * other.denom,
        }
    }
}

impl Sub for Rational {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        // a/b - c/d = (ad - bc) / bd
        Self {
            num: self.num * other.denom - other.num * self.denom,
            denom: self.denom * other.denom,
        }
    }
}

impl Mul for Rational {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            num: self.num * other.num,
            denom: self.denom * other.denom,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_arithmetic() {
        let a = Rational::new(1, 2); // 1/2
        let b = Rational::new(1, 3); // 1/3

        // 1/2 + 1/3 = 5/6
        let sum = a + b;
        assert_eq!(sum.num * 6, 5 * sum.denom);

        // 1/2 - 1/3 = 1/6
        let diff = a - b;
        assert_eq!(diff.num * 6, 1 * diff.denom);

        // 1/2 * 1/3 = 1/6
        let prod = a * b;
        assert_eq!(prod.num * 6, 1 * prod.denom);

        // -1/2
        let neg = -a;
        assert_eq!(neg.num, -1);
        assert_eq!(neg.denom, 2);
    }

    #[test]
    fn rational_ordering() {
        let half = Rational::new(1, 2);
        let third = Rational::new(1, 3);
        let two_thirds = Rational::new(2, 3);

        assert!(third < half);
        assert!(half < two_thirds);
        assert!(third < two_thirds);

        // Same value different representation
        let three_sixths = Rational::new(3, 6);
        assert_eq!(half.cmp(&three_sixths), Ordering::Equal);
    }

    #[test]
    fn rational_negative_ordering() {
        let neg_half = Rational::new(-1, 2);
        let half = Rational::new(1, 2);
        let zero = Rational::new(0, 1);

        assert!(neg_half < zero);
        assert!(zero < half);
        assert!(neg_half < half);
    }

    #[test]
    fn rational_from_int() {
        let five = Rational::from_int(5);
        assert_eq!(five.num, 5);
        assert_eq!(five.denom, 1);
        assert_eq!(five.to_f64(), 5.0);
    }
}
