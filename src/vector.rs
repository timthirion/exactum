//! Vector types for 2D and 3D geometry.

use std::ops::{Add, Neg, Sub};

/// A 2D displacement vector with integer components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Vector2<T> {
    pub x: T,
    pub y: T,
}

impl<T> Vector2<T> {
    /// Creates a new 2D vector.
    #[inline]
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl<T: Add<Output = T>> Add for Vector2<T> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }
}

impl<T: Sub<Output = T>> Sub for Vector2<T> {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }
}

impl<T: Neg<Output = T>> Neg for Vector2<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}

/// A 3D displacement vector with integer components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Vector3<T> {
    /// Creates a new 3D vector.
    #[inline]
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl<T: Add<Output = T>> Add for Vector3<T> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl<T: Sub<Output = T>> Sub for Vector3<T> {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl<T: Neg<Output = T>> Neg for Vector3<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector2_add() {
        let a = Vector2::new(1, 2);
        let b = Vector2::new(3, 4);
        assert_eq!(a + b, Vector2::new(4, 6));
    }

    #[test]
    fn vector2_neg() {
        let v = Vector2::new(3, -5);
        assert_eq!(-v, Vector2::new(-3, 5));
    }
}
