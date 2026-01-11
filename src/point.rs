//! Point types for 2D and 3D geometry.

use std::ops::{Add, Sub};

use crate::vector::{Vector2, Vector3};

/// A 2D point with integer coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point2<T> {
    pub x: T,
    pub y: T,
}

impl<T> Point2<T> {
    /// Creates a new 2D point.
    #[inline]
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl<T: Sub<Output = T> + Copy> Sub for Point2<T> {
    type Output = Vector2<T>;

    #[inline]
    fn sub(self, other: Self) -> Vector2<T> {
        Vector2::new(self.x - other.x, self.y - other.y)
    }
}

impl<T: Add<Output = T>> Add<Vector2<T>> for Point2<T> {
    type Output = Self;

    #[inline]
    fn add(self, v: Vector2<T>) -> Self {
        Self::new(self.x + v.x, self.y + v.y)
    }
}

/// A 3D point with integer coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Point3<T> {
    /// Creates a new 3D point.
    #[inline]
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl<T: Sub<Output = T> + Copy> Sub for Point3<T> {
    type Output = Vector3<T>;

    #[inline]
    fn sub(self, other: Self) -> Vector3<T> {
        Vector3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl<T: Add<Output = T>> Add<Vector3<T>> for Point3<T> {
    type Output = Self;

    #[inline]
    fn add(self, v: Vector3<T>) -> Self {
        Self::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point2_sub() {
        let a = Point2::new(5, 3);
        let b = Point2::new(2, 1);
        let v = a - b;
        assert_eq!(v, Vector2::new(3, 2));
    }

    #[test]
    fn point2_add_vector() {
        let p = Point2::new(1, 2);
        let v = Vector2::new(3, 4);
        assert_eq!(p + v, Point2::new(4, 6));
    }
}
