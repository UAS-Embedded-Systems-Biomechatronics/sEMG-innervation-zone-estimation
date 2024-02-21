/* 
* Copyright 2023 Malte Mechtenberg
* 
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

use num_traits::FromPrimitive;

#[derive(Debug)]
pub struct LineX<T> {
    pub m: T,
    pub b: T,

    pub is_positive: bool,
}

impl<T> LineX<T>
where
    T: std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::AddAssign
        + std::ops::Sub<Output = T>
        + std::cmp::PartialOrd<T>
        + num_traits::Float
        //+ std::cmp::PartialEq
        + num_traits::cast::FromPrimitive
        + std::convert::Into<f64>
        + Copy,
    f64: From<T>,
{
    pub fn from_points(x_vec: &[T], y_vec: &[T]) -> Vec<LineX<T>> {
        assert_eq!(
            x_vec.len(), y_vec.len(),
            "x_vec and y_vec have to be the same length."
        );
        assert!(x_vec.len() >= 2, "at least two data points needed.");

        let mut lines: Vec<LineX<T>> = Vec::new();
        lines.reserve(x_vec.len());

        for idx in 0..(x_vec.len() - 1) {
            let line_x: LineX<T> =
                LineX::new(x_vec[idx], y_vec[idx], x_vec[idx + 1], y_vec[idx + 1]);
            lines.push(line_x);
        }

        return lines;
    }

    pub fn new(x0: T, y0: T, x1: T, y1: T) -> LineX<T> {
        let m: T = (y0 - y1) / (x0 - x1);
        let b: T = y0 - (x0 * m);

        LineX {
            m,
            b,
            is_positive: m >= FromPrimitive::from_usize(0).unwrap(),
        }
    }

    pub fn get_intersection(&self, other: &LineX<T>) -> Option<LineIntersection<T>> {
        if self.m == other.m {
            return None;
        }

        let x = (other.b - self.b) / (self.m - other.m);
        let y = self.m * x + self.b;

        if x.is_nan() || y.is_nan() {
            return None;
        }

        Some(LineIntersection { x, y })
    }
}

#[derive(Debug, Copy, Clone)]
pub struct LineIntersection<T> {
    pub x: T,
    pub y: T,
}

impl<T> std::ops::DivAssign<T> for LineIntersection<T>
where
    T: Copy + std::ops::Div<Output = T>,
{
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

impl<T> std::ops::Div<T> for LineIntersection<T>
where
    T: Copy + std::ops::Div<Output = T>,
{
    type Output = LineIntersection<T>;
    fn div(self, rhs: T) -> LineIntersection<T> {
        return LineIntersection {
            x: self.x / rhs,
            y: self.y / rhs,
        };
    }
}

impl<T> std::ops::AddAssign<LineIntersection<T>> for LineIntersection<T>
where
    T: Copy
        + std::ops::Div<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::AddAssign
        + num_traits::cast::FromPrimitive,
{
    fn add_assign(&mut self, rhs: LineIntersection<T>) {
        self.x = self.x + rhs.x;
        self.y = self.y + rhs.y;
    }
}

impl<T> std::ops::Sub<LineIntersection<T>> for LineIntersection<T>
where
    T: Copy + std::ops::Sub<Output = T>,
{
    type Output = LineIntersection<T>;
    fn sub(self, _rhs: LineIntersection<T>) -> LineIntersection<T> {
        let x = self.x - _rhs.x;
        let y = self.y - _rhs.y;

        LineIntersection { x, y }
    }
}

impl<T> std::ops::Add<LineIntersection<T>> for LineIntersection<T>
where
    T: Copy
        + std::ops::Div<Output = T>
        + std::ops::Add<Output = T>
        + num_traits::cast::FromPrimitive,
{
    type Output = LineIntersection<T>;
    fn add(self, _rhs: LineIntersection<T>) -> LineIntersection<T> {
        let x = self.x + _rhs.x;
        let y = self.y + _rhs.y;

        LineIntersection { x, y }
    }
}

impl<T> LineIntersection<T>
where
    T: Copy
        + std::ops::Div<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::AddAssign
        + std::ops::DivAssign
        + num_traits::cast::FromPrimitive,
{
    pub fn as_vec(&self) -> Vec<T> {
        let mut v = Vec::new();
        v.push(self.x);
        v.push(self.y);

        return v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_lines() {
        //let x_s = &[1.0, 0.0, 1.0];
        let lines = LineX::from_points(&[1.0, 0.0, 1.0], &[1.0, 0.0, -1.0]);

        assert_eq!(lines.len(), 2);
        assert!(lines[0].is_positive);
        assert!(!lines[1].is_positive);

        let intersect = lines[0].get_intersection(&lines[1]).unwrap();
        assert_eq!(intersect.x, 0.0);
        assert_eq!(intersect.y, 0.0);

        let lines = LineX::from_points(&[1.0, 0.0, 1.0], &[2.0, 1.0, 0.0]);
        let intersect = lines[0].get_intersection(&lines[1]).unwrap();

        assert_eq!(intersect.x, 0.0);
        assert_eq!(intersect.y, 1.0);

        let lines = LineX::from_points(&[2.0, 1.0, 2.0], &[-2.0, -1.0, 0.0]);
        let intersect = lines[0].get_intersection(&lines[1]).unwrap();

        assert!(!lines[0].is_positive);
        assert!(lines[1].is_positive);
        assert_eq!(intersect.x, 1.0);
        assert_eq!(intersect.y, -1.0);
    }

    #[test]
    fn operators() {
        let mut a = LineIntersection { x: 1.1, y: 1.2 };
        let b = LineIntersection { x: 1.5, y: 1.2 };

        let mut c = LineIntersection { x: 0.0, y: 0.0 };

        c += a;

        assert_eq!(c.x, 1.1);
        assert_eq!(c.y, 1.2);

        a += b;

        assert_eq!(a.x, 2.6);
        assert_eq!(a.y, 2.4);

        a /= 4.0;

        assert_eq!(a.x, 0.65);
        assert_eq!(a.y, 0.6);
    }
}
