mod dok {
    use std::collections::HashMap;
    use std::ops::Index;
    // use std::ops::{Add, Sub, Mul, Index, IndexMut};

    type Coords = (u64, u64);
    type CoordMap<T> = HashMap<Coords, T>;

    const ZERO_F64: f64 = 0.0;
    const ONE_F64: f64 = 1.0;

    static ZERO_F64_REF: &'static f64 = &ZERO_F64;
    static ONE_F64_REF: &'static f64 = &ONE_F64;

    pub trait Zero {
        fn zero() -> &'static Self;
    }

    impl Zero for f64 {
        fn zero() -> &'static f64 {
            ZERO_F64_REF
        }
    }

    pub trait One {
        fn one() -> &'static Self;
    }

    impl One for f64 {
        fn one() -> &'static f64 {
            &ONE_F64_REF
        }
    }

    pub trait MatrixElem: Zero + One + Copy {}
    impl<T: Zero + One + Copy> MatrixElem for T {}

    /// A Dictionary-of-Keys Sparse Matrix
    pub struct DOKMatrix<T: 'static>
        where T: MatrixElem
    {
        pub nrows: u64,
        pub ncols: u64,
        elems: CoordMap<T>,
    }

    impl<T> DOKMatrix<T>
        where T: MatrixElem + Copy
    {
        /// Create a DOKMatrix.
        ///
        /// # Arguments
        ///
        /// * `nrows` - Number of rows in the new matrix.
        /// * `ncols` - Number of columns in the returned matrix.
        /// * `elems` - Map from indices of non-zero elements to values.
        pub fn new(nrows: u64, ncols: u64, elems: CoordMap<T>) -> Self {
            DOKMatrix {
                nrows: nrows,
                ncols: ncols,
                elems: elems,
            }
        }

        /// Create a DOKMatrix with all zero elements.
        ///
        /// # Arguments
        ///
        /// * `nrows` - Number of rows in the new matrix.
        /// * `ncols` - Number of columns in the returned matrix.
        pub fn zeros(nrows: u64, ncols: u64) -> Self {
            Self::new(nrows, ncols, CoordMap::<T>::new())
        }

        pub fn identity(size: u64) -> Self {
            let mut map = HashMap::<Coords, T>::new();
            for i in 0..size {
                map.insert((i, i), *T::one());
            }
            Self::new(size, size, map)
        }

        pub fn transposed(&self) -> Self {
            let mut map = HashMap::<Coords, T>::new();
            for (&(i, j), v) in (&self.elems).into_iter() {
                map.insert((j, i), *v);
            }
            return Self::new(self.ncols, self.nrows, map);
        }
    }

    impl<T> Index<(u64, u64)> for DOKMatrix<T>
        where T: MatrixElem
    {
        type Output = T;

        /// Get the element at coordinate (row, col).
        fn index<'a>(&'a self, (row, col): (u64, u64)) -> &'a T {
            if row >= self.nrows || col >= self.ncols {
                panic!("Out of bounds index ({row}, {col}) for sparse matrix \
                        of shape ({nrows}, {ncols})",
                       row = row,
                       col = col,
                       nrows = self.nrows,
                       ncols = self.ncols)
            }
            let elem = self.elems.get(&(row, col));
            match elem {
                None => T::zero(),
                Some(elem) => elem,
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use util::itertools::cartesian_product;

    use super::dok::DOKMatrix;
    type FloatMatrix = DOKMatrix<f64>;

    #[test]
    fn test_zeros() {
        let m = FloatMatrix::zeros(5, 5);
        for (i, j) in cartesian_product(0..5, 0..5) {
            assert_eq!(m[(i, j)], 0.0);
        }
    }

    #[test]
    fn test_manual_zeros() {
        let elems = HashMap::<(u64, u64), f64>::new();
        let m = FloatMatrix::new(5, 5, elems);
        for (i, j) in cartesian_product(0..5, 0..5) {
            assert_eq!(m[(i, j)], 0.0);
        }
    }

    fn check_identity(m: DOKMatrix<f64>) {
        for (i, j) in cartesian_product(0..m.nrows, 0..m.ncols) {
            if i == j {
                assert_eq!(m[(i, j)], 1.0);
            } else {
                assert_eq!(m[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn test_identity() {
        check_identity(FloatMatrix::identity(5));
    }

    #[test]
    fn test_manual_identity() {
        let mut elems = HashMap::<(u64, u64), f64>::new();
        for i in 0..5 {
            elems.insert((i, i), 1.0);
        }
        check_identity(FloatMatrix::new(5, 5, elems));
    }

    #[test]
    fn test_transpose() {
        let mut m = HashMap::new();
        let keys = [(0, 1), (0, 2), (0, 3), (2, 2)];
        let values = [1.0, 2.0, 3.0, -4.0];
        for (k, v) in keys.iter().zip(values.iter()) {
            m.insert(*k, *v);
        }

        let mat = FloatMatrix::new(4, 8, m);
        assert_eq!(mat[(0, 1)], 1.0);
        assert_eq!(mat[(0, 2)], 2.0);
        assert_eq!(mat[(0, 3)], 3.0);
        assert_eq!(mat[(2, 2)], -4.0);

        let transposed = mat.transposed();
        assert_eq!(transposed[(1, 0)], 1.0);
        assert_eq!(transposed[(2, 0)], 2.0);
        assert_eq!(transposed[(3, 0)], 3.0);
        assert_eq!(transposed[(2, 2)], -4.0);
    }
}
