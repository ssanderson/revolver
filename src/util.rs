pub mod itertools {
    use std::iter::Iterator;

    #[derive(Debug)]
    pub struct CartesianProduct<I, J>
        where I: Iterator
    {
        first: I,
        saved_first: Option<I::Item>,
        second: J,
        second_clone: J,
    }

    pub fn cartesian_product<I, J>(mut i: I, j: J) -> CartesianProduct<I, J>
        where I: Iterator,
              J: Clone
    {
        CartesianProduct {
            saved_first: i.next(),
            first: i,
            second_clone: j.clone(),
            second: j,
        }
    }

    impl<I, J> Iterator for CartesianProduct<I, J>
        where I: Iterator,
              I::Item: Clone,
              J: Iterator + Clone
    {
        type Item = (I::Item, J::Item);

        fn next(&mut self) -> Option<(I::Item, J::Item)> {

            // We need to get the second element first because if `second` is
            // exhausted then we need to advance the first iterator.
            let second_item = match self.second.next() {
                Some(item) => item,
                None => {
                    // Restart the second iterator.
                    self.second = self.second_clone.clone();
                    match self.second.next() {
                        Some(item) => {
                            self.saved_first = self.first.next();
                            item
                        }
                        None => return None,
                    }
                }
            };
            match self.saved_first {
                // We need to clone the first item because it will be yielded
                // multiple times.
                Some(ref first_item) => Some((first_item.clone(), second_item)),
                None => None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::itertools::cartesian_product;

    #[test]
    fn test_cartesian_product() {
        let mut it = cartesian_product(0..5, 0..5);
        for expected_i in 0..5 {
            for expected_j in 0..5 {
                let (i, j) = it.next().expect("Iterator terminated early.");
                assert_eq!(i, expected_i);
                assert_eq!(j, expected_j);
            }
        }
    }

    #[test]
    fn test_cartesian_product_empty() {
        for (_, _) in cartesian_product(0..0, 0..5) {
            panic!("Cartesian product of empty input should not yield.");
        }
        for (_, _) in cartesian_product(0..5, 0..0) {
            panic!("Cartesian product of empty input should not yield.");
        }
        for (_, _) in cartesian_product(0..0, 0..0) {
            panic!("Cartesian product of empty input should not yield.");
        }
    }
}
