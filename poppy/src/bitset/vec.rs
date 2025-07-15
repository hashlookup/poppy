#[derive(Debug, Clone)]
pub struct VecBitSet(Vec<u8>);

#[inline(always)]
pub(crate) fn byte_cap_from_bit_cap(bit_cap: usize) -> usize {
    ((bit_cap as f64) / 8.0).ceil() as usize
}

impl VecBitSet {
    /// Creates a new bitset with a given capacity
    pub fn with_bit_capacity(capacity: usize) -> Self {
        let byte_size = byte_cap_from_bit_cap(capacity);
        Self(vec![0; byte_size])
    }

    /// Returns true if bitset is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.0.as_mut_slice()
    }

    /// Set the value of the nth bit (set to 1)
    ///
    /// # Panics
    ///
    /// Panic on index out of bound
    #[inline(always)]
    pub fn set_nth_bit(&mut self, index: usize) -> bool {
        self.set_value_nth_bit(index, true)
    }

    /// Unset the value of the nth bit (set to 0)
    ///
    /// # Panics
    ///
    /// Panic on index out of bound
    #[inline(always)]
    pub fn unset_nth_bit(&mut self, index: usize) -> bool {
        self.set_value_nth_bit(index, false)
    }

    #[inline(always)]
    fn set_value_nth_bit(&mut self, index: usize, value: bool) -> bool {
        let iblock = index / 8;
        // equivalent to index % 8
        let mask = (value as u8) << (index & 7);
        let old = self.0[iblock] & mask == mask;
        if value {
            self.0[iblock] |= mask;
        } else {
            self.0[iblock] &= mask;
        }
        old
    }

    /// Get the value of the nth bit
    ///
    /// # Panics
    ///
    /// Panic on index out of bound
    #[inline(always)]
    pub fn get_nth_bit(&self, index: usize) -> bool {
        let iblock = index / 8;
        // equivalent to index % 8
        let mask = 1u8 << (index & 7);
        self.0[iblock] & mask == mask
    }

    /// Clears out a bitset (set all the bits to 0)
    #[inline]
    pub fn clear(&mut self) {
        self.0.iter_mut().for_each(|b| *b = 0);
    }

    /// Returns the size in bits of the bitset
    #[inline]
    pub fn bit_len(&self) -> usize {
        self.0.len() * 8
    }

    /// Returns the size in bytes of the bitset
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.0.len()
    }

    /// Make the union of the current set with another one
    ///
    /// # Panics
    ///
    /// This function panics if the two sets have different lengths
    #[inline]
    pub fn union(&mut self, other: &Self) {
        assert_eq!(
            self.byte_len(),
            other.byte_len(),
            "bitsets must have same lengths"
        );
        (0..self.0.len()).for_each(|i| self.0[i] |= other.0[i])
    }

    /// Make the intersection of the current set with another one
    ///
    /// # Panics
    ///
    /// This function panics if the two sets have different lengths
    #[inline]
    pub fn intersection(&mut self, other: &Self) {
        assert_eq!(
            self.byte_len(),
            other.byte_len(),
            "bitsets must have same lengths"
        );
        (0..self.0.len()).for_each(|i| self.0[i] &= other.0[i])
    }

    /// Count bits to one in common between two sets
    ///
    /// # Panics
    ///
    /// This function panics if the two sets have different lengths
    #[inline]
    pub fn count_ones_in_common(&self, other: &Self) -> usize {
        assert_eq!(
            self.byte_len(),
            other.byte_len(),
            "bitsets must have same lengths"
        );
        (0..self.0.len())
            .map(|i| (self.0[i] & other.0[i]).count_ones() as usize)
            .sum()
    }

    /// Counts bits to one in the current set
    #[inline]
    pub fn count_ones(&self) -> usize {
        self.0.iter().map(|b| b.count_ones() as usize).sum()
    }

    /// Counts bits to zero in the current set
    #[inline]
    pub fn count_zeros(&self) -> usize {
        self.0.iter().map(|b| b.count_zeros() as usize).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_bit_capacity() {
        let capacity = 10;
        let bitset = VecBitSet::with_bit_capacity(capacity);
        assert_eq!(bitset.bit_len(), 16); // because of byte alignment, with capacity 10 it's 16 bits (2 bytes)
        assert_eq!(bitset.byte_len(), 2);
    }

    #[test]
    fn test_is_empty() {
        let bitset = VecBitSet::with_bit_capacity(0);
        assert!(bitset.is_empty());
    }

    #[test]
    fn test_set_nth_bit() {
        let mut bitset = VecBitSet::with_bit_capacity(8);
        assert!(!bitset.get_nth_bit(0));
        bitset.set_nth_bit(0);
        assert!(bitset.get_nth_bit(0));
    }

    #[test]
    fn test_unset_nth_bit() {
        let mut bitset = VecBitSet::with_bit_capacity(8);
        bitset.set_nth_bit(0);
        assert!(bitset.get_nth_bit(0));
        bitset.unset_nth_bit(0);
        assert!(!bitset.get_nth_bit(0));
    }

    #[test]
    fn test_clear() {
        let mut bitset = VecBitSet::with_bit_capacity(8);
        bitset.set_nth_bit(0);
        bitset.set_nth_bit(1);
        bitset.clear();
        assert_eq!(bitset.count_ones(), 0);
    }

    #[test]
    fn test_union() {
        let mut bitset1 = VecBitSet::with_bit_capacity(8);
        let mut bitset2 = VecBitSet::with_bit_capacity(8);
        bitset1.set_nth_bit(0);
        bitset2.set_nth_bit(1);
        bitset1.union(&bitset2);
        assert!(bitset1.get_nth_bit(0));
        assert!(bitset1.get_nth_bit(1));
    }

    #[test]
    fn test_intersection() {
        let mut bitset1 = VecBitSet::with_bit_capacity(8);
        let mut bitset2 = VecBitSet::with_bit_capacity(8);
        bitset1.set_nth_bit(0);
        bitset1.set_nth_bit(1);
        bitset2.set_nth_bit(1);
        bitset2.set_nth_bit(2);
        bitset1.intersection(&bitset2);
        assert!(!bitset1.get_nth_bit(0));
        assert!(bitset1.get_nth_bit(1));
        assert!(!bitset1.get_nth_bit(2));
    }

    #[test]
    fn test_count_ones_in_common() {
        let mut bitset1 = VecBitSet::with_bit_capacity(8);
        let mut bitset2 = VecBitSet::with_bit_capacity(8);
        bitset1.set_nth_bit(0);
        bitset1.set_nth_bit(1);
        bitset2.set_nth_bit(1);
        bitset2.set_nth_bit(2);
        assert_eq!(bitset1.count_ones_in_common(&bitset2), 1);
    }

    #[test]
    fn test_count_ones() {
        let mut bitset = VecBitSet::with_bit_capacity(8);
        bitset.set_nth_bit(0);
        bitset.set_nth_bit(1);
        assert_eq!(bitset.count_ones(), 2);
    }

    #[test]
    fn test_count_zeros() {
        let mut bitset = VecBitSet::with_bit_capacity(8);
        bitset.set_nth_bit(0);
        bitset.set_nth_bit(1);
        assert_eq!(bitset.count_zeros(), 6);
    }

    #[test]
    #[should_panic]
    fn test_out_of_bounds_access() {
        let bitset = VecBitSet::with_bit_capacity(8);
        // should panic but handled in the test if it does.
        bitset.get_nth_bit(8);
    }
}
