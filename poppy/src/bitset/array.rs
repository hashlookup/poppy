use core::f32;

/// Array based bitset implementation.
/// N gives the size in Bytes of the bucket
#[derive(Debug, Clone)]
pub struct BitSet<const N: usize>([u8; N]);

impl<const N: usize> Default for BitSet<N> {
    fn default() -> Self {
        Self([0u8; N])
    }
}

impl<const N: usize> BitSet<N> {
    /// Creates a new bitset
    pub const fn new() -> Self {
        Self([0; N])
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

    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.0.as_mut_slice()
    }

    /// Clears out a bitset (set all the bits to 0)
    #[inline]
    pub fn clear(&mut self) {
        self.0.iter_mut().for_each(|b| *b = 0);
    }

    /// Returns the size in bits of the bitset
    #[inline]
    pub const fn bit_len(&self) -> usize {
        N * 8
    }

    /// Returns the size in bytes of the bitset
    #[inline]
    pub const fn byte_len(&self) -> usize {
        N
    }

    /// Counts bits to one in the current set
    #[inline]
    pub fn count_ones(&self) -> usize {
        self.0.iter().map(|b| b.count_ones() as usize).sum()
    }

    /// Make the union of the current set with another one
    #[inline]
    pub fn union(&mut self, other: &Self) {
        (0..self.0.len()).for_each(|i| self.0[i] |= other.0[i])
    }

    /// Make the intersection of the current set with another one
    #[inline]
    pub fn intersection(&mut self, other: &Self) {
        (0..self.0.len()).for_each(|i| self.0[i] &= other.0[i])
    }

    /// Count bits to one in common between two sets
    #[inline]
    pub const fn count_ones_in_common(&self, other: &Self) -> usize {
        let mut count = 0;
        let mut i = 0;
        loop {
            if i == self.0.len() {
                break;
            }
            count += (self.0[i] & other.0[i]).count_ones() as usize;
            i += 1;
        }
        count
    }

    /// Counts bits to zero in the current set
    #[inline]
    pub fn count_zeros(&self) -> usize {
        self.0.iter().map(|b| b.count_zeros() as usize).sum()
    }

    /// Returns the size in bytes of the bitset
    #[inline]
    pub const fn byte_size() -> usize {
        N
    }

    /// Returns the size in bytes of the bitset
    #[inline]
    pub const fn bit_size() -> usize {
        N * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_bitset() {
        let bitset: BitSet<2> = BitSet::new();
        assert_eq!(bitset.as_slice(), &[0u8; 2]);
    }

    #[test]
    fn test_default_bitset() {
        let bitset: BitSet<2> = BitSet::default();
        assert_eq!(bitset.as_slice(), &[0u8; 2]);
    }

    #[test]
    fn test_set_nth_bit() {
        let mut bitset: BitSet<2> = BitSet::new();
        assert_eq!(bitset.set_nth_bit(0), false); // Bit was 0, now set to 1
        assert_eq!(bitset.get_nth_bit(0), true);
    }

    #[test]
    fn test_unset_nth_bit() {
        let mut bitset: BitSet<2> = BitSet::new();
        bitset.set_nth_bit(0); // Set bit to 1 first
        assert_eq!(bitset.unset_nth_bit(0), true); // Bit was 1, now set to 0
        assert_eq!(bitset.get_nth_bit(0), false);
    }

    #[test]
    fn test_get_nth_bit() {
        let mut bitset: BitSet<2> = BitSet::new();
        bitset.set_nth_bit(0);
        assert_eq!(bitset.get_nth_bit(0), true);
        assert_eq!(bitset.get_nth_bit(1), false);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_set_nth_bit_out_of_bounds() {
        let mut bitset: BitSet<2> = BitSet::new();
        bitset.set_nth_bit(16); // Assuming N=2, so 16 bits is out of bounds
    }

    #[test]
    fn test_clear() {
        let mut bitset: BitSet<2> = BitSet::new();
        bitset.set_nth_bit(0);
        bitset.set_nth_bit(1);
        bitset.clear();
        assert_eq!(bitset.as_slice(), &[0u8; 2]);
    }

    #[test]
    fn test_count_ones() {
        let mut bitset: BitSet<2> = BitSet::new();
        assert_eq!(bitset.count_ones(), 0);
        bitset.set_nth_bit(0);
        bitset.set_nth_bit(1);
        assert_eq!(bitset.count_ones(), 2);
    }

    #[test]
    fn test_count_zeros() {
        let mut bitset: BitSet<2> = BitSet::new();
        assert_eq!(bitset.count_zeros(), 16); // 2 bytes = 16 bits, all zeros initially
        bitset.set_nth_bit(0);
        assert_eq!(bitset.count_zeros(), 15);
    }

    #[test]
    fn test_union() {
        let mut bitset1: BitSet<2> = BitSet::new();
        let mut bitset2: BitSet<2> = BitSet::new();
        bitset1.set_nth_bit(0);
        bitset2.set_nth_bit(1);
        bitset1.union(&bitset2);
        assert_eq!(bitset1.get_nth_bit(0), true);
        assert_eq!(bitset1.get_nth_bit(1), true);
    }

    #[test]
    fn test_intersection() {
        let mut bitset1: BitSet<2> = BitSet::new();
        let mut bitset2: BitSet<2> = BitSet::new();
        bitset1.set_nth_bit(0);
        bitset1.set_nth_bit(1);
        bitset2.set_nth_bit(1);
        bitset1.intersection(&bitset2);
        assert_eq!(bitset1.get_nth_bit(0), false);
        assert_eq!(bitset1.get_nth_bit(1), true);
    }

    #[test]
    fn test_count_ones_in_common() {
        let mut bitset1: BitSet<2> = BitSet::new();
        let mut bitset2: BitSet<2> = BitSet::new();
        bitset1.set_nth_bit(0);
        bitset1.set_nth_bit(1);
        bitset2.set_nth_bit(1);
        assert_eq!(bitset1.count_ones_in_common(&bitset2), 1);
    }

    #[test]
    fn test_bit_len() {
        let bitset: BitSet<2> = BitSet::new();
        assert_eq!(bitset.bit_len(), 16); // 2 bytes = 16 bits
    }

    #[test]
    fn test_byte_len() {
        let bitset: BitSet<2> = BitSet::new();
        assert_eq!(bitset.byte_len(), 2);
    }

    #[test]
    fn test_byte_size() {
        assert_eq!(BitSet::<2>::byte_size(), 2);
    }

    #[test]
    fn test_bit_size() {
        assert_eq!(BitSet::<2>::bit_size(), 16);
    }
}
