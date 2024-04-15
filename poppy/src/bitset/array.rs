/// Array based bitset implementation.
/// N gives the size in Bytes of the bucket
#[derive(Debug, Clone)]
pub struct BitSet<const N: usize>([u8; N]);

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
        self.0[iblock] |= mask;
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
