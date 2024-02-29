use std::{io, slice};

use thiserror::Error;

use crate::ahash::BloomAHasher;

// N gives the size in Bytes of the bucket
#[derive(Debug, Clone)]
pub struct BitSet<const N: usize>([u8; N]);

impl<const N: usize> BitSet<N> {
    pub const fn new() -> Self {
        Self([0; N])
    }

    #[inline(always)]
    fn set_nth_bit(&mut self, index: usize) -> bool {
        let iblock = index / 8;
        // equivalent to index % 8
        let mask = 1u8 << (index & 7);
        let old = self.0[iblock] & mask == mask;
        self.0[iblock] |= mask;
        old
    }

    #[inline(always)]
    fn get_nth_bit(&self, index: usize) -> bool {
        let iblock = index / 8;
        // equivalent to index % 8
        let mask = 1u8 << (index & 7);
        self.0[iblock] & mask == mask
    }

    #[inline]
    pub fn clear(&mut self) {
        self.0.iter_mut().for_each(|b| *b = 0);
    }

    #[inline]
    pub const fn bit_len(&self) -> usize {
        N * 8
    }

    #[inline]
    pub const fn bit_size() -> usize {
        N * 8
    }

    #[inline]
    pub fn count_ones(&self) -> usize {
        self.0.iter().map(|b| b.count_ones() as usize).sum()
    }

    #[inline]
    pub fn union(&mut self, other: &Self) {
        (0..self.0.len()).for_each(|i| self.0[i] |= other.0[i])
    }

    #[inline]
    pub fn intersection(&mut self, other: &Self) {
        (0..self.0.len()).for_each(|i| self.0[i] &= other.0[i])
    }

    #[inline]
    pub fn count_ones_in_common(&self, other: &Self) -> usize {
        (0..self.0.len())
            .map(|i| (self.0[i] & other.0[i]).count_ones() as usize)
            .sum()
    }

    #[inline]
    pub fn count_zeros(&self) -> usize {
        self.0.iter().map(|b| b.count_zeros() as usize).sum()
    }

    #[inline]
    pub const fn byte_size() -> usize {
        N
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct IndexIterator<const M: u64> {
    h1: u64,
    h2: u64,
    i: u64,
    count: u64,
}

impl<const M: u64> IndexIterator<M> {
    fn new(count: u64) -> Self {
        Self {
            count,
            ..Default::default()
        }
    }

    #[inline(always)]
    fn bucket_hash(&self) -> u64 {
        // we use xor shift to pseudo randomize bucket hash
        // decrease collisions on h % m
        let mut xs = self.h1;
        xs ^= xs.wrapping_shl(13);
        xs ^= xs.wrapping_shr(17);
        xs ^= xs.wrapping_shl(5);
        xs
    }

    #[inline(always)]
    fn init_with_data<S: AsRef<[u8]>>(mut self, data: S) -> Self {
        self.h1 = BloomAHasher::digest(&data);
        self.h2 = 0;
        self.i = 0;
        self
    }
}

impl<const M: u64> Iterator for IndexIterator<M> {
    type Item = u64;

    #[inline]
    // implements double hashing technique
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.count {
            if self.i > 0 {
                // we init h2 only if we passed already one iteration
                if self.i == 1 {
                    self.h2 = BloomAHasher::digest(self.h1.to_be_bytes());
                }
                self.h1 = self.h1.wrapping_add(self.h2);
                self.h2 = self.h2.wrapping_add(self.i);
            }
            self.i = self.i.wrapping_add(1);

            // shortcut if we are a power of 2
            if M.is_power_of_two() {
                return Some(self.h1 & (M - 1));
            }
            return Some(self.h1 % M);
        }
        None
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    IoError(#[from] io::Error),
    #[error("invalid version {0}")]
    InvalidVersion(u8),
    #[error("union error: {0}")]
    Merge(String),
    #[error("too many entries, false positive rate cannot be met")]
    TooManyEntries,
}

const BUCKET_SIZE: usize = 4096;
const BIT_SET_MOD: u64 = (BUCKET_SIZE * 8) as u64;
type Bucket = BitSet<BUCKET_SIZE>;

#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// the maximum of entries the filter can contains without
    /// impacting the desired false positive probability
    capacity: u64,
    /// desired false positive probability
    fpp: f64,
    /// number of hash functions
    n_hash_buck: u64,
    /// estimated number of elements in the filter
    count: u64,
    /// buckets of small but optimized for speed bloom filters
    buckets: Vec<Bucket>,
    /// arbitrary data that we can attach to the filter
    pub data: Vec<u8>,
}

impl BloomFilter {
    pub fn with_capacity(cap: u64, proba: f64) -> Self {
        // the capacity per bucket given a fpp
        let bucket_cap = crate::cap_from_bit_size(Bucket::bit_size() as u64, proba);
        let bucket_bit_size = Bucket::bit_size() as u64;
        // the number of buckets we need to store all data
        let n_bucket = (cap as f64 / bucket_cap as f64).ceil() as u64;

        // number of hash func per bucket
        let n_hash_buck: u64 = crate::k(bucket_bit_size, bucket_cap);

        Self {
            buckets: vec![Bucket::new(); n_bucket as usize],
            capacity: cap,
            fpp: proba,
            n_hash_buck,
            count: 0,
            data: vec![],
        }
    }

    // moving this as a structure member does not improve perfs
    #[inline(always)]
    fn index_iter(&self) -> IndexIterator<BIT_SET_MOD> {
        IndexIterator::new(self.n_hash_buck)
    }

    #[inline(always)]
    pub fn is_optimal(&self) -> bool {
        self.buckets.len().is_power_of_two()
    }

    /// clears out the bloom filter
    #[inline(always)]
    pub fn clear(&mut self) {
        self.buckets.iter_mut().for_each(|bucket| bucket.clear());
        self.count = 0;
    }

    #[inline]
    pub fn insert_bytes<D: AsRef<[u8]>>(&mut self, data: D) -> Result<bool, Error> {
        let mut new = false;
        let it = self.index_iter().init_with_data(data);

        let h = it.bucket_hash();
        let ibucket = {
            if self.is_optimal() {
                h & (self.buckets.len() as u64 - 1)
            } else {
                h % self.buckets.len() as u64
            }
        };

        let bucket = self
            .buckets
            .get_mut(ibucket as usize)
            .expect("ibucket out of bound");

        let reached_cap = self.count == self.capacity;

        for ibit in it {
            if reached_cap && !bucket.get_nth_bit(ibit as usize) {
                return Err(Error::TooManyEntries);
            }
            if !bucket.set_nth_bit(ibit as usize) {
                new = true
            }
            debug_assert!(bucket.get_nth_bit(ibit as usize));
        }

        if new {
            self.count += 1;
        }

        Ok(new)
    }

    #[inline(always)]
    pub fn insert<S: Sized>(&mut self, value: S) -> Result<bool, Error> {
        self.insert_bytes(unsafe {
            slice::from_raw_parts(&value as *const S as *const u8, core::mem::size_of::<S>())
        })
    }

    #[inline]
    pub fn contains_bytes<D: AsRef<[u8]>>(&self, data: D) -> bool {
        let it = self.index_iter().init_with_data(data);

        let h = it.bucket_hash();

        let ibucket = {
            if self.is_optimal() {
                h & (self.buckets.len() as u64 - 1)
            } else {
                h % self.buckets.len() as u64
            }
        };
        let bucket = self
            .buckets
            .get(ibucket as usize)
            .expect("ibucket out of bound");

        for ibit in it {
            if !bucket.get_nth_bit(ibit as usize) {
                return false;
            }
        }

        true
    }

    #[inline(always)]
    pub fn contains<S: Sized>(&mut self, value: S) -> bool {
        self.contains_bytes(unsafe {
            slice::from_raw_parts(&value as *const S as *const u8, core::mem::size_of::<S>())
        })
    }

    /// counts all the set bits in the bloom filter
    #[inline]
    pub fn count_ones(&self) -> usize {
        self.buckets.iter().map(|u| u.count_ones()).sum()
    }

    /// counts all the unset bits in the bloom filter
    #[inline]
    pub fn count_zeros(&self) -> usize {
        self.buckets.iter().map(|u| u.count_zeros()).sum()
    }

    #[inline]
    pub fn size_in_bytes(&self) -> usize {
        self.buckets.len() * Bucket::byte_size()
    }

    /// this function estimates the number of common elements between two bloom filters
    /// without having to intersect them
    pub fn estimate_common_elements(&self, other: &Self) -> Result<usize, Error> {
        if !self.has_same_params(other) {
            return Err(Error::Merge(
                "cannot compare bloom filters with different parameters".into(),
            ));
        }

        Ok((0..self.buckets.len())
            .map(|i| {
                let b1 = &self.buckets[i];
                let b2 = &other.buckets[i];
                (-(b1.bit_len() as f64
                    * f64::ln(1.0 - (b1.count_ones_in_common(b2) as f64 / b1.bit_len() as f64)))
                    / self.n_hash_buck as f64) as usize
            })
            .sum())
    }

    /// function used to update the estimated count of entries
    fn update_count(&mut self) {
        // we sum up the estimated number of elements in every bucket
        self.count = self
            .buckets
            .iter()
            .map(|b| {
                (-(b.bit_len() as f64
                    * f64::ln(1.0 - (b.count_ones() as f64 / b.bit_len() as f64)))
                    / self.n_hash_buck as f64) as u64
            })
            .sum();
    }

    /// returns an estimate of the number of element in the filter
    /// the exact number of element cannot be known as there might
    /// be collisions
    #[inline(always)]
    pub fn count_estimate(&self) -> u64 {
        self.count
    }

    #[inline(always)]
    pub fn has_same_params(&self, other: &Self) -> bool {
        self.capacity == other.capacity
            && self.fpp == other.fpp
            && self.n_hash_buck == other.n_hash_buck
            && self.buckets.len() == other.buckets.len()
    }

    /// Union of two [BloomFilter]
    pub fn union(mut self, other: Self) -> Result<Self, Error> {
        self.union_merge(&other)?;
        Ok(self)
    }

    /// In place union of two [BloomFilter]
    pub fn union_merge(&mut self, other: &Self) -> Result<(), Error> {
        if !self.has_same_params(other) {
            return Err(Error::Merge(
                "cannot make union of bloom filters with different parameters".into(),
            ));
        }
        (0..self.buckets.len()).for_each(|i| self.buckets[i].union(&other.buckets[i]));
        // we need to update the count at the end of merge operation
        self.update_count();
        Ok(())
    }

    /// Intersection of two [BloomFilter]
    pub fn intersection(mut self, other: Self) -> Result<Self, Error> {
        self.intersection_merge(&other)?;
        Ok(self)
    }

    /// In place intersection of two [BloomFilter]
    pub fn intersection_merge(&mut self, other: &Self) -> Result<(), Error> {
        if !self.has_same_params(other) {
            return Err(Error::Merge(
                "cannot make intersection of bloom filters with different parameters".into(),
            ));
        }
        (0..self.buckets.len()).for_each(|i| self.buckets[i].intersection(&other.buckets[i]));
        // we need to update the count at the end of merge operation
        self.update_count();
        Ok(())
    }
}

/// macro to ease bloom filter creation
/// # Example
/// ```
/// use poppy::bloom;
///
/// let mut b = bloom!(100, 0.1);
/// b.insert("hello");
/// assert!(b.contains("hello"));
/// ```
///
/// # Other Example
/// ```
/// use poppy::bloom;
///
/// let mut b = bloom!(100, 0.1, ["hello", "world"]);
/// assert!(b.contains_bytes("hello"));
/// assert!(b.contains_bytes("world"));
/// ```
macro_rules! bloom {
        ($cap:expr, $proba:expr) => {
            $crate::bloom2::BloomFilter::with_capacity($cap, $proba)
        };
        ($cap:expr, $proba:expr, [$($values:literal),*]) => {
            {
                let mut b=bloom!($cap, $proba);
                $(b.insert_bytes($values).unwrap();)*
                b
            }
        };
    }

#[cfg(test)]
mod test {

    use std::{
        collections::HashSet,
        fs,
        io::{self, BufRead},
    };

    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::utils::{time_it, ByteSize};

    use super::*;

    macro_rules! current_dir {
        () => {{
            let file = std::path::PathBuf::from(file!());
            std::path::PathBuf::from(file.parent().unwrap())
        }};
    }

    #[test]
    fn test_bitset() {
        let mut b = Bucket::new();

        for i in 0..Bucket::bit_size() {
            assert!(!b.set_nth_bit(i));
            assert!(b.set_nth_bit(i));
            assert!(b.get_nth_bit(i));
        }

        for i in 0..Bucket::bit_size() {
            assert!(b.get_nth_bit(i));
        }
    }

    #[test]
    fn test_bloom() {
        let mut b = BloomFilter::with_capacity(10000, 0.001);
        assert!(!b.contains_bytes("value"));
        b.insert_bytes("value").unwrap();
        assert!(b.contains_bytes("value"));
        assert_eq!(b.count, 1);
        assert!(!b.contains_bytes("unknown"));
    }

    #[test]
    fn test_union() {
        let mut b = bloom!(1000, 0.0001, ["hello", "world"]);
        let o = bloom!(1000, 0.0001, ["union", "test"]);

        b.union_merge(&o).unwrap();

        // estimate count should be exact for a small test like this
        assert_eq!(b.count_estimate(), 4);

        ["hello", "world", "union", "test"]
            .into_iter()
            .for_each(|v| {
                assert!(b.contains_bytes(v), "\"{v}\" not in filter");
            });
    }

    #[test]
    fn test_intersection() {
        let b = bloom!(
            1000,
            0.0001,
            ["hello", "world", "testing", "bloom", "filters"]
        );
        let o = bloom!(
            1000,
            0.0001,
            ["hello", "from", "intersecting", "two", "filters"]
        );

        assert_eq!(b.estimate_common_elements(&o).unwrap(), 2);

        let inter = b.intersection(o).unwrap();

        // estimate count should be exact for a small test like this
        assert_eq!(inter.count_estimate(), 2);

        ["hello", "filters"].into_iter().for_each(|v| {
            assert!(inter.contains_bytes(v), "\"{v}\" not in filter");
        });
    }

    #[test]
    #[ignore]
    fn benchmark_bloom() {
        let mut rng: StdRng = SeedableRng::from_seed([42; 32]);
        let test_files = vec![current_dir!().join("data/sample.txt")];

        let mut lines = HashSet::new();
        let mut dataset_size = 0;
        for t in test_files {
            let f = fs::File::open(t).unwrap();
            let reader = io::BufReader::new(f);
            for line in reader.lines() {
                let line = line.unwrap();
                let size = line.as_bytes().len();
                if lines.insert(line) {
                    dataset_size += size
                }
            }
        }

        let count = lines.len() as u64;
        let fp_rate = 0.001;

        let mut b = BloomFilter::with_capacity(count, fp_rate);
        let mb_size = dataset_size as f64 / 1_048_576.0;
        let runs = 5;

        let insert_dur = time_it(
            || {
                lines.iter().for_each(|l| {
                    b.insert_bytes(l).unwrap();
                })
            },
            runs,
        );

        // we verify that everything we entered is in the filter
        lines.iter().for_each(|l| assert!(b.contains_bytes(l)));

        let bit_size = crate::bit_size(count, fp_rate);
        //eprintln!("version of bloom-filter: {}", b.version());
        eprintln!("count: {}", count);
        eprintln!("proba: {}", fp_rate);
        eprintln!(
            "bit_size:{} optimized: {} expected_proba: {}",
            ByteSize::from_bits(bit_size as usize),
            bit_size.is_power_of_two(),
            crate::estimate_p(lines.len() as u64, bit_size),
        );
        eprintln!(
            "data: {} entries -> {}",
            count,
            ByteSize::from_bytes(dataset_size)
        );

        eprintln!("\nInsertion performance:");
        eprintln!("\tinsert duration: {:?}", insert_dur);
        eprintln!(
            "\tinsertion speed: {:.1} entries/s -> {:.1} MB/s",
            count as f64 / insert_dur.as_secs_f64(),
            mb_size / insert_dur.as_secs_f64()
        );

        eprintln!("\nQuery performance:");
        for mut_prob in (0..=100).step_by(10) {
            let mut tmp = lines.iter().cloned().collect::<Vec<String>>();
            let mutated_lines = tmp
                .iter_mut()
                .map(|e| {
                    let mut mutated = false;
                    if rng.gen_range(0..=100) < mut_prob {
                        mutated = true;
                        unsafe { e.as_bytes_mut() }
                            .iter_mut()
                            .for_each(|b| *b ^= rng.gen_range(0..=255));
                    }
                    (mutated, e.clone())
                })
                .collect::<Vec<(bool, String)>>();

            let mut fp_count = 0usize;
            let mut tn_count = 0usize;
            let query_dur = time_it(
                || {
                    mutated_lines.iter().for_each(|(m, l)| {
                        let is_in_bf = b.contains_bytes(l);
                        // data has been mutated
                        if *m {
                            if is_in_bf {
                                fp_count += 1
                            } else {
                                tn_count += 1
                            }
                        }
                    })
                },
                runs,
            );

            eprintln!(
                "\tcondition: {}% of queried values are in filter",
                100 - mut_prob
            );
            eprintln!("\tquery duration: {:?}", query_dur);
            eprintln!(
                "\tquery speed: {:.1} entries/s -> {:.1} MB/s",
                count as f64 / query_dur.as_secs_f64(),
                mb_size / query_dur.as_secs_f64()
            );
            eprintln!("\tfp = {}", fp_count);
            eprintln!("\ttn = {}", tn_count);
            eprintln!(
                "\tfp rate = {:3}",
                fp_count as f64 / (fp_count + tn_count) as f64
            );
            eprintln!();
        }

        eprintln!("Size in bytes: {}", ByteSize::from_bytes(b.size_in_bytes()));
        eprintln!("\nCollision information:");
        eprintln!("\texpected collision rate: {}", fp_rate);
    }
}
