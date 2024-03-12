use std::{
    io::{self, BufWriter, Read, Write},
    slice,
};

use core::mem::size_of;

use crate::{
    ahash::BloomAHasher,
    utils::{read_le_f64, read_le_u64},
    OptLevel,
};

// N gives the size in Bytes of the bucket
#[derive(Debug, Clone)]
pub struct VecBitSet(Vec<u8>);

impl VecBitSet {
    pub fn with_bit_capacity(cap: usize) -> Self {
        let byte_size = ((cap as f64) / 8.0).ceil() as usize;
        Self(vec![0; byte_size])
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.0.as_mut_slice()
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
    pub fn bit_len(&self) -> usize {
        self.0.len() * 8
    }

    #[inline]
    pub fn byte_len(&self) -> usize {
        self.0.len()
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
}

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

#[inline(always)]
fn xor_shift(mut xs: u64) -> u64 {
    xs ^= xs.wrapping_shl(13);
    xs ^= xs.wrapping_shr(17);
    xs ^= xs.wrapping_shl(5);
    xs
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
        xor_shift(self.h1)
    }

    #[inline(always)]
    fn init_with_data<S: AsRef<[u8]>>(mut self, data: S) -> Self {
        if data.as_ref().len() > size_of::<u64>() {
            self.h1 = BloomAHasher::digest(&data);
        } else {
            // if data is smaller than u64 we don't need to hash it
            let mut tmp = [0u8; size_of::<u64>()];
            data.as_ref()
                .iter()
                .enumerate()
                .for_each(|(i, &b)| tmp[i] = b);
            self.h1 = u64::from_le_bytes(tmp);
        }
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
                    // here we propagate hash algorithm collision to index
                    // collision. This means we will never be able to achieve
                    // fpp < hash_collision_rate. However AHash has a very low
                    // collision rate (i.e. 0% for decent dataset sizes we want
                    // to store in a BF) so it should not be an issue.
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

type Error = crate::Error;

const BUCKET_SIZE: usize = 4096;
const BIT_SET_MOD: u64 = (BUCKET_SIZE * 8) as u64;
type Bucket = BitSet<BUCKET_SIZE>;

#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// this is used to speed up lookup and stabilize it
    /// it is not used by default as it increases bloom
    /// filter size
    index_cache: VecBitSet,
    /// the maximum of entries the filter can contains without
    /// impacting the desired false positive probability
    pub(crate) capacity: u64,
    /// desired false positive probability
    pub(crate) fpp: f64,
    /// number of hash functions
    pub(crate) n_hash_buck: u64,
    /// estimated number of elements in the filter
    pub(crate) count: u64,
    /// buckets of small but optimized for speed bloom filters
    pub(crate) buckets: Vec<Bucket>,
    /// arbitrary data that we can attach to the filter
    pub data: Vec<u8>,
}

impl BloomFilter {
    #[inline]
    pub fn from_reader<R: Read>(r: R) -> Result<Self, Error> {
        let mut br = io::BufReader::new(r);
        let r = &mut br;

        let capacity = read_le_u64(r)?;
        let fpp = read_le_f64(r)?;
        let n_hash_buck = read_le_u64(r)?;
        let count = read_le_u64(r)?;

        let cache_bit_size = read_le_u64(r)? as usize;
        let mut index_cache = VecBitSet::with_bit_capacity(cache_bit_size);
        r.read_exact(index_cache.as_mut_slice())?;

        // we read the number of buckets
        let n_buckets = read_le_u64(r)? as usize;
        let mut buckets = vec![Bucket::new(); n_buckets];

        // we read all the buckets as byte buffers
        for bucket in buckets.iter_mut().take(n_buckets) {
            r.read_exact(bucket.0.as_mut_slice())?;
        }

        // reading data
        let mut data = Vec::new();
        r.read_to_end(&mut data)?;

        Ok(Self {
            index_cache,
            capacity,
            fpp,
            n_hash_buck,
            count,
            buckets,
            data,
        })
    }

    #[inline]
    pub fn write<W: Write>(&self, w: &mut W) -> Result<(), Error> {
        let mut w = BufWriter::new(w);

        w.write_all(&self.capacity.to_le_bytes())?;
        w.write_all(&self.fpp.to_le_bytes())?;
        w.write_all(&self.n_hash_buck.to_le_bytes())?;
        w.write_all(&self.count.to_le_bytes())?;

        // bit capacity of index cache
        w.write_all(&(self.index_cache.bit_len() as u64).to_le_bytes())?;
        // we write index cache
        w.write_all(self.index_cache.as_slice())?;

        // the number of buckets as u64
        w.write_all(&(self.buckets.len() as u64).to_le_bytes())?;

        // we serialize buckets
        for i in self.buckets.iter() {
            w.write_all(&i.0)?;
        }

        w.write_all(&self.data)?;
        Ok(())
    }

    pub(crate) fn make(capacity: u64, fpp: f64, opt: OptLevel) -> Self {
        // the capacity per bucket given a fpp
        let bucket_cap = crate::cap_from_bit_size(Bucket::bit_size() as u64, fpp);
        let bucket_bit_size = Bucket::bit_size() as u64;
        // the number of buckets we need to store all data

        let mut n_bucket = (capacity as f64 / bucket_cap as f64).ceil() as u64;
        let mut n_hash_buck = (crate::k(bucket_bit_size, bucket_cap) as f64).ceil() as u64;

        let mut cache_bit_size = 0;
        let bits_per_entry = (bucket_bit_size as f64 / (bucket_cap as f64)).round();

        match opt {
            OptLevel::None | OptLevel::Space => {}
            OptLevel::Speed => {
                cache_bit_size = capacity.next_power_of_two();
                n_bucket = n_bucket.next_power_of_two();
                n_hash_buck =
                    (f64::ln(2.0) * crate::estimate_p(capacity, cache_bit_size) * (bits_per_entry))
                        .ceil() as u64;
            }
            OptLevel::Best => {
                cache_bit_size = capacity.next_power_of_two();
                n_hash_buck =
                    (f64::ln(2.0) * crate::estimate_p(capacity, cache_bit_size) * (bits_per_entry))
                        .ceil() as u64;
            }
        }

        Self {
            index_cache: VecBitSet::with_bit_capacity(cache_bit_size as usize),
            buckets: vec![Bucket::new(); n_bucket as usize],
            capacity,
            fpp,
            n_hash_buck,
            count: 0,
            data: vec![],
        }
    }

    pub fn with_capacity(cap: u64, proba: f64) -> Self {
        Self::make(cap, proba, OptLevel::Speed)
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

        if !self.index_cache.is_empty() {
            self.index_cache
                .set_nth_bit(h as usize & (self.index_cache.bit_len() - 1));
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

        if !self.index_cache.is_empty()
            && !self
                .index_cache
                .get_nth_bit(h as usize & (self.index_cache.bit_len() - 1))
        {
            return false;
        }

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
        self.buckets.len() * Bucket::byte_size() + self.index_cache.byte_len()
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
            && self.index_cache.byte_len() == other.index_cache.byte_len()
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
        self.index_cache.union(&other.index_cache);
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
        self.index_cache.intersection(&other.index_cache);
        (0..self.buckets.len()).for_each(|i| self.buckets[i].intersection(&other.buckets[i]));
        // we need to update the count at the end of merge operation
        self.update_count();
        Ok(())
    }
}

#[cfg(test)]
mod test {

    use std::{
        collections::HashSet,
        fs,
        io::{self, BufRead},
    };

    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::{
        estimate_p,
        utils::{time_it, ByteSize, Stats},
    };

    use super::*;

    macro_rules! current_dir {
        () => {{
            let file = std::path::PathBuf::from(file!());
            std::path::PathBuf::from(file.parent().unwrap())
        }};
    }

    macro_rules! bloom {
        ($cap:expr, $proba:expr) => {
            $crate::bloom::v2::BloomFilter::with_capacity($cap, $proba)
        };
        ($cap:expr, $proba:expr, [$($values:literal),*]) => {
            {
                let mut b=bloom!($cap, $proba);
                $(b.insert_bytes($values).unwrap();)*
                b
            }
        };
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
    fn test_bitset_fpp() {
        //let cap = 424103097usize;
        let cap = (1usize << 22) + 1;
        let mut b = VecBitSet::with_bit_capacity(cap.next_power_of_two());
        //let mut b = VecBitSet::with_bit_capacity(cap);

        fn get_index(b: &VecBitSet, i: usize) -> usize {
            (BloomAHasher::digest(i.to_le_bytes()) & (b.bit_len() as u64 - 1)) as usize
        }

        for i in 0..cap {
            b.set_nth_bit(get_index(&b, i));
        }

        for i in 0..cap {
            assert!(b.get_nth_bit(get_index(&b, i)));
        }

        let mut s = Stats::new();
        for i in cap..cap * 10 {
            if b.get_nth_bit(get_index(&b, i)) {
                s.inc_fp()
            } else {
                s.inc_tn()
            }
        }

        println!(
            "estimated fp bitset: {}",
            estimate_p(cap as u64, b.bit_len() as u64)
        );
        println!("real fp rate bitset: {}", s.fp_rate())
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
    fn test_small_data() {
        // 25% tolerance on fpp
        let tol = 0.25;
        let count = 1000000;
        let mut b = BloomFilter::make(count, 0.0001, OptLevel::None);
        let mut s = Stats::new();

        for i in 0..count {
            b.insert(i).unwrap();
            assert!(b.contains(i));
        }

        for i in count..count * 2 {
            if b.contains(i) {
                s.inc_fp();
            } else {
                s.inc_tn();
            }
        }

        println!(
            "fpp={} fp_rate={} n_hash_buck={}",
            b.fpp,
            s.fp_rate(),
            b.n_hash_buck
        );
        assert!(s.fp_rate() < b.fpp * (1.0 + tol))
    }

    #[test]
    fn test_serialization() {
        let mut b = bloom!(1000, 0.0001, ["deserialization", "test"]);
        let mut data = vec![0; 256];
        data.iter_mut().enumerate().for_each(|(i, b)| *b = i as u8);

        b.data = data.clone();

        let mut cursor = io::Cursor::new(vec![]);
        b.write(&mut cursor).unwrap();
        println!("cursor position: {}", cursor.position());
        cursor.set_position(0);
        // deserializing the stuff out
        let b = BloomFilter::from_reader(cursor).unwrap();
        assert_eq!(b.fpp, 0.0001);
        assert!(b.contains_bytes("deserialization"));
        assert!(b.contains_bytes("test"));
        assert!(!b.contains_bytes("hello"));
        assert!(!b.contains_bytes("world"));
        assert_eq!(b.data, data);
        // last element must be 255
        assert_eq!(b.data.last().unwrap(), &255);
    }

    #[test]
    #[ignore]
    fn benchmark_bloom() {
        let mut rng: StdRng = SeedableRng::from_seed([42; 32]);
        let test_files = vec![current_dir!().join("../data/sample.txt")];

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
        eprintln!("n_hash_buck: {}", b.n_hash_buck);
        eprintln!("\nCollision information:");
        eprintln!("\texpected collision rate: {}", fp_rate);
    }
}
