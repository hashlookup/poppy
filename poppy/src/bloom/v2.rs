use std::{
    io::{self, BufWriter, Read, Seek, Write},
    marker::PhantomData,
};

use crate::{
    bitset::{
        array::BitSet,
        vec::{byte_cap_from_bit_cap, VecBitSet},
    },
    hash::{digest, wyhash::WyHasher, PoppyHash, PoppyHasher},
    read_flags,
    utils::{read_le_f64, read_le_u64},
    Flags, OptLevel, Params,
};
use core::mem::size_of;

#[derive(Debug, Default, Clone)]
pub struct IndexIterator<H: PoppyHasher, const M: u64> {
    init: bool,
    h1: u64,
    h2: u64,
    i: u64,
    count: u64,
    h: PhantomData<H>,
}

impl<H: PoppyHasher + Clone, const M: u64> std::marker::Copy for IndexIterator<H, M> {}

#[inline(always)]
fn xorshift_star(mut seed: u64) -> u64 {
    seed ^= seed.wrapping_shl(12);
    seed ^= seed.wrapping_shr(25);
    seed ^= seed.wrapping_shl(27);
    seed.wrapping_mul(2685821657736338717)
}

impl<H: PoppyHasher, const M: u64> IndexIterator<H, M> {
    fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    fn set_count(&mut self, count: u64) {
        self.count = count;
    }

    #[inline(always)]
    fn bucket_hash(&self) -> u64 {
        // we use xor shift to pseudo randomize bucket hash
        // decrease collisions on h % m
        xorshift_star(self.h1)
    }

    #[inline(always)]
    #[allow(dead_code)] // this method is used to check compatibility with former implementation
    fn init_with_slice<S: AsRef<[u8]>>(mut self, data: S) -> Self {
        self.init = true;
        if data.as_ref().len() > size_of::<u64>() {
            self.h1 = digest::<H, S>(data);
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

    #[inline(always)]
    fn init_with_hashable<D: PoppyHash + ?Sized>(mut self, data: &D) -> Self {
        self.init = true;
        self.h1 = data.hash_pop::<H>();
        self.h2 = 0;
        self.i = 0;
        self
    }

    #[inline(always)]
    const fn is_init(&self) -> bool {
        self.init && self.i == 0
    }
}

impl<H: PoppyHasher, const M: u64> Iterator for IndexIterator<H, M> {
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
                    // fpp < hash_collision_rate. However WiHash has a very low
                    // collision rate (i.e. 0% for decent dataset sizes we want
                    // to store in a BF) so it should not be an issue.
                    self.h2 = H::hash_one(self.h1.to_be_bytes());
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

/// Faster and more accurate implementation than [crate::v1::BloomFilter]
/// Until further notice, this is the structure to use by default.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    flags: Flags,
    /// this is used to speed up lookup and stabilize it
    /// it is not used by default as it increases bloom
    /// filter size
    index_cache: VecBitSet,
    /// these two fields are needed to store partially read filter size
    n_buckets: u64,
    cache_byte_size: u64,
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

impl TryFrom<Params> for BloomFilter {
    type Error = Error;
    fn try_from(p: Params) -> Result<Self, Error> {
        Self::make(p.capacity as u64, p.fpp, p.opt)
    }
}

impl BloomFilter {
    #[inline]
    pub fn from_reader<R: Read + Seek>(r: R) -> Result<Self, Error> {
        Self::_from_reader(r, false)
    }

    #[inline]
    pub(crate) fn _from_reader<R: Read + Seek>(r: R, partial: bool) -> Result<Self, Error> {
        let mut br = io::BufReader::new(r);
        let r = &mut br;

        let flags = read_flags(r)?;

        if flags.version != 2 {
            return Err(Error::InvalidVersion(flags.version));
        }

        Self::from_reader_with_flags(r, flags, partial)
    }

    #[inline]
    pub(crate) fn from_reader_with_flags<R: Read + Seek>(
        r: R,
        flags: Flags,
        partial: bool,
    ) -> Result<Self, Error> {
        let mut br = io::BufReader::new(r);
        let r = &mut br;

        let capacity = read_le_u64(r)?;
        let fpp = read_le_f64(r)?;

        // we control fpp
        if !(f64::MIN_POSITIVE..=1.0).contains(&fpp) {
            return Err(Error::WrongFpp(fpp));
        }

        let n_hash_buck = read_le_u64(r)?;
        let count = read_le_u64(r)?;

        let cache_bit_size = read_le_u64(r)?;
        let cache_byte_size = byte_cap_from_bit_cap(cache_bit_size as usize) as u64;
        let index_cache = {
            if partial {
                r.seek_relative(cache_byte_size as i64)?;
                VecBitSet::with_bit_capacity(0)
            } else {
                let mut ic = VecBitSet::with_bit_capacity(cache_bit_size as usize);
                r.read_exact(ic.as_mut_slice())?;
                ic
            }
        };

        // we read the number of buckets
        let n_buckets = read_le_u64(r)? as usize;

        let buckets = {
            if partial {
                r.seek_relative((n_buckets * core::mem::size_of::<Bucket>()) as i64)?;
                vec![]
            } else {
                let mut buckets = vec![Bucket::new(); n_buckets];
                // we read all the buckets as byte buffers
                for bucket in buckets.iter_mut().take(n_buckets) {
                    r.read_exact(bucket.as_mut_slice())?;
                }
                buckets
            }
        };

        // reading data
        let mut data = Vec::new();
        r.read_to_end(&mut data)?;

        Ok(Self {
            flags,
            index_cache,
            n_buckets: n_buckets as u64,
            cache_byte_size,
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

        // writting version
        w.write_all(&self.flags.to_bytes())?;

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
            w.write_all(i.as_slice())?;
        }

        w.write_all(&self.data)?;
        Ok(())
    }

    pub(crate) fn make(capacity: u64, fpp: f64, opt: OptLevel) -> Result<Self, Error> {
        if !(f64::MIN_POSITIVE..=1.0).contains(&fpp) {
            return Err(Error::WrongFpp(fpp));
        }

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

        let index_cache = VecBitSet::with_bit_capacity(cache_bit_size as usize);
        let cache_bit_size = index_cache.byte_len() as u64;

        Ok(Self {
            flags: Flags::new(2).opt(opt),
            index_cache,
            n_buckets: n_bucket,
            cache_byte_size: cache_bit_size,
            buckets: vec![Bucket::new(); n_bucket as usize],
            capacity,
            fpp,
            n_hash_buck,
            count: 0,
            data: vec![],
        })
    }

    pub fn with_capacity(cap: u64, proba: f64) -> Result<Self, Error> {
        Self::make(cap, proba, OptLevel::None)
    }

    #[allow(dead_code)]
    #[inline(always)]
    // we keep this dead code as it allow to test compatibility with old implementation
    // moving this as a structure member does not improve perfs
    fn index_iter(&self) -> IndexIterator<WyHasher, BIT_SET_MOD> {
        let mut it = IndexIterator::<_, BIT_SET_MOD>::new();
        it.set_count(self.n_hash_buck);
        it
    }

    #[inline(always)]
    fn has_optimal_buckets(&self) -> bool {
        self.buckets.len().is_power_of_two()
    }

    /// Clears out the bloom filter
    #[inline(always)]
    pub fn clear(&mut self) {
        self.buckets.iter_mut().for_each(|bucket| bucket.clear());
        self.count = 0;
    }

    #[inline(always)]
    /// Function to implement hash one insert many use cases. An [IndexIterator] can
    /// be obtained from [Self::prep_index_iter] method.
    pub fn insert_iter(&mut self, it: IndexIterator<WyHasher, BIT_SET_MOD>) -> Result<bool, Error> {
        let mut it = it;

        if !it.is_init() {
            return Err(Error::UninitIter);
        }

        it.set_count(self.n_hash_buck);

        if self.capacity == 0 {
            return Err(Error::TooManyEntries);
        }

        let mut new = false;

        let h = it.bucket_hash();
        let ibucket = {
            if self.has_optimal_buckets() {
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

    #[inline]
    fn _insert_bytes_old<D: AsRef<[u8]>>(&mut self, data: D) -> Result<bool, Error> {
        self.insert_iter(self.index_iter().init_with_slice(data))
    }

    #[inline]
    /// Insert a byte slice into the filter. This function is kept to support backward
    /// compatibility with old API.
    pub fn insert_bytes<D: AsRef<[u8]>>(&mut self, data: D) -> Result<bool, Error> {
        self.insert(&data.as_ref())
    }

    #[inline]
    /// Generic insert any data implementing [PoppyHash] trait
    pub fn insert<H: PoppyHash>(&mut self, data: &H) -> Result<bool, Error> {
        self.insert_iter(Self::build_index_iter(data))
    }

    #[inline]
    /// Function to implement hash one contains many use cases. An [IndexIterator] can
    /// be obtained from [Self::prep_index_iter] method.
    ///
    /// # Example
    ///
    /// ```
    /// use poppy_filters::v2::BloomFilter;
    ///
    /// let mut b: BloomFilter = BloomFilter::with_capacity(10000, 0.001).unwrap();
    ///
    /// /// we prepare the data to be inserted and/or checked
    /// /// this way, the cost of hashing the data is done only once
    /// let prep = (0..1000).map(|d| BloomFilter::build_index_iter(&d)).collect::<Vec<_>>();
    ///
    /// for p in prep {
    ///     b.insert_iter(p).unwrap();
    ///     b.contains_iter(p).unwrap();
    /// }
    /// ```
    pub fn contains_iter(&self, it: IndexIterator<WyHasher, BIT_SET_MOD>) -> Result<bool, Error> {
        let mut it = it;

        if !it.is_init() {
            return Err(Error::UninitIter);
        }

        it.set_count(self.n_hash_buck);

        if self.capacity == 0 {
            return Ok(false);
        }

        let h = it.bucket_hash();

        if !self.index_cache.is_empty()
            && !self
                .index_cache
                .get_nth_bit(h as usize & (self.index_cache.bit_len() - 1))
        {
            return Ok(false);
        }

        let ibucket = {
            if self.has_optimal_buckets() {
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
                return Ok(false);
            }
        }

        Ok(true)
    }

    #[inline]
    fn _contains_bytes_old<D: AsRef<[u8]>>(&self, data: D) -> bool {
        let it = self.index_iter().init_with_slice(data);
        // this cannot panic as our iterator has been inititialized
        self.contains_iter(it).unwrap()
    }

    #[inline]
    pub fn contains_bytes<S: AsRef<[u8]>>(&self, data: S) -> bool {
        self.contains(&data.as_ref())
    }

    #[inline]
    pub fn contains<H: PoppyHash + ?Sized>(&self, data: &H) -> bool {
        // this cannot panic as our iterator has been inititialized
        self.contains_iter(Self::build_index_iter(data)).unwrap()
    }

    #[inline]
    pub fn build_index_iter<H: PoppyHash + ?Sized>(
        data: &H,
    ) -> IndexIterator<WyHasher, BIT_SET_MOD> {
        IndexIterator::new().init_with_hashable(data)
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
        self.n_buckets as usize * Bucket::byte_size() + self.cache_byte_size as usize
    }

    #[inline(always)]
    pub fn fpp(&self) -> f64 {
        self.fpp
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity as usize
    }

    #[inline(always)]
    /// Returns true if the filter is full
    pub fn is_full(&self) -> bool {
        self.count_estimate() as usize == self.capacity()
    }

    /// this function estimates the number of common elements between two bloom filters
    /// without having to intersect them
    pub fn count_common_entries(&self, other: &Self) -> Result<usize, Error> {
        if !self.has_same_params(other) {
            return Err(Error::Merge(
                "cannot compare filters with different parameters".into(),
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

    #[inline]
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
        self.flags == other.flags
            && self.capacity == other.capacity
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

    use md5::{Digest, Md5};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use sha1::Sha1;
    use sha2::Sha256;

    use crate::{
        estimate_p,
        hash::PoppyHasher,
        utils::{benchmark, ByteSize, Stats},
    };

    use super::*;

    fn md5(data: &[u8]) -> Vec<u8> {
        let mut hasher = Md5::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }

    fn sha1(data: &[u8]) -> Vec<u8> {
        let mut hasher = Sha1::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }

    fn sha256(data: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }

    macro_rules! bloom {
        ($cap:expr, $proba:expr) => {
            $crate::bloom::v2::BloomFilter::with_capacity($cap, $proba).unwrap()
        };
        ($cap:expr, $proba:expr, [$($values:literal),*]) => {
            {
                let mut b=bloom!($cap, $proba);
                $(b.insert(&$values).unwrap();)*
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
            (WyHasher::hash_one(i.to_le_bytes()) & (b.bit_len() as u64 - 1)) as usize
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
        let mut b = BloomFilter::with_capacity(10000, 0.001).unwrap();
        assert!(!b._contains_bytes_old("hello"));
        b.insert_bytes("hello").unwrap();
        assert!(b._contains_bytes_old("hello"));
        assert!(b.contains(&"hello"));
        assert!(b.contains_bytes("hello"));
        assert_eq!(b.count, 1);
        assert!(!b.contains(&"unknown"));
    }

    #[test]
    fn test_poppy_hash_compatibility() {
        let mut b: BloomFilter = BloomFilter::with_capacity(10000, 0.001).unwrap();
        assert!(!b.contains(&"hello"));
        b.insert(&"hello").unwrap();
        b.insert(&String::from("some string")).unwrap();
        b._insert_bytes_old("some old string").unwrap();

        assert!(b._contains_bytes_old("hello"));
        assert!(b._contains_bytes_old("some string"));
        assert!(b._contains_bytes_old("some old string"));

        assert!(b.contains(&"hello"));
        assert!(b.contains(&"some string"));
        assert!(b.contains(&"some old string"));

        assert!(!b.contains(&"unknown"));
    }

    #[test]
    fn test_insert_contains_by_iter() {
        let mut b: BloomFilter = BloomFilter::with_capacity(10000, 0.001).unwrap();

        let prep = (0..1000)
            .map(|d| BloomFilter::build_index_iter(&d))
            .collect::<Vec<_>>();

        for p in prep {
            b.insert_iter(p).unwrap();
            b.contains_iter(p).unwrap();
        }
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

        assert_eq!(b.count_common_entries(&o).unwrap(), 2);

        let inter = b.intersection(o).unwrap();

        // estimate count should be exact for a small test like this
        assert_eq!(inter.count_estimate(), 2);

        ["hello", "filters"].into_iter().for_each(|v| {
            assert!(inter.contains_bytes(v), "\"{v}\" not in filter");
        });
    }

    fn test_real_fpp(data: Vec<Vec<u8>>, tol: f64) {
        let mut rng: StdRng = SeedableRng::from_seed([42; 32]);
        let mut fpps = vec![];

        let mut dataset = HashSet::new();
        for d in data {
            dataset.insert(d);
        }

        let count = dataset.len() as u64;
        let fpp = 0.001;

        let mut b = BloomFilter::with_capacity(count, fpp).unwrap();
        //let mut b = BloomFilter::make(count, fpp, OptLevel::None);

        // we insert data into filter
        dataset.iter().for_each(|buf| {
            b.insert_bytes(buf).unwrap();
            assert!(b.contains_bytes(buf))
        });

        println!("dataset size: {}", dataset.len());
        // we apply mutations to the dataset
        for mut_prob in (0..=100).step_by(10) {
            let mut tmp = dataset.iter().cloned().collect::<Vec<_>>();
            let mutated_lines = tmp
                .iter_mut()
                .map(|e| {
                    let mut mutated = false;
                    if rng.gen_range(0..=100) < mut_prob {
                        mutated = true;
                        e.iter_mut().for_each(|b| *b ^= rng.gen_range(0..=255));
                    }
                    (mutated, e.clone())
                })
                .collect::<Vec<(bool, Vec<u8>)>>();

            let mut s = Stats::new();

            let mut tested = HashSet::new();
            mutated_lines.iter().for_each(|(m, l)| {
                let is_in_bf = b.contains_bytes(l);
                // data has been mutated, has not been tested yet and
                // is not part of entry dataset
                if *m && !dataset.contains(l) && !tested.contains(l) {
                    if is_in_bf {
                        s.inc_fp()
                    } else {
                        s.inc_tn()
                    }
                }
                tested.insert(l.clone());
            });

            if !s.fp_rate().is_nan() {
                println!(
                    "mut: {}% real fpp: {} tot: {}",
                    mut_prob,
                    s.fp_rate(),
                    s.total()
                );
                fpps.push(s.fp_rate());
            }
        }

        let avg_fpp = fpps.iter().sum::<f64>() / (fpps.len() as f64);
        println!("average fpp: {avg_fpp}");
        if !avg_fpp.is_nan() {
            assert!(
                avg_fpp < (fpp * (1.0 + tol)),
                "real fpp: {} VS expected: {}",
                avg_fpp,
                fpp,
            );
        }
    }

    #[test]
    #[allow(unreachable_code)]
    fn test_fpps() {
        #[cfg(debug_assertions)]
        {
            println!("test is ignored because it takes too long in debug");
            return;
        }

        let tol = 0.2;
        let root = std::env::current_dir().unwrap();
        let test_files = vec![
            root.join("src/data/words/french.txt"),
            root.join("src/data/words/english.txt"),
        ];

        for t in test_files {
            let f = fs::File::open(&t).unwrap();
            let data = io::BufReader::new(f)
                .lines()
                .map(|l| l.unwrap().as_bytes().to_vec())
                .collect::<Vec<Vec<u8>>>();
            println!("testing: {}", t.to_string_lossy());
            test_real_fpp(data, 0.1);
            println!();
        }

        println!("testing with u8");
        test_real_fpp(
            (0..u8::MAX).map(|u| u.to_le_bytes().to_vec()).collect(),
            tol,
        );
        println!();

        println!("testing with u16");
        test_real_fpp(
            (0..u16::MAX).map(|u| u.to_le_bytes().to_vec()).collect(),
            tol,
        );
        println!();

        println!("testing with u32");
        let max_it = u32::MAX >> 14;
        test_real_fpp(
            (0..u32::MAX >> 14)
                .map(|u| u.to_le_bytes().to_vec())
                .collect(),
            tol,
        );
        println!();

        println!("testing with u64");
        test_real_fpp(
            (0..max_it as u64)
                .map(|u| u.to_le_bytes().to_vec())
                .collect(),
            tol,
        );
        println!();

        println!("testing with u128");
        test_real_fpp(
            (0..max_it as u128)
                .map(|u| u.to_le_bytes().to_vec())
                .collect(),
            tol,
        );
        println!();

        println!("testing with md5");
        test_real_fpp(
            (0..max_it as u128)
                .map(|u| md5(&u.to_le_bytes()[..]))
                .collect(),
            tol,
        );
        println!();

        println!("testing with sha1");
        test_real_fpp(
            (0..max_it as u128)
                .map(|u| sha1(&u.to_le_bytes()[..]))
                .collect(),
            tol,
        );
        println!();

        println!("testing with sha256");
        test_real_fpp(
            (0..max_it as u128)
                .map(|u| sha256(&u.to_le_bytes()[..]))
                .collect(),
            tol,
        );
        println!();
    }

    #[test]
    fn test_serialization() {
        let mut b = bloom!(1000, 0.0001, ["deserialization", "test"]);
        let mut data = vec![0; 256];
        data.iter_mut().enumerate().for_each(|(i, b)| *b = i as u8);

        b.data = data.clone();

        let mut cursor = io::Cursor::new(vec![]);
        b.write(&mut cursor).unwrap();
        cursor.set_position(0);
        // deserializing the stuff out
        let new = BloomFilter::from_reader(cursor).unwrap();
        assert_eq!(new.fpp, 0.0001);
        assert_eq!(new.size_in_bytes(), new.size_in_bytes());
        assert!(new.contains_bytes("deserialization"));
        assert!(new.contains_bytes("test"));
        assert!(!new.contains_bytes("hello"));
        assert!(!new.contains_bytes("world"));
        assert_eq!(new.data, data);
    }

    #[test]
    fn test_partial_deserialization() {
        let p = Params::new(1000, 0.0001).opt(OptLevel::Best);
        let mut b = p.try_into_v2().unwrap();
        b.insert(&"hello").unwrap();
        b.insert(&"world").unwrap();
        let mut data = vec![0; 256];
        data.iter_mut().enumerate().for_each(|(i, b)| *b = i as u8);

        b.data = data.clone();

        let mut cursor = io::Cursor::new(vec![]);
        b.write(&mut cursor).unwrap();
        cursor.set_position(0);
        // deserializing the stuff out
        let new = BloomFilter::_from_reader(cursor, true).unwrap();
        assert_eq!(new.fpp, 0.0001);
        assert_eq!(new.capacity, 1000);
        assert_eq!(new.count, 2);
        assert_eq!(new.data, data);
        assert_eq!(new.size_in_bytes(), b.size_in_bytes());
    }

    #[test]
    fn test_contains_on_empty() {
        let b = bloom!(0, 0.001);
        assert!(!b.contains_bytes("toto"))
    }

    #[test]
    #[ignore]
    fn benchmark_bloom() {
        let mut rng: StdRng = SeedableRng::from_seed([42; 32]);
        // package root
        let root = std::env::current_dir().unwrap();
        let test_files = vec![root.join("src/data/sample.txt")];

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
        let fpp = 0.001;

        let mut b = BloomFilter::with_capacity(count, fpp).unwrap();
        let mb_size = dataset_size as f64 / 1_048_576.0;
        let runs = 5;

        let insert_dur = benchmark(
            || {
                lines.iter().for_each(|l| {
                    b.insert_bytes(l).unwrap();
                })
            },
            runs,
        );

        // we verify that everything we entered is in the filter
        lines.iter().for_each(|l| assert!(b.contains_bytes(l)));

        let bit_size = crate::bit_size(count, fpp);
        eprintln!("count: {}", count);
        eprintln!("proba: {}", fpp);
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

            let mut s = Stats::new();
            let query_dur = benchmark(
                || {
                    mutated_lines.iter().for_each(|(m, l)| {
                        let is_in_bf = b.contains_bytes(l);
                        // data has been mutated
                        if *m {
                            if is_in_bf {
                                s.inc_fp();
                            } else {
                                s.inc_tn();
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
            eprintln!("\tfp rate = {:3}", s.fp_rate());
            eprintln!();
        }

        eprintln!("Size in bytes: {}", ByteSize::from_bytes(b.size_in_bytes()));
        eprintln!("n_hash_buck: {}", b.n_hash_buck);
        eprintln!("\nCollision information:");
        eprintln!("\texpected collision rate: {}", fpp);
    }
}
