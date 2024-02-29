use crate::fingerprint::Fingerprint;
use crate::utils::is_power_of_2;
use core::slice;
use std::hash::Hash;
use std::ops::{Div, Mul, Sub};

use std::{
    collections::HashSet,
    io::{self, BufWriter, Read, Write},
};
use thiserror::Error;

mod utils;
pub use utils::*;

pub const DEFAULT_VERSION: u8 = 2;

fn read_le_u64<R: Read>(r: &mut R) -> Result<u64, io::Error> {
    let mut bytes = [0u8; 8];
    r.read_exact(bytes.as_mut_slice())?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_le_f64<R: Read>(r: &mut R) -> Result<f64, io::Error> {
    let mut bytes = [0u8; 8];
    r.read_exact(bytes.as_mut_slice())?;
    Ok(f64::from_le_bytes(bytes))
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    IoError(#[from] io::Error),
    #[error("invalid version {0}")]
    InvalidVersion(u8),
    #[error("union error: {0}")]
    Union(String),
    #[error("too many entries, false positive rate cannot be met")]
    TooManyEntries,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Params {
    pub cap: u64,
    pub proba: f64,
}

impl Eq for Params {}

impl Hash for Params {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.cap);
        state.write(&self.proba.to_le_bytes());
    }
}

impl Params {
    pub fn into_bloom_filter(self) -> BloomFilter {
        self.into()
    }

    #[inline]
    pub fn estimate_p(&self, n: u64) -> f64 {
        utils::estimate_p(n, self.bit_size())
    }

    #[inline]
    pub fn bit_size(&self) -> u64 {
        utils::bit_size(self.cap, self.proba)
    }
}

#[derive(Debug, Clone)]
pub struct BloomFilter {
    version: u8,
    // M == bitset.len()
    bitset: Vec<u64>,
    // n: desired maximum number of elements
    cap: u64,
    // desired false positive probability
    p: f64,
    // k: number of hash functions
    n_hash: u64,
    // m: number of bits
    bit_size: u64,
    // N: number of elements in the filter
    count: u64,
    // arbitrary data that we can attach to the filter
    pub data: Vec<u8>,
    fingerprint: Fingerprint,
}

fn prev_power_of_2(n: u64) -> u64 {
    let mut power = 1;
    while power < n {
        power <<= 1;
    }
    power >> 1
}

fn next_power_of_2(n: u64) -> u64 {
    let mut power = 1;
    while power < n {
        power <<= 1;
    }
    power
}

impl From<Params> for BloomFilter {
    fn from(value: Params) -> Self {
        Self::with_capacity(value.cap, value.proba)
    }
}

impl BloomFilter {
    fn pow2_caps_for_proba(proba: f64, max_bit_size: u64) -> Vec<u64> {
        let step = 100;
        let mut marked = HashSet::new();
        for i in 0..63 {
            let cap = utils::cap_from_bit_size(1 << i, proba);
            let start = {
                if cap >= step {
                    cap - step
                } else {
                    0
                }
            };

            let stop = match cap.checked_add(100) {
                Some(c) => c,
                _ => u64::MAX,
            };

            for c in start..stop {
                let bs = utils::bit_size(c, proba);
                if is_power_of_2(bs) && !marked.contains(&c) && bs <= max_bit_size {
                    marked.insert(c);
                }
            }
        }
        let mut v: Vec<u64> = marked.into_iter().collect();
        v.sort();
        v
    }

    pub fn possible_optimizations(n: u64, proba: f64) -> Vec<Params> {
        let mut params = HashSet::new();

        let bs = utils::bit_size(n, proba);

        for bs in [prev_power_of_2(bs), next_power_of_2(bs)] {
            let mut p = 0.5;
            while p > proba / 10.0 {
                let caps = BloomFilter::pow2_caps_for_proba(p, bs);
                if let Some(c) = caps
                    .into_iter()
                    .find(|c| *c >= n && (*c as f64) < (n as f64 + 0.5 * n as f64))
                {
                    params.insert(Params { cap: c, proba: p });
                }
                // prevents having lots of numbers in fract
                p = p.sub(0.0005).mul(1_000_000.0).round().div(1_000_000.0)
            }
        }

        let mut params = params.into_iter().collect::<Vec<Params>>();
        params.sort_by(|a, b| {
            a.proba
                .partial_cmp(&b.proba)
                .unwrap()
                .then_with(|| a.cap.cmp(&b.cap))
        });
        params
    }

    pub fn optimize(n: u64, proba: f64) -> Option<Params> {
        Self::possible_optimizations(n, proba).first().cloned()
    }

    pub fn with_capacity(cap: u64, proba: f64) -> Self {
        let version = DEFAULT_VERSION;

        // size in bits, computed from the capacity we want and the probability
        let bit_size = utils::bit_size(cap, proba);

        // size in u64
        let u64_size = f64::ceil(bit_size as f64 / 64.0) as usize;

        let n_hash_fn = utils::k(bit_size, cap);

        Self {
            version,
            bitset: vec![0; u64_size],
            cap,
            p: proba,
            n_hash: n_hash_fn,
            bit_size,
            count: 0,
            data: vec![],
            fingerprint: Fingerprint::new(version, n_hash_fn, bit_size),
        }
    }

    pub fn with_version(mut self, version: u8) -> Result<Self, Error> {
        self.version = version;
        self.check_version()?;
        self.fingerprint = Fingerprint::new(self.version, self.n_hash, self.bit_size);
        Ok(self)
    }

    fn check_version(&self) -> Result<(), Error> {
        if !matches!(self.version, 1 | 2) {
            return Err(Error::InvalidVersion(self.version));
        }
        Ok(())
    }

    #[inline(always)]
    pub fn estimated_p(&self) -> f64 {
        utils::estimate_p(self.count_estimate(), self.bit_size)
    }

    #[inline(always)]
    pub fn version(&self) -> u8 {
        self.version
    }

    #[inline(always)]
    pub fn cap(&self) -> usize {
        self.cap as usize
    }

    #[inline(always)]
    pub fn proba(&self) -> f64 {
        self.p
    }

    #[inline(always)]
    pub fn n_hash_fn(&self) -> u64 {
        self.n_hash
    }

    #[inline(always)]
    pub fn n_bits(&self) -> u64 {
        self.bit_size
    }

    #[inline]
    pub fn from_reader<R: Read>(r: R) -> Result<Self, Error> {
        let mut r = io::BufReader::new(r);

        let flags = read_le_u64(&mut r)?;
        let version = (flags & 0xff) as u8;

        let cap = read_le_u64(&mut r)?;
        let p = read_le_f64(&mut r)?;
        let n_hash = read_le_u64(&mut r)?;
        let bit_size = read_le_u64(&mut r)?;
        let count = read_le_u64(&mut r)?;

        // initializing bitset
        let u64_size = f64::ceil((bit_size as f64) / 64.0) as u64;
        let mut bitset = vec![0; u64_size as usize];

        // reading the bloom filter
        for i in bitset.iter_mut() {
            *i = read_le_u64(&mut r)?;
        }

        // reading data
        let data = std::io::read_to_string(r)?.as_bytes().to_vec();

        let fingerprint = Fingerprint::new(version, n_hash, bit_size);

        let b = BloomFilter {
            version,
            bitset,
            cap,
            p,
            n_hash,
            bit_size,
            count,
            data,
            fingerprint,
        };

        b.check_version()?;

        Ok(b)
    }

    #[inline]
    pub fn write<W: Write>(&self, w: &mut W) -> Result<(), Error> {
        let mut w = BufWriter::new(w);

        let flags = self.version as u64;
        w.write_all(&flags.to_le_bytes())?;
        w.write_all(&self.cap.to_le_bytes())?;
        w.write_all(&self.p.to_le_bytes())?;
        w.write_all(&self.n_hash.to_le_bytes())?;
        w.write_all(&self.bit_size.to_le_bytes())?;
        w.write_all(&self.count.to_le_bytes())?;

        for i in self.bitset.iter() {
            w.write_all(&i.to_le_bytes())?;
        }

        w.write_all(&self.data)?;
        Ok(())
    }

    #[inline(always)]
    /// get the nth bit value
    fn get_nth_bit(&self, index: u64) -> bool {
        let iblock = index / 64;
        let ibit = index % 64;
        // we cannot overflow shift as ibit < 64
        let bit = 1u64.wrapping_shl(ibit as u32);
        self.bitset[iblock as usize] & bit == bit
    }

    pub fn bits(&self) -> Vec<bool> {
        (0..self.bit_size)
            .map(|i| self.get_nth_bit(i))
            .collect::<Vec<bool>>()
    }

    #[inline(always)]
    pub fn insert<S: Sized>(&mut self, value: S) -> Result<(), Error> {
        self.insert_bytes(unsafe {
            slice::from_raw_parts(&value as *const S as *const u8, core::mem::size_of::<S>())
        })
    }

    #[inline(always)]
    pub fn insert_unchecked<S: Sized>(&mut self, value: S) {
        let _ = self.insert(value);
    }

    /// inserts a value into the bloom filter, as bloom filters are not easily
    /// growable an error is returned if we try to insert too many entries
    #[inline(always)]
    pub fn insert_bytes<S: AsRef<[u8]>>(&mut self, value: S) -> Result<(), Error> {
        let mut new = false;

        for index in self.fingerprint.fingerprint(value) {
            let iblock = index / 64;
            let ibit = index % 64;

            let entry = self
                .bitset
                .get_mut(iblock as usize)
                .expect("block index out of bound");

            // we cannot overflow shift as ibit < 64
            let bit = 1u64.wrapping_shl(ibit as u32);

            // this is the old bit value
            let old = *entry & bit == bit;
            if !old && self.count >= self.cap {
                return Err(Error::TooManyEntries);
            }

            // we update entry
            *entry |= bit;

            if !old {
                new = true
            }

            debug_assert!(self.get_nth_bit(index))
        }

        if new {
            self.count += 1;
        }

        Ok(())
    }

    #[inline(always)]
    pub fn insert_bytes_unchecked<S: AsRef<[u8]>>(&mut self, value: S) {
        let _ = self.insert_bytes(value);
    }

    /// clears out the bloom filter
    #[inline(always)]
    pub fn clear(&mut self) {
        self.bitset.iter_mut().for_each(|bucket| *bucket = 0);
        self.count = 0;
    }

    /// checks if an entry is contained in the bloom filter
    #[inline(always)]
    pub fn contains_bytes<S: AsRef<[u8]>>(&self, value: S) -> bool {
        for index in self.fingerprint.fingerprint(value) {
            if !self.get_nth_bit(index) {
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
    #[inline(always)]
    pub fn count_ones(&self) -> usize {
        self.bitset.iter().map(|u| u.count_ones() as usize).sum()
    }

    /// counts all the unset bits in the bloom filter
    #[inline(always)]
    pub fn count_zeros(&self) -> usize {
        self.bitset.iter().map(|u| u.count_zeros() as usize).sum()
    }

    /// function used to update the estimated count of entries
    fn update_count(&mut self) {
        self.count = (-(self.bit_size as f64
            * f64::ln(1.0 - (self.count_ones() as f64 / self.bit_size as f64)))
            / self.n_hash as f64) as u64;
    }

    /// returns an estimate of the number of element in the filter
    /// the exact number of element cannot be known as there might
    /// be collisions
    #[inline(always)]
    pub fn count_estimate(&self) -> u64 {
        self.count
    }

    #[inline(always)]
    pub fn size_in_u64(&self) -> usize {
        self.bitset.len()
    }

    #[inline(always)]
    pub fn size_in_bytes(&self) -> usize {
        self.bitset.len() * core::mem::size_of::<u64>()
    }

    #[inline(always)]
    pub fn has_same_params(&self, other: &Self) -> bool {
        self.version == other.version
            && self.cap == other.cap
            && self.p == other.p
            && self.n_hash == other.n_hash
            && self.bit_size == other.bit_size
            && self.bitset.len() == other.bitset.len()
    }

    /// makes the union of self with another bloom filter (having the same
    /// parameters)
    #[inline]
    pub fn union(&mut self, other: &Self) -> Result<(), Error> {
        if !self.has_same_params(other) {
            return Err(Error::Union(
                "cannot make union of bloom filters with different parameters".into(),
            ));
        }

        for (i, e) in self.bitset.iter_mut().enumerate() {
            *e |= other.bitset[i];
        }

        // we need to update the estimated number of elements after a union
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
#[macro_export]
macro_rules! bloom {
        ($cap:expr, $proba:expr) => {
            $crate::BloomFilter::with_capacity($cap, $proba)
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
        path::PathBuf,
    };

    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::utils::{time_it, ByteSize, Stats};

    use super::*;

    macro_rules! current_dir {
        () => {{
            let file = PathBuf::from(file!());
            PathBuf::from(file.parent().unwrap())
        }};
    }

    #[test]
    fn test_bloom() {
        let mut b = bloom!(100000, 0.001);
        assert!(!b.contains_bytes("value"));
        b.insert_bytes_unchecked("value");
        assert!(b.contains_bytes("value"));
        assert_eq!(b.count, 1);
        assert!(!b.contains_bytes("unknown"));
    }

    #[test]
    fn test_macro() {
        let b = bloom!(1000, 0.0001, ["hello", "world"]);
        assert!(b.contains_bytes("hello"));
        assert!(b.contains_bytes("world"));
    }

    #[test]
    fn test_serialization() {
        let b = bloom!(1000, 0.0001, ["deserialization", "test"]);
        let mut cursor = io::Cursor::new(vec![]);
        b.write(&mut cursor).unwrap();
        cursor.set_position(0);
        // deserializing the stuff out
        let b = BloomFilter::from_reader(cursor).unwrap();
        assert_eq!(b.proba(), 0.0001);
        assert!(b.contains_bytes("deserialization"));
        assert!(b.contains_bytes("test"));
        assert!(!b.contains_bytes("hello"));
        assert!(!b.contains_bytes("world"));
    }

    #[test]
    fn test_deserialization() {
        // this test file has been generated with Go bloom cli
        let data = include_bytes!("./data/test.bloom");
        let pb = bloom!(10000, 0.01).with_version(1).unwrap();
        let b = BloomFilter::from_reader(io::BufReader::new(io::Cursor::new(data))).unwrap();
        // proba it's been serialzed with with Go implementation
        assert!(pb.has_same_params(&b));
        assert!(b.contains_bytes("hello"));
        assert!(b.contains_bytes("world"));
        assert!(!b.contains_bytes("hello world"));
    }

    #[test]
    fn test_union() {
        let mut b = bloom!(1000, 0.0001, ["hello", "world"]);
        let o = bloom!(1000, 0.0001, ["union", "test"]);

        b.union(&o).unwrap();

        ["hello", "world", "union", "test"]
            .into_iter()
            .for_each(|v| {
                assert!(b.contains_bytes(v));
            });

        // estimate count should be exact for a small test like this
        assert_eq!(b.count_estimate(), 4);
    }

    #[test]
    fn test_union_failure() {
        let mut b = bloom!(1000, 0.0001, ["hello", "world"]);
        let o = bloom!(100, 0.0001, ["union", "test"]);

        assert!(b.union(&o).is_err())
    }

    #[test]
    fn test_clear() {
        let mut b = bloom!(1000, 0.0001, ["hello", "world"]);

        assert_eq!(b.count_estimate(), 2);
        b.clear();
        assert!(!b.contains_bytes("hello"));
        assert!(!b.contains_bytes("world"));
        assert_eq!(b.count_estimate(), 0);
    }

    #[test]
    fn test_too_many_entries() {
        let mut b = bloom!(5, 0.0001, ["hello", "world", "toasting", "bloom", "filter"]);

        assert_eq!(b.count_estimate(), 5);
        assert!(matches!(
            b.insert_bytes("everything should explode, OMG !"),
            Err(Error::TooManyEntries)
        ));
    }

    #[test]
    #[ignore]
    fn benchmark_reader() {
        let mut b = bloom!(2097152, 0.001).with_version(2).unwrap();
        let sample = current_dir!().join("data").join("sample.txt");

        let read = time_it(
            || {
                let f = fs::File::open(&sample).unwrap();
                let reader = io::BufReader::new(f);
                for line in reader.lines() {
                    b.insert_bytes_unchecked(line.unwrap());
                }
            },
            10,
        );

        println!("read time: {read:?}")
    }

    #[test]
    fn test_pow_2() {
        let n = 483409697;
        let proba = 0.001;

        let bucket_size = 4096;
        let c = utils::cap_from_bit_size(bucket_size * 8, proba);
        let n_bucket = (n as f64 / c as f64).ceil() as u64;
        println!(
            "cap_per_bucket={c} n_bucket={n_bucket} size={} one_vec={}",
            ByteSize::from_bytes((bucket_size * n_bucket) as usize),
            ByteSize::from_bits(utils::bit_size(n, proba) as usize),
        )
    }

    #[test]
    fn test_optimize() {
        let n = 425712207;
        let proba = 0.001;

        for p in BloomFilter::possible_optimizations(n, proba) {
            println!(
                "n={n} cap={} proba={} estimated_p={} bs={}",
                p.cap,
                p.proba,
                p.estimate_p(n),
                ByteSize::from_bits(p.bit_size() as usize),
            )
        }

        let p = BloomFilter::optimize(n, proba).unwrap();

        let bs = p.bit_size();

        println!(
            "initial: cap={n} proba={proba} bs={}",
            ByteSize::from_bits(utils::bit_size(n, proba) as usize)
        );
        println!(
            "optimized: cap={} proba={} bs={bs} -> {} estimated_proba={}",
            p.cap,
            p.proba,
            ByteSize::from_bits(bs as usize),
            utils::estimate_p(n, bs),
        );

        assert!(is_power_of_2(utils::bit_size(p.cap, p.proba)));
        assert!(p.cap >= n);
    }

    #[test]
    fn test_fp_rate() {
        let exp_fp_rate = 0.1;
        // 5% tolerance on the fp_rate
        let treshold = 0.05;

        // this value of n produces a bitset with a size power of 2
        let n = 109397;
        let mut b = bloom!(n, exp_fp_rate).with_version(2).unwrap();

        assert!(is_power_of_2(b.bit_size));

        for i in 0..n {
            b.insert(i).unwrap();
            assert!(b.contains(i));
        }

        let mut s = Stats::new();
        for i in n..n * 2 {
            if b.contains(i) {
                s.inc_fp()
            } else {
                s.inc_tn()
            }
        }

        println!("real fp rate = {}", s.fp_rate());
        assert!(s.fp_rate() < exp_fp_rate + exp_fp_rate * treshold);
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
        //let p = BloomFilter::optimize(count, fp_rate).unwrap();
        let p = Params {
            cap: count,
            proba: fp_rate,
        };

        let mut b = p.into_bloom_filter().with_version(1).unwrap();
        let mb_size = dataset_size as f64 / 1_048_576.0;
        let runs = 5;

        let insert_dur = time_it(
            || lines.iter().for_each(|l| b.insert_bytes_unchecked(l)),
            runs,
        );

        let bit_size = utils::bit_size(count, fp_rate);
        eprintln!("version of bloom-filter: {}", b.version());
        eprintln!("count: {}", count);
        eprintln!("proba: {}", fp_rate);
        eprintln!(
            "bit_size:{} optimized: {} expected_proba: {}",
            ByteSize::from_bits(bit_size as usize),
            is_power_of_2(bit_size),
            utils::estimate_p(lines.len() as u64, bit_size),
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

        eprintln!(
            "Count bit set to 1: {}",
            b.bits().iter().filter(|&&b| b).count()
        );
        eprintln!(
            "Count bit set to 0: {}",
            b.bits().iter().filter(|&&b| !b).count()
        );
        eprintln!("Size in bytes: {}", ByteSize::from_bytes(b.size_in_bytes()));
        eprintln!("\nCollision information:");
        eprintln!("\texpected collision rate: {}", fp_rate);
        eprintln!(
            "\treal collision rate: {:.5}%",
            1.0 - (b.count_estimate() as f64) / count as f64
        );
    }
}
