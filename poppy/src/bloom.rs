use core::slice;

use std::io::{self, Read, Write};
use thiserror::Error;

mod utils;
pub use utils::*;

pub mod v1;
pub mod v2;

pub const DEFAULT_VERSION: u8 = 2;

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Flags {
    version: u8,
    opt: OptLevel,
    reserved: [u8; 6],
}

pub(crate) fn read_flags<R: Read>(r: &mut R) -> Result<Flags, Error> {
    let mut bytes = [0u8; 8];
    r.read_exact(bytes.as_mut_slice())?;
    Flags::from_bytes(bytes)
}

impl Flags {
    pub(crate) fn new(version: u8) -> Self {
        Self {
            version,
            opt: OptLevel::None,
            reserved: [0; 6],
        }
    }

    pub(crate) fn opt(mut self, o: OptLevel) -> Self {
        self.opt = o;
        self
    }

    pub(crate) fn to_bytes(&self) -> [u8; 8] {
        [self.version, self.opt as u8, 0, 0, 0, 0, 0, 0]
    }

    pub(crate) fn from_bytes(bytes: [u8; 8]) -> Result<Self, Error> {
        Ok(Self {
            version: bytes[0],
            opt: bytes[1].try_into()?,
            reserved: [bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]],
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum OptLevel {
    None,
    Space,
    Speed,
    Best,
}

impl OptLevel {
    fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Space => "space",
            Self::Speed => "speed",
            Self::Best => "best",
        }
    }
}

impl std::fmt::Display for OptLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Error)]
pub enum OptLevelError {
    #[error("invalid opt-level={0}")]
    Invalid(u8),
}

impl TryFrom<u8> for OptLevel {
    type Error = OptLevelError;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            v if v == Self::None as u8 => Ok(Self::None),
            v if v == Self::Space as u8 => Ok(Self::Space),
            v if v == Self::Speed as u8 => Ok(Self::Speed),
            v if v == Self::Best as u8 => Ok(Self::Best),
            _ => Err(OptLevelError::Invalid(value)),
        }
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
    #[error("{0}")]
    OptLevel(#[from] OptLevelError),
    #[error("too many entries, false positive rate cannot be met")]
    TooManyEntries,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Params {
    pub version: u8,
    pub capacity: usize,
    pub fpp: f64,
    pub opt: OptLevel,
}

impl Params {
    pub fn new(capacity: usize, fpp: f64) -> Self {
        Params {
            version: DEFAULT_VERSION,
            capacity,
            fpp,
            opt: OptLevel::None,
        }
    }

    pub fn version(mut self, version: u8) -> Self {
        self.version = version;
        self
    }

    pub fn opt(mut self, opt: OptLevel) -> Self {
        self.opt = opt;
        self
    }

    pub fn into_v1(self) -> v1::BloomFilter {
        self.into()
    }

    pub fn into_v2(self) -> v2::BloomFilter {
        self.into()
    }
}

impl TryFrom<Params> for BloomFilter {
    type Error = Error;
    fn try_from(p: Params) -> Result<Self, Self::Error> {
        Self::with_version_capacity_opt(p.version, p.capacity, p.fpp, p.opt)
    }
}

#[derive(Debug, Clone)]
pub enum BloomFilter {
    V1(v1::BloomFilter),
    V2(v2::BloomFilter),
}

impl BloomFilter {
    pub fn with_version_capacity_opt(
        v: u8,
        cap: usize,
        proba: f64,
        opt: OptLevel,
    ) -> Result<Self, Error> {
        match v {
            1 => Ok(Self::V1(v1::BloomFilter::with_capacity(cap as u64, proba))),
            2 => Ok(Self::V2(v2::BloomFilter::make(cap as u64, proba, opt))),
            _ => Err(Error::InvalidVersion(v)),
        }
    }

    pub fn with_version_capacity(v: u8, cap: usize, proba: f64) -> Result<Self, Error> {
        Self::with_version_capacity_opt(v, cap, proba, OptLevel::None)
    }

    pub fn with_capacity(cap: usize, proba: f64) -> Self {
        Self::V2(v2::BloomFilter::with_capacity(cap as u64, proba))
    }

    pub fn fill<S: AsRef<[u8]>>(&mut self, dataset: Vec<S>) -> Result<(), Error> {
        for entry in dataset {
            // cannot panic as we built the filter to be able to hold all dataset
            self.insert_bytes(entry)?;
        }
        Ok(())
    }

    pub fn from_reader<R: Read>(r: R) -> Result<Self, Error> {
        let mut r = io::BufReader::new(r);

        let flags = read_flags(&mut r)?;

        match flags.version {
            1 => Ok(Self::V1(v1::BloomFilter::from_reader_with_flags(r, flags)?)),
            2 => Ok(Self::V2(v2::BloomFilter::from_reader_with_flags(r, flags)?)),
            _ => Err(Error::InvalidVersion(flags.version)),
        }
    }

    #[inline]
    pub fn insert_bytes<D: AsRef<[u8]>>(&mut self, data: D) -> Result<bool, Error> {
        match self {
            Self::V1(b) => b.insert_bytes(data),
            Self::V2(b) => b.insert_bytes(data),
        }
    }

    #[inline]
    pub fn insert<S: Sized>(&mut self, value: S) -> Result<bool, Error> {
        self.insert_bytes(unsafe {
            slice::from_raw_parts(&value as *const S as *const u8, core::mem::size_of::<S>())
        })
    }

    #[inline]
    pub fn contains_bytes<D: AsRef<[u8]>>(&self, data: D) -> bool {
        match self {
            Self::V1(b) => b.contains_bytes(data),
            Self::V2(b) => b.contains_bytes(data),
        }
    }

    #[inline]
    pub fn contains<S: Sized>(&mut self, value: S) -> bool {
        self.contains_bytes(unsafe {
            slice::from_raw_parts(&value as *const S as *const u8, core::mem::size_of::<S>())
        })
    }

    #[inline]
    pub fn write<W: Write>(&self, w: &mut W) -> Result<(), Error> {
        match self {
            Self::V1(b) => b.write(w),
            Self::V2(b) => b.write(w),
        }
    }

    #[inline(always)]
    pub const fn version(&self) -> u8 {
        match self {
            Self::V1(_) => 1,
            Self::V2(_) => 2,
        }
    }

    #[inline]
    pub fn has_same_params(&self, other: &Self) -> bool {
        if self.version() != other.version() {
            return false;
        }

        match (self, other) {
            (Self::V1(s), Self::V1(o)) => s.has_same_params(o),
            (Self::V2(s), Self::V2(o)) => s.has_same_params(o),
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn union_merge(&mut self, other: &Self) -> Result<(), Error> {
        if !self.has_same_params(other) {
            return Err(Error::Merge(
                "cannot merge filters with different parameters".into(),
            ));
        }

        match (self, other) {
            (Self::V1(s), Self::V1(o)) => s.union_merge(o),
            (Self::V2(s), Self::V2(o)) => s.union_merge(o),
            _ => unreachable!(),
        }
    }

    /// This methods estimates the number of common entries between two filters
    #[inline]
    pub fn count_common_entries(&self, other: &Self) -> Result<usize, Error> {
        if !self.has_same_params(other) {
            return Err(Error::Merge(
                "cannot compare filters with different parameters".into(),
            ));
        }

        match (self, other) {
            (Self::V1(s), Self::V1(o)) => s.count_common_entries(o),
            (Self::V2(s), Self::V2(o)) => s.count_common_entries(o),
            _ => unreachable!(),
        }
    }

    pub fn capacity(&self) -> usize {
        match self {
            Self::V1(b) => b.capacity as usize,
            Self::V2(b) => b.capacity as usize,
        }
    }

    /// clears all bits (set to 0) of the filter
    pub fn clear(&mut self) {
        match self {
            Self::V1(b) => b.clear(),
            Self::V2(b) => b.clear(),
        }
    }

    /// false positive probability of the bloom filter
    pub fn fpp(&self) -> f64 {
        match self {
            Self::V1(b) => b.fpp,
            Self::V2(b) => b.fpp,
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::V1(b) => b.size_in_bytes(),
            Self::V2(b) => b.size_in_bytes(),
        }
    }

    pub fn count_estimate(&self) -> u64 {
        match self {
            Self::V1(b) => b.count_estimate(),
            Self::V2(b) => b.count_estimate(),
        }
    }

    pub fn data(&self) -> &[u8] {
        match self {
            Self::V1(b) => b.data.as_slice(),
            Self::V2(b) => b.data.as_slice(),
        }
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
    use crate::OptLevel;

    #[test]
    fn test_opt_level() {
        assert_eq!(OptLevel::try_from(0u8).unwrap(), OptLevel::None);
        assert_eq!(OptLevel::try_from(1u8).unwrap(), OptLevel::Space);
        assert_eq!(OptLevel::try_from(2u8).unwrap(), OptLevel::Speed);
        assert_eq!(OptLevel::try_from(3u8).unwrap(), OptLevel::Best);
    }
}
