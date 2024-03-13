use crate::utils::read_le_u64;
use core::slice;
use std::hash::Hash;

use std::io::{self, Read, Write};
use thiserror::Error;

mod utils;
pub use utils::*;

pub mod v1;
pub mod v2;

pub const DEFAULT_VERSION: u8 = 2;

#[derive(Debug, PartialEq)]
#[repr(u8)]
pub enum OptLevel {
    None,
    Space,
    Speed,
    Best,
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
    pub fn into_v1(self) -> v1::BloomFilter {
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
pub enum BloomFilter {
    V1(v1::BloomFilter),
    V2(v2::BloomFilter),
}

impl BloomFilter {
    pub fn with_version_capacity_opt(
        v: u8,
        cap: u64,
        proba: f64,
        opt: OptLevel,
    ) -> Result<Self, Error> {
        match v {
            1 => Ok(Self::V1(v1::BloomFilter::with_capacity(cap, proba))),
            2 => Ok(Self::V2(v2::BloomFilter::make(cap, proba, opt))),
            _ => Err(Error::InvalidVersion(v)),
        }
    }

    pub fn with_version_capacity(v: u8, cap: u64, proba: f64) -> Result<Self, Error> {
        Self::with_version_capacity_opt(v, cap, proba, OptLevel::None)
    }

    pub fn with_capacity(cap: u64, proba: f64) -> Self {
        Self::V2(v2::BloomFilter::with_capacity(cap, proba))
    }

    pub fn from_reader<R: Read>(r: R) -> Result<Self, Error> {
        let mut r = io::BufReader::new(r);

        let flags = read_le_u64(&mut r)?;
        let version = (flags & 0xff) as u8;

        match version {
            1 => Ok(Self::V1(v1::BloomFilter::from_reader_skip_version(r)?)),
            2 => Ok(Self::V2(v2::BloomFilter::from_reader_skip_version(r)?)),
            _ => Err(Error::InvalidVersion(version)),
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

    pub fn cap(&self) -> usize {
        match self {
            Self::V1(b) => b.cap(),
            Self::V2(b) => b.capacity as usize,
        }
    }
    /// false positive probability of the bloom filter
    pub fn fpp(&self) -> f64 {
        match self {
            Self::V1(b) => b.proba(),
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
