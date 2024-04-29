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

/// Structure used for easy filter creation from a set
/// of parameters
///
/// # Example
///
/// ```
/// use poppy_filters::Params;
///
/// let p = Params::new(1000, 0.1);
/// // will create a V2 version of BloomFilter from
/// // parameters
/// let bf = p.into_v2();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Params {
    /// version of the filter
    pub version: u8,
    /// capacity of the filter
    pub capacity: usize,
    /// false positive probability
    pub fpp: f64,
    /// optimization level
    pub opt: OptLevel,
}

impl Params {
    /// Creates new parameters given a filter capacity and
    /// false positive probability (fpp)
    pub fn new(capacity: usize, fpp: f64) -> Self {
        Params {
            version: DEFAULT_VERSION,
            capacity,
            fpp,
            opt: OptLevel::None,
        }
    }

    /// Sets the version of the filter we want to create
    pub fn version(mut self, version: u8) -> Self {
        self.version = version;
        self
    }

    /// Sets the optimization level of the filter
    pub fn opt(mut self, opt: OptLevel) -> Self {
        self.opt = opt;
        self
    }

    /// Turn Params into [v1::BloomFilter]
    pub fn into_v1(self) -> v1::BloomFilter {
        self.into()
    }

    /// Turn Params into [v2::BloomFilter]
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
/// Common BloomFilter implementation supporting both
/// [v1::BloomFilter] and [v2::BloomFilter].
///
/// # Example
///
/// ```
/// use poppy_filters::BloomFilter;
///
/// let mut bf = BloomFilter::with_capacity(1042, 0.1);
///
/// bf.insert_bytes("toto");
/// assert!(bf.contains_bytes(String::from("toto")));
/// ```
pub enum BloomFilter {
    V1(v1::BloomFilter),
    V2(v2::BloomFilter),
}

impl BloomFilter {
    /// Creates a new filter given the settings passed as parameters
    pub fn with_version_capacity_opt(
        version: u8,
        cap: usize,
        fpp: f64,
        opt: OptLevel,
    ) -> Result<Self, Error> {
        match version {
            1 => Ok(Self::V1(v1::BloomFilter::with_capacity(cap as u64, fpp))),
            2 => Ok(Self::V2(v2::BloomFilter::make(cap as u64, fpp, opt))),
            _ => Err(Error::InvalidVersion(version)),
        }
    }

    /// Creates a new filter given a version, capacity and false positive probability (fpp)
    pub fn with_version_capacity(version: u8, cap: usize, fpp: f64) -> Result<Self, Error> {
        Self::with_version_capacity_opt(version, cap, fpp, OptLevel::None)
    }

    /// Creates a new filter with a given capacity and false positive probability (fpp)
    pub fn with_capacity(cap: usize, fpp: f64) -> Self {
        Self::V2(v2::BloomFilter::with_capacity(cap as u64, fpp))
    }

    /// Fill a bloom filter with several entries
    pub fn fill<S: AsRef<[u8]>>(&mut self, dataset: Vec<S>) -> Result<(), Error> {
        for entry in dataset {
            self.insert_bytes(entry)?;
        }
        Ok(())
    }

    /// Creates a BloomFilter from a reader. The data read is supposed
    /// to be in the appropriate format, if not the function returns an
    /// Err.
    pub fn from_reader<R: Read>(r: R) -> Result<Self, Error> {
        let mut r = io::BufReader::new(r);

        let flags = read_flags(&mut r)?;

        match flags.version {
            1 => Ok(Self::V1(v1::BloomFilter::from_reader_with_flags(r, flags)?)),
            2 => Ok(Self::V2(v2::BloomFilter::from_reader_with_flags(r, flags)?)),
            _ => Err(Error::InvalidVersion(flags.version)),
        }
    }

    /// Insert data inside the filter. An error is returned if insertion failed.
    #[inline]
    pub fn insert_bytes<D: AsRef<[u8]>>(&mut self, data: D) -> Result<bool, Error> {
        match self {
            Self::V1(b) => b.insert_bytes(data),
            Self::V2(b) => b.insert_bytes(data),
        }
    }

    /// Returns true if bytes are contained in the filter.
    #[inline]
    pub fn contains_bytes<D: AsRef<[u8]>>(&self, data: D) -> bool {
        match self {
            Self::V1(b) => b.contains_bytes(data),
            Self::V2(b) => b.contains_bytes(data),
        }
    }

    /// Write the filter into a writer implementing [Write]
    #[inline]
    pub fn write<W: Write>(&self, w: &mut W) -> Result<(), Error> {
        match self {
            Self::V1(b) => b.write(w),
            Self::V2(b) => b.write(w),
        }
    }

    /// Returns the version of the filter
    #[inline(always)]
    pub const fn version(&self) -> u8 {
        match self {
            Self::V1(_) => 1,
            Self::V2(_) => 2,
        }
    }

    /// Returns true if the current filter and other have the same parameters
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

    /// Merge current filter with another one. The result of the union is
    /// the current filter.
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

    /// Returns the capacity of the filter
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        match self {
            Self::V1(b) => b.capacity as usize,
            Self::V2(b) => b.capacity as usize,
        }
    }

    /// Clears all bits (set to 0) of the filter
    pub fn clear(&mut self) {
        match self {
            Self::V1(b) => b.clear(),
            Self::V2(b) => b.clear(),
        }
    }

    /// Returns the false positive probability of the filter
    #[inline(always)]
    pub fn fpp(&self) -> f64 {
        match self {
            Self::V1(b) => b.fpp,
            Self::V2(b) => b.fpp,
        }
    }

    /// Return the size in bytes of the filter
    #[inline(always)]
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::V1(b) => b.size_in_bytes(),
            Self::V2(b) => b.size_in_bytes(),
        }
    }

    /// Returns the estimated count of element inserted into the  filter
    #[inline(always)]
    pub fn count_estimate(&self) -> u64 {
        match self {
            Self::V1(b) => b.count_estimate(),
            Self::V2(b) => b.count_estimate(),
        }
    }

    /// Returns true if the filter is full
    pub fn is_full(&self) -> bool {
        self.count_estimate() as usize == self.capacity()
    }

    /// Returns the blob of data joined to the filter
    #[inline(always)]
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
/// use poppy_filters::bloom;
///
/// let mut b = bloom!(100, 0.1);
/// b.insert_bytes("hello");
/// assert!(b.contains_bytes("hello"));
/// ```
///
/// # Other Example
/// ```
/// use poppy_filters::bloom;
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

    #[test]
    fn test_is_full() {
        let mut b = bloom!(10, 0.001);
        assert!(!b.is_full());
        (0..10i32).for_each(|i| {
            b.insert_bytes(i.to_le_bytes()).unwrap();
        });
        assert!(b.is_full())
    }
}
