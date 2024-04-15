use statrs::statistics::Statistics;
use std::{
    fmt::Display,
    hash::Hash,
    io::{self, Read},
    time::Duration,
};

pub(crate) fn read_le_u64<R: Read>(r: &mut R) -> Result<u64, io::Error> {
    let mut bytes = [0u8; 8];
    r.read_exact(bytes.as_mut_slice())?;
    Ok(u64::from_le_bytes(bytes))
}

pub(crate) fn read_le_f64<R: Read>(r: &mut R) -> Result<f64, io::Error> {
    let mut bytes = [0u8; 8];
    r.read_exact(bytes.as_mut_slice())?;
    Ok(f64::from_le_bytes(bytes))
}

pub enum ByteSize {
    Bytes(usize),
    Kilo(usize),
    Mega(usize),
    Giga(usize),
}

impl Eq for ByteSize {}

impl Hash for ByteSize {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.in_bytes().hash(state)
    }
}

impl PartialEq for ByteSize {
    fn eq(&self, other: &Self) -> bool {
        self.in_bytes() == other.in_bytes()
    }
}

impl PartialOrd for ByteSize {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.in_bytes().cmp(&other.in_bytes()))
    }
}

const KILO: usize = 1 << 10;
const MEGA: usize = 1 << 20;
const GIGA: usize = 1 << 30;

impl Display for ByteSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1}{}", self.in_unit(), self.unit_str())
    }
}

impl ByteSize {
    #[inline(always)]
    pub fn from_bits(b: usize) -> Self {
        Self::from_bytes(b / 8)
    }

    #[inline(always)]
    pub fn from_bytes(b: usize) -> Self {
        if b < KILO {
            Self::Bytes(b)
        } else if b < MEGA {
            Self::Kilo(b)
        } else if b < GIGA {
            Self::Mega(b)
        } else {
            Self::Giga(b)
        }
    }

    #[inline(always)]
    pub fn from_mb(mb: usize) -> Self {
        Self::from_bytes(mb * MEGA)
    }

    #[inline(always)]
    fn unit_str(&self) -> &'static str {
        match self {
            Self::Bytes(_) => "B",
            Self::Kilo(_) => "KB",
            Self::Mega(_) => "MB",
            Self::Giga(_) => "GB",
        }
    }

    #[inline(always)]
    pub fn in_bytes(&self) -> usize {
        match self {
            Self::Bytes(b) => *b,
            Self::Kilo(b) => *b,
            Self::Mega(b) => *b,
            Self::Giga(b) => *b,
        }
    }

    #[inline(always)]
    pub fn in_mb(&self) -> f64 {
        match self {
            Self::Bytes(b) => *b as f64 / MEGA as f64,
            Self::Kilo(b) => *b as f64 / MEGA as f64,
            Self::Mega(b) => *b as f64 / MEGA as f64,
            Self::Giga(b) => *b as f64 / MEGA as f64,
        }
    }

    #[inline(always)]
    pub fn as_bytes(self) -> Self {
        match self {
            Self::Bytes(b) => Self::Bytes(b),
            Self::Kilo(b) => Self::Bytes(b),
            Self::Mega(b) => Self::Bytes(b),
            Self::Giga(b) => Self::Bytes(b),
        }
    }

    #[inline(always)]
    pub fn in_unit(&self) -> f64 {
        match self {
            Self::Bytes(b) => *b as f64,
            Self::Kilo(b) => *b as f64 / KILO as f64,
            Self::Mega(b) => *b as f64 / MEGA as f64,
            Self::Giga(b) => *b as f64 / GIGA as f64,
        }
    }
}

#[inline(always)]
pub fn benchmark<F: FnMut()>(mut f: F, run: u32) -> Duration {
    let it = run;
    let mut times = Vec::with_capacity(run as usize);
    for _ in 0..it {
        let start_time = std::time::Instant::now();
        f();
        let end_time = std::time::Instant::now();
        times.push((end_time - start_time).as_secs_f64())
    }

    if run == 1 {
        return Duration::from_secs_f64(times[0]);
    }

    // Calculate the mean and standard deviation of the data
    let mean = times.as_slice().mean();
    let std_dev = times.as_slice().std_dev();

    // Define the threshold for outliers (e.g., 3 standard deviations from the mean)
    let threshold = 3.0 * std_dev;

    // Remove elements that are outside the threshold
    times.retain(|&x| (x - mean).abs() <= threshold);
    Duration::from_secs_f64(times.mean())
}

pub fn time_it_once<F: FnMut()>(f: F) -> Duration {
    benchmark(f, 1)
}

#[derive(Debug, Default)]
pub struct Stats {
    fp: f64,
    tp: f64,
    tn: f64,
    _fn: f64,
    total: usize,
}

impl Stats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fp_rate(&self) -> f64 {
        self.fp / (self.tn + self.fp)
    }

    pub fn inc_tp(&mut self) {
        self.tp += 1.0;
        self.total += 1
    }

    pub fn inc_fn(&mut self) {
        self._fn += 1.0;
        self.total += 1
    }

    pub fn inc_fp(&mut self) {
        self.fp += 1.0;
        self.total += 1
    }

    pub fn inc_tn(&mut self) {
        self.tn += 1.0;
        self.total += 1
    }

    pub fn total(&self) -> usize {
        self.total
    }
}
