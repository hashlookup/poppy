use std::{
    fs,
    io::{self, Read},
    path::PathBuf,
};

use poppy::Params;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};

#[pyclass]
pub struct BloomFilter(poppy::BloomFilter);

struct Error(poppy::Error);

impl From<poppy::Error> for Error {
    fn from(value: poppy::Error) -> Self {
        Self(value)
    }
}

impl From<Error> for PyErr {
    fn from(value: Error) -> Self {
        PyValueError::new_err(value.0.to_string())
    }
}

#[pyfunction]
/// Loads a filter from bytes
pub fn loads(bytes: Vec<u8>) -> PyResult<BloomFilter> {
    let br = io::Cursor::new(bytes);
    Ok(BloomFilter(
        poppy::BloomFilter::from_reader(br).map_err(Error::from)?,
    ))
}

#[pyfunction]
/// Loads a filter from a given path
pub fn load(path: PathBuf) -> PyResult<BloomFilter> {
    Ok(BloomFilter(
        poppy::BloomFilter::from_reader(fs::File::open(path)?).map_err(Error::from)?,
    ))
}

#[pymethods]
impl BloomFilter {
    #[new]
    /// Creates a new filter with the given capacity and false positive probability
    fn new(capacity: usize, fpp: f64) -> PyResult<Self> {
        Ok(Self(
            poppy::BloomFilter::with_capacity(capacity, fpp).map_err(Error::from)?,
        ))
    }

    #[staticmethod]
    /// Creates a new filter with a given version. Pass version=1 if you
    /// want the filter being compatible with DCSO bloom filter tools.
    fn with_version(version: u8, capacity: usize, fpp: f64) -> PyResult<Self> {
        Ok(Self(
            poppy::BloomFilter::with_version_capacity(version, capacity, fpp)
                .map_err(Error::from)?,
        ))
    }

    #[staticmethod]
    /// Creates a new filter with given parameters
    fn with_params(version: u8, capacity: usize, fpp: f64, opt: u8) -> PyResult<Self> {
        let p = Params::new(capacity, fpp).version(version).opt(
            opt.try_into()
                .map_err(poppy::Error::from)
                .map_err(Error::from)?,
        );

        Ok(Self(p.try_into().map_err(Error::from)?))
    }

    /// Insert bytes into the filter
    pub fn insert_bytes(&mut self, data: &[u8]) -> PyResult<bool> {
        Ok(self.0.insert_bytes(data).map_err(Error::from)?)
    }

    /// Insert a str into the filter
    pub fn insert_str(&mut self, s: &str) -> PyResult<bool> {
        Ok(self.0.insert_bytes(s).map_err(Error::from)?)
    }

    /// Check if argument is contained in the filter
    pub fn contains_bytes(&mut self, data: &[u8]) -> bool {
        self.0.contains_bytes(data)
    }

    /// Check if argument is contained in the filter
    pub fn contains_str(&mut self, s: &str) -> bool {
        self.0.contains_bytes(s)
    }

    /// Returns true if filter is full
    pub fn is_full(&self) -> bool {
        self.0.is_full()
    }

    /// Merge two filters, doing the union of them this methods does an
    /// in-place merging into the current filter
    pub fn union_merge(&mut self, o: &Self) -> PyResult<()> {
        Ok(self.0.union_merge(&o.0).map_err(Error::from)?)
    }

    /// Estimate the number of common entries between two filters
    pub fn count_common_entries(&self, o: &Self) -> PyResult<usize> {
        Ok(self.0.count_common_entries(&o.0).map_err(Error::from)?)
    }

    /// Dumps bloom filter into a binary form
    pub fn dumps<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        let mut cursor = io::Cursor::new(vec![]);
        self.0.write(&mut cursor).map_err(Error::from)?;
        cursor.set_position(0);
        let b = PyBytes::new(py, cursor.bytes().flatten().collect::<Vec<u8>>().as_slice());
        Ok(b)
    }

    /// Save filter into a file
    pub fn save(&self, path: PathBuf) -> PyResult<()> {
        let mut f = fs::File::create(path)?;
        Ok(self.0.write(&mut f).map_err(Error::from)?)
    }

    // gather all the getters here

    #[getter]
    pub fn version(&self) -> u8 {
        self.0.version()
    }

    #[getter]
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    #[getter]
    pub fn fpp(&self) -> f64 {
        self.0.fpp()
    }

    #[getter]
    pub fn count_estimate(&self) -> usize {
        self.0.count_estimate() as usize
    }

    #[getter]
    pub fn data(&self) -> Vec<u8> {
        self.0.data().to_vec()
    }
}

/// Python bindings to Poppy bloom filter library (written in Rust)
#[pymodule]
#[pyo3(name = "poppy")]
fn poppy_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BloomFilter>()?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(loads, m)?)?;
    Ok(())
}
