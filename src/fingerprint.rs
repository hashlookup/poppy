pub(crate) mod v1;
pub(crate) mod v2;

pub trait Fingerprinter: Iterator<Item = u64> + std::fmt::Debug + Default + Clone {
    fn new(count: u64, modulo: u64) -> Self;
    fn fingerprint<S: AsRef<[u8]>>(self, data: S) -> Self;
}

#[derive(Debug, Clone, Copy)]
pub enum Fingerprint {
    V1(v1::Fingerprint),
    V2(v2::Fingerprint),
}

impl Fingerprint {
    pub fn new(version: u8, count: u64, modulo: u64) -> Self {
        match version {
            1 => Self::V1(v1::Fingerprint::new(count, modulo)),
            _ => Self::V2(v2::Fingerprint::new(count, modulo)),
        }
    }

    pub fn fingerprint<S: AsRef<[u8]>>(mut self, data: S) -> Self {
        match &mut self {
            Self::V1(f) => {
                *f = f.fingerprint(data);
            }
            Self::V2(f) => {
                *f = f.fingerprint(data);
            }
        };
        self
    }
}

impl Iterator for Fingerprint {
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::V1(f) => f.next(),
            Self::V2(f) => f.next(),
        }
    }
}
