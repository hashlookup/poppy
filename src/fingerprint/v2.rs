use std::u64;

use crate::{ahash::BloomAHasher, utils::is_power_of_2};

use super::Fingerprinter;

pub type Slow = AHashFingerprint<false>;
pub type Fast = AHashFingerprint<true>;

#[derive(Debug, Clone, Copy)]
pub enum Fingerprint {
    Slow(Slow),
    Fast(Fast),
}

impl Fingerprint {
    #[inline]
    pub fn new(count: u64, modulo: u64) -> Self {
        if is_power_of_2(modulo) {
            Self::Fast(Fast::new(count, modulo))
        } else {
            Self::Slow(Slow::new(count, modulo))
        }
    }

    #[inline(always)]
    pub fn fingerprint<S: AsRef<[u8]>>(mut self, data: S) -> Self {
        match &mut self {
            Self::Slow(f) => {
                *f = f.fingerprint(data);
            }
            Self::Fast(f) => {
                *f = f.fingerprint(data);
            }
        }
        self
    }
}

impl Iterator for Fingerprint {
    type Item = u64;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Slow(f) => f.next(),
            Self::Fast(f) => f.next(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct AHashFingerprint<const IS_POW_2: bool> {
    h1: u64,
    h2: u64,
    i: u64,
    modulo: u64,
    count: u64,
}

impl<const IS_POW_2: bool> Fingerprinter for AHashFingerprint<IS_POW_2> {
    fn new(count: u64, modulo: u64) -> Self {
        Self {
            modulo,
            count,
            ..Default::default()
        }
    }

    #[inline(always)]
    fn fingerprint<S: AsRef<[u8]>>(mut self, data: S) -> Self {
        self.h1 = BloomAHasher::digest(&data);
        self.h2 = BloomAHasher::digest(self.h1.to_be_bytes());
        self.i = 0;
        self
    }
}

impl<const IS_POW_2: bool> Iterator for AHashFingerprint<IS_POW_2> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.count {
            if self.i > 0 {
                self.h1 = self.h1.wrapping_add(self.h2);
                self.h2 = self.h2.wrapping_add(self.i);
            }
            self.i = self.i.wrapping_add(1);

            // shortcut if we are a power of 2
            if IS_POW_2 {
                return Some(self.h1 & (self.modulo - 1));
            }
            return Some(self.h1 % self.modulo);
        }
        None
    }
}
