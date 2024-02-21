use std::hash::Hasher;

const FNV_OFFSET: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

#[derive(Debug)]
pub struct Fnv1Hasher {
    sum: u64,
}

impl Default for Fnv1Hasher {
    fn default() -> Self {
        Self { sum: FNV_OFFSET }
    }
}

impl Hasher for Fnv1Hasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.sum
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        bytes.iter().for_each(|b| {
            self.sum = self.sum.wrapping_mul(FNV_PRIME);
            self.sum ^= *b as u64;
        })
    }
}

impl Fnv1Hasher {
    // ToDo move this to a generic taking a Hasher as param
    pub fn digest<S: AsRef<[u8]>>(s: S) -> u64 {
        let mut h = Self::default();
        h.write(s.as_ref());
        h.finish()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    // this test checks that FnvHasher returns the exact same value as the
    // go library (used in DCSO implementation)
    fn test_fnv() {
        assert_eq!(Fnv1Hasher::digest("Hello, World!"), 8889723880822884486);
        assert_eq!(
            Fnv1Hasher::digest("Let's rustify all this"),
            13581150826273240441
        );
    }

    #[test]
    fn test_fnv_update() {
        let mut hasher = Fnv1Hasher::default();
        hasher.write("Hello, ".as_bytes());
        hasher.write("World!".as_bytes());
        assert_eq!(hasher.finish(), 8889723880822884486)
    }
}
