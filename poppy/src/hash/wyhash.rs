use std::hash::Hasher;

use wyhash::WyHash;

use super::PoppyHasher;

// poppy seed
const SEED: u64 = 0x706f707079533d42;

#[derive(Clone)]
pub struct WyHasher {
    h: WyHash,
}

impl Default for WyHasher {
    fn default() -> Self {
        Self {
            h: WyHash::with_seed(SEED),
        }
    }
}

impl Hasher for WyHasher {
    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        self.h.write(bytes)
    }

    #[inline(always)]
    fn finish(&self) -> u64 {
        self.h.finish()
    }
}

impl PoppyHasher for WyHasher {}

#[cfg(test)]
mod test {

    use crate::hash::PoppyHasher;

    use super::WyHasher;

    #[test]
    fn test_hasher() {
        println!("{}", WyHasher::hash_one("poppy"));
        assert_eq!(16507271990128044474, WyHasher::hash_one("poppy"));
    }
}
