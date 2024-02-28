use std::hash::{BuildHasher, Hasher};

use ahash::{AHasher, RandomState};

// poppy seed
const SEED: u64 = 0x706f707079;

#[derive(Debug)]
pub struct BloomAHasher {
    h: AHasher,
}

impl Default for BloomAHasher {
    fn default() -> Self {
        Self {
            h: RandomState::with_seeds(SEED, SEED, SEED, SEED).build_hasher(),
        }
    }
}

impl Hasher for BloomAHasher {
    fn write(&mut self, bytes: &[u8]) {
        self.h.write(bytes)
    }

    fn finish(&self) -> u64 {
        self.h.finish()
    }
}

impl BloomAHasher {
    #[inline(always)]
    pub fn digest<S: AsRef<[u8]>>(s: S) -> u64 {
        let mut h = Self::default();
        h.write(s.as_ref());
        h.finish()
    }
}

#[cfg(test)]
mod test {
    use std::{
        collections::HashSet,
        io::{self, BufRead},
    };

    use super::BloomAHasher;

    #[test]
    fn test_randomness() {
        let h = BloomAHasher::digest("toto");
        assert_eq!(h, 14712558333585361687);
    }

    #[test]
    #[ignore]
    fn test_ahash_collisions() {
        let data = include_bytes!("./data/sample.txt");
        let reader = io::BufReader::new(io::Cursor::new(data));
        let lines: HashSet<String> = reader
            .lines()
            .map_while(Result::ok)
            .collect::<HashSet<String>>();

        let mut collisions = HashSet::new();
        let mut total = 0f64;
        let mut n_collisions = 0f64;
        let mut tn = 0f64;

        for line in &lines {
            if !collisions.insert(BloomAHasher::digest(line)) {
                n_collisions += 1.0
            } else {
                tn += 1.0
            }
            total += 1.0
        }

        println!("total number of values: {}", total);
        println!("% ahash collisions: {:.2}%", n_collisions * 100.0 / total);
        println!("% ahash fp rate: {:.4}", n_collisions / (tn + n_collisions));
    }
}
