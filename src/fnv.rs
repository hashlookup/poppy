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

    #[test]
    #[ignore]
    fn test_fnv_collisions() {
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

        for line in lines {
            if !collisions.insert(Fnv1Hasher::digest(line)) {
                n_collisions += 1.0
            } else {
                tn += 1.0
            }
            total += 1.0
        }

        println!("total number of values: {total}");
        println!(
            "% fnv collisions: {} -> {:.2}%",
            n_collisions,
            n_collisions * 100.0 / total
        );
        println!("% fnv fp rate: {:.4}", n_collisions / (tn + n_collisions));
    }
}
