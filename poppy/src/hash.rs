use std::hash::Hasher;

pub(crate) mod fnv;
pub(crate) mod wyhash;

pub trait PoppyHasher: Hasher + Default {
    #[inline(always)]
    fn hash_one<S: AsRef<[u8]>>(s: S) -> u64 {
        let mut h = Self::default();
        h.write(s.as_ref());
        h.finish()
    }
}
