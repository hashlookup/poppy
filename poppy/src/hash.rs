use std::hash::{Hash, Hasher};
use std::mem::size_of;

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

/// Trait to implement for custom types to be inserted into filters
///
/// # Example
///
/// ```
/// use poppy_filters::PoppyHash;
///
/// #[derive(Hash)]
/// struct MyStruct {
///     some_int: i64,
///     s: String,
/// }
///
/// // PoppyHash can simply be implemented like this
/// // for any structure also implementing Hash trait
/// impl PoppyHash for MyStruct {};
///
/// ```
pub trait PoppyHash: Hash {
    fn hash_pop<H: PoppyHasher>(&self) -> u64 {
        let mut hasher = H::default();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

impl<T> PoppyHash for Option<T> where T: Hash {}

macro_rules! impl_poppy_hash {
    ($($t:ty),*) => {
        $(impl PoppyHash for $t {})*
    };
}

impl_poppy_hash!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

impl PoppyHash for &[u8] {
    fn hash_pop<H: PoppyHasher>(&self) -> u64 {
        if self.len() > size_of::<u64>() {
            digest::<H, &[u8]>(self)
        } else {
            // if data is smaller than u64 we don't need to hash it
            let mut tmp = [0u8; size_of::<u64>()];
            self.iter().enumerate().for_each(|(i, &b)| tmp[i] = b);
            u64::from_le_bytes(tmp)
        }
    }
}

impl PoppyHash for Vec<u8> {
    fn hash_pop<H: PoppyHasher>(&self) -> u64 {
        self.as_slice().hash_pop::<H>()
    }
}

impl PoppyHash for &str {
    fn hash_pop<H: PoppyHasher>(&self) -> u64 {
        self.as_bytes().hash_pop::<H>()
    }
}

impl PoppyHash for String {
    fn hash_pop<H: PoppyHasher>(&self) -> u64 {
        self.as_bytes().hash_pop::<H>()
    }
}

pub(crate) fn digest<H: Hasher + Default, S: AsRef<[u8]>>(s: S) -> u64 {
    let mut h = H::default();
    h.write(s.as_ref());
    h.finish()
}
