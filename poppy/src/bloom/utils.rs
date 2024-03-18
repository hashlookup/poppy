#[inline(always)]
pub(crate) fn k(bit_size: u64, cap: u64) -> u64 {
    f64::ceil(f64::ln(2.0) * bit_size as f64 / (cap as f64)) as u64
}

#[inline(always)]
pub fn bit_size(cap: u64, proba: f64) -> u64 {
    f64::abs(f64::ceil(
        (cap as f64) * f64::ln(proba) / f64::ln(2.0).powf(2.0),
    )) as u64
}

#[inline(always)]
/// estimates the false positive probability given a number of element in a bloom
/// filter and its size in bits
pub(crate) fn estimate_p(n: u64, bit_size: u64) -> f64 {
    let k = k(bit_size, n);
    (1.0 - f64::exp(-(k as f64) * n as f64 / bit_size as f64)).powf(k as f64)
}

#[inline(always)]
pub(crate) fn cap_from_bit_size(bit_size: u64, proba: f64) -> u64 {
    f64::abs(bit_size as f64 * f64::ln(2.0).powf(2.0) / f64::ln(proba)) as u64
}
