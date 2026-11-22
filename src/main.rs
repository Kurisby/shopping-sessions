// Shopping Sessions pipeline: native, multithreaded, std only.
// Reads $DATA_DIR, writes Parquet to $OUTPUT_DIR/user_daily and $OUTPUT_DIR/campaign_daily.

use std::arch::x86_64::*;
use std::fs::{self, File};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::*};
use std::time::Instant;

mod parquet;
use parquet::{Col, Data, Kind};

const DAY: i64 = 86_400_000;
const WEEK: i64 = 7 * DAY;
const GAP: u32 = 1_800_000;

// ---------------------------------------------------------------- memory

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
}
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_PRIVATE: i32 = 2;
const MAP_FIXED: i32 = 0x10;
const MAP_ANON: i32 = 0x20;
const MADV_HUGEPAGE: i32 = 14;
const MADV_DONTNEED: i32 = 4;

/// Big mappings (start, len), released in parallel before exit so the kernel's
/// single-threaded teardown has little left to do.
static REGIONS: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());
static NO_THP: AtomicBool = AtomicBool::new(false);

fn note_region(p: *mut u8, len: usize) {
    REGIONS.lock().unwrap().push((p as usize, len));
}

fn release_regions(nthreads: usize) {
    const PIECE: usize = 32 << 20;
    let regs = std::mem::take(&mut *REGIONS.lock().unwrap());
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    for (p, len) in regs {
        let len = (len + 4095) & !4095;
        let mut o = 0;
        while o < len {
            let n = PIECE.min(len - o);
            pieces.push((p + o, n));
            o += n;
        }
    }
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..nthreads {
            s.spawn(|| loop {
                let j = next.fetch_add(1, Relaxed);
                if j >= pieces.len() {
                    break;
                }
                unsafe {
                    madvise(pieces[j].0 as *mut u8, pieces[j].1, MADV_DONTNEED);
                }
            });
        }
    });
}

/// Maps a file read-only with at least one zero page after it, so SIMD loads may run past the end.
fn map_file(path: &Path) -> &'static [u8] {
    let f = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let len = f.metadata().unwrap().len() as usize;
    if len == 0 {
        return &[];
    }
    unsafe {
        let total = (len + 4095) / 4096 * 4096 + 4096;
        let base = mmap(std::ptr::null_mut(), total, PROT_READ, MAP_PRIVATE | MAP_ANON, -1, 0);
        if base as isize == -1 {
            panic!("mmap anon failed");
        }
        let p = mmap(base, len, PROT_READ, MAP_PRIVATE | MAP_FIXED, f.as_raw_fd(), 0);
        if p as isize == -1 || p != base {
            panic!("mmap failed for {}", path.display());
        }
        note_region(p, len);
        std::slice::from_raw_parts(p, len)
    }
}

/// Zeroed anonymous memory with transparent huge pages requested.
fn zeroed<T>(n: usize) -> &'static mut [T] {
    let bytes = (n * std::mem::size_of::<T>()).max(1);
    unsafe {
        let total = (bytes + (2 << 20) - 1) / (2 << 20) * (2 << 20);
        let p = mmap(std::ptr::null_mut(), total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if p as isize == -1 {
            panic!("mmap anon failed");
        }
        if !NO_THP.load(Relaxed) {
            madvise(p, total, MADV_HUGEPAGE);
        }
        note_region(p, total);
        std::slice::from_raw_parts_mut(p as *mut T, n)
    }
}

/// Small files read into padded heap memory.
fn read_padded(path: &Path) -> &'static [u8] {
    let mut v = fs::read(path).unwrap_or_default();
    let n = v.len();
    v.extend_from_slice(&[0u8; 64]);
    let s: &'static [u8] = Box::leak(v.into_boxed_slice());
    &s[..n]
}

fn csv_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |x| x == "csv") && p.is_file())
            .collect(),
        Err(_) => Vec::new(),
    };
    v.sort();
    v
}

#[inline]
fn memchr_nl(s: &[u8], from: usize) -> usize {
    match s[from..].iter().position(|&c| c == b'\n') {
        Some(p) => from + p,
        None => s.len(),
    }
}

/// Splits a file (header skipped) into chunks of whole lines.
fn chunks_of(file: &'static [u8], target: usize, out: &mut Vec<&'static [u8]>) {
    let first = memchr_nl(file, 0);
    if first >= file.len() {
        return;
    }
    let mut s = first + 1;
    while s < file.len() {
        let mut e = s + target;
        if e >= file.len() {
            e = file.len();
        } else {
            e = memchr_nl(file, e);
            if e < file.len() {
                e += 1;
            }
        }
        out.push(&file[s..e]);
        s = e;
    }
}

// ---------------------------------------------------------------- refs and hashes

#[inline(always)]
fn mkref(p: *const u8, n: usize) -> u64 {
    (p as u64) | ((n.min(0xFFFF) as u64) << 48)
}
#[inline(always)]
fn deref(r: u64) -> &'static [u8] {
    unsafe { std::slice::from_raw_parts((r & 0xFFFF_FFFF_FFFF) as *const u8, (r >> 48) as usize) }
}

const S0: u64 = 0xa0761d6478bd642f;
const S1: u64 = 0xe7037ed1a0b428db;
const S3: u64 = 0x589965cc75374cc3;

/// 64x64 -> 128 multiply, folded: every output bit depends on every input bit.
#[inline(always)]
fn mum(a: u64, b: u64) -> u64 {
    let r = (a as u128) * (b as u128);
    (r as u64) ^ ((r >> 64) as u64)
}

/// wyhash-style string hash (full 64-bit avalanche; the old multiply-rotate chain
/// collided on 9-byte IDs that share a prefix). Never returns 0.
/// Reads up to 7 bytes past the end when n < 8: only for padded memory.
#[inline(always)]
unsafe fn hash_ptr(p: *const u8, n: usize) -> u64 {
    let rd = |i: usize| std::ptr::read_unaligned(p.add(i) as *const u64);
    let mut seed = S1 ^ n as u64;
    let (a, b) = if n < 8 {
        if n == 0 {
            (0, 0)
        } else {
            (rd(0) & (u64::MAX >> (64 - 8 * n)), 0)
        }
    } else if n <= 16 {
        (rd(0), rd(n - 8))
    } else {
        let mut i = 0;
        while n - i > 16 {
            seed = mum(rd(i) ^ S0, rd(i + 8) ^ seed);
            i += 16;
        }
        (rd(n - 16), rd(n - 8))
    };
    let h = mum(S3 ^ n as u64, mum(a ^ S0, b ^ seed));
    if h == 0 {
        1
    } else {
        h
    }
}

/// Same value as hash_ptr, safe on any slice.
fn hash_bytes(b: &[u8]) -> u64 {
    if b.len() >= 8 {
        unsafe { hash_ptr(b.as_ptr(), b.len()) }
    } else {
        let mut buf = [0u8; 8];
        buf[..b.len()].copy_from_slice(b);
        unsafe { hash_ptr(buf.as_ptr(), b.len()) }
    }
}

// ---------------------------------------------------------------- u64 -> value open addressing map

struct U64Map<V: Copy + Default> {
    keys: Vec<u64>,
    vals: Vec<V>,
    mask: usize,
    len: usize,
}

impl<V: Copy + Default> U64Map<V> {
    fn with_capacity(n: usize) -> Self {
        let cap = (n * 2).next_power_of_two().max(16);
        U64Map { keys: vec![0; cap], vals: vec![V::default(); cap], mask: cap - 1, len: 0 }
    }
    fn clear(&mut self) {
        if self.len > 0 {
            self.keys.iter_mut().for_each(|k| *k = 0);
            self.len = 0;
        }
    }
    #[inline]
    fn get(&self, k: u64) -> Option<V> {
        let mut i = (k as usize) & self.mask;
        loop {
            let kk = unsafe { *self.keys.get_unchecked(i) };
            if kk == k {
                return Some(unsafe { *self.vals.get_unchecked(i) });
            }
            if kk == 0 {
                return None;
            }
            i = (i + 1) & self.mask;
        }
    }
    /// Returns (slot, inserted)
    #[inline]
    fn slot(&mut self, k: u64) -> (usize, bool) {
        if (self.len + 1) * 2 > self.keys.len() {
            self.grow();
        }
        let mut i = (k as usize) & self.mask;
        loop {
            let kk = unsafe { *self.keys.get_unchecked(i) };
            if kk == k {
                return (i, false);
            }
            if kk == 0 {
                self.keys[i] = k;
                self.len += 1;
                return (i, true);
            }
            i = (i + 1) & self.mask;
        }
    }
    fn grow(&mut self) {
        let old_k = std::mem::take(&mut self.keys);
        let old_v = std::mem::take(&mut self.vals);
        let cap = old_k.len() * 2;
        self.keys = vec![0; cap];
        self.vals = vec![V::default(); cap];
        self.mask = cap - 1;
        for (k, v) in old_k.into_iter().zip(old_v) {
            if k != 0 {
                let mut i = (k as usize) & self.mask;
                while self.keys[i] != 0 {
                    i = (i + 1) & self.mask;
                }
                self.keys[i] = k;
                self.vals[i] = v;
            }
        }
    }
}

// ---------------------------------------------------------------- timestamps

#[inline(always)]
fn d2(v: &[u8], i: usize) -> Option<i64> {
    let a = v[i].wrapping_sub(b'0');
    let b = v[i + 1].wrapping_sub(b'0');
    if a > 9 || b > 9 {
        None
    } else {
        Some(a as i64 * 10 + b as i64)
    }
}

#[inline(always)]
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((m + 9) % 12) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[inline(always)]
fn is_leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

#[inline(always)]
fn parse_iso(v: &[u8]) -> Option<i64> {
    if v[7] != b'-' || v[10] != b'T' || v[13] != b':' || v[16] != b':' {
        return None;
    }
    let y = d2(v, 0)? * 100 + d2(v, 2)?;
    let mo = d2(v, 5)?;
    let d = d2(v, 8)?;
    let h = d2(v, 11)?;
    let mi = d2(v, 14)?;
    let s = d2(v, 17)?;
    if mo < 1 || mo > 12 || d < 1 || h > 23 || mi > 59 || s > 59 {
        return None;
    }
    let dim = match mo {
        2 => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    if d > dim {
        return None;
    }
    let mut ms = (days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + s) * 1000;
    if v.len() == 20 {
        if v[19] != b'Z' {
            return None;
        }
    } else {
        let sign = v[19];
        if (sign != b'+' && sign != b'-') || v[22] != b':' {
            return None;
        }
        let oh = d2(v, 20)?;
        let om = d2(v, 23)?;
        if oh > 23 || om > 59 {
            return None;
        }
        let off = (oh * 60 + om) * 60_000;
        if sign == b'+' {
            ms -= off;
        } else {
            ms += off;
        }
    }
    Some(ms)
}

/// Epoch milliseconds, or None for a bad timestamp (or one far outside any window).
#[inline(always)]
fn parse_ts(v: &[u8]) -> Option<i64> {
    let n = v.len();
    if n == 0 {
        return None;
    }
    if (n == 20 || n == 25) && v[4] == b'-' {
        return parse_iso(v);
    }
    let mut i = 0;
    while i < n && v[i] == b'0' {
        i += 1;
    }
    if n - i > 18 {
        return None;
    }
    let mut x: i64 = 0;
    while i < n {
        let d = v[i].wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        x = x * 10 + d as i64;
        i += 1;
    }
    Some(x)
}

#[inline(always)]
fn parse_price(v: &[u8]) -> Option<u32> {
    let n = v.len();
    if n == 0 {
        return None;
    }
    let mut i = 0;
    while i < n && v[i] == b'0' {
        i += 1;
    }
    if n - i > 8 {
        return None;
    }
    let mut x: u32 = 0;
    while i < n {
        let d = v[i].wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        x = x * 10 + d as u32;
        i += 1;
    }
    if x < 1 || x > 10_000_000 {
        None
    } else {
        Some(x)
    }
}


// ---------------------------------------------------------------- fast timestamp paths

static mut MONTH_START: [i32; 4800] = [0; 4800]; // days since epoch of the 1st, years 1900..2300
static mut MONTH_LEN: [u8; 4800] = [0; 4800];
/// (year - 1900) * 16 + month -> days-since-epoch of the 1st << 8 | month length
static mut TBL16: [i32; 6400] = [0; 6400];

fn init_month_tables() {
    unsafe {
        for y in 1900..2300i64 {
            for m in 1..=12i64 {
                let i = ((y - 1900) * 12 + m - 1) as usize;
                MONTH_START[i] = days_from_civil(y, m, 1) as i32;
                MONTH_LEN[i] = match m {
                    2 => {
                        if is_leap(y) {
                            29
                        } else {
                            28
                        }
                    }
                    4 | 6 | 9 | 11 => 30,
                    _ => 31,
                };
                TBL16[((y - 1900) * 16 + m) as usize] = (MONTH_START[i] << 8) | MONTH_LEN[i] as i32;
            }
        }
    }
}

#[inline(always)]
fn all_digits(x: u64) -> bool {
    ((x & 0xF0F0F0F0F0F0F0F0) == 0x3030303030303030) & (((x.wrapping_add(0x0606060606060606)) & 0xF0F0F0F0F0F0F0F0) == 0x3030303030303030)
}

#[inline(always)]
fn parse8(x: u64) -> u64 {
    let v = x.wrapping_sub(0x3030303030303030);
    let v = v.wrapping_mul(10).wrapping_add(v >> 8);
    (((v & 0x000000FF000000FF).wrapping_mul(100 + (1000000 << 32)))
        .wrapping_add(((v >> 16) & 0x000000FF000000FF).wrapping_mul(1 + (10000 << 32))))
        >> 32
}

const BAD_TS: i64 = i64::MIN;

/// `YYYY-MM-DDTHH:MM:SSZ` (F1, n == 20) or `YYYY-MM-DDTHH:MM:SS+HH:MM` (F2, n == 25).
/// Returns BAD_TS when not valid. F2 reads 3 bytes past its end (padded memory only).
#[inline(always)]
unsafe fn parse_f12<const F2: bool>(p: *const u8) -> i64 {
    const S0: u64 = 0xFF0000FF00000000;
    const E0: u64 = 0x2D00002D00000000;
    const S1: u64 = 0x0000FF0000FF0000;
    const E1: u64 = 0x00003A0000540000;
    const S2: u64 = 0xFF0000FF;
    const E2: u64 = 0x5A00003A;
    const Z: u64 = 0x3030303030303030;
    let w0 = std::ptr::read_unaligned(p as *const u64);
    let w1 = std::ptr::read_unaligned(p.add(8) as *const u64);
    let w2 = std::ptr::read_unaligned(p.add(16) as *const u32) as u64;
    let x0 = (w0 & !S0) | (Z & S0);
    let x1 = (w1 & !S1) | (Z & S1);
    let x2 = (w2 & !S2) | (Z & S2) | 0x3030303000000000;
    let tail = if F2 { (w2 & 0xFF) == 0x3A } else { (w2 & S2) == E2 };
    let ok = ((w0 & S0) == E0) & ((w1 & S1) == E1) & tail & all_digits(x0) & all_digits(x1) & all_digits(x2);
    if !ok {
        return BAD_TS;
    }
    let v0 = x0.wrapping_sub(Z);
    let t0 = v0.wrapping_mul(10).wrapping_add(v0 >> 8);
    let v1 = x1.wrapping_sub(Z);
    let t1 = v1.wrapping_mul(10).wrapping_add(v1 >> 8);
    let v2 = x2.wrapping_sub(Z);
    let t2 = v2.wrapping_mul(10).wrapping_add(v2 >> 8);
    let year = (t0 & 0xFF) * 100 + ((t0 >> 16) & 0xFF);
    let month = (t0 >> 40) & 0xFF;
    let day = t1 & 0xFF;
    let hour = (t1 >> 24) & 0xFF;
    let minute = (t1 >> 48) & 0xFF;
    let second = (t2 >> 8) & 0xFF;
    if year < 1900 || year >= 2300 || month < 1 || month > 12 || hour > 23 || minute > 59 || second > 59 {
        if year < 1900 || year >= 2300 {
            return parse_iso(sl(p, if F2 { 25 } else { 20 })).unwrap_or(BAD_TS);
        }
        return BAD_TS;
    }
    let idx = ((year - 1900) * 12 + month - 1) as usize;
    if day < 1 || day > MONTH_LEN[idx] as u64 {
        return BAD_TS;
    }
    let ms = ((MONTH_START[idx] as i64 + day as i64 - 1) * 86_400 + (hour * 3600 + minute * 60 + second) as i64) * 1000;
    if !F2 {
        return ms;
    }
    // offset: sign at 19, "HH:MM" at 20..25
    let sign = *p.add(19);
    let w3 = std::ptr::read_unaligned(p.add(20) as *const u64) & 0xFF_FFFF_FFFF;
    let x3 = (w3 & !0xFF_0000) | 0x30_0000 | 0x3030_3000_0000_0000;
    if (sign != b'+' && sign != b'-') | ((w3 >> 16) & 0xFF != b':' as u64) | !all_digits(x3) {
        return BAD_TS;
    }
    let oh = ((w3 & 0xFF) - 0x30) * 10 + ((w3 >> 8) & 0xFF) - 0x30;
    let om = (((w3 >> 24) & 0xFF) - 0x30) * 10 + ((w3 >> 32) & 0xFF) - 0x30;
    if oh > 23 || om > 59 {
        return BAD_TS;
    }
    let off = ((oh * 60 + om) * 60_000) as i64;
    if sign == b'+' {
        ms - off
    } else {
        ms + off
    }
}

const fn v128(b: [u8; 16]) -> __m128i {
    unsafe { std::mem::transmute(b) }
}
const fn w128(w: [i16; 8]) -> __m128i {
    unsafe { std::mem::transmute(w) }
}

/// SIMD form of `parse_f12`: same results. F2 reads 3 bytes past its end (padded memory only).
#[inline(always)]
unsafe fn parse_f12_simd<const F2: bool>(p: *const u8) -> i64 {
    const PA: __m128i = v128(*b"0000-00-00T00:00");
    const LA: __m128i = v128([9, 9, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9]);
    const PB1: __m128i = v128(*b"-00-00T00:00:00Z");
    const LB1: __m128i = v128([0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0]);
    const PB2: __m128i = v128(*b"-00-00T00:00:00\0");
    const LB2: __m128i = v128([0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 255]);
    const SHA: __m128i = v128([0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 0x80, 0x80, 0x80, 0x80]);
    const SHB: __m128i = v128([0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 13, 14, 0x80, 0x80]);
    const W10: __m128i = v128([10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1]);
    const LO: __m128i = w128([19, 0, 1, 1, 0, 0, 0, 0]);
    const HI: __m128i = w128([22, 99, 12, 31, 23, 59, 59, 0]);
    const WM: __m128i = w128([1600, 16, 1, 0, 3600, 60, 1, 0]);
    let a = _mm_loadu_si128(p as *const __m128i);
    let b = _mm_loadu_si128(p.add(4) as *const __m128i);
    let xa = _mm_xor_si128(a, PA);
    let xb = _mm_xor_si128(b, if F2 { PB2 } else { PB1 });
    let lb = if F2 { LB2 } else { LB1 };
    let ok = _mm_and_si128(_mm_cmpeq_epi8(_mm_max_epu8(xa, LA), LA), _mm_cmpeq_epi8(_mm_max_epu8(xb, lb), lb));
    if _mm_movemask_epi8(ok) != 0xFFFF {
        return BAD_TS;
    }
    let v = _mm_or_si128(_mm_shuffle_epi8(xa, SHA), _mm_shuffle_epi8(xb, SHB));
    let vals = _mm_maddubs_epi16(v, W10); // [YY, YY, MM, DD, hh, mm, ss, 0]
    let bad = _mm_or_si128(_mm_cmpgt_epi16(LO, vals), _mm_cmpgt_epi16(vals, HI));
    if _mm_movemask_epi8(bad) != 0 {
        // out-of-table year (exact slow path) or a bad field
        return parse_iso(sl(p, if F2 { 25 } else { 20 })).unwrap_or(BAD_TS);
    }
    let r = _mm_madd_epi16(vals, WM); // [ym16, month, hh*3600 + mm*60, ss]
    let ym = _mm_cvtsi128_si32(r) + _mm_extract_epi32::<1>(r) - 1900 * 16;
    let hms = _mm_extract_epi32::<2>(r) + _mm_extract_epi32::<3>(r);
    let d = _mm_extract_epi16::<3>(vals);
    let packed = *TBL16.get_unchecked(ym as usize);
    if (d - 1) as u32 >= (packed & 0xFF) as u32 {
        return BAD_TS;
    }
    let ms = (((packed >> 8) + d - 1) as i64 * 86_400 + hms as i64) * 1000;
    if !F2 {
        return ms;
    }
    let sign = *p.add(19);
    let w3 = std::ptr::read_unaligned(p.add(20) as *const u64) & 0xFF_FFFF_FFFF;
    let x3 = (w3 & !0xFF_0000) | 0x30_0000 | 0x3030_3000_0000_0000;
    if (sign != b'+' && sign != b'-') | ((w3 >> 16) & 0xFF != b':' as u64) | !all_digits(x3) {
        return BAD_TS;
    }
    let oh = ((w3 & 0xFF) - 0x30) * 10 + ((w3 >> 8) & 0xFF) - 0x30;
    let om = (((w3 >> 24) & 0xFF) - 0x30) * 10 + ((w3 >> 32) & 0xFF) - 0x30;
    if oh > 23 || om > 59 {
        return BAD_TS;
    }
    let off = ((oh * 60 + om) * 60_000) as i64;
    if sign == b'+' {
        ms - off
    } else {
        ms + off
    }
}

/// Any timestamp form; BAD_TS when bad.
#[inline(always)]
unsafe fn parse_ts_fast(p: *const u8, n: usize) -> i64 {
    if n == 20 {
        return parse_f12_simd::<false>(p);
    }
    if n == 25 {
        return parse_f12_simd::<true>(p);
    }
    if n == 13 {
        let a = std::ptr::read_unaligned(p as *const u64);
        let b = std::ptr::read_unaligned(p.add(5) as *const u64);
        if all_digits(a) & all_digits(b) {
            return (parse8(a) * 100_000 + parse8(b) % 100_000) as i64;
        }
        return BAD_TS;
    }
    match parse_ts(sl(p, n)) {
        Some(t) => t,
        None => BAD_TS,
    }
}

// ---------------------------------------------------------------- CSV dialect (scalar, for small files)

#[inline(always)]
fn split_line<const N: usize>(line: &'static [u8], out: &mut [&'static [u8]; N]) -> bool {
    let n = line.len();
    let mut f = 0usize;
    let mut start = 0usize;
    let mut q = 0u32;
    let mut i = 0usize;
    loop {
        let c = if i < n { line[i] } else { b',' };
        if c == b',' {
            if f == N {
                return false;
            }
            let mut s = start;
            let mut e = i;
            if q != 0 {
                if q != 2 || e - s < 2 || line[s] != b'"' || line[e - 1] != b'"' {
                    return false;
                }
                s += 1;
                e -= 1;
            }
            out[f] = &line[s..e];
            f += 1;
            if i >= n {
                break;
            }
            start = i + 1;
            q = 0;
        } else if c == b'"' {
            q += 1;
        }
        i += 1;
    }
    f == N
}

fn for_each_line(chunk: &'static [u8], mut f: impl FnMut(&'static [u8])) {
    let n = chunk.len();
    let mut s = 0;
    while s < n {
        let e = memchr_nl(chunk, s);
        f(&chunk[s..e]);
        s = e + 1;
    }
}

// ---------------------------------------------------------------- SIMD line scanner

#[inline(always)]
unsafe fn block_masks(p: *const u8) -> (u64, u64) {
    let a = _mm256_loadu_si256(p as *const __m256i);
    let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
    let comma = _mm256_set1_epi8(b',' as i8);
    let nl = _mm256_set1_epi8(b'\n' as i8);
    let qt = _mm256_set1_epi8(b'"' as i8);
    let sa = _mm256_or_si256(_mm256_cmpeq_epi8(a, comma), _mm256_cmpeq_epi8(a, nl));
    let sb = _mm256_or_si256(_mm256_cmpeq_epi8(b, comma), _mm256_cmpeq_epi8(b, nl));
    let s = (_mm256_movemask_epi8(sa) as u32 as u64) | ((_mm256_movemask_epi8(sb) as u32 as u64) << 32);
    let q = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, qt)) as u32 as u64)
        | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, qt)) as u32 as u64) << 32);
    (s, q)
}


/// (',' or '\n' mask, '"' mask, '\n' mask) of 64 bytes.
#[inline(always)]
unsafe fn block_masks3(p: *const u8) -> (u64, u64, u64) {
    let a = _mm256_loadu_si256(p as *const __m256i);
    let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
    let comma = _mm256_set1_epi8(b',' as i8);
    let nl = _mm256_set1_epi8(b'\n' as i8);
    let qt = _mm256_set1_epi8(b'"' as i8);
    let na = _mm256_cmpeq_epi8(a, nl);
    let nb = _mm256_cmpeq_epi8(b, nl);
    let sa = _mm256_or_si256(_mm256_cmpeq_epi8(a, comma), na);
    let sb = _mm256_or_si256(_mm256_cmpeq_epi8(b, comma), nb);
    let s = (_mm256_movemask_epi8(sa) as u32 as u64) | ((_mm256_movemask_epi8(sb) as u32 as u64) << 32);
    let n = (_mm256_movemask_epi8(na) as u32 as u64) | ((_mm256_movemask_epi8(nb) as u32 as u64) << 32);
    let q = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, qt)) as u32 as u64)
        | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, qt)) as u32 as u64) << 32);
    (s, q, n)
}

// ---------------------------------------------------------------- region-indexed line splitter
// Each region (<= ~64 KB, whole lines) first gets a flat index of every ',' / '\n' offset
// (branch-light, 8 per block unconditionally), plus per-block quote counts. Lines then take
// their N separators straight from the index: no per-block refill branch per field.

const REGION: usize = 1 << 16;

struct Flat {
    idx: Vec<u32>, // every ',' / '\n' offset, then 16 copies of len
    nl: Vec<u32>,  // every '\n' offset, then 4 copies of len
    qcum: Vec<u32>,
    qm: Vec<u64>,
}

impl Flat {
    fn new() -> Self {
        Flat { idx: Vec::new(), nl: Vec::new(), qcum: Vec::new(), qm: Vec::new() }
    }
    #[inline(always)]
    unsafe fn build(&mut self, start: *const u8, len: usize) {
        let nb = (len + 63) / 64;
        if self.idx.capacity() < len + 96 {
            self.idx = Vec::with_capacity(len + 96);
            self.nl = Vec::with_capacity(len + 96);
        }
        if self.qcum.capacity() < nb + 2 {
            self.qcum = Vec::with_capacity(nb + 2);
            self.qm = Vec::with_capacity(nb + 2);
        }
        let ip = self.idx.as_mut_ptr();
        let np = self.nl.as_mut_ptr();
        let qc = self.qcum.as_mut_ptr();
        let qmp = self.qm.as_mut_ptr();
        let mut n = 0usize;
        let mut m = 0usize;
        let mut qsum = 0u32;
        for b in 0..nb {
            let (mut s, mut q, mut nm) = block_masks3(start.add(b * 64));
            let rem = len - b * 64;
            if rem < 64 {
                let k = (1u64 << rem) - 1;
                s &= k;
                q &= k;
                nm &= k;
            }
            *qc.add(b) = qsum;
            *qmp.add(b) = q;
            qsum += q.count_ones();
            let base = (b * 64) as u32;
            let cnt = s.count_ones() as usize;
            let o = ip.add(n);
            for k in 0..8 {
                *o.add(k) = base + s.trailing_zeros();
                s &= s.wrapping_sub(1);
            }
            if cnt > 8 {
                for k in 8..16 {
                    *o.add(k) = base + s.trailing_zeros();
                    s &= s.wrapping_sub(1);
                }
                if cnt > 16 {
                    let mut k = 16;
                    while s != 0 {
                        *o.add(k) = base + s.trailing_zeros();
                        s &= s.wrapping_sub(1);
                        k += 1;
                    }
                }
            }
            n += cnt;
            let cn = nm.count_ones() as usize;
            let o = np.add(m);
            *o = base + nm.trailing_zeros();
            nm &= nm.wrapping_sub(1);
            *o.add(1) = base + nm.trailing_zeros();
            if cn > 2 {
                nm &= nm.wrapping_sub(1);
                let mut k = 2;
                while nm != 0 {
                    *o.add(k) = base + nm.trailing_zeros();
                    nm &= nm.wrapping_sub(1);
                    k += 1;
                }
            }
            m += cn;
        }
        for k in 0..16 {
            *ip.add(n + k) = len as u32;
        }
        for k in 0..4 {
            *np.add(m + k) = len as u32;
        }
        *qc.add(nb) = qsum;
        *qmp.add(nb) = 0;
    }
    #[inline(always)]
    unsafe fn qbefore(&self, off: u32) -> u32 {
        let b = (off >> 6) as usize;
        let r = off & 63;
        *self.qcum.as_ptr().add(b) + (*self.qm.as_ptr().add(b) & ((1u64 << r) - 1)).count_ones()
    }
}

/// Calls `f` with the N fields (quotes stripped) of every well-formed line of `chunk`
/// (whole lines; the last may lack its '\n').
#[inline(always)]
unsafe fn for_lines<const N: usize>(flat: &mut Flat, chunk: &'static [u8], mut f: impl FnMut(&[*const u8; N], &[usize; N])) {
    let cs = chunk.as_ptr();
    let clen = chunk.len();
    // 8 slots (the vector path writes 8 lanes); callers see the first N
    let mut fpa: [*const u8; 8] = [std::ptr::null(); 8];
    let mut fla: [usize; 8] = [0; 8];
    let fp: &mut [*const u8; N] = &mut *(fpa.as_mut_ptr() as *mut [*const u8; N]);
    let fl: &mut [usize; N] = &mut *(fla.as_mut_ptr() as *mut [usize; N]);
    let mut rs = 0usize;
    while rs < clen {
        let mut re = rs + REGION;
        if re >= clen {
            re = clen;
        } else {
            let mut p = re;
            while p > rs && *cs.add(p - 1) != b'\n' {
                p -= 1;
            }
            if p > rs {
                re = p;
            } else {
                re = memchr_nl(chunk, re);
                re = if re < clen { re + 1 } else { clen };
            }
        }
        let base = cs.add(rs);
        let len = (re - rs) as u32;
        flat.build(base, re - rs);
        let ix = flat.idx.as_ptr();
        let nx = flat.nl.as_ptr();
        let mut j = 0usize; // index (in ix) of the line's first separator
        let mut k = 0usize; // index (in nx) of the line's newline
        let mut ls = 0u32;
        let mut qls = 0u32; // quotes before ls
        while ls < len {
            let s = ix.add(j);
            let e = *nx.add(k); // the line's end: its '\n', or len
            k += 1;
            let last = *s.add(N - 1);
            // well-formed: the N-th separator is the line's first newline (or the end),
            // so the N-1 before it are commas
            if (last != e) | (*s.add(N - 2) >= last) {
                if e >= len {
                    break;
                }
                let mut jj = j;
                while *ix.add(jj) != e {
                    jj += 1;
                }
                j = jj + 1;
                ls = e + 1;
                qls = flat.qbefore(ls);
                continue;
            }
            if N == 7 || N == 4 {
                // starts = [ls, s0 + 1, s1 + 1, ...], lens = s - starts, in 8 lanes at once
                let sv = _mm256_loadu_si256(s as *const __m256i);
                let prev = _mm256_permutevar8x32_epi32(sv, _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6));
                let prev = _mm256_blend_epi32::<1>(prev, _mm256_set1_epi32(ls as i32 - 1));
                let starts = _mm256_sub_epi32(prev, _mm256_set1_epi32(-1));
                let lens = _mm256_sub_epi32(sv, starts);
                let bv = _mm256_set1_epi64x(base as i64);
                let fpp = fp.as_mut_ptr() as *mut __m256i;
                let flp = fl.as_mut_ptr() as *mut __m256i;
                _mm256_storeu_si256(fpp, _mm256_add_epi64(bv, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(starts))));
                _mm256_storeu_si256(flp, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(lens)));
                if N == 7 {
                    // lanes 4..7 (the 8th is scratch, written into fp/fl's padding)
                    _mm256_storeu_si256(fpp.add(1), _mm256_add_epi64(bv, _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(starts))));
                    _mm256_storeu_si256(flp.add(1), _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(lens)));
                }
            } else {
                fp[0] = base.add(ls as usize);
                fl[0] = (*s - ls) as usize;
                for i in 1..N {
                    let a = *s.add(i - 1) + 1;
                    fp[i] = base.add(a as usize);
                    fl[i] = (*s.add(i) - a) as usize;
                }
            }
            // the line's '\n' is no quote, so quotes before the next line = quotes before `last`
            let ql = flat.qbefore(last);
            let q = ql - qls;
            qls = ql;
            ls = last + 1;
            j += N;
            if q != 0 {
                // fields wrapped in a quote pair lose it; the line is well-formed only
                // if those pairs are all its quotes
                let mut fq = 0u32;
                for i in 0..N {
                    let p = fp[i];
                    let n = fl[i];
                    let both = (n >= 2) & (*p == b'"') & (*p.add(n.max(1) - 1) == b'"');
                    fq += both as u32;
                    fp[i] = p.add(both as usize);
                    fl[i] = n - 2 * both as usize;
                }
                if q != 2 * fq {
                    continue;
                }
            }
            f(fp, fl);
        }
        rs = re;
    }
}

#[inline(always)]
unsafe fn sl(p: *const u8, n: usize) -> &'static [u8] {
    std::slice::from_raw_parts(p, n)
}

// ---------------------------------------------------------------- lookups

struct Lookup {
    prefix: Vec<u8>,
    pword: u64, // prefix bytes, little-endian
    direct: Vec<u32>,
    other: U64Map<u32>,
    has_other: bool,
}

const ABSENT: u32 = u32::MAX;

impl Lookup {
    fn new(prefix: &[u8]) -> Self {
        let mut pw = [0u8; 8];
        pw[..prefix.len().min(8)].copy_from_slice(&prefix[..prefix.len().min(8)]);
        Lookup { prefix: prefix.to_vec(), pword: u64::from_le_bytes(pw), direct: Vec::new(), other: U64Map::with_capacity(16), has_other: false }
    }
    /// `canonical` for keys inside padded memory (8 readable bytes past the key's start
    /// and past the prefix): word compares instead of byte loops.
    #[inline(always)]
    unsafe fn canonical_fast(&self, k: &[u8]) -> Option<usize> {
        let p = self.prefix.len();
        let n = k.len().wrapping_sub(p); // digits
        if n.wrapping_sub(1) >= 8 || p > 8 {
            return self.canonical(k);
        }
        let w = std::ptr::read_unaligned(k.as_ptr() as *const u64);
        let pm = u64::MAX >> (64 - 8 * p as u32);
        if (w & pm) != self.pword {
            return None;
        }
        let v = std::ptr::read_unaligned(k.as_ptr().add(p) as *const u64);
        // digits into the high bytes, '0's below (leading zeros)
        let x = (v << (8 * (8 - n) as u32)) | 0x3030303030303030u64.checked_shr(8 * n as u32).unwrap_or(0);
        if !all_digits(x) || ((v & 0xFF) == b'0' as u64 && n > 1) {
            return None;
        }
        Some(parse8(x) as usize)
    }
    #[inline(always)]
    fn canonical(&self, k: &[u8]) -> Option<usize> {
        let p = self.prefix.len();
        if k.len() <= p || k.len() > p + 9 || k[..p] != self.prefix[..] {
            return None;
        }
        let digits = &k[p..];
        if digits[0] == b'0' && digits.len() > 1 {
            return None;
        }
        let mut x = 0usize;
        for &c in digits {
            let d = c.wrapping_sub(b'0');
            if d > 9 {
                return None;
            }
            x = x * 10 + d as usize;
        }
        Some(x)
    }
    fn insert(&mut self, k: &[u8], v: u32) {
        match self.canonical(k) {
            Some(x) if x < 10_000_000 => {
                if self.direct.len() <= x {
                    self.direct.resize(x + 1, ABSENT);
                }
                self.direct[x] = v;
            }
            _ => {
                let (s, _) = self.other.slot(hash_bytes(k));
                self.other.vals[s] = v + 1;
                self.has_other = true;
            }
        }
    }
    #[inline(always)]
    /// `k` must lie in padded memory (mapped input files are).
    fn get(&self, k: &[u8]) -> u32 {
        if let Some(x) = unsafe { self.canonical_fast(k) } {
            return if x < self.direct.len() { self.direct[x] } else { ABSENT };
        }
        if self.has_other {
            match self.other.get(hash_bytes(k)) {
                Some(v) if v != 0 => v - 1,
                _ => ABSENT,
            }
        } else {
            ABSENT
        }
    }
}

fn read_small_csv(path: &Path) -> Vec<(&'static [u8], &'static [u8])> {
    let data = read_padded(path);
    let mut rows = Vec::new();
    let first = memchr_nl(data, 0);
    if first >= data.len() {
        return rows;
    }
    for_each_line(&data[first + 1..], |line| {
        let mut f: [&'static [u8]; 2] = [&[]; 2];
        if split_line(line, &mut f) && !f[0].is_empty() {
            rows.push((f[0], f[1]));
        }
    });
    rows
}


// ---------------------------------------------------------------- user IDs: one lock-free table for every ID

#[repr(C, align(16))]
struct Slot {
    key: AtomicU64,
    rep: AtomicU32, // 0 = this slot is its own person, else representative slot + 1
    _pad: u32,
}

struct Ids {
    slots: &'static [Slot],
    prefs: &'static [AtomicU64], // the ID's bytes (ptr | len << 48)
    mask: usize,
    bits: u32,
    overflow: AtomicBool,
}

impl Ids {
    fn new(cap: usize) -> Self {
        let cap = cap.next_power_of_two().max(1024);
        Ids {
            slots: zeroed::<Slot>(cap),
            prefs: zeroed::<AtomicU64>(cap),
            mask: cap - 1,
            bits: cap.trailing_zeros(),
            overflow: AtomicBool::new(false),
        }
    }
    #[inline(always)]
    fn slot_ptr(&self, h: u64) -> *const Slot {
        unsafe { self.slots.as_ptr().add((h as usize) & self.mask) }
    }
    /// Slot of the ID, inserting it when new. Returns (slot, inserted).
    #[inline(always)]
    fn intern(&self, h: u64, r: u64) -> (u32, bool) {
        let mut i = (h as usize) & self.mask;
        let mut probes = 0u32;
        loop {
            let s = unsafe { self.slots.get_unchecked(i) };
            let k = s.key.load(Relaxed);
            if k == h {
                return (i as u32, false);
            }
            if k == 0 {
                match s.key.compare_exchange(0, h, Relaxed, Relaxed) {
                    Ok(_) => {
                        unsafe { self.prefs.get_unchecked(i) }.store(r, Relaxed);
                        return (i as u32, true);
                    }
                    Err(k2) => {
                        if k2 == h {
                            return (i as u32, false);
                        }
                    }
                }
            }
            probes += 1;
            if probes > 512 {
                self.overflow.store(true, Relaxed);
                return (0, false);
            }
            i = (i + 1) & self.mask;
        }
    }
    /// Person (representative slot) of an ID seen in events or clicks.
    #[inline(always)]
    fn person(&self, h: u64, r: u64) -> u32 {
        let (i, ins) = self.intern(h, r);
        if ins {
            return i;
        }
        let rep = unsafe { self.slots.get_unchecked(i as usize) }.rep.load(Relaxed);
        if rep == 0 {
            i
        } else {
            rep - 1
        }
    }
}

#[inline(always)]
fn uf_find(parent: &[AtomicU32], mut x: u32) -> u32 {
    loop {
        let p = parent[x as usize].load(Relaxed);
        if p == 0 {
            return x;
        }
        let p = p - 1;
        let gp = parent[p as usize].load(Relaxed);
        if gp != 0 {
            let _ = parent[x as usize].compare_exchange_weak(p + 1, gp, Relaxed, Relaxed);
        }
        x = p;
    }
}

#[inline(always)]
fn uf_union(parent: &[AtomicU32], a: u32, b: u32) {
    let (mut a, mut b) = (a, b);
    loop {
        a = uf_find(parent, a);
        b = uf_find(parent, b);
        if a == b {
            return;
        }
        let (hi, lo) = if a > b { (a, b) } else { (b, a) };
        if parent[hi as usize].compare_exchange(0, lo + 1, AcqRel, Relaxed).is_ok() {
            return;
        }
    }
}

/// Joins linked IDs; every linked ID's slot gets its representative (byte-wise smallest ID).
fn build_links(ids: &Ids, link_chunks: &[&'static [u8]], nthreads: usize) {
    let cap = ids.mask + 1;
    let parent: &'static [AtomicU32] = zeroed::<AtomicU32>(cap);
    let best: &'static [AtomicU32] = zeroed::<AtomicU32>(cap);
    let next = AtomicUsize::new(0);
    let mut inserted: Vec<Vec<u32>> = Vec::new();
    const LB: usize = 32;
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..nthreads)
            .map(|_| {
                let next = &next;
                s.spawn(move || {
                    let mut mine: Vec<u32> = Vec::new();
                    let mut flat = Flat::new();
                    // batches: hash + prefetch slots, then intern + prefetch parents, then union
                    let mut batch: Vec<(u64, u64, u64, u64)> = Vec::with_capacity(LB);
                    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(LB);
                    let mut flush = |batch: &mut Vec<(u64, u64, u64, u64)>, mine: &mut Vec<u32>| {
                        pairs.clear();
                        for &(ha, ra, hb, rb) in batch.iter() {
                            let (a, ia) = ids.intern(ha, ra);
                            let (b, ib) = ids.intern(hb, rb);
                            if ia {
                                mine.push(a);
                            }
                            if ib {
                                mine.push(b);
                            }
                            if a != b {
                                unsafe {
                                    _mm_prefetch(parent.as_ptr().add(a as usize) as *const i8, _MM_HINT_T0);
                                    _mm_prefetch(parent.as_ptr().add(b as usize) as *const i8, _MM_HINT_T0);
                                }
                                pairs.push((a, b));
                            }
                        }
                        for &(a, b) in pairs.iter() {
                            uf_union(parent, a, b);
                        }
                        batch.clear();
                    };
                    loop {
                        let j = next.fetch_add(1, Relaxed);
                        if j >= link_chunks.len() {
                            break;
                        }
                        unsafe {
                            for_lines::<2>(&mut flat, link_chunks[j], |fp, fl| {
                                if fl[0] == 0 || fl[1] == 0 {
                                    return;
                                }
                                let ha = hash_ptr(fp[0], fl[0]);
                                let hb = hash_ptr(fp[1], fl[1]);
                                _mm_prefetch(ids.slot_ptr(ha) as *const i8, _MM_HINT_T0);
                                _mm_prefetch(ids.slot_ptr(hb) as *const i8, _MM_HINT_T0);
                                batch.push((ha, mkref(fp[0], fl[0]), hb, mkref(fp[1], fl[1])));
                                if batch.len() == LB {
                                    flush(&mut batch, &mut mine);
                                }
                            });
                        }
                    }
                    flush(&mut batch, &mut mine);
                    mine
                })
            })
            .collect();
        for h in hs {
            inserted.push(h.join().unwrap());
        }
    });
    if ids.overflow.load(Relaxed) {
        return;
    }
    let name = |i: u32| deref(ids.prefs[i as usize].load(Relaxed));
    const PD: usize = 8;
    std::thread::scope(|s| {
        for v in &inserted {
            s.spawn(move || {
                for (k, &i) in v.iter().enumerate() {
                    if k + PD < v.len() {
                        let j = v[k + PD] as usize;
                        unsafe {
                            _mm_prefetch(parent.as_ptr().add(j) as *const i8, _MM_HINT_T0);
                            _mm_prefetch(ids.prefs.as_ptr().add(j) as *const i8, _MM_HINT_T0);
                        }
                    }
                    if k + PD / 2 < v.len() {
                        let r = ids.prefs[v[k + PD / 2] as usize].load(Relaxed);
                        unsafe {
                            _mm_prefetch((r & 0xFFFF_FFFF_FFFF) as *const i8, _MM_HINT_T0);
                        }
                    }
                    let r = uf_find(parent, i) as usize;
                    let my = name(i);
                    loop {
                        let cur = best[r].load(Relaxed);
                        if cur != 0 && name(cur - 1) <= my {
                            break;
                        }
                        if best[r].compare_exchange(cur, i + 1, Relaxed, Relaxed).is_ok() {
                            break;
                        }
                    }
                }
            });
        }
    });
    std::thread::scope(|s| {
        for v in &inserted {
            s.spawn(move || {
                for (k, &i) in v.iter().enumerate() {
                    if k + PD < v.len() {
                        let j = v[k + PD] as usize;
                        unsafe {
                            _mm_prefetch(parent.as_ptr().add(j) as *const i8, _MM_HINT_T0);
                            _mm_prefetch(ids.slots.as_ptr().add(j) as *const i8, _MM_HINT_T0);
                        }
                    }
                    let r = uf_find(parent, i) as usize;
                    let b = best[r].load(Relaxed) - 1;
                    if b != i {
                        ids.slots[i as usize].rep.store(b + 1, Relaxed);
                    }
                }
            });
        }
    });
}

// ---------------------------------------------------------------- bucketed output buffers in huge-page arenas

struct Buckets<T: Copy + 'static> {
    base: *mut T,
    cap: usize,
    cur: Vec<[*mut T; 2]>, // write pointer, end of the bucket's region
    spill: Vec<Vec<T>>,
}
unsafe impl<T: Copy + 'static> Send for Buckets<T> {}
unsafe impl<T: Copy + 'static> Sync for Buckets<T> {}

impl<T: Copy + 'static> Buckets<T> {
    fn new(n: usize, cap: usize) -> Self {
        let mem = zeroed::<T>(n * cap);
        let base = mem.as_mut_ptr();
        let cur = (0..n).map(|b| unsafe { [base.add(b * cap), base.add((b + 1) * cap)] }).collect();
        Buckets { base, cap, cur, spill: (0..n).map(|_| Vec::new()).collect() }
    }
    #[inline(always)]
    fn push(&mut self, b: usize, v: T) {
        unsafe {
            let c = self.cur.get_unchecked_mut(b);
            let p = c[0];
            if p < c[1] {
                *p = v;
                c[0] = p.add(1);
            } else {
                self.spill.get_unchecked_mut(b).push(v);
            }
        }
    }
    #[inline(always)]
    fn main(&self, b: usize) -> &mut [T] {
        unsafe {
            let s = self.base.add(b * self.cap);
            std::slice::from_raw_parts_mut(s, self.cur[b][0].offset_from(s) as usize)
        }
    }
    #[inline(always)]
    fn spilled(&self, b: usize) -> &mut [T] {
        let v = &self.spill[b];
        unsafe { std::slice::from_raw_parts_mut(v.as_ptr() as *mut T, v.len()) }
    }
    fn len(&self, b: usize) -> usize {
        self.main(b).len() + self.spill[b].len()
    }
}

// ---------------------------------------------------------------- records

#[derive(Clone, Copy)]
#[repr(C)]
struct DKey {
    ehash: u64,
    ingest: i32,
    rowid: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct PRec {
    pid: u32,
    ts: u32,
    pc: u32, // price << 8 | category (price 0 = not a purchase)
    rowid: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct CRec {
    pid: u32,
    ts: u32,
    camp: u32,
    _pad: u32,
    kref: u64,
}

struct Ctx {
    ws: i64,
    we: i64,
    base: i64, // ws - 7 days; all u32 times are relative to this
    products: Lookup,
    campaigns: Lookup,
    ncat: usize,
    unknown_cat: u32,
    ncamp: usize,
    first_day: i64,
    ndays: usize,
    ids: Ids,
    pshift: u32,
}

const EB_BITS: u32 = 9;
const NEB: usize = 1 << EB_BITS;
const NPB: usize = 512;
const RING: usize = 16;
const INVALID: u32 = u32::MAX;

#[inline(always)]
fn ebucket(h: u64) -> usize {
    (h >> (64 - EB_BITS)) as usize
}

#[derive(Clone, Copy, Default)]
struct Pending {
    uhash: u64,
    uref: u64,
    a: u32,
    b: u32,
    c: u32,
}

struct ThreadOut {
    dk: Buckets<DKey>,
    pe: Buckets<PRec>,
    pc: Buckets<CRec>,
    rows: u32,
    tid: u32,
    ring: [Pending; RING],
    rn: usize,
    rh: usize,
    flat: Flat,
}

impl ThreadOut {
    #[inline(always)]
    fn pop_event(&mut self, ctx: &Ctx) {
        let e = self.ring[self.rh];
        self.rh = (self.rh + 1) % RING;
        self.rn -= 1;
        let pid = ctx.ids.person(e.uhash, e.uref);
        self.pe.push((pid >> ctx.pshift) as usize, PRec { pid, ts: e.a, pc: e.b, rowid: e.c });
    }
    #[inline(always)]
    fn push_pending(&mut self, p: Pending, ctx: &Ctx) {
        unsafe {
            _mm_prefetch(ctx.ids.slot_ptr(p.uhash) as *const i8, _MM_HINT_T0);
        }
        if self.rn == RING {
            self.pop_event(ctx);
        }
        let tail = (self.rh + self.rn) % RING;
        self.ring[tail] = p;
        self.rn += 1;
    }
}

#[derive(Clone, Copy, Default)]
struct PendingClick {
    uhash: u64,
    uref: u64,
    kref: u64,
    ts: u32,
    camp: u32,
}

struct ClickRing {
    ring: [PendingClick; RING],
    rn: usize,
    rh: usize,
}

impl ClickRing {
    #[inline(always)]
    fn pop(&mut self, ctx: &Ctx, out: &mut ThreadOut) {
        let e = self.ring[self.rh];
        self.rh = (self.rh + 1) % RING;
        self.rn -= 1;
        let pid = ctx.ids.person(e.uhash, e.uref);
        out.pc.push((pid >> ctx.pshift) as usize, CRec { pid, ts: e.ts, camp: e.camp, _pad: 0, kref: e.kref });
    }
    #[inline(always)]
    fn push(&mut self, p: PendingClick, ctx: &Ctx, out: &mut ThreadOut) {
        unsafe {
            _mm_prefetch(ctx.ids.slot_ptr(p.uhash) as *const i8, _MM_HINT_T0);
        }
        if self.rn == RING {
            self.pop(ctx, out);
        }
        let tail = (self.rh + self.rn) % RING;
        self.ring[tail] = p;
        self.rn += 1;
    }
}

unsafe fn parse_events(ctx: &Ctx, chunk: &'static [u8], out: &mut ThreadOut) {
    let ws = ctx.ws;
    let we = ctx.we;
    let base = ctx.base;
    let mut flat = std::mem::replace(&mut out.flat, Flat::new());
    for_lines::<7>(&mut flat, chunk, |fp, fl| {
        if fl[0] == 0 {
            return;
        }
        let ehash = hash_ptr(fp[0], fl[0]);
        let ingest = parse_ts_fast(fp[3], fl[3]);
        let rowid = (out.tid << 27) | out.rows;
        out.rows += 1;
        let ingest_rel = if ingest == BAD_TS { i32::MAX } else { ((ingest - base).div_euclid(1000)).clamp(i32::MIN as i64, (i32::MAX - 1) as i64) as i32 };
        out.dk.push(ebucket(ehash), DKey { ehash, ingest: ingest_rel, rowid });
        // validation
        if fl[1] == 0 || fl[5] == 0 {
            return;
        }
        let ts = parse_ts_fast(fp[2], fl[2]);
        if ts < ws || ts >= we {
            return;
        }
        let mut pc = 0u32;
        if fl[4] == 4 {
            let w = std::ptr::read_unaligned(fp[4] as *const u32);
            if w != u32::from_le_bytes(*b"view") && w != u32::from_le_bytes(*b"cart") {
                return;
            }
        } else if fl[4] == 8 && std::ptr::read_unaligned(fp[4] as *const u64) == u64::from_le_bytes(*b"purchase") {
            let price = match parse_price(sl(fp[6], fl[6])) {
                Some(p) => p,
                None => return,
            };
            let cat = match ctx.products.get(sl(fp[5], fl[5])) {
                ABSENT => ctx.unknown_cat,
                c => c,
            };
            pc = (price << 8) | cat;
        } else {
            return;
        }
        if ingest == BAD_TS || ingest - ts > DAY {
            return;
        }
        let uhash = hash_ptr(fp[1], fl[1]);
        out.push_pending(Pending { uhash, uref: mkref(fp[1], fl[1]), a: (ts - base) as u32, b: pc, c: rowid }, ctx);
    });
    out.flat = flat;
    while out.rn > 0 {
        out.pop_event(ctx);
    }
}

unsafe fn parse_clicks(ctx: &Ctx, chunk: &'static [u8], out: &mut ThreadOut) {
    let lo = ctx.base;
    let we = ctx.we;
    let mut ring = ClickRing { ring: [PendingClick::default(); RING], rn: 0, rh: 0 };
    let mut flat = std::mem::replace(&mut out.flat, Flat::new());
    for_lines::<4>(&mut flat, chunk, |fp, fl| {
        if fl[0] == 0 || fl[1] == 0 {
            return;
        }
        let camp = ctx.campaigns.get(sl(fp[2], fl[2]));
        if camp == ABSENT {
            return;
        }
        let ts = parse_ts_fast(fp[3], fl[3]);
        if ts < lo || ts >= we {
            return;
        }
        let uhash = hash_ptr(fp[1], fl[1]);
        ring.push(PendingClick { uhash, uref: mkref(fp[1], fl[1]), kref: mkref(fp[0], fl[0]), ts: (ts - lo) as u32, camp }, ctx, out);
    });
    out.flat = flat;
    while ring.rn > 0 {
        ring.pop(ctx, out);
    }
}

// ---------------------------------------------------------------- output buffers

struct UserDaily {
    n: usize,
    names: Vec<u8>, // PLAIN byte arrays, one per person (rows of a person are contiguous)
    nnames: u32,
    uidx: Vec<u32>,
    day: Vec<u32>, // day index from first_day
    sessions: Vec<u64>,
    events: Vec<u64>,
    revenue: Vec<u64>,
    top_def: Vec<u8>,
    top_idx: Vec<u32>,
}

impl UserDaily {
    fn new(rows: usize) -> Self {
        UserDaily {
            n: 0,
            names: Vec::with_capacity(rows * 8),
            nnames: 0,
            uidx: Vec::with_capacity(rows),
            day: Vec::with_capacity(rows),
            sessions: Vec::with_capacity(rows),
            events: Vec::with_capacity(rows),
            revenue: Vec::with_capacity(rows),
            top_def: Vec::with_capacity(rows),
            top_idx: Vec::with_capacity(rows / 8),
        }
    }
}

#[inline]
fn put_str(buf: &mut Vec<u8>, s: &[u8]) {
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s);
}

#[derive(Clone, Copy)]
struct CKey {
    key: u64, // ts << 32 | camp
    kref: u64,
}

struct PersonWork {
    cat_sum: Vec<u64>,
    wcnt: Vec<u32>,
    wpos: Vec<u32>,
    active: Vec<u32>,
    plist: Vec<u32>,
    credited: Vec<[i64; 3]>,
    assisted: Vec<u32>,
    cnt: Vec<u32>,
    ccnt: Vec<u32>,
    ev: Vec<u64>,  // ts << 32 | pc, grouped by person
    ck: Vec<CKey>, // grouped by person
}

#[inline(always)]
fn sort_small<T: Copy>(v: &mut [T], key: impl Fn(&T) -> u64) {
    if v.len() <= 24 {
        for i in 1..v.len() {
            let x = v[i];
            let kx = key(&x);
            let mut j = i;
            while j > 0 && key(&v[j - 1]) > kx {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[j] = x;
        }
    } else {
        v.sort_unstable_by_key(|r| key(r));
    }
}

static TM: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
unsafe fn process_person_bucket(ctx: &Ctx, cat_names: &[Vec<u8>], outs: &[ThreadOut], losers: &[&'static [AtomicU64]], b: usize, w: &mut PersonWork, ud: &mut UserDaily) {
    let tq = Instant::now();
    let range = 1usize << ctx.pshift;
    let lo_pid = (b << ctx.pshift) as u32;
    w.cnt.clear();
    w.cnt.resize(range + 1, 0);
    // count (and drop dedup losers in place)
    let mut total = 0usize;
    for o in outs {
        for part in [o.pe.main(b), o.pe.spilled(b)] {
            for r in part.iter_mut() {
                let t = (r.rowid >> 27) as usize;
                let i = (r.rowid & ((1 << 27) - 1)) as usize;
                if losers[t][i >> 6].load(Relaxed) & (1u64 << (i & 63)) != 0 {
                    r.pid = INVALID;
                } else {
                    *w.cnt.get_unchecked_mut((r.pid - lo_pid) as usize) += 1;
                    total += 1;
                }
            }
        }
    }
    if total == 0 {
        return;
    }
    // exclusive prefix sums
    let mut acc = 0u32;
    for c in w.cnt.iter_mut() {
        let x = *c;
        *c = acc;
        acc += x;
    }
    w.ev.clear();
    w.ev.reserve(total);
    w.ev.set_len(total);
    for o in outs {
        for part in [o.pe.main(b), o.pe.spilled(b)] {
            for r in part.iter() {
                if r.pid != INVALID {
                    let p = (r.pid - lo_pid) as usize;
                    let at = *w.cnt.get_unchecked(p);
                    *w.cnt.get_unchecked_mut(p) = at + 1;
                    *w.ev.get_unchecked_mut(at as usize) = ((r.ts as u64) << 32) | r.pc as u64;
                }
            }
        }
    }
    TM[0].fetch_add(tq.elapsed().as_nanos() as u64, Relaxed);
    let tq = Instant::now();
    // now cnt[p] = end of p; start of p = cnt[p-1] (0 for p = 0)
    // clicks of persons that have events
    w.ccnt.clear();
    w.ccnt.resize(range + 1, 0);
    let mut ctotal = 0usize;
    for o in outs {
        for part in [o.pc.main(b), o.pc.spilled(b)] {
            for c in part.iter() {
                *w.ccnt.get_unchecked_mut((c.pid - lo_pid) as usize) += 1;
                ctotal += 1;
            }
        }
    }
    let mut acc = 0u32;
    for c in w.ccnt.iter_mut() {
        let x = *c;
        *c = acc;
        acc += x;
    }
    w.ck.clear();
    w.ck.reserve(ctotal);
    w.ck.set_len(ctotal);
    for o in outs {
        for part in [o.pc.main(b), o.pc.spilled(b)] {
            for c in part.iter() {
                let p = (c.pid - lo_pid) as usize;
                let at = *w.ccnt.get_unchecked(p);
                *w.ccnt.get_unchecked_mut(p) = at + 1;
                *w.ck.get_unchecked_mut(at as usize) = CKey { key: ((c.ts as u64) << 32) | c.camp as u64, kref: c.kref };
            }
        }
    }

    TM[1].fetch_add(tq.elapsed().as_nanos() as u64, Relaxed);
    let tq = Instant::now();
    let base = ctx.base;
    let ndays = ctx.ndays;
    let organic = ctx.ncamp;
    let prefs = ctx.ids.prefs;
    w.plist.clear();
    {
        let mut prev = 0u32;
        for p in 0..range {
            let e = *w.cnt.get_unchecked(p);
            if e != prev {
                w.plist.push(p as u32);
            }
            prev = e;
        }
    }
    let np = w.plist.len();
    for ii in 0..np {
        if ii + 6 < np {
            let r = prefs.get_unchecked(lo_pid as usize + *w.plist.get_unchecked(ii + 6) as usize).load(Relaxed);
            _mm_prefetch((r & 0xFFFF_FFFF_FFFF) as *const i8, _MM_HINT_T0);
        }
        let p = *w.plist.get_unchecked(ii) as usize;
        let es = if p == 0 { 0 } else { *w.cnt.get_unchecked(p - 1) as usize };
        let ee = *w.cnt.get_unchecked(p) as usize;
        let cs = if p == 0 { 0 } else { *w.ccnt.get_unchecked(p - 1) as usize };
        let ce = *w.ccnt.get_unchecked(p) as usize;
        sort_small(w.ev.get_unchecked_mut(es..ee), |&x| x);
        if ce - cs > 1 {
            sort_small(w.ck.get_unchecked_mut(cs..ce), |c| c.key);
        }
        let evs = w.ev.get_unchecked(es..ee);
        let clicks = w.ck.get_unchecked(cs..ce);
        let nc = clicks.len();
        let name = deref(prefs.get_unchecked(lo_pid as usize + p).load(Relaxed));
        put_str(&mut ud.names, name);
        ud.nnames += 1;
        let n = evs.len();
        let mut lo = 0usize;
        let mut hi = 0usize;
        let mut cur_day: i64 = i64::MIN;
        let mut d_sessions = 0i64;
        let mut d_events = 0i64;
        let mut d_rev = 0i64;
        let mut d_purch = false;
        let mut k = 0usize;
        while k < n {
            let start = (*evs.get_unchecked(k) >> 32) as u32;
            let mut prev = start;
            let mut rev = 0i64;
            let mut m = k;
            let day = (base + start as i64).div_euclid(DAY);
            if day != cur_day {
                if cur_day != i64::MIN {
                    flush_day(ud, (cur_day - ctx.first_day) as u32, d_sessions, d_events, d_rev, d_purch, &mut w.cat_sum, cat_names);
                }
                cur_day = day;
                d_sessions = 0;
                d_events = 0;
                d_rev = 0;
                d_purch = false;
            }
            while m < n {
                let r = *evs.get_unchecked(m);
                let t = (r >> 32) as u32;
                if t - prev > GAP {
                    break;
                }
                prev = t;
                let pc = r as u32;
                let price = (pc >> 8) as i64;
                if price > 0 {
                    rev += price;
                    *w.cat_sum.get_unchecked_mut((pc & 0xFF) as usize) += price as u64;
                    d_purch = true;
                }
                m += 1;
            }
            d_sessions += 1;
            d_events += (m - k) as i64;
            d_rev += rev;
            let dayidx = (day - ctx.first_day) as usize;
            let camp = if nc == 0 {
                organic
            } else {
                // slide the 7-day click window [start - 7d, start]
                while hi < nc && ((clicks.get_unchecked(hi).key >> 32) as u32) <= start {
                    let c = clicks.get_unchecked(hi).key as u32 as usize;
                    let x = *w.wcnt.get_unchecked(c);
                    if x == 0 {
                        *w.wpos.get_unchecked_mut(c) = w.active.len() as u32;
                        w.active.push(c as u32);
                    }
                    *w.wcnt.get_unchecked_mut(c) = x + 1;
                    hi += 1;
                }
                let low = start - WEEK as u32;
                while lo < hi && ((clicks.get_unchecked(lo).key >> 32) as u32) < low {
                    let c = clicks.get_unchecked(lo).key as u32 as usize;
                    let x = *w.wcnt.get_unchecked(c) - 1;
                    *w.wcnt.get_unchecked_mut(c) = x;
                    if x == 0 {
                        let pos = *w.wpos.get_unchecked(c) as usize;
                        let last = w.active.pop().unwrap_unchecked();
                        if last as usize != c {
                            *w.active.get_unchecked_mut(pos) = last;
                            *w.wpos.get_unchecked_mut(last as usize) = pos as u32;
                        }
                    }
                    lo += 1;
                }
                for &c in w.active.iter() {
                    *w.assisted.get_unchecked_mut(c as usize * ndays + dayidx) += 1;
                }
                if hi > lo {
                    let mut bi = hi - 1;
                    let t = clicks.get_unchecked(bi).key >> 32;
                    let mut x = bi;
                    while x > lo && (clicks.get_unchecked(x - 1).key >> 32) == t {
                        x -= 1;
                        if deref(clicks.get_unchecked(x).kref) < deref(clicks.get_unchecked(bi).kref) {
                            bi = x;
                        }
                    }
                    (clicks.get_unchecked(bi).key as u32) as usize
                } else {
                    organic
                }
            };
            let cc = w.credited.get_unchecked_mut(camp * ndays + dayidx);
            cc[0] += 1;
            if rev > 0 {
                cc[1] += 1;
                cc[2] += rev;
            }
            k = m;
        }
        for &c in w.active.iter() {
            *w.wcnt.get_unchecked_mut(c as usize) = 0;
        }
        w.active.clear();
        flush_day(ud, (cur_day - ctx.first_day) as u32, d_sessions, d_events, d_rev, d_purch, &mut w.cat_sum, cat_names);
    }
    TM[2].fetch_add(tq.elapsed().as_nanos() as u64, Relaxed);
}

#[inline]
fn flush_day(ud: &mut UserDaily, dayidx: u32, sessions: i64, events: i64, rev: i64, purch: bool, cat_sum: &mut [u64], _cat_names: &[Vec<u8>]) {
    ud.n += 1;
    ud.uidx.push(ud.nnames - 1);
    ud.day.push(dayidx);
    ud.sessions.push(sessions as u64);
    ud.events.push(events as u64);
    ud.revenue.push(rev as u64);
    if purch {
        let mut best = 0usize;
        let mut bv = 0u64;
        for (c, v) in cat_sum.iter_mut().enumerate() {
            if *v > bv {
                bv = *v;
                best = c;
            }
            *v = 0;
        }
        ud.top_def.push(1);
        ud.top_idx.push(best as u32);
    } else {
        ud.top_def.push(0);
    }
}

// ---------------------------------------------------------------- main

fn env_or(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn main() {
    let t0 = Instant::now();
    init_month_tables();
    if std::env::var("SS_NOTHP").is_ok() {
        NO_THP.store(true, Relaxed);
    }
    let verbose = std::env::var("SS_VERBOSE").is_ok();
    let lap = |what: &str| {
        if verbose {
            eprintln!("{what:>12} {:8.1} ms", t0.elapsed().as_secs_f64() * 1000.0);
        }
    };
    let args: Vec<String> = std::env::args().collect();
    let (data_dir, out_dir) = if args.len() >= 3 {
        (PathBuf::from(&args[args.len() - 2]), PathBuf::from(&args[args.len() - 1]))
    } else {
        (PathBuf::from(env_or("DATA_DIR", "/data")), PathBuf::from(env_or("OUTPUT_DIR", "/output")))
    };
    let nthreads: usize = std::env::var("SS_THREADS")
        .ok()
        .or_else(|| std::env::var("NCPU").ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |n| n.get()))
        .clamp(1, 32);

    // manifest
    let manifest = fs::read_to_string(data_dir.join("manifest.json")).expect("manifest");
    let get = |key: &str| -> i64 {
        let at = manifest.find(&format!("\"{key}\"")).expect("key");
        let rest = &manifest[at + key.len() + 2..];
        let q1 = rest.find('"').unwrap();
        let rest = &rest[q1 + 1..];
        let q2 = rest.find('"').unwrap();
        parse_ts(rest[..q2].as_bytes()).expect("manifest ts")
    };
    let ws = get("window_start");
    let we = get("window_end");
    let base = ws - WEEK;
    assert!(we - base < u32::MAX as i64 - 1, "window too long");

    // products / campaigns: parsed on a side thread while files are mapped and links are joined
    let pc_dir = data_dir.clone();
    let mut pc_handle = Some(std::thread::spawn(move || {
        let prows = read_small_csv(&pc_dir.join("products.csv"));
        // categories in first-appearance order, then sorted
        let mut cmap: std::collections::HashMap<&'static [u8], u32> = std::collections::HashMap::new();
        let mut uniq: Vec<&'static [u8]> = Vec::new();
        let mut pcat: Vec<u32> = Vec::with_capacity(prows.len());
        for &(_, cat) in &prows {
            let id = *cmap.entry(cat).or_insert_with(|| {
                uniq.push(cat);
                (uniq.len() - 1) as u32
            });
            pcat.push(id);
        }
        if !cmap.contains_key(&b"unknown"[..]) {
            uniq.push(b"unknown");
        }
        let mut cat_names: Vec<Vec<u8>> = uniq.iter().map(|c| c.to_vec()).collect();
        cat_names.sort();
        assert!(cat_names.len() <= 255, "too many categories");
        let remap: Vec<u32> = uniq.iter().map(|c| cat_names.binary_search_by(|x| x.as_slice().cmp(c)).unwrap() as u32).collect();
        let mut products = Lookup::new(b"p");
        for (i, &(pid, _)) in prows.iter().enumerate() {
            products.insert(pid, remap[pcat[i] as usize]);
        }
        let unknown_cat = cat_names.binary_search(&b"unknown".to_vec()).unwrap() as u32;
        let crows = read_small_csv(&pc_dir.join("campaigns.csv"));
        let mut campaigns = Lookup::new(b"c_");
        for (i, (cid, _)) in crows.iter().enumerate() {
            campaigns.insert(cid, i as u32);
        }
        (cat_names, products, unknown_cat, crows, campaigns)
    }));
    let mut cat_names: Vec<Vec<u8>> = Vec::new();
    let mut products = Lookup::new(b"p");
    let mut unknown_cat = 0u32;
    let mut crows: Vec<(&'static [u8], &'static [u8])> = Vec::new();
    let mut campaigns = Lookup::new(b"c_");

    // files (mapped in parallel)
    let ev_paths = csv_files(&data_dir.join("events"));
    let ck_paths = csv_files(&data_dir.join("clicks"));
    let ln_paths = csv_files(&data_dir.join("links"));
    let all_paths: Vec<&PathBuf> = ev_paths.iter().chain(ck_paths.iter()).chain(ln_paths.iter()).collect();
    let mapped: Vec<AtomicU64> = (0..all_paths.len() * 2).map(|_| AtomicU64::new(0)).collect();
    {
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..nthreads {
                s.spawn(|| loop {
                    let j = next.fetch_add(1, Relaxed);
                    if j >= all_paths.len() {
                        break;
                    }
                    let m = map_file(all_paths[j]);
                    mapped[2 * j].store(m.as_ptr() as u64, Relaxed);
                    mapped[2 * j + 1].store(m.len() as u64, Relaxed);
                });
            }
        });
    }
    let file_at = |j: usize| -> &'static [u8] {
        let p = mapped[2 * j].load(Relaxed) as *const u8;
        let n = mapped[2 * j + 1].load(Relaxed) as usize;
        if n == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(p, n) }
        }
    };
    let ev_files: Vec<&'static [u8]> = (0..ev_paths.len()).map(|j| file_at(j)).collect();
    let ck_files: Vec<&'static [u8]> = (0..ck_paths.len()).map(|j| file_at(ev_paths.len() + j)).collect();
    let ln_files: Vec<&'static [u8]> = (0..ln_paths.len()).map(|j| file_at(ev_paths.len() + ck_paths.len() + j)).collect();
    lap("mapped");

    let first_day = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - first_day + 1).max(1) as usize;

    let ev_bytes: usize = ev_files.iter().map(|f| f.len()).sum();
    let ck_bytes: usize = ck_files.iter().map(|f| f.len()).sum();
    let link_bytes: usize = ln_files.iter().map(|f| f.len()).sum();
    let mut link_chunks = Vec::new();
    let lchunk = (link_bytes / (nthreads * 8)).clamp(1 << 14, 1 << 22);
    for &f in &ln_files {
        chunks_of(f, lchunk, &mut link_chunks);
    }
    let mut jobs: Vec<(u8, &'static [u8])> = Vec::new();
    {
        let chunk_target = ((ev_bytes + ck_bytes) / (nthreads * 24)).clamp(1 << 16, 4 << 20);
        let mut v = Vec::new();
        for &f in &ev_files {
            chunks_of(f, chunk_target, &mut v);
        }
        jobs.extend(v.into_iter().map(|c| (0u8, c)));
        let mut v = Vec::new();
        for &f in &ck_files {
            chunks_of(f, chunk_target, &mut v);
        }
        jobs.extend(v.into_iter().map(|c| (1u8, c)));
    }
    jobs.sort_by_key(|j| std::cmp::Reverse(j.1.len()));
    let est_rows = ev_bytes / 80 + 1024;
    let est_ck = ck_bytes / 48 + 1024;

    let mut grow = 1usize;
    loop {
        // ID table sized for linked IDs plus the users seen only in events
        // ~26-byte link lines, each ID in ~1.5 of them; plus users seen only in events
        let est_ids = link_bytes / 20 + est_rows / 60 + 4096;
        let ids = Ids::new(est_ids * 3 / 2 * grow);
        build_links(&ids, &link_chunks, nthreads);
        if let Some(h) = pc_handle.take() {
            let r = h.join().unwrap();
            cat_names = r.0;
            products = r.1;
            unknown_cat = r.2;
            crows = r.3;
            campaigns = r.4;
        }
        let ncamp = crows.len();
        lap("links");
        let pshift = ids.bits.saturating_sub(NPB.trailing_zeros());
        let npb = 1usize << (ids.bits - pshift);
        let ctx = Ctx { ws, we, base, products, campaigns, ncat: cat_names.len(), unknown_cat, ncamp, first_day, ndays, ids, pshift };

        // ---- phase 1: parse events and clicks
        let per_eb = est_rows / (nthreads * NEB) * 3 / 2 + 256;
        let per_pb = est_rows / (nthreads * npb) * 3 / 2 + 256;
        let per_cb = est_ck / (nthreads * npb) * 3 / 2 + 256;
        let next = AtomicUsize::new(0);
        let mut outs: Vec<ThreadOut> = Vec::with_capacity(nthreads);
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for t in 0..nthreads {
                let ctx = &ctx;
                let jobs = &jobs;
                let next = &next;
                handles.push(s.spawn(move || {
                    let mut out = ThreadOut {
                        dk: Buckets::new(NEB, per_eb),
                        pe: Buckets::new(npb, per_pb),
                        pc: Buckets::new(npb, per_cb),
                        rows: 0,
                        tid: t as u32,
                        ring: [Pending::default(); RING],
                        rn: 0,
                        rh: 0,
                        flat: Flat::new(),
                    };
                    loop {
                        let j = next.fetch_add(1, Relaxed);
                        if j >= jobs.len() {
                            break;
                        }
                        let (kind, chunk) = jobs[j];
                        unsafe {
                            if kind == 0 {
                                parse_events(ctx, chunk, &mut out);
                            } else {
                                parse_clicks(ctx, chunk, &mut out);
                            }
                        }
                    }
                    out
                }));
            }
            for h in handles {
                outs.push(h.join().unwrap());
            }
        });
        lap("parsed");
        if ctx.ids.overflow.load(Relaxed) {
            if verbose {
                eprintln!("id table overflow, retrying larger");
            }
            grow *= 4;
            let Ctx { products: p2, campaigns: c2, .. } = ctx;
            products = p2;
            campaigns = c2;
            continue;
        }

        // ---- phase 2: dedup event copies (smallest ingest_ts wins) -> loser bitmap
        let losers: Vec<&'static [AtomicU64]> = outs.iter().map(|o| &*zeroed::<AtomicU64>((o.rows as usize + 63) / 64 + 1)).collect();
        {
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..nthreads {
                    let outs = &outs;
                    let losers = &losers;
                    let next = &next;
                    s.spawn(move || {
                        let mut keys: Vec<u64> = Vec::new();
                        let mut vals: Vec<u64> = Vec::new();
                        let mut used: Vec<u32> = Vec::new();
                        let mark = |rowid: u32| {
                            let t = (rowid >> 27) as usize;
                            let i = (rowid & ((1 << 27) - 1)) as usize;
                            losers[t][i >> 6].fetch_or(1u64 << (i & 63), Relaxed);
                        };
                        loop {
                            let b = next.fetch_add(1, Relaxed);
                            if b >= NEB {
                                break;
                            }
                            let total: usize = outs.iter().map(|o| o.dk.len(b)).sum();
                            let cap = (total * 2).next_power_of_two().max(16);
                            let mask = cap - 1;
                            if keys.len() < cap {
                                // grow (fresh zeroed memory); otherwise reuse: only used slots are dirty
                                keys = vec![0; cap];
                                vals = vec![0; cap];
                            }
                            used.clear();
                            for o in outs.iter() {
                                for part in [o.dk.main(b), o.dk.spilled(b)] {
                                    for r in part.iter() {
                                        let v = ((r.ingest as i64 - i32::MIN as i64) as u64) << 32 | r.rowid as u64;
                                        let mut p = (r.ehash as usize) & mask;
                                        loop {
                                            let k = unsafe { *keys.get_unchecked(p) };
                                            if k == 0 {
                                                unsafe {
                                                    *keys.get_unchecked_mut(p) = r.ehash;
                                                    *vals.get_unchecked_mut(p) = v;
                                                }
                                                used.push(p as u32);
                                                break;
                                            }
                                            if k == r.ehash {
                                                let old = vals[p];
                                                if (v >> 32) < (old >> 32) {
                                                    vals[p] = v;
                                                    mark(old as u32);
                                                } else {
                                                    mark(v as u32);
                                                }
                                                break;
                                            }
                                            p = (p + 1) & mask;
                                        }
                                    }
                                }
                            }
                            for &p in used.iter() {
                                unsafe {
                                    *keys.get_unchecked_mut(p as usize) = 0;
                                }
                            }
                        }
                    });
                }
            });
        }
        lap("dedup");

        // ---- phase 3: per person bucket sessionize + attribute
        let sizes: Vec<usize> = (0..npb).map(|b| outs.iter().map(|o| o.pe.len(b)).sum()).collect();
        let mut order: Vec<usize> = (0..npb).collect();
        order.sort_by_key(|&b| std::cmp::Reverse(sizes[b]));
        let next = AtomicUsize::new(0);
        let mut results: Vec<(UserDaily, Vec<[i64; 4]>)> = Vec::new();
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for _t in 0..nthreads {
                let next = &next;
                let outs = &outs;
                let losers = &losers;
                let order = &order;
                let ctx = &ctx;
                let cat_names = &cat_names;
                handles.push(s.spawn(move || {
                    let mut w = PersonWork {
                        cat_sum: vec![0; ctx.ncat],
                        wcnt: vec![0; ctx.ncamp + 1],
                        wpos: vec![0; ctx.ncamp + 1],
                        active: Vec::with_capacity(ctx.ncamp + 1),
                        plist: Vec::new(),
                        credited: vec![[0; 3]; (ctx.ncamp + 1) * ctx.ndays],
                        assisted: vec![0; (ctx.ncamp + 1) * ctx.ndays],
                        cnt: Vec::new(),
                        ccnt: Vec::new(),
                        ev: Vec::new(),
                        ck: Vec::new(),
                    };
                    let mut ud = UserDaily::new(est_rows / 9 / nthreads + 1024);
                    loop {
                        let j = next.fetch_add(1, Relaxed);
                        if j >= npb {
                            break;
                        }
                        unsafe {
                            process_person_bucket(ctx, cat_names, outs, losers, order[j], &mut w, &mut ud);
                        }
                    }
                    let cc: Vec<[i64; 4]> = w.credited.iter().zip(w.assisted.iter()).map(|(c, &a)| [c[0], c[1], c[2], a as i64]).collect();
                    (ud, cc)
                }));
            }
            for h in handles {
                results.push(h.join().unwrap());
            }
        });
        lap("sessions");
        if verbose { eprintln!("  person: events {:.1} clicks {:.1} walk {:.1} (thread-ms)", TM[0].load(Relaxed) as f64/1e6, TM[1].load(Relaxed) as f64/1e6, TM[2].load(Relaxed) as f64/1e6); }

        // ---- phase 4: write outputs
        let ud_dir = out_dir.join("user_daily");
        let cd_dir = out_dir.join("campaign_daily");
        fs::create_dir_all(&ud_dir).unwrap();
        fs::create_dir_all(&cd_dir).unwrap();
        let mut camp_total = vec![[0i64; 4]; (ncamp + 1) * ndays];
        for (_, cc) in &results {
            for (a, b) in camp_total.iter_mut().zip(cc.iter()) {
                for x in 0..4 {
                    a[x] += b[x];
                }
            }
        }
        let mut day_dict_v: Vec<u8> = Vec::new();
        for d in 0..ndays {
            day_dict_v.extend_from_slice(&((first_day + d as i64) as i32).to_le_bytes());
        }
        let mut cat_dict_v: Vec<u8> = Vec::new();
        for c in &cat_names {
            put_str(&mut cat_dict_v, c);
        }
        let day_dict = &day_dict_v[..];
        let cat_dict = &cat_dict_v[..];
        let cat_names = &cat_names;
        std::thread::scope(|s| {
            for (t, (ud, _)) in results.iter().enumerate() {
                let ud_dir = &ud_dir;
                s.spawn(move || {
                    if ud.n == 0 && t != 0 {
                        return;
                    }
                    let tw = Instant::now();
                    let (sd, sn, si) = parquet::dict_i64(&ud.sessions);
                    let (ed, en, ei) = parquet::dict_i64(&ud.events);
                    let (rd, rn, ri) = parquet::dict_i64(&ud.revenue);
                    let n = ud.n;
                    let cols = [
                        Col { name: "user_id", kind: Kind::Str, optional: false, defs: &[], data: Data::Dict { dict: &ud.names, n: ud.nnames as usize, idx: &ud.uidx } },
                        Col { name: "day", kind: Kind::Date, optional: false, defs: &[], data: Data::Dict { dict: day_dict, n: ndays, idx: &ud.day } },
                        Col { name: "sessions", kind: Kind::I64, optional: false, defs: &[], data: Data::Dict { dict: &sd, n: sn, idx: &si } },
                        Col { name: "events", kind: Kind::I64, optional: false, defs: &[], data: Data::Dict { dict: &ed, n: en, idx: &ei } },
                        Col { name: "revenue_cents", kind: Kind::I64, optional: false, defs: &[], data: Data::Dict { dict: &rd, n: rn, idx: &ri } },
                        Col { name: "top_category", kind: Kind::Str, optional: true, defs: &ud.top_def, data: Data::Dict { dict: cat_dict, n: cat_names.len(), idx: &ud.top_idx } },
                    ];
                    let tb = tw.elapsed().as_secs_f64() * 1e3;
                    let f = fs::File::create(ud_dir.join(format!("part-{t:03}.parquet"))).unwrap();
                    let mut bw = std::io::BufWriter::with_capacity(1 << 16, f);
                    parquet::write(&mut bw, n, &cols).unwrap();
                    if verbose {
                        eprintln!("   part {t}: encode {:.1} ms write {:.1} ms", tb, tw.elapsed().as_secs_f64() * 1e3 - tb);
                    }
                });
            }
            let mut n = 0usize;
            let (mut cid, mut ch, mut day, mut s0, mut s1, mut s2, mut s3) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for c in 0..=ncamp {
                for d in 0..ndays {
                    let v = camp_total[c * ndays + d];
                    if v[0] == 0 && v[3] == 0 {
                        continue;
                    }
                    n += 1;
                    if c == ncamp {
                        put_str(&mut cid, b"organic");
                        put_str(&mut ch, b"organic");
                    } else {
                        put_str(&mut cid, &crows[c].0);
                        put_str(&mut ch, &crows[c].1);
                    }
                    day.extend_from_slice(&((first_day + d as i64) as i32).to_le_bytes());
                    s0.extend_from_slice(&v[0].to_le_bytes());
                    s1.extend_from_slice(&v[1].to_le_bytes());
                    s2.extend_from_slice(&v[2].to_le_bytes());
                    s3.extend_from_slice(&v[3].to_le_bytes());
                }
            }
            let cols = [
                Col { name: "campaign_id", kind: Kind::Str, optional: false, defs: &[], data: Data::Plain(&cid) },
                Col { name: "channel", kind: Kind::Str, optional: false, defs: &[], data: Data::Plain(&ch) },
                Col { name: "day", kind: Kind::Date, optional: false, defs: &[], data: Data::Plain(&day) },
                Col { name: "sessions", kind: Kind::I64, optional: false, defs: &[], data: Data::Plain(&s0) },
                Col { name: "purchasing_sessions", kind: Kind::I64, optional: false, defs: &[], data: Data::Plain(&s1) },
                Col { name: "revenue_cents", kind: Kind::I64, optional: false, defs: &[], data: Data::Plain(&s2) },
                Col { name: "assisted_sessions", kind: Kind::I64, optional: false, defs: &[], data: Data::Plain(&s3) },
            ];
            let f = fs::File::create(cd_dir.join("part-000.parquet")).unwrap();
            let mut bw = std::io::BufWriter::with_capacity(1 << 16, f);
            parquet::write(&mut bw, n, &cols).unwrap();
        });
        lap("written");
        if std::env::var("SS_NOREL").is_err() {
            release_regions(nthreads);
            lap("released");
        }
        std::process::exit(0);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }
    fn gen(r: &mut Rng, f2: bool) -> Vec<u8> {
        let years = [1899, 1900, 1901, 1969, 1970, 2000, 2024, 2026, 2027, 2028, 2099, 2100, 2200, 2299, 2300, 2400, 0, 9999];
        let y = if r.below(4) == 0 { r.below(10000) } else { years[r.below(years.len() as u64) as usize] };
        let mo = if r.below(3) == 0 { r.below(100) } else { 1 + r.below(12) };
        let d = if r.below(3) == 0 { r.below(100) } else { [1, 28, 29, 30, 31, 15][r.below(6) as usize] };
        let h = if r.below(4) == 0 { r.below(100) } else { r.below(24) };
        let mi = if r.below(4) == 0 { r.below(100) } else { r.below(60) };
        let s = if r.below(4) == 0 { r.below(100) } else { r.below(60) };
        let mut v = format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}").into_bytes();
        if f2 {
            let oh = if r.below(4) == 0 { r.below(100) } else { r.below(24) };
            let om = if r.below(4) == 0 { r.below(100) } else { r.below(60) };
            v.extend_from_slice(format!("{}{oh:02}:{om:02}", if r.below(2) == 0 { '+' } else { '-' }).as_bytes());
        } else {
            v.push(b'Z');
        }
        // mutations
        let nm = r.below(4);
        let alphabet = b"0123456789-:TZ+ /a\x00\xff9";
        for _ in 0..nm {
            if r.below(2) == 0 {
                let i = r.below(v.len() as u64) as usize;
                v[i] = alphabet[r.below(alphabet.len() as u64) as usize];
            }
        }
        v
    }
    #[test]
    fn fuzz_canonical() {
        let mut r = Rng(12345);
        let mut buf = vec![0u8; 64];
        for pre in [&b"p"[..], &b"c_"[..]] {
            let l = Lookup::new(pre);
            for _ in 0..5_000_000 {
                let n = r.below(14) as usize;
                let alphabet = b"0123456789pc_ x";
                for i in 0..n {
                    buf[i] = if r.below(4) == 0 { alphabet[r.below(alphabet.len() as u64) as usize] } else { b'0' + r.below(10) as u8 };
                }
                if r.below(2) == 0 && n >= pre.len() {
                    buf[..pre.len()].copy_from_slice(pre);
                }
                for k in n..64 {
                    buf[k] = (r.next() & 0xFF) as u8;
                }
                let k = &buf[..n];
                let want = l.canonical(k);
                let got = unsafe { l.canonical_fast(k) };
                assert_eq!(got, want, "key {:?}", String::from_utf8_lossy(k));
            }
        }
    }
    #[test]
    fn fuzz_ts() {
        init_month_tables();
        let mut r = Rng(0x9E3779B97F4A7C15);
        let mut buf = vec![0u8; 64];
        let mut valid = 0u64;
        for it in 0..20_000_000u64 {
            let v = gen(&mut r, it & 1 == 1);
            buf[..v.len()].copy_from_slice(&v);
            for k in v.len()..64 {
                buf[k] = (r.next() & 0xFF) as u8;
            }
            let want = parse_ts(&v).unwrap_or(BAD_TS);
            let got = unsafe { parse_ts_fast(buf.as_ptr(), v.len()) };
            let old = unsafe { if v.len() == 20 { parse_f12::<false>(buf.as_ptr()) } else { parse_f12::<true>(buf.as_ptr()) } };
            assert_eq!(got, want, "input {:?}", String::from_utf8_lossy(&v));
            assert_eq!(old, want, "old input {:?}", String::from_utf8_lossy(&v));
            valid += (want != BAD_TS) as u64;
        }
        eprintln!("valid {valid}");
    }
}
