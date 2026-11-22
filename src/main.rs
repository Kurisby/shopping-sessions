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
    fn munmap(addr: *mut u8, len: usize) -> i32;
    fn mremap(old: *mut u8, old_len: usize, new_len: usize, flags: i32, ...) -> *mut u8;
}

/// Big heap blocks (>= 1 MB) come straight from mmap with huge pages requested: page
/// faults are costly on the scorer, and 2 MB pages take 512x fewer of them.
struct HugeAlloc;
const BIG: usize = 1 << 20;
const HUGE: usize = 2 << 20;
const MREMAP_MAYMOVE: i32 = 1;

#[inline]
fn big_len(n: usize) -> usize {
    (n + HUGE - 1) & !(HUGE - 1)
}

unsafe impl std::alloc::GlobalAlloc for HugeAlloc {
    #[inline]
    unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
        if l.size() < BIG || l.align() > 4096 {
            return std::alloc::System.alloc(l);
        }
        let n = big_len(l.size());
        let p = mmap(std::ptr::null_mut(), n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if p as isize == -1 {
            return std::ptr::null_mut();
        }
        if !NO_THP.load(Relaxed) {
            madvise(p, n, MADV_HUGEPAGE);
        }
        p
    }
    #[inline]
    unsafe fn alloc_zeroed(&self, l: std::alloc::Layout) -> *mut u8 {
        if l.size() < BIG || l.align() > 4096 {
            return std::alloc::System.alloc_zeroed(l);
        }
        self.alloc(l) // fresh anonymous memory is zero
    }
    #[inline]
    unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
        if l.size() < BIG || l.align() > 4096 {
            return std::alloc::System.dealloc(p, l);
        }
        munmap(p, big_len(l.size()));
    }
    unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, new: usize) -> *mut u8 {
        let old_big = l.size() >= BIG && l.align() <= 4096;
        let new_big = new >= BIG && l.align() <= 4096;
        if old_big && new_big {
            let (a, b) = (big_len(l.size()), big_len(new));
            if a == b {
                return p;
            }
            let q = mremap(p, a, b, MREMAP_MAYMOVE);
            if q as isize == -1 {
                return std::ptr::null_mut();
            }
            if b > a && !NO_THP.load(Relaxed) {
                madvise(q, b, MADV_HUGEPAGE);
            }
            return q;
        }
        if !old_big && !new_big {
            return std::alloc::System.realloc(p, l, new);
        }
        // crossing the threshold: allocate, copy, free
        let nl = std::alloc::Layout::from_size_align_unchecked(new, l.align());
        let q = self.alloc(nl);
        if !q.is_null() {
            std::ptr::copy_nonoverlapping(p, q, l.size().min(new));
            self.dealloc(p, l);
        }
        q
    }
}

#[global_allocator]
static GLOBAL: HugeAlloc = HugeAlloc;
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

#[repr(C, align(64))]
struct HMask([u8; 96]);
/// 48 x 0xFF then 48 x 0: 16 bytes from offset 48 - n + 16 j mask block j of an n-byte ID.
static HMASK: HMask = HMask({
    let mut m = [0u8; 96];
    let mut i = 0;
    while i < 48 {
        m[i] = 0xFF;
        i += 1;
    }
    m
});

/// AES-round string hash (AES-NI) of an ID. Up to 48 bytes it is branch-free: three 16-byte
/// blocks, masked to the ID's length, each absorbed by one round, then the length. Rounds are
/// bijections on the 128-bit state, so distinct IDs of up to 16 bytes never collide before
/// the 64-bit truncation. Every block byte passes at least two rounds after it is absorbed
/// (full diffusion); the length passes one, which moves it into the low four bytes.
/// Never returns 0. Reads 48 bytes from `p` whatever `n` is: only for padded memory.
#[inline(always)]
unsafe fn hash_ptr(p: *const u8, n: usize) -> u64 {
    if n > 48 {
        return hash_long(p, n);
    }
    let k0 = _mm_set_epi64x(S0 as i64, S1 as i64);
    let m = HMASK.0.as_ptr().add(48 - n);
    let x0 = _mm_and_si128(_mm_loadu_si128(p as *const __m128i), _mm_loadu_si128(m as *const __m128i));
    let x1 = _mm_and_si128(_mm_loadu_si128(p.add(16) as *const __m128i), _mm_loadu_si128(m.add(16) as *const __m128i));
    let x2 = _mm_and_si128(_mm_loadu_si128(p.add(32) as *const __m128i), _mm_loadu_si128(m.add(32) as *const __m128i));
    let s = _mm_aesenc_si128(_mm_aesenc_si128(_mm_xor_si128(x0, k0), x1), x2);
    let s = _mm_aesenc_si128(_mm_aesenc_si128(s, _mm_set1_epi64x(n as i64)), k0);
    let h = _mm_cvtsi128_si64(s) as u64;
    if h == 0 {
        1
    } else {
        h
    }
}

/// `hash_ptr` for IDs over 48 bytes.
#[cold]
#[inline(never)]
unsafe fn hash_long(p: *const u8, n: usize) -> u64 {
    let k0 = _mm_set_epi64x(S0 as i64, S1 as i64);
    let k1 = _mm_xor_si128(_mm_set_epi64x(S3 as i64, S0 as i64), _mm_cvtsi64_si128(n as i64));
    let mut s = _mm_xor_si128(_mm_loadu_si128(p as *const __m128i), k0);
    let mut i = 16;
    while i + 16 < n {
        s = _mm_aesenc_si128(s, _mm_loadu_si128(p.add(i) as *const __m128i));
        i += 16;
    }
    s = _mm_aesenc_si128(s, _mm_loadu_si128(p.add(n - 16) as *const __m128i));
    s = _mm_aesenc_si128(_mm_aesenc_si128(s, k1), k0);
    let h = _mm_cvtsi128_si64(_mm_aesenc_si128(s, k1)) as u64;
    if h == 0 {
        1
    } else {
        h
    }
}

/// Same value as hash_ptr, safe on any slice.
fn hash_bytes(b: &[u8]) -> u64 {
    if b.len() > 48 {
        unsafe { hash_ptr(b.as_ptr(), b.len()) }
    } else {
        let mut buf = [0u8; 48];
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
/// (year - 2000) * 512 + month * 32 + day -> seconds since the epoch of that day's start,
/// years 2000..2063; 0 for anything that is not a date.
static mut DAYSECS: [u32; 32768] = [0; 32768];

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
                if (2000..2064).contains(&y) {
                    for d in 1..=MONTH_LEN[i] as i64 {
                        DAYSECS[((y - 2000) * 512 + m * 32 + d) as usize] = ((MONTH_START[i] as i64 + d - 1) * 86_400) as u32;
                    }
                }
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

/// Whole seconds of `YYYY-MM-DDTHH:MM:SS` + `Z` (or + any byte when F2) at `p`, years
/// 2000..2063 only; u64::MAX when it is anything else (the caller then parses it exactly).
/// One branch covers the format and the field ranges; the day table rejects bad dates.
#[inline(always)]
unsafe fn iso_secs<const F2: bool>(p: *const u8) -> u64 {
    const PA: __m128i = v128(*b"0000-00-00T00:00");
    const LA: __m128i = v128([9, 9, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9]);
    const PB1: __m128i = v128(*b"-00-00T00:00:00Z");
    const LB1: __m128i = v128([0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0]);
    const PB2: __m128i = v128(*b"-00-00T00:00:00\0");
    const LB2: __m128i = v128([0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 255]);
    const SHA: __m128i = v128([0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 0x80, 0x80, 0x80, 0x80]);
    const SHB: __m128i = v128([0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 13, 14, 0x80, 0x80]);
    const W10: __m128i = v128([10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1]);
    // [YY, YY, MM, DD, hh, mm, ss, 0]: century 20, and every field small enough for the table
    const LO: __m128i = w128([20, 0, 0, 0, 0, 0, 0, 0]);
    const HI: __m128i = w128([20, 63, 15, 31, 23, 59, 59, 0]);
    const WM: __m128i = w128([0, 512, 32, 1, 3600, 60, 1, 0]);
    let a = _mm_loadu_si128(p as *const __m128i);
    let b = _mm_loadu_si128(p.add(4) as *const __m128i);
    let xa = _mm_xor_si128(a, PA);
    let xb = _mm_xor_si128(b, if F2 { PB2 } else { PB1 });
    let lb = if F2 { LB2 } else { LB1 };
    let ok = _mm_and_si128(_mm_cmpeq_epi8(_mm_max_epu8(xa, LA), LA), _mm_cmpeq_epi8(_mm_max_epu8(xb, lb), lb));
    let v = _mm_or_si128(_mm_shuffle_epi8(xa, SHA), _mm_shuffle_epi8(xb, SHB));
    let vals = _mm_maddubs_epi16(v, W10);
    let bad = _mm_or_si128(_mm_cmpgt_epi16(LO, vals), _mm_cmpgt_epi16(vals, HI));
    if _mm_movemask_epi8(_mm_andnot_si128(bad, ok)) != 0xFFFF {
        return u64::MAX;
    }
    let r = _mm_madd_epi16(vals, WM); // [year * 512, month * 32 + day, hh * 3600 + mm * 60, ss]
    let q = _mm_add_epi32(r, _mm_srli_epi64::<32>(r)); // [table index, _, seconds of day, _]
    let t = *DAYSECS.get_unchecked(_mm_cvtsi128_si32(q) as u32 as usize);
    if t == 0 {
        return u64::MAX;
    }
    t as u64 + _mm_extract_epi32::<2>(q) as u32 as u64
}

/// `YYYY-MM-DDTHH:MM:SSZ` (n == 20) or, F2, `YYYY-MM-DDTHH:MM:SS+HH:MM` (n == 25) in ms;
/// BAD_TS when not valid. F2 reads 3 bytes past its end (padded memory only).
#[inline(always)]
unsafe fn parse_f12_simd<const F2: bool>(p: *const u8) -> i64 {
    let secs = iso_secs::<F2>(p);
    if secs == u64::MAX {
        // anything else of this length: the exact parser (other years, bad fields, digits)
        return parse_ts_cold(p, if F2 { 25 } else { 20 });
    }
    let ms = secs as i64 * 1000;
    if !F2 {
        return ms;
    }
    let sign = *p.add(19);
    let w3 = std::ptr::read_unaligned(p.add(20) as *const u64) & 0xFF_FFFF_FFFF;
    let x3 = (w3 & !0xFF_0000) | 0x30_0000 | 0x3030_3000_0000_0000;
    // `&`, not `&&`: the sign is random, so a short-circuit branch on it would mispredict
    if ((sign != b'+') & (sign != b'-')) | ((w3 >> 16) & 0xFF != b':' as u64) | !all_digits(x3) {
        return BAD_TS;
    }
    let oh = ((w3 & 0xFF) - 0x30) * 10 + ((w3 >> 8) & 0xFF) - 0x30;
    let om = (((w3 >> 24) & 0xFF) - 0x30) * 10 + ((w3 >> 32) & 0xFF) - 0x30;
    if oh > 23 || om > 59 {
        return BAD_TS;
    }
    let off = ((oh * 60 + om) * 60_000) as i64;
    // ms - off for '+', ms + off for '-', without a branch
    let neg = -((sign == b'+') as i64);
    ms + ((off ^ neg) - neg)
}

/// `parse_f12_simd::<false>` in whole seconds (BAD_TS when bad): the `Z` form has no
/// sub-second part, so this is exactly its value / 1000.
#[inline(always)]
unsafe fn parse_z_secs(p: *const u8) -> i64 {
    let secs = iso_secs::<false>(p);
    if secs == u64::MAX {
        let ms = parse_ts_cold(p, 20);
        return if ms == BAD_TS { BAD_TS } else { ms.div_euclid(1000) };
    }
    secs as i64
}

/// `parse_ts` of `n` bytes at `p`, BAD_TS when bad: the rare inputs the fast paths decline.
#[cold]
#[inline(never)]
unsafe fn parse_ts_cold(p: *const u8, n: usize) -> i64 {
    parse_ts(sl(p, n)).unwrap_or(BAD_TS)
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
        }
        if self.qm.capacity() < nb + 8 {
            self.qm = Vec::with_capacity(nb + 8);
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
        for k in 0..8 {
            *qmp.add(nb + k) = 0;
        }
    }
    #[inline(always)]
    unsafe fn qbefore(&self, off: u32) -> u32 {
        let b = (off >> 6) as usize;
        let r = off & 63;
        *self.qcum.as_ptr().add(b) + (*self.qm.as_ptr().add(b) & ((1u64 << r) - 1)).count_ones()
    }
}

/// Strips the quote pair of every field wrapped in one; false when the line has other quotes.
#[inline(always)]
unsafe fn strip_quotes<const N: usize>(fp: &mut [*const u8; N], fl: &mut [usize; N], q: u32) -> bool {
    let mut fq = 0u32;
    for i in 0..N {
        let p = fp[i];
        let n = fl[i];
        let both = (n >= 2) & (*p == b'"') & (*p.add(n.max(1) - 1) == b'"');
        fq += both as u32;
        fp[i] = p.add(both as usize);
        fl[i] = n - 2 * both as usize;
    }
    q == 2 * fq
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
                let mut starts = _mm256_sub_epi32(prev, _mm256_set1_epi32(-1));
                let mut lens = _mm256_sub_epi32(sv, starts);
                // the line's '\n' is no quote, so quotes before the next line = quotes before `last`
                let ql = flat.qbefore(last);
                let q = ql - qls;
                qls = ql;
                let lsl = ls;
                ls = last + 1;
                j += N;
                let mut slow = false;
                // most lines have quotes, so the check runs on every line: a branch on q would
                // mispredict about one line in four, and the check costs less than that
                {
                    if last - lsl < 250 {
                        // quote bits of the line's 256 bytes from lsl, as 8 x 32-bit words
                        let qp = flat.qm.as_ptr().add((lsl >> 6) as usize) as *const __m256i;
                        let r = (lsl & 63) as i64;
                        let w = _mm256_or_si256(
                            _mm256_srl_epi64(_mm256_loadu_si256(qp), _mm_cvtsi64_si128(r)),
                            _mm256_sll_epi64(_mm256_loadu_si256((qp as *const u64).add(1) as *const __m256i), _mm_cvtsi64_si128(64 - r)),
                        );
                        let rs = _mm256_sub_epi32(starts, _mm256_set1_epi32(lsl as i32));
                        let re = _mm256_sub_epi32(sv, _mm256_set1_epi32(lsl as i32 + 1));
                        let m31 = _mm256_set1_epi32(31);
                        let fb = _mm256_srlv_epi32(_mm256_permutevar8x32_epi32(w, _mm256_srli_epi32::<5>(rs)), _mm256_and_si256(rs, m31));
                        let lb = _mm256_srlv_epi32(_mm256_permutevar8x32_epi32(w, _mm256_srli_epi32::<5>(re)), _mm256_and_si256(re, m31));
                        let lane1 = if N == 7 { _mm256_setr_epi32(1, 1, 1, 1, 1, 1, 1, 0) } else { _mm256_setr_epi32(1, 1, 1, 1, 0, 0, 0, 0) };
                        // fields wrapped in a quote pair lose it; the line is well-formed only
                        // if those pairs are all its quotes
                        let both = _mm256_and_si256(
                            _mm256_and_si256(fb, lb),
                            _mm256_and_si256(_mm256_cmpgt_epi32(lens, _mm256_set1_epi32(1)), lane1),
                        );
                        let fq = (_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32::<31>(both))) as u32).count_ones();
                        if q != 2 * fq {
                            continue;
                        }
                        starts = _mm256_add_epi32(starts, both);
                        lens = _mm256_sub_epi32(lens, _mm256_add_epi32(both, both));
                    } else {
                        slow = true;
                    }
                }
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
                if slow && !strip_quotes(fp, fl, q) {
                    continue;
                }
            } else {
                fp[0] = base.add(ls as usize);
                fl[0] = (*s - ls) as usize;
                for i in 1..N {
                    let a = *s.add(i - 1) + 1;
                    fp[i] = base.add(a as usize);
                    fl[i] = (*s.add(i) - a) as usize;
                }
                let ql = flat.qbefore(last);
                let q = ql - qls;
                qls = ql;
                ls = last + 1;
                j += N;
                if q != 0 && !strip_quotes(fp, fl, q) {
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

/// An ID's bytes: up to 15 inline (bytes 0..15, length in byte 15),
/// else a ref (ptr | len << 48) in w0 and 0xFF in byte 15.
#[repr(C, align(16))]
struct Name {
    w0: AtomicU64,
    w1: AtomicU64,
}

/// The first 7 bytes of `x` (little-endian bytes) as a big-endian 56-bit number.
#[inline(always)]
fn be7(x: u64) -> u64 {
    (x << 8).swap_bytes()
}

impl Name {
    /// Stores the ID that `r` (a mkref of padded memory) points at.
    #[inline(always)]
    fn set(&self, r: u64) {
        let n = (r >> 48) as usize;
        let p = (r & 0xFFFF_FFFF_FFFF) as *const u8;
        let (a, b) = unsafe {
            let rd = |i: usize| std::ptr::read_unaligned(p.add(i) as *const u64);
            if n > 15 {
                // long: the ref, and the first 7 bytes (for comparisons without a dereference)
                (r, (0xFFu64 << 56) | be7(rd(0)))
            } else if n > 8 {
                (rd(0), (rd(n - 8) >> (8 * (16 - n))) | ((n as u64) << 56))
            } else if n == 8 {
                (rd(0), 8u64 << 56)
            } else if n == 0 {
                (0, 0)
            } else {
                (rd(0) & (u64::MAX >> (64 - 8 * n)), (n as u64) << 56)
            }
        };
        self.w0.store(a, Relaxed);
        // w1 is never 0 once set (IDs are not empty), so it doubles as the "ready" flag
        self.w1.store(b, Release);
    }
    /// The name's first 7 bytes, zero-padded, big-endian: when two names' keys differ, they
    /// order the names as their bytes do. Waits for a name another thread may still be storing
    /// (its slot key is already visible).
    #[inline(always)]
    fn key7_sync(&self) -> u64 {
        let mut w1 = self.w1.load(Acquire);
        while w1 == 0 {
            std::hint::spin_loop();
            w1 = self.w1.load(Acquire);
        }
        let inline = be7(self.w0.load(Relaxed));
        if (w1 >> 56) == 0xFF {
            w1 & 0x00FF_FFFF_FFFF_FFFF
        } else {
            inline
        }
    }
    #[inline(always)]
    fn is_long(&self) -> bool {
        (self.w1.load(Relaxed) >> 56) > 15
    }
    #[inline(always)]
    fn ptr(&self) -> *const u8 {
        (self.w0.load(Relaxed) & 0xFFFF_FFFF_FFFF) as *const u8
    }
    #[inline(always)]
    fn get(&'static self) -> &'static [u8] {
        let l = (self.w1.load(Relaxed) >> 56) as usize;
        if l <= 15 {
            unsafe { std::slice::from_raw_parts(self as *const Name as *const u8, l) }
        } else {
            deref(self.w0.load(Relaxed))
        }
    }
}

struct Ids {
    slots: &'static [Slot],
    prefs: &'static [Name], // the ID's bytes
    mask: usize,
    bits: u32,
    overflow: AtomicBool,
}

impl Ids {
    fn new(cap: usize) -> Self {
        let cap = cap.next_power_of_two().max(1024);
        Ids {
            slots: zeroed::<Slot>(cap),
            prefs: zeroed::<Name>(cap),
            mask: cap - 1,
            bits: cap.trailing_zeros(),
            overflow: AtomicBool::new(false),
        }
    }
    /// The name entry of slot group `h`'s first slot (the group's names share a line).
    #[inline(always)]
    fn name_ptr(&self, h: u64) -> *const Name {
        unsafe { self.prefs.as_ptr().add((h as usize) & self.mask & !3) }
    }
    #[inline(always)]
    fn slot_ptr(&self, h: u64) -> *const Slot {
        unsafe { self.slots.as_ptr().add((h as usize) & self.mask) }
    }
    /// Slot of the ID, inserting it when new. Returns (slot, inserted).
    /// Probes whole 64-byte groups of 4 slots with one vector compare (no per-slot branches).
    #[inline(always)]
    fn intern(&self, h: u64, r: u64) -> (u32, bool) {
        let mut g = (h as usize) & self.mask & !3;
        unsafe {
            let hv = _mm256_set1_epi64x(h as i64);
            let zero = _mm256_setzero_si256();
            let mut groups = 0u32;
            loop {
                let p = self.slots.as_ptr().add(g) as *const __m256i;
                let a = _mm256_load_si256(p);
                let b = _mm256_load_si256(p.add(1));
                // keys are the even 64-bit lanes: slot k of the group is bit 2k
                let hit = (_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(a, hv))) & 5)
                    | ((_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(b, hv))) & 5) << 4);
                if hit != 0 {
                    return ((g + (hit.trailing_zeros() / 2) as usize) as u32, false);
                }
                let empty = (_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(a, zero))) & 5)
                    | ((_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(b, zero))) & 5) << 4);
                if empty != 0 {
                    let i = g + (empty.trailing_zeros() / 2) as usize;
                    match self.slots.get_unchecked(i).key.compare_exchange(0, h, Relaxed, Relaxed) {
                        Ok(_) => {
                            self.prefs.get_unchecked(i).set(r);
                            return (i as u32, true);
                        }
                        Err(k2) => {
                            if k2 == h {
                                return (i as u32, false);
                            }
                            continue; // another ID took the slot: look at the group again
                        }
                    }
                }
                groups += 1;
                if groups > 128 {
                    self.overflow.store(true, Relaxed);
                    return (0, false);
                }
                g = (g + 4) & self.mask;
            }
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
            // path halving with a plain store: x is no root (no union writes it), and gp, an
            // ancestor of x, stays one, so a racing halving that this store undoes loses
            // nothing but a shortcut
            parent[x as usize].store(gp, Relaxed);
        }
        x = p;
    }
}

/// Joins the sets of a and b, always hanging the root with the byte-wise larger ID under
/// the other: every parent's ID is smaller than its child's, so each root is its set's
/// smallest ID (the person's representative) and no cycle can form.
#[inline(always)]
fn uf_union(ids: &Ids, parent: &[AtomicU32], a: u32, b: u32) {
    let (mut a, mut b) = (a, b);
    loop {
        a = uf_find(parent, a);
        b = uf_find(parent, b);
        if a == b {
            return;
        }
        let (pa, pb) = (&ids.prefs[a as usize], &ids.prefs[b as usize]);
        let (ka, kb) = (pa.key7_sync(), pb.key7_sync());
        // the 7-byte keys settle nearly every pair; only ties read the full IDs
        let a_gt = if ka != kb { ka > kb } else { pa.get() > pb.get() };
        let (hi, lo) = if a_gt { (a, b) } else { (b, a) };
        if parent[hi as usize].compare_exchange(0, lo + 1, AcqRel, Relaxed).is_ok() {
            return;
        }
    }
}

/// Joins linked IDs; every linked ID's slot gets its representative (byte-wise smallest ID).
fn build_links(ids: &Ids, link_chunks: &[&'static [u8]], nthreads: usize) {
    let tb0 = Instant::now();
    let cap = ids.mask + 1;
    let parent: &'static [AtomicU32] = zeroed::<AtomicU32>(cap);
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
                                    _mm_prefetch(ids.prefs.as_ptr().add(a as usize) as *const i8, _MM_HINT_T0);
                                    _mm_prefetch(ids.prefs.as_ptr().add(b as usize) as *const i8, _MM_HINT_T0);
                                }
                                pairs.push((a, b));
                            }
                        }
                        for &(a, b) in pairs.iter() {
                            uf_union(ids, parent, a, b);
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
                                // and the names of the slots' groups (one line per group of 4):
                                // a new ID's name store then hits, so the slot CAS (a locked
                                // instruction, which waits for every earlier store) waits on nothing
                                if PF_NAMES {
                                    _mm_prefetch(ids.name_ptr(ha) as *const i8, _MM_HINT_ET0);
                                    _mm_prefetch(ids.name_ptr(hb) as *const i8, _MM_HINT_ET0);
                                }
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
    if std::env::var("SS_VERBOSE").is_ok() {
        eprintln!("  links pass 1 {:.1} ms: {} inserted", tb0.elapsed().as_secs_f64() * 1e3, inserted.iter().map(|v| v.len()).sum::<usize>());
    }
    const PD: usize = 8;
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
                    let r = uf_find(parent, i);
                    if r != i {
                        ids.slots[i as usize].rep.store(r + 1, Relaxed);
                    }
                }
            });
        }
    });
}

// ---------------------------------------------------------------- bucketed output buffers in huge-page arenas

/// Per-bucket record lists, as chunks of `ch` records bump-allocated from one arena per thread,
/// so memory is touched only as records arrive (fixed per-bucket regions had to be sized for the
/// busiest thread and touched it all under huge pages).
struct Buckets<T: Copy + 'static> {
    ch: usize,
    cur: Vec<[*mut T; 2]>,    // write pointer, end of the current chunk (null before the first)
    chunks: Vec<Vec<*mut T>>, // chunk starts in fill order; the last is the current one
    next: *mut T,             // arena bump pointer
    left: usize,              // chunks left in the arena
    arena: usize,             // chunks per arena region
    all: Vec<*mut T>,         // every chunk start, in allocation order (a ref's chunk number)
    rend: Vec<u32>,           // ref one past the bucket's current chunk
    tag: u32,                 // high bits of every ref (the thread)
}
unsafe impl<T: Copy + 'static> Send for Buckets<T> {}
unsafe impl<T: Copy + 'static> Sync for Buckets<T> {}

impl<T: Copy + 'static> Buckets<T> {
    /// `n` buckets, chunks of `ch` records, arena regions sized for about `expect` records.
    fn new(n: usize, ch: usize, expect: usize) -> Self {
        let arena = expect / ch + n + 16;
        let mem = zeroed::<T>(arena * ch);
        Buckets {
            ch,
            cur: vec![[std::ptr::null_mut(); 2]; n],
            chunks: (0..n).map(|_| Vec::new()).collect(),
            next: mem.as_mut_ptr(),
            left: arena,
            arena,
            all: Vec::new(),
            rend: vec![0; n],
            tag: 0,
        }
    }
    /// `push`, returning the record's ref: tag | chunk number << 10 | slot (chunks of 1024).
    #[inline(always)]
    fn push_ref(&mut self, b: usize, v: T) -> u32 {
        unsafe {
            let c = self.cur.get_unchecked_mut(b);
            if c[0] >= c[1] {
                self.refill(b);
            }
            let c = self.cur.get_unchecked_mut(b);
            let p = c[0];
            *p = v;
            c[0] = p.add(1);
            *self.rend.get_unchecked(b) - c[1].offset_from(p) as u32
        }
    }
    /// The record of a ref from `push_ref` (without its tag).
    #[inline(always)]
    unsafe fn at(&self, r: u32) -> *mut T {
        self.all.get_unchecked((r >> 10) as usize).add((r & 1023) as usize)
    }
    #[inline(always)]
    fn push(&mut self, b: usize, v: T) {
        // (the record does not go to the cold refill: passed by reference, it would go
        // through the stack, and a 12-byte record stored as u32s then loaded as a u64
        // stalls on store forwarding)
        unsafe {
            let c = self.cur.get_unchecked_mut(b);
            if c[0] >= c[1] {
                self.refill(b);
            }
            let c = self.cur.get_unchecked_mut(b);
            let p = c[0];
            *p = v;
            c[0] = p.add(1);
        }
    }
    #[cold]
    #[inline(never)]
    fn refill(&mut self, b: usize) {
        unsafe {
            if self.left == 0 {
                self.next = zeroed::<T>(self.arena * self.ch).as_mut_ptr();
                self.left = self.arena;
            }
            let s = self.next;
            self.next = s.add(self.ch);
            self.left -= 1;
            self.chunks[b].push(s);
            self.cur[b] = [s, s.add(self.ch)];
            if self.ch == 1024 {
                assert!(self.all.len() < (1 << 17) - 1, "too many chunks");
                self.rend[b] = self.tag | ((self.all.len() as u32) << 10) + 1024;
            }
            self.all.push(s);
        }
    }
    /// The bucket's records, chunk by chunk.
    #[inline(always)]
    fn parts(&self, b: usize) -> impl Iterator<Item = &mut [T]> + '_ {
        let cs = &self.chunks[b];
        let k = cs.len();
        let end = self.cur[b][0];
        let ch = self.ch;
        cs.iter().enumerate().map(move |(i, &s)| unsafe {
            let n = if i + 1 < k { ch } else { end.offset_from(s) as usize };
            std::slice::from_raw_parts_mut(s, n)
        })
    }
    fn len(&self, b: usize) -> usize {
        let cs = &self.chunks[b];
        match cs.last() {
            None => 0,
            Some(&s) => (cs.len() - 1) * self.ch + unsafe { self.cur[b][0].offset_from(s) as usize },
        }
    }
}

// ---------------------------------------------------------------- records

#[derive(Clone, Copy)]
#[repr(C)]
struct DKey {
    ehash: u64,
    ingest: i32,
    pref: u32, // the event's PRec (thread << 27 | ref), NONE when it has none
}

#[derive(Clone, Copy)]
#[repr(C)]
struct PRec {
    pid: u32, // INVALID once dedup finds a copy that wins
    ts: u32,
    pc: u32, // price << 8 | category (price 0 = not a purchase)
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

const LPD: usize = 24;
const PF_NAMES: bool = true;
static BMAX: AtomicU64 = AtomicU64::new(0);
static BMAXJ: AtomicU64 = AtomicU64::new(0);

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

const EB_BITS: u32 = 11;
const NEB: usize = 1 << EB_BITS;
const NPB: usize = 512;
const BATCH: usize = 64;
const INVALID: u32 = u32::MAX;
const NONE: u32 = u32::MAX;

#[inline(always)]
fn ebucket(h: u64) -> usize {
    (h >> (64 - EB_BITS)) as usize
}

/// Queue of a `ts` field's form: 0 = ISO `Z` (20 bytes), 1 = epoch ms (13), 2 = ISO offset
/// (25), 3 = anything else. Rows wait in one queue per form, so each queue is parsed by one
/// form's code: the form, random from row to row, costs no branch mispredict.
#[inline(always)]
fn ts_form(n: usize) -> usize {
    const F: [u8; 32] = {
        let mut f = [3u8; 32];
        f[20] = 0;
        f[13] = 1;
        f[25] = 2;
        f
    };
    F[n.min(31)] as usize
}

/// Parses a `ts` field of form `F` (see ts_form); BAD_TS when bad.
#[inline(always)]
unsafe fn parse_ts_form<const F: usize>(p: *const u8, n: usize) -> i64 {
    match F {
        0 => parse_f12_simd::<false>(p),
        1 => {
            let a = std::ptr::read_unaligned(p as *const u64);
            let b = std::ptr::read_unaligned(p.add(5) as *const u64);
            if all_digits(a) & all_digits(b) {
                (parse8(a) * 100_000 + parse8(b) % 100_000) as i64
            } else {
                BAD_TS
            }
        }
        2 => parse_f12_simd::<true>(p),
        _ => match parse_ts(sl(p, n)) {
            Some(t) => t,
            None => BAD_TS,
        },
    }
}

#[derive(Clone, Copy, Default)]
struct Pending {
    uhash: u64,
    uref: u64,
    tsp: u64, // the ts field
    ingest: i64,
    ehash: u64,
    irel: i32, // ingest for dedup
    tsn: u32,
    pc: u32,
}

struct ThreadOut {
    dk: Buckets<DKey>,
    pe: Buckets<PRec>,
    pc: Buckets<CRec>,
    tid: u32,
    pq: [[Pending; BATCH]; 4], // by ts form
    pn: [usize; 4],
    pt: usize, // rows pending in all queues
    flat: Flat,
}

impl ThreadOut {
    /// Parses the queued rows' ts, drops those outside the window or ingested over a day
    /// after it, and buckets the rest by person.
    #[inline(always)]
    fn finish_events<const F: usize>(&mut self, ctx: &Ctx) {
        let n = self.pn[F];
        self.pn[F] = 0;
        for i in 0..n {
            let e = unsafe { *self.pq[F].get_unchecked(i) };
            let ts = unsafe { parse_ts_form::<F>(e.tsp as *const u8, e.tsn as usize) };
            // (ingest is never BAD_TS here)
            let pref = if (ts < ctx.ws) | (ts >= ctx.we) | (e.ingest.wrapping_sub(ts) > DAY) {
                NONE
            } else {
                let pid = ctx.ids.person(e.uhash, e.uref);
                self.pe.push_ref((pid >> ctx.pshift) as usize, PRec { pid, ts: (ts - ctx.base) as u32, pc: e.pc })
            };
            self.dk.push(ebucket(e.ehash), DKey { ehash: e.ehash, ingest: e.irel, pref });
        }
    }
    /// Queues a row (its person's ID slot prefetched); every 64 rows all queues are
    /// flushed, so each slot is used about 64 rows after its prefetch, whatever the form.
    #[inline(always)]
    fn push_pending(&mut self, p: Pending, ctx: &Ctx) {
        let k = ts_form(p.tsn as usize);
        unsafe {
            _mm_prefetch(ctx.ids.slot_ptr(p.uhash) as *const i8, _MM_HINT_T0);
            let m = *self.pn.get_unchecked(k);
            *self.pq.get_unchecked_mut(k).get_unchecked_mut(m) = p;
            *self.pn.get_unchecked_mut(k) = m + 1;
        }
        self.pt += 1;
        if self.pt == BATCH {
            self.flush_all_events(ctx);
        }
    }
    #[inline(never)]
    fn flush_all_events(&mut self, ctx: &Ctx) {
        self.pt = 0;
        self.finish_events::<0>(ctx);
        self.finish_events::<1>(ctx);
        self.finish_events::<2>(ctx);
        self.finish_events::<3>(ctx);
    }
}

#[derive(Clone, Copy, Default)]
struct PendingClick {
    uhash: u64,
    uref: u64,
    kref: u64,
    tsp: u64,
    tsn: u32,
    camp: u32,
}

struct ClickBatch {
    q: [[PendingClick; BATCH]; 4], // by ts form
    n: [usize; 4],
    t: usize, // clicks pending in all queues
}

impl ClickBatch {
    #[inline(never)]
    fn flush(&mut self, ctx: &Ctx, out: &mut ThreadOut) {
        self.t = 0;
        self.finish::<0>(ctx, out);
        self.finish::<1>(ctx, out);
        self.finish::<2>(ctx, out);
        self.finish::<3>(ctx, out);
    }
    /// Parses the queued clicks' ts, drops those outside [window start - 7 days, window end),
    /// and buckets the rest by person.
    #[inline(always)]
    fn finish<const F: usize>(&mut self, ctx: &Ctx, out: &mut ThreadOut) {
        let n = self.n[F];
        self.n[F] = 0;
        for i in 0..n {
            let e = unsafe { *self.q[F].get_unchecked(i) };
            let ts = unsafe { parse_ts_form::<F>(e.tsp as *const u8, e.tsn as usize) };
            if ts < ctx.base || ts >= ctx.we {
                continue;
            }
            let pid = ctx.ids.person(e.uhash, e.uref);
            out.pc.push((pid >> ctx.pshift) as usize, CRec { pid, ts: (ts - ctx.base) as u32, camp: e.camp, _pad: 0, kref: e.kref });
        }
    }
    #[inline(always)]
    fn push(&mut self, p: PendingClick, ctx: &Ctx, out: &mut ThreadOut) {
        let k = ts_form(p.tsn as usize);
        unsafe {
            _mm_prefetch(ctx.ids.slot_ptr(p.uhash) as *const i8, _MM_HINT_T0);
            let m = *self.n.get_unchecked(k);
            *self.q.get_unchecked_mut(k).get_unchecked_mut(m) = p;
            *self.n.get_unchecked_mut(k) = m + 1;
        }
        self.t += 1;
        if self.t == BATCH {
            self.flush(ctx, out);
        }
    }
}

unsafe fn parse_events(ctx: &Ctx, chunk: &'static [u8], out: &mut ThreadOut) {
    let base = ctx.base;
    // the seconds path gives the same ingest_rel as the ms path only when base is a whole
    // second (it is when the manifest's window bounds are ISO timestamps)
    let base_secs = base.div_euclid(1000);
    let zfast = base.rem_euclid(1000) == 0;
    // fast-path ingest seconds (years 2000..2063) minus base_secs fit an i32 below i32::MAX
    // without clamping when base is in 2000..2034
    let noclamp = zfast & (946_684_800..946_684_800 + (1 << 30)).contains(&base_secs);
    let mut flat = std::mem::replace(&mut out.flat, Flat::new());
    for_lines::<7>(&mut flat, chunk, |fp, fl| {
        if fl[0] == 0 {
            return;
        }
        let ehash = hash_ptr(fp[0], fl[0]);
        // ingest_ts: always the `Z` form in practice, parsed straight to seconds
        let fsecs = if noclamp & (fl[3] == 20) { iso_secs::<false>(fp[3]) } else { u64::MAX };
        let (ingest, ingest_rel) = if fsecs != u64::MAX {
            (fsecs as i64 * 1000, (fsecs as i64 - base_secs) as i32)
        } else if zfast & (fl[3] == 20) {
            let secs = parse_z_secs(fp[3]);
            if secs == BAD_TS {
                (BAD_TS, i32::MAX)
            } else {
                (secs * 1000, (secs - base_secs).clamp(i32::MIN as i64, (i32::MAX - 1) as i64) as i32)
            }
        } else {
            let ingest = parse_ts_fast(fp[3], fl[3]);
            (ingest, if ingest == BAD_TS { i32::MAX } else { ((ingest - base).div_euclid(1000)).clamp(i32::MIN as i64, (i32::MAX - 1) as i64) as i32 })
        };
        // a row that fails validation still takes part in dedup
        let nope = DKey { ehash, ingest: ingest_rel, pref: NONE };
        if fl[1] == 0 || fl[5] == 0 {
            out.dk.push(ebucket(ehash), nope);
            return;
        }
        let mut pc = 0u32;
        if fl[4] == 4 {
            let w = std::ptr::read_unaligned(fp[4] as *const u32);
            if !((w == u32::from_le_bytes(*b"view")) | (w == u32::from_le_bytes(*b"cart"))) {
                out.dk.push(ebucket(ehash), nope);
                return;
            }
        } else if fl[4] == 8 && std::ptr::read_unaligned(fp[4] as *const u64) == u64::from_le_bytes(*b"purchase") {
            let price = match parse_price(sl(fp[6], fl[6])) {
                Some(p) => p,
                None => {
                    out.dk.push(ebucket(ehash), nope);
                    return;
                }
            };
            let cat = match ctx.products.get(sl(fp[5], fl[5])) {
                ABSENT => ctx.unknown_cat,
                c => c,
            };
            pc = (price << 8) | cat;
        } else {
            out.dk.push(ebucket(ehash), nope);
            return;
        }
        if ingest == BAD_TS {
            out.dk.push(ebucket(ehash), nope);
            return;
        }
        // ts (window and ingest-delay checks too) is parsed with its queue, by form
        let uhash = hash_ptr(fp[1], fl[1]);
        out.push_pending(Pending { uhash, uref: mkref(fp[1], fl[1]), tsp: fp[2] as u64, ingest, ehash, irel: ingest_rel, tsn: fl[2] as u32, pc }, ctx);
    });
    out.flat = flat;
    out.flush_all_events(ctx);
}

unsafe fn parse_clicks(ctx: &Ctx, chunk: &'static [u8], out: &mut ThreadOut) {
    let mut ring = ClickBatch { q: [[PendingClick::default(); BATCH]; 4], n: [0; 4], t: 0 };
    let mut flat = std::mem::replace(&mut out.flat, Flat::new());
    for_lines::<4>(&mut flat, chunk, |fp, fl| {
        if fl[0] == 0 || fl[1] == 0 {
            return;
        }
        let camp = ctx.campaigns.get(sl(fp[2], fl[2]));
        if camp == ABSENT {
            return;
        }
        let uhash = hash_ptr(fp[1], fl[1]);
        ring.push(PendingClick { uhash, uref: mkref(fp[1], fl[1]), kref: mkref(fp[0], fl[0]), tsp: fp[3] as u64, tsn: fl[3] as u32, camp }, ctx, out);
    });
    out.flat = flat;
    ring.flush(ctx, out);
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
    small: [u64; 33],
    small_ck: [CKey; 33],
    cat_sum: Vec<u64>,
    wcnt: Vec<u32>,
    wpos: Vec<u32>,
    since: Vec<u32>, // an active campaign's first session (person-local index) not yet counted
    active: Vec<u32>,
    plist: Vec<u32>,
    credited: Vec<[i64; 3]>,
    assisted: Vec<u32>,
    cnt: Vec<u32>,
    ccnt: Vec<u32>,
    ev: Vec<u64>,  // ts << 32 | pc, grouped by person
    ck: Vec<CKey>, // grouped by person
    tmp_ev: Vec<u64>,
    tmp_ck: Vec<CKey>,
}

/// Sorts by the key's high 32 bits (times; ties in any order are fine for every caller).
/// Comparison sort for small slices, LSD radix (3 x 11 bits) for large ones (whales).
#[inline(always)]
fn sort_small<T: Copy>(v: &mut [T], tmp: &mut Vec<T>, key: impl Fn(&T) -> u64) {
    if v.len() <= 1 {
        return;
    }
    if v.len() < RADIX_MIN {
        v.sort_unstable_by_key(|r| key(r));
    } else {
        radix_hi32(v, tmp, key);
    }
}

const RADIX_MIN: usize = 2048;

/// Sorts a person's events (ts << 32 | pc) by time (the order of equal times is free) and
/// returns them: in place, or for a small person in `small`.
#[inline(always)]
unsafe fn sort_events<'a>(v: &'a mut [u64], tmp: &mut Vec<u64>, small: &'a mut [u64; 33]) -> &'a [u64] {
    let n = v.len();
    if n <= 8 {
        if n <= 1 {
            return v;
        }
        rank_sort::<2>(v, small);
    } else if n <= 16 {
        rank_sort::<4>(v, small);
    } else if n <= 32 {
        rank_sort::<8>(v, small);
    } else if n < RADIX_MIN {
        merge_sort_mid(v, tmp, small);
        return v;
    } else {
        sort_small(v, tmp, |&x| x);
        return v;
    }
    &small[..n]
}

/// Branch-free sort of 33..RADIX_MIN events (a comparison sort would mispredict about every
/// other comparison on these random orders): rank-sorted runs of 32, then merges whose
/// only branches are loop exits.
#[inline(never)]
unsafe fn merge_sort_mid(v: &mut [u64], tmp: &mut Vec<u64>, small: &mut [u64; 33]) {
    let n = v.len();
    tmp.clear();
    tmp.reserve(n);
    tmp.set_len(n);
    let mut b = 0;
    while b + 32 <= n {
        rank32(v.as_ptr().add(b), tmp.as_mut_ptr().add(b));
        b += 32;
    }
    if b < n {
        rank_sort::<8>(v.get_unchecked(b..n), small);
        std::ptr::copy_nonoverlapping(small.as_ptr(), tmp.as_mut_ptr().add(b), n - b);
    }
    let (mut src, mut dst) = (tmp.as_mut_ptr(), v.as_mut_ptr());
    let mut width = 32;
    while width < n {
        let mut start = 0;
        while start < n {
            let mid = (start + width).min(n);
            let end = (start + 2 * width).min(n);
            let (mut i, mut j, mut k) = (start, mid, start);
            if mid < end {
                // the heads stay in registers and both successors load every step (the
                // index of the last one clamped), so a step waits on a compare and a move,
                // not on the load its own compare picked
                let mut a = *src.add(i);
                let mut c = *src.add(j);
                loop {
                    let na = *src.add(i + 1); // i + 1 <= mid < end: in bounds
                    let nc = *src.add((j + 1).min(end - 1));
                    let tb = c < a;
                    *dst.add(k) = if tb { c } else { a };
                    k += 1;
                    j += tb as usize;
                    i += !tb as usize;
                    if (i >= mid) | (j >= end) {
                        break;
                    }
                    a = if tb { a } else { na };
                    c = if tb { nc } else { c };
                }
            }
            std::ptr::copy_nonoverlapping(src.add(i), dst.add(k), mid - i);
            k += mid - i;
            std::ptr::copy_nonoverlapping(src.add(j), dst.add(k), end - j);
            start = end;
        }
        std::mem::swap(&mut src, &mut dst);
        width *= 2;
    }
    if src != v.as_mut_ptr() {
        std::ptr::copy_nonoverlapping(src, v.as_mut_ptr(), n);
    }
}

/// `rank_sort::<8>` of exactly 32 events from `v` into `out`, keys built 4 at a time.
#[inline(always)]
unsafe fn rank32(v: *const u64, out: *mut u64) {
    let mut kk = [0i64; 32];
    let mut blk = [_mm256_setzero_si256(); 8];
    let mut acc = [_mm256_setzero_si256(); 8];
    let iota = _mm256_setr_epi64x(0, 1, 2, 3);
    for m in 0..8 {
        let x = _mm256_loadu_si256(v.add(4 * m) as *const __m256i);
        // (ts << 5 | position): unique, positive (ts < 2^31), in (time, position) order
        let k = _mm256_or_si256(_mm256_slli_epi64::<5>(_mm256_srli_epi64::<32>(x)), _mm256_add_epi64(iota, _mm256_set1_epi64x(4 * m as i64)));
        blk[m] = k;
        _mm256_storeu_si256((kk.as_mut_ptr() as *mut __m256i).add(m), k);
    }
    for j in 0..32 {
        let b = _mm256_set1_epi64x(*kk.get_unchecked(j));
        for m in 0..8 {
            acc[m] = _mm256_sub_epi64(acc[m], _mm256_cmpgt_epi64(blk[m], b));
        }
    }
    let mut r = [0u64; 32];
    for m in 0..8 {
        _mm256_storeu_si256((r.as_mut_ptr() as *mut __m256i).add(m), acc[m]);
    }
    for i in 0..32 {
        *out.add(*r.get_unchecked(i) as usize) = *v.add(i);
    }
}

/// Branch-free sort of 2..=4*NB events into `out` (most persons have few): each event's
/// rank is the number of events before it in (time, position) order, counted with vector
/// compares of every event against every other. Loops run a fixed 4*NB times.
#[inline(always)]
unsafe fn rank_sort<const NB: usize>(v: &[u64], out: &mut [u64; 33]) {
    let n = v.len();
    debug_assert!(n >= 1 && n <= 4 * NB);
    let mut k = [0i64; 32];
    let mut val = [0u64; 32];
    for i in 0..4 * NB {
        let x = *v.get_unchecked(i.min(n - 1));
        // ts < 2^31, so (ts << 5 | i) is a unique, positive key in time order;
        // padding lanes get the largest key, rank n, and land in out[n] (never read)
        *k.get_unchecked_mut(i) = if i < n { (((x >> 32) << 5) | i as u64) as i64 } else { i64::MAX };
        *val.get_unchecked_mut(i) = x;
    }
    let kp = k.as_ptr() as *const __m256i;
    let mut blk = [_mm256_setzero_si256(); NB];
    let mut acc = [_mm256_setzero_si256(); NB];
    for m in 0..NB {
        blk[m] = _mm256_loadu_si256(kp.add(m));
    }
    for j in 0..4 * NB {
        let b = _mm256_set1_epi64x(*k.get_unchecked(j));
        for m in 0..NB {
            // lanes whose key is greater than key j gain one
            acc[m] = _mm256_sub_epi64(acc[m], _mm256_cmpgt_epi64(blk[m], b));
        }
    }
    let mut r = [0u64; 32];
    for m in 0..NB {
        _mm256_storeu_si256((r.as_mut_ptr() as *mut __m256i).add(m), acc[m]);
    }
    for i in 0..4 * NB {
        *out.get_unchecked_mut(*r.get_unchecked(i) as usize) = *val.get_unchecked(i);
    }
}

/// `rank_sort` for clicks: 2..=4*NB clicks into `out` in (time, position) order.
#[inline(always)]
unsafe fn rank_sort_ck<const NB: usize>(v: &[CKey], out: &mut [CKey; 33]) {
    let n = v.len();
    debug_assert!(n >= 1 && n <= 4 * NB);
    let mut k = [0i64; 32];
    for i in 0..4 * NB {
        let x = v.get_unchecked(i.min(n - 1)).key;
        *k.get_unchecked_mut(i) = if i < n { (((x >> 32) << 5) | i as u64) as i64 } else { i64::MAX };
    }
    let kp = k.as_ptr() as *const __m256i;
    let mut blk = [_mm256_setzero_si256(); NB];
    let mut acc = [_mm256_setzero_si256(); NB];
    for m in 0..NB {
        blk[m] = _mm256_loadu_si256(kp.add(m));
    }
    for j in 0..4 * NB {
        let b = _mm256_set1_epi64x(*k.get_unchecked(j));
        for m in 0..NB {
            acc[m] = _mm256_sub_epi64(acc[m], _mm256_cmpgt_epi64(blk[m], b));
        }
    }
    let mut r = [0u64; 32];
    for m in 0..NB {
        _mm256_storeu_si256((r.as_mut_ptr() as *mut __m256i).add(m), acc[m]);
    }
    for i in 0..4 * NB {
        *out.get_unchecked_mut(*r.get_unchecked(i) as usize) = *v.get_unchecked(i.min(n - 1));
    }
}

#[inline(never)]
fn radix_hi32<T: Copy>(v: &mut [T], tmp: &mut Vec<T>, key: impl Fn(&T) -> u64) {
    let n = v.len();
    tmp.clear();
    tmp.reserve(n);
    unsafe { tmp.set_len(n) };
    let mut h = vec![[0u32; 2048]; 3];
    for x in v.iter() {
        let k = (key(x) >> 32) as u32;
        h[0][(k & 2047) as usize] += 1;
        h[1][((k >> 11) & 2047) as usize] += 1;
        h[2][(k >> 22) as usize] += 1;
    }
    for d in 0..3 {
        let mut acc = 0u32;
        for c in h[d].iter_mut() {
            let x = *c;
            *c = acc;
            acc += x;
        }
    }
    unsafe {
        let (a, b) = (v.as_mut_ptr(), tmp.as_mut_ptr());
        for (d, (src, dst)) in [(a, b), (b, a), (a, b)].into_iter().enumerate() {
            let hd = &mut h[d];
            let sh = 11 * d as u32;
            for i in 0..n {
                let x = *src.add(i);
                let k = (((key(&x) >> 32) as u32) >> sh) & 2047;
                let at = hd.get_unchecked_mut(k as usize);
                *dst.add(*at as usize) = x;
                *at += 1;
            }
        }
        std::ptr::copy_nonoverlapping(b, a, n);
    }
}

static TM: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
unsafe fn process_person_bucket(ctx: &Ctx, cat_names: &[Vec<u8>], outs: &[ThreadOut], b: usize, w: &mut PersonWork, ud: &mut UserDaily) {
    let tq = Instant::now();
    let range = 1usize << ctx.pshift;
    let lo_pid = (b << ctx.pshift) as u32;
    w.cnt.clear();
    w.cnt.resize(range + 1, 0);
    // count per person; dedup losers (pid INVALID) go to the extra slot `range`, after
    // everyone, so neither pass branches on them
    let r32 = range as u32;
    let mut total = 0usize;
    for o in outs {
        for part in o.pe.parts(b) {
            total += part.len();
            for r in part.iter() {
                *w.cnt.get_unchecked_mut(r.pid.wrapping_sub(lo_pid).min(r32) as usize) += 1;
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
        for part in o.pe.parts(b) {
            for r in part.iter() {
                let p = r.pid.wrapping_sub(lo_pid).min(r32) as usize;
                let at = *w.cnt.get_unchecked(p);
                *w.cnt.get_unchecked_mut(p) = at + 1;
                *w.ev.get_unchecked_mut(at as usize) = ((r.ts as u64) << 32) | r.pc as u64;
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
        for part in o.pc.parts(b) {
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
        for part in o.pc.parts(b) {
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
        // persons with events (cnt changes), listed without a branch per slot
        w.plist.reserve(range);
        let pl = w.plist.as_mut_ptr();
        let mut np = 0usize;
        let mut prev = 0u32;
        for p in 0..range {
            let e = *w.cnt.get_unchecked(p);
            *pl.add(np) = p as u32;
            np += (e != prev) as usize;
            prev = e;
        }
        w.plist.set_len(np);
    }
    let np = w.plist.len();
    for ii in 0..np {
        // two-stage prefetch: the person's prefs entry, then (once cached) the name bytes
        if ii + 16 < np {
            _mm_prefetch(prefs.as_ptr().add(lo_pid as usize + *w.plist.get_unchecked(ii + 16) as usize) as *const i8, _MM_HINT_T0);
        }
        if ii + 6 < np {
            let e = prefs.get_unchecked(lo_pid as usize + *w.plist.get_unchecked(ii + 6) as usize);
            if e.is_long() {
                _mm_prefetch(e.ptr() as *const i8, _MM_HINT_T0);
            }
        }
        let p = *w.plist.get_unchecked(ii) as usize;
        let es = if p == 0 { 0 } else { *w.cnt.get_unchecked(p - 1) as usize };
        let ee = *w.cnt.get_unchecked(p) as usize;
        let cs = if p == 0 { 0 } else { *w.ccnt.get_unchecked(p - 1) as usize };
        let ce = *w.ccnt.get_unchecked(p) as usize;
        let evs: &[u64] = sort_events(w.ev.get_unchecked_mut(es..ee), &mut w.tmp_ev, &mut w.small);
        let ncl = ce - cs;
        if ncl > 1 {
            if ncl <= 32 {
                // branch-free, like the events (a comparison sort mispredicts on these)
                let v = w.ck.get_unchecked_mut(cs..ce);
                if ncl <= 8 {
                    rank_sort_ck::<2>(v, &mut w.small_ck);
                } else if ncl <= 16 {
                    rank_sort_ck::<4>(v, &mut w.small_ck);
                } else {
                    rank_sort_ck::<8>(v, &mut w.small_ck);
                }
                v.copy_from_slice(&w.small_ck[..ncl]);
            } else {
                sort_small(w.ck.get_unchecked_mut(cs..ce), &mut w.tmp_ck, |c| c.key);
            }
        }

        let clicks = w.ck.get_unchecked(cs..ce);
        let nc = clicks.len();
        let name = prefs.get_unchecked(lo_pid as usize + p).get();
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
        // assisted sessions are counted lazily: an active campaign c owes `sc - since[c]`
        // sessions to the current day, paid when it leaves the window, the day changes or the
        // person ends (so a session costs nothing per active campaign)
        let mut sc = 0u32;
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
                    let od = (cur_day - ctx.first_day) as usize;
                    for &c in w.active.iter() {
                        let c = c as usize;
                        *w.assisted.get_unchecked_mut(c * ndays + od) += sc - *w.since.get_unchecked(c);
                        *w.since.get_unchecked_mut(c) = sc;
                    }
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
                        *w.since.get_unchecked_mut(c) = sc;
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
                        *w.assisted.get_unchecked_mut(c * ndays + dayidx) += sc - *w.since.get_unchecked(c);
                        let pos = *w.wpos.get_unchecked(c) as usize;
                        let last = w.active.pop().unwrap_unchecked();
                        if last as usize != c {
                            *w.active.get_unchecked_mut(pos) = last;
                            *w.wpos.get_unchecked_mut(last as usize) = pos as u32;
                        }
                    }
                    lo += 1;
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
            cc[1] += (rev > 0) as i64;
            cc[2] += rev;
            sc += 1;
            k = m;
        }
        let ld = (cur_day - ctx.first_day) as usize;
        for &c in w.active.iter() {
            let c = c as usize;
            *w.assisted.get_unchecked_mut(c * ndays + ld) += sc - *w.since.get_unchecked(c);
            *w.wcnt.get_unchecked_mut(c) = 0;
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
            // first maximum, without a branch per category
            let m = 0u64.wrapping_sub((*v > bv) as u64);
            bv = (bv & !m) | (*v & m);
            best = (best & !(m as usize)) | (c & m as usize);
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
            let st = fs::read_to_string("/proc/self/status").unwrap_or_default();
            let g = |k: &str| st.lines().find(|l| l.starts_with(k)).map(|l| l.split_whitespace().nth(1).unwrap_or("?").to_string()).unwrap_or_default();
            eprintln!("{what:>12} {:8.1} ms  anon {} file {} kB", t0.elapsed().as_secs_f64() * 1000.0, g("RssAnon"), g("RssFile"));
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
        let tl = Instant::now();
        let ids = Ids::new(est_ids * 3 / 2 * grow);
        build_links(&ids, &link_chunks, nthreads);
        if verbose {
            eprintln!("  build_links {:.1} ms ({} slots)", tl.elapsed().as_secs_f64() * 1e3, ids.mask + 1);
        }
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
        // arenas sized for twice a thread's share; a busier thread takes another region
        let per_t = est_rows * 2 / nthreads;
        let per_tc = est_ck * 2 / nthreads;
        let next = AtomicUsize::new(0);
        let mut outs: Vec<ThreadOut> = Vec::with_capacity(nthreads);
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for t in 0..nthreads {
                let ctx = &ctx;
                let jobs = &jobs;
                let next = &next;
                handles.push(s.spawn(move || {
                    let mut pe = Buckets::new(npb, 1024, per_t);
                    pe.tag = (t as u32) << 27;
                    let mut out = ThreadOut {
                        dk: Buckets::new(NEB, 256, per_t),
                        pe,
                        pc: Buckets::new(npb, 256, per_tc),
                        tid: t as u32,
                        pq: [[Pending::default(); BATCH]; 4],
                        pn: [0; 4],
                        pt: 0,
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
        if verbose {
            let mb = |x: usize| x as f64 / 1048576.0;
            let (mut u, mut u2, mut u3, mut c) = (0, 0, 0, 0);
            for o in outs.iter() {
                for b in 0..o.dk.cur.len() { u += o.dk.len(b) * 16; c += o.dk.chunks[b].len() * o.dk.ch * 16; }
                for b in 0..o.pe.cur.len() { u2 += o.pe.len(b) * 12; c += o.pe.chunks[b].len() * o.pe.ch * 12; }
                for b in 0..o.pc.cur.len() { u3 += o.pc.len(b) * 24; c += o.pc.chunks[b].len() * o.pc.ch * 24; }
            }
            eprintln!("  mem dk {:.0} pe {:.0} pc {:.0} MB used, {:.0} MB in chunks, ids {:.0} MB, rows {:?}", mb(u), mb(u2), mb(u3), mb(c), mb(ctx.ids.mask as usize * 16), outs.iter().map(|o| (0..NEB).map(|b| o.dk.len(b)).sum::<usize>()).collect::<Vec<_>>());
        }
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

        // ---- phase 2: dedup event copies (smallest ingest_ts wins); a losing copy's PRec gets pid INVALID
        {
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..nthreads {
                    let outs = &outs;
                    let next = &next;
                    s.spawn(move || {
                        let mut tab: Vec<[u64; 2]> = Vec::new(); // (ehash, ingest << 32 | pref)
                        let mut used: Vec<u32> = Vec::new();
                        let mut filt: Vec<u64> = Vec::new();
                        let mut multi: Vec<u64> = Vec::new();
                        let mut cands: Vec<DKey> = Vec::new();
                        // each PRec has one DKey, so no two threads write the same record
                        let mark = |pref: u32| {
                            if pref != NONE {
                                unsafe {
                                    let p = outs.get_unchecked((pref >> 27) as usize).pe.at(pref & ((1 << 27) - 1));
                                    std::ptr::write_volatile(std::ptr::addr_of_mut!((*p).pid), INVALID);
                                }
                            }
                        };
                        loop {
                            let b = next.fetch_add(1, Relaxed);
                            if b >= NEB {
                                break;
                            }
                            let total: usize = outs.iter().map(|o| o.dk.len(b)).sum();
                            if total == 0 {
                                continue;
                            }
                            // two filter passes (no data-dependent branches) keep only the keys whose
                            // filter bit another key also set: every copy of a duplicated event, plus
                            // a few percent of false hits. Only those go through the hash table.
                            let fb = (total * 16).next_power_of_two().max(1024);
                            let fmask = (fb - 1) as u64;
                            filt.clear();
                            filt.resize(fb / 64, 0);
                            multi.clear();
                            multi.resize(fb / 64, 0);
                            let fp = filt.as_mut_ptr();
                            let mp = multi.as_mut_ptr();
                            for o in outs.iter() {
                                for part in o.dk.parts(b) {
                                    for r in part.iter() {
                                        let i = r.ehash & fmask;
                                        let w = (i >> 6) as usize;
                                        let bit = 1u64 << (i & 63);
                                        unsafe {
                                            let f = *fp.add(w);
                                            *mp.add(w) |= f & bit;
                                            *fp.add(w) = f | bit;
                                        }
                                    }
                                }
                            }
                            cands.clear();
                            cands.reserve(total);
                            let cp = cands.as_mut_ptr();
                            let mut nc = 0usize;
                            for o in outs.iter() {
                                for part in o.dk.parts(b) {
                                    for r in part.iter() {
                                        let i = r.ehash & fmask;
                                        unsafe {
                                            let hit = (*mp.add((i >> 6) as usize) >> (i & 63)) & 1;
                                            *cp.add(nc) = *r;
                                            nc += hit as usize;
                                        }
                                    }
                                }
                            }
                            unsafe {
                                cands.set_len(nc);
                            }
                            let cap = (nc * 2).next_power_of_two().max(16);
                            let mask = cap - 1;
                            if tab.len() < cap {
                                // grow (fresh zeroed memory); otherwise reuse: only used slots are dirty
                                tab = vec![[0; 2]; cap];
                            }
                            used.clear();
                            for r in cands.iter() {
                                let v = ((r.ingest as i64 - i32::MIN as i64) as u64) << 32 | r.pref as u64;
                                let mut p = (r.ehash as usize) & mask;
                                loop {
                                    let e = unsafe { tab.get_unchecked_mut(p) };
                                    let k = e[0];
                                    if k == 0 {
                                        *e = [r.ehash, v];
                                        used.push(p as u32);
                                        break;
                                    }
                                    if k == r.ehash {
                                        let old = e[1];
                                        if (v >> 32) < (old >> 32) {
                                            e[1] = v;
                                            mark(old as u32);
                                        } else {
                                            mark(v as u32);
                                        }
                                        break;
                                    }
                                    p = (p + 1) & mask;
                                }
                            }
                            for &p in used.iter() {
                                unsafe {
                                    tab.get_unchecked_mut(p as usize)[0] = 0;
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
                let order = &order;
                let ctx = &ctx;
                let cat_names = &cat_names;
                handles.push(s.spawn(move || {
                    let mut w = PersonWork {
                        small: [0; 33],
                        small_ck: [CKey { key: 0, kref: 0 }; 33],
                        cat_sum: vec![0; ctx.ncat],
                        wcnt: vec![0; ctx.ncamp + 1],
                        wpos: vec![0; ctx.ncamp + 1],
                        since: vec![0; ctx.ncamp + 1],
                        active: Vec::with_capacity(ctx.ncamp + 1),
                        plist: Vec::new(),
                        credited: vec![[0; 3]; (ctx.ncamp + 1) * ctx.ndays],
                        assisted: vec![0; (ctx.ncamp + 1) * ctx.ndays],
                        cnt: Vec::new(),
                        ccnt: Vec::new(),
                        ev: Vec::new(),
                        ck: Vec::new(),
                        tmp_ev: Vec::new(),
                        tmp_ck: Vec::new(),
                    };
                    let mut ud = UserDaily::new(est_rows / 9 / nthreads + 1024);
                    loop {
                        let j = next.fetch_add(1, Relaxed);
                        if j >= npb {
                            break;
                        }
                        let tb = Instant::now();
                        unsafe {
                            process_person_bucket(ctx, cat_names, outs, order[j], &mut w, &mut ud);
                        }
                        let dt = tb.elapsed().as_micros() as u64;
                        if dt > BMAX.load(Relaxed) {
                            BMAX.store(dt, Relaxed);
                            BMAXJ.store(j as u64, Relaxed);
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
        if verbose {
            let j = BMAXJ.load(Relaxed) as usize;
            eprintln!("  max bucket {} us (rank {j}, {} events); top sizes {:?}", BMAX.load(Relaxed), sizes[order[j]], (0..6).map(|k| sizes[order[k]]).collect::<Vec<_>>());
        }
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
        if verbose {
            let st = fs::read_to_string("/proc/self/stat").unwrap_or_default();
            let f: Vec<&str> = st.rsplit(')').next().unwrap_or("").split_whitespace().collect();
            eprintln!("  minflt {} majflt {}", f.get(7).unwrap_or(&"?"), f.get(9).unwrap_or(&"?"));
        }
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
    fn name_roundtrip() {
        let mut buf = vec![0u8; 256];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = b'a' + (i % 26) as u8;
        }
        let buf: &'static [u8] = Box::leak(buf.into_boxed_slice());
        let names: &'static [Name] = zeroed::<Name>(64);
        for n in 1..=48usize {
            for off in 0..8 {
                let e = &names[n];
                e.set(mkref(unsafe { buf.as_ptr().add(off) }, n));
                assert_eq!(e.get(), &buf[off..off + n], "n={n} off={off}");
                assert_eq!(e.is_long(), n > 15);
            }
        }
    }

    #[test]
    fn name_key7_orders_like_bytes() {
        let mut r = Rng(0x5eed_1234);
        let alpha = [0u8, b'0', b'9', b'a', b'u', b'z', 0xFF];
        let mut buf: Vec<u8> = Vec::new();
        let mut spans = Vec::new();
        for _ in 0..400 {
            let n = 1 + (r.next() % 40) as usize;
            let s = buf.len();
            for _ in 0..n {
                buf.push(alpha[(r.next() % alpha.len() as u64) as usize]);
            }
            spans.push((s, n));
        }
        buf.extend_from_slice(&[0u8; 64]);
        let buf: &'static [u8] = Box::leak(buf.into_boxed_slice());
        let names: &'static [Name] = zeroed::<Name>(spans.len());
        for (i, &(s, n)) in spans.iter().enumerate() {
            names[i].set(mkref(unsafe { buf.as_ptr().add(s) }, n));
        }
        for i in 0..spans.len() {
            for j in 0..spans.len() {
                let (a, b) = (&names[i], &names[j]);
                let (ka, kb) = (a.key7_sync(), b.key7_sync());
                let want = a.get() > b.get();
                if ka != kb {
                    assert_eq!(ka > kb, want, "{:?} {:?}", a.get(), b.get());
                }
            }
        }
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
            let mut v = v;
            if it % 7 == 3 {
                // digit strings and stray bytes of the ISO lengths
                let k = (r.next() % v.len() as u64) as usize;
                v[k] = if r.below(2) == 0 { b'0' + r.below(10) as u8 } else { (r.next() & 0xFF) as u8 };
                if r.below(3) == 0 {
                    let z = r.below(v.len() as u64) as usize;
                    for (i, b) in v.iter_mut().enumerate() {
                        *b = if i < z { b'0' } else { b'0' + r.below(10) as u8 };
                    }
                }
                buf[..v.len()].copy_from_slice(&v);
            }
            let want = parse_ts(&v).unwrap_or(BAD_TS);
            let got = unsafe { parse_ts_fast(buf.as_ptr(), v.len()) };
            let old = unsafe { if v.len() == 20 { parse_f12::<false>(buf.as_ptr()) } else { parse_f12::<true>(buf.as_ptr()) } };
            assert_eq!(got, want, "input {:?}", String::from_utf8_lossy(&v));
            if v[4] == b'-' {
                assert_eq!(old, want, "old input {:?}", String::from_utf8_lossy(&v));
            }
            if v.len() == 20 {
                let zs = unsafe { parse_z_secs(buf.as_ptr()) };
                assert_eq!(zs, if want == BAD_TS { BAD_TS } else { want.div_euclid(1000) }, "z input {:?}", String::from_utf8_lossy(&v));
            }
            valid += (want != BAD_TS) as u64;
        }
        eprintln!("valid {valid}");
    }

    #[test]
    fn rank_sort_matches_sort() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let mut tmp = Vec::new();
        let mut small = [0u64; 33];
        for round in 0..200_000 {
            let n = 1 + (rng.next() % 40) as usize;
            let spread = if round % 3 == 0 { 4 } else { 1 << 31 };
            let mut v: Vec<u64> = (0..n).map(|_| ((rng.next() % spread) << 32) | (rng.next() & 0xFFFF_FFFF)).collect();
            let mut want: Vec<u64> = v.iter().map(|x| x >> 32).collect();
            want.sort();
            let mut ms: Vec<u64> = v.clone();
            ms.sort();
            let got = unsafe { sort_events(&mut v, &mut tmp, &mut small) }.to_vec();
            assert_eq!(got.iter().map(|x| x >> 32).collect::<Vec<_>>(), want, "n={n}");
            let mut g2 = got.clone();
            g2.sort();
            assert_eq!(g2, ms, "same multiset n={n}");
        }
    }

    #[test]
    fn sort_events_orders_times() {
        let mut r = Rng(0x5EED);
        let mut tmp = Vec::new();
        let mut small = [0u64; 33];
        for it in 0..4000u64 {
            let n = if it % 3 == 0 { 1 + r.below(40) as usize } else { 1 + r.below(2100) as usize };
            let spread = [4u64, 100, 1 << 20, 1 << 31][(it % 4) as usize];
            let v: Vec<u64> = (0..n).map(|_| (r.below(spread) << 32) | r.below(1 << 20)).collect();
            let mut w = v.clone();
            let got: Vec<u64> = unsafe { sort_events(&mut w, &mut tmp, &mut small).to_vec() };
            let mut want = v.clone();
            want.sort_unstable();
            let mut g2 = got.clone();
            g2.sort_unstable();
            assert_eq!(g2, want, "same multiset, n {n}");
            assert!(got.windows(2).all(|p| (p[0] >> 32) <= (p[1] >> 32)), "time order, n {n}");
        }
    }

    #[test]
    fn hash_quality() {
        let mut r = Rng(0xABCDEF12345);
        let mut seen = std::collections::HashMap::new();
        let mut pad = vec![0u8; 128];
        let alnum = b"0123456789abcdefghijklmnopqrstuvwxyz";
        for it in 0..3_000_000u64 {
            let n = match it % 4 { 0 => 9, 1 => 33, 2 => 8, _ => 1 + (r.next() % 80) as usize };
            let mut v: Vec<u8> = (0..n).map(|_| alnum[(r.next() % if n == 33 { 16 } else { 36 }) as usize]).collect();
            v[0] = b'u';
            for (k, b) in pad.iter_mut().enumerate() {
                *b = if k < n { v[k] } else { (r.next() & 0xFF) as u8 };
            }
            let h = unsafe { hash_ptr(pad.as_ptr(), n) };
            assert_eq!(h, hash_bytes(&v));
            assert_ne!(h, 0);
            if let Some(old) = seen.insert(h, v.clone()) {
                assert_eq!(old, v, "collision");
            }
        }
    }

    fn ref_fields(line: &[u8], n: usize) -> Option<Vec<Vec<u8>>> {
        let f: Vec<&[u8]> = line.split(|&c| c == b',').collect();
        if f.len() != n {
            return None;
        }
        let mut out = Vec::new();
        for x in f {
            let mut x = x;
            if x.contains(&b'"') {
                if x.len() < 2 || x[0] != b'"' || x[x.len() - 1] != b'"' || x[1..x.len() - 1].contains(&b'"') {
                    return None;
                }
                x = &x[1..x.len() - 1];
            }
            out.push(x.to_vec());
        }
        Some(out)
    }

    fn check_lines<const N: usize>(chunk: &[u8]) {
        let mut pad = chunk.to_vec();
        pad.extend_from_slice(&[0u8; 128]);
        let pad: &'static [u8] = Box::leak(pad.into_boxed_slice());
        let c = &pad[..chunk.len()];
        let mut want = Vec::new();
        let mut ls: Vec<&[u8]> = chunk.split(|&b| b == b'\n').collect();
        if chunk.ends_with(b"\n") || chunk.is_empty() {
            ls.pop();
        }
        for l in ls {
            if let Some(f) = ref_fields(l, N) {
                want.push(f);
            }
        }
        let mut got = Vec::new();
        let mut flat = Flat::new();
        unsafe {
            for_lines::<N>(&mut flat, c, |fp, fl| {
                got.push((0..N).map(|i| sl(fp[i], fl[i]).to_vec()).collect::<Vec<_>>());
            });
        }
        assert_eq!(got.len(), want.len(), "chunk {:?}", String::from_utf8_lossy(chunk));
        assert!(got == want, "chunk {:?}", String::from_utf8_lossy(chunk));
    }

    #[test]
    fn fuzz_lines() {
        let mut r = Rng(0x1234_5678_9ABC_DEF1);
        let alpha = b"ab\",,,\"\"x9\n";
        for it in 0..60_000u64 {
            let mut chunk = Vec::new();
            let nl = 1 + r.next() % 40;
            for _ in 0..nl {
                let nf = if r.next() % 4 == 0 { 1 + r.next() % 9 } else if it & 1 == 0 { 7 } else { 4 };
                for f in 0..nf {
                    if f > 0 {
                        chunk.push(b',');
                    }
                    let kind = r.next() % 6;
                    let len = match r.next() % 10 { 0 => 0, 1 => 1, 2 => 2, 9 => (r.next() % 300) as usize, _ => (r.next() % 12) as usize };
                    let mut body: Vec<u8> = (0..len).map(|_| if kind == 0 { alpha[(r.next() % alpha.len() as u64) as usize] } else { b'a' + (r.next() % 26) as u8 }).collect();
                    if kind == 0 { body.retain(|&b| b != b'\n' || r.next() % 8 == 0); }
                    if kind >= 3 && kind < 5 {
                        chunk.push(b'"');
                        chunk.extend_from_slice(&body);
                        chunk.push(b'"');
                    } else {
                        chunk.extend_from_slice(&body);
                    }
                }
                chunk.push(b'\n');
            }
            if r.next() % 3 == 0 {
                chunk.pop();
            }
            if it & 1 == 0 {
                check_lines::<7>(&chunk);
            } else {
                check_lines::<4>(&chunk);
            }
        }
    }

}
