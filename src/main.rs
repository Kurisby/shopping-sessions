// Shopping Sessions pipeline: native, multithreaded, std only.
// Reads $DATA_DIR, writes Parquet to $OUTPUT_DIR/user_daily and $OUTPUT_DIR/campaign_daily.

use std::arch::x86_64::*;
use std::fs::{self, File};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::*};
use std::time::Instant;

mod parquet;

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
const MAP_POPULATE: i32 = 0x8000;
const MADV_HUGEPAGE: i32 = 14;
const MADV_WILLNEED: i32 = 3;

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
        let p = mmap(base, len, PROT_READ, MAP_PRIVATE | MAP_FIXED | MAP_POPULATE, f.as_raw_fd(), 0);
        if p as isize == -1 || p != base {
            panic!("mmap failed for {}", path.display());
        }
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
        madvise(p, total, MADV_HUGEPAGE);
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

#[inline(always)]
fn fmix(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

const HK0: u64 = 0x243F6A8885A308D3;
const HK1: u64 = 0x9E3779B97F4A7C15;
const HK2: u64 = 0xD6E8FEB86659FD93;

/// Injective for strings of equal length (every step is a bijection); random-like across lengths.
/// Reads up to 7 bytes past the end: only for padded memory.
#[inline(always)]
unsafe fn hash_ptr(p: *const u8, n: usize) -> u64 {
    let mut h = HK0 ^ (n as u64).wrapping_mul(HK1);
    let mut i = 0;
    while i + 8 <= n {
        let w = std::ptr::read_unaligned(p.add(i) as *const u64);
        h = (h ^ w).wrapping_mul(HK1).rotate_left(31);
        i += 8;
    }
    if i < n {
        let r = n - i;
        let w = std::ptr::read_unaligned(p.add(i) as *const u64) & (u64::MAX >> (64 - 8 * r));
        h = (h ^ w).wrapping_mul(HK2).rotate_left(29);
    }
    let h = fmix(h);
    if h == 0 {
        1
    } else {
        h
    }
}

/// Same value as hash_ptr, safe on any slice.
fn hash_bytes(b: &[u8]) -> u64 {
    let n = b.len();
    let mut h = HK0 ^ (n as u64).wrapping_mul(HK1);
    let mut i = 0;
    while i + 8 <= n {
        let w = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        h = (h ^ w).wrapping_mul(HK1).rotate_left(31);
        i += 8;
    }
    if i < n {
        let mut w = 0u64;
        let mut k = 0;
        while i < n {
            w |= (b[i] as u64) << k;
            k += 8;
            i += 1;
        }
        h = (h ^ w).wrapping_mul(HK2).rotate_left(29);
    }
    let h = fmix(h);
    if h == 0 {
        1
    } else {
        h
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

struct Scanner {
    p: *const u8, // current 64-byte block
    end: *const u8,
    sep: u64,
    qm: u64,
    qcum: u32, // quotes before block p
}

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

impl Scanner {
    #[inline(always)]
    unsafe fn new(start: *const u8, end: *const u8) -> Self {
        let mut s = Scanner { p: start, end, sep: 0, qm: 0, qcum: 0 };
        s.load();
        s
    }
    #[inline(always)]
    unsafe fn load(&mut self) {
        let (mut s, mut q) = block_masks(self.p);
        let rem = self.end.offset_from(self.p);
        if rem < 64 {
            let m = if rem <= 0 { 0 } else { (1u64 << rem) - 1 };
            s &= m;
            q &= m;
        }
        self.sep = s;
        self.qm = q;
    }
    /// Next ',' or '\n' position, or `end`.
    #[inline(always)]
    unsafe fn next_sep(&mut self) -> *const u8 {
        while self.sep == 0 {
            if self.p >= self.end {
                return self.end;
            }
            self.qcum += self.qm.count_ones();
            self.p = self.p.add(64);
            if self.p >= self.end {
                self.qm = 0;
                return self.end;
            }
            self.load();
        }
        let tz = self.sep.trailing_zeros();
        self.sep &= self.sep - 1;
        self.p.add(tz as usize)
    }
    /// Quotes strictly before `pos`; pos must be in the current block or at/after its start.
    #[inline(always)]
    unsafe fn quotes_before(&self, pos: *const u8) -> u32 {
        let off = pos.offset_from(self.p);
        if off <= 0 {
            self.qcum
        } else if off >= 64 {
            self.qcum + self.qm.count_ones()
        } else {
            self.qcum + (self.qm & ((1u64 << off) - 1)).count_ones()
        }
    }
}

/// Splits the next line into N fields. Returns (ok, next line start).
/// Fields are (ptr, len) with whole-field quotes stripped.
#[inline(always)]
unsafe fn next_line<const N: usize>(sc: &mut Scanner, ls: *const u8, qs: &mut u32, fp: &mut [*const u8; N], fl: &mut [usize; N]) -> (bool, *const u8) {
    let end = sc.end;
    let mut seps: [*const u8; N] = [std::ptr::null(); N];
    for i in 0..N {
        seps[i] = sc.next_sep();
    }
    let mut good = true;
    for i in 0..N - 1 {
        good &= (seps[i] < end) & (*seps[i] == b',');
    }
    let last = seps[N - 1];
    good &= (last >= end) | (*last == b'\n');
    if !good {
        // malformed: find where this line really ends, then restart the scanner there
        let mut le = std::ptr::null();
        for i in 0..N {
            if seps[i] >= end || *seps[i] == b'\n' {
                le = seps[i];
                break;
            }
        }
        if le.is_null() {
            let mut p = last;
            while p < end && *p != b'\n' {
                p = p.add(1);
            }
            le = p;
        }
        let nls = if le >= end { end } else { le.add(1) };
        *sc = Scanner::new(nls, end);
        *qs = 0;
        return (false, nls);
    }
    fp[0] = ls;
    fl[0] = seps[0].offset_from(ls) as usize;
    for i in 1..N {
        fp[i] = seps[i - 1].add(1);
        fl[i] = seps[i].offset_from(fp[i]) as usize;
    }
    let qe = sc.quotes_before(last);
    let q = qe - *qs;
    *qs = qe;
    let nls = last.add(1);
    if q != 0 {
        let mut fq = 0u32;
        let mut okq = true;
        for i in 0..N {
            let p = fp[i];
            let n = fl[i];
            let first_q = (n > 0) & (*p == b'"');
            let last_q = (n >= 2) & (*p.add(n.saturating_sub(1)) == b'"');
            okq &= !first_q | last_q;
            fq += first_q as u32;
            fp[i] = p.add(first_q as usize);
            fl[i] = n.wrapping_sub(2 * first_q as usize);
        }
        if !okq || q != 2 * fq {
            return (false, nls);
        }
    }
    (true, nls)
}

#[inline(always)]
unsafe fn sl(p: *const u8, n: usize) -> &'static [u8] {
    std::slice::from_raw_parts(p, n)
}

// ---------------------------------------------------------------- lookups

struct Lookup {
    prefix: Vec<u8>,
    direct: Vec<u32>,
    other: U64Map<u32>,
    has_other: bool,
}

const ABSENT: u32 = u32::MAX;

impl Lookup {
    fn new(prefix: &[u8]) -> Self {
        Lookup { prefix: prefix.to_vec(), direct: Vec::new(), other: U64Map::with_capacity(16), has_other: false }
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
    fn get(&self, k: &[u8]) -> u32 {
        if let Some(x) = self.canonical(k) {
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

fn read_small_csv(path: &Path) -> Vec<(Vec<u8>, Vec<u8>)> {
    let data = read_padded(path);
    let mut rows = Vec::new();
    let first = memchr_nl(data, 0);
    if first >= data.len() {
        return rows;
    }
    for_each_line(&data[first + 1..], |line| {
        let mut f: [&'static [u8]; 2] = [&[]; 2];
        if split_line(line, &mut f) && !f[0].is_empty() {
            rows.push((f[0].to_vec(), f[1].to_vec()));
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
            if probes > 2048 {
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
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..nthreads)
            .map(|_| {
                let next = &next;
                s.spawn(move || {
                    let mut mine: Vec<u32> = Vec::new();
                    loop {
                        let j = next.fetch_add(1, Relaxed);
                        if j >= link_chunks.len() {
                            break;
                        }
                        for_each_line(link_chunks[j], |line| {
                            let mut f: [&'static [u8]; 2] = [&[]; 2];
                            if !split_line(line, &mut f) || f[0].is_empty() || f[1].is_empty() {
                                return;
                            }
                            let (a, ia) = ids.intern(hash_bytes(f[0]), mkref(f[0].as_ptr(), f[0].len()));
                            let (b, ib) = ids.intern(hash_bytes(f[1]), mkref(f[1].as_ptr(), f[1].len()));
                            if ia {
                                mine.push(a);
                            }
                            if ib {
                                mine.push(b);
                            }
                            if a != b {
                                uf_union(parent, a, b);
                            }
                        });
                    }
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
    std::thread::scope(|s| {
        for v in &inserted {
            s.spawn(move || {
                for &i in v {
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
                for &i in v {
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
    lens: Vec<u32>,
    spill: Vec<Vec<T>>,
}
unsafe impl<T: Copy + 'static> Send for Buckets<T> {}
unsafe impl<T: Copy + 'static> Sync for Buckets<T> {}

impl<T: Copy + 'static> Buckets<T> {
    fn new(n: usize, cap: usize) -> Self {
        let mem = zeroed::<T>(n * cap);
        Buckets { base: mem.as_mut_ptr(), cap, lens: vec![0; n], spill: (0..n).map(|_| Vec::new()).collect() }
    }
    #[inline(always)]
    fn push(&mut self, b: usize, v: T) {
        unsafe {
            let l = *self.lens.get_unchecked(b) as usize;
            if l < self.cap {
                *self.base.add(b * self.cap + l) = v;
                *self.lens.get_unchecked_mut(b) = (l + 1) as u32;
            } else {
                self.spill.get_unchecked_mut(b).push(v);
            }
        }
    }
    #[inline(always)]
    fn main(&self, b: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.base.add(b * self.cap), self.lens[b] as usize) }
    }
    #[inline(always)]
    fn spilled(&self, b: usize) -> &mut [T] {
        let v = &self.spill[b];
        unsafe { std::slice::from_raw_parts_mut(v.as_ptr() as *mut T, v.len()) }
    }
    fn len(&self, b: usize) -> usize {
        self.lens[b] as usize + self.spill[b].len()
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
    let start = chunk.as_ptr();
    let end = start.add(chunk.len());
    let mut sc = Scanner::new(start, end);
    let mut ls = start;
    let mut qs = 0u32;
    let mut fp: [*const u8; 7] = [std::ptr::null(); 7];
    let mut fl: [usize; 7] = [0; 7];
    while ls < end {
        let (ok, nls) = next_line::<7>(&mut sc, ls, &mut qs, &mut fp, &mut fl);
        ls = nls;
        if !ok || fl[0] == 0 {
            continue;
        }
        let ehash = hash_ptr(fp[0], fl[0]);
        let ingest = parse_ts(sl(fp[3], fl[3]));
        let rowid = (out.tid << 27) | out.rows;
        out.rows += 1;
        let ingest_rel = match ingest {
            Some(t) => ((t - base).div_euclid(1000)).clamp(i32::MIN as i64, (i32::MAX - 1) as i64) as i32,
            None => i32::MAX,
        };
        out.dk.push(ebucket(ehash), DKey { ehash, ingest: ingest_rel, rowid });
        // validation
        if fl[1] == 0 || fl[5] == 0 {
            continue;
        }
        let ts = match parse_ts(sl(fp[2], fl[2])) {
            Some(t) if t >= ws && t < we => t,
            _ => continue,
        };
        let mut pc = 0u32;
        if fl[4] == 4 {
            let w = std::ptr::read_unaligned(fp[4] as *const u32);
            if w != u32::from_le_bytes(*b"view") && w != u32::from_le_bytes(*b"cart") {
                continue;
            }
        } else if fl[4] == 8 && std::ptr::read_unaligned(fp[4] as *const u64) == u64::from_le_bytes(*b"purchase") {
            let price = match parse_price(sl(fp[6], fl[6])) {
                Some(p) => p,
                None => continue,
            };
            let cat = match ctx.products.get(sl(fp[5], fl[5])) {
                ABSENT => ctx.unknown_cat,
                c => c,
            };
            pc = (price << 8) | cat;
        } else {
            continue;
        }
        match ingest {
            Some(i) if i - ts <= DAY => {}
            _ => continue,
        }
        let uhash = hash_ptr(fp[1], fl[1]);
        out.push_pending(Pending { uhash, uref: mkref(fp[1], fl[1]), a: (ts - base) as u32, b: pc, c: rowid }, ctx);
    }
    while out.rn > 0 {
        out.pop_event(ctx);
    }
}

unsafe fn parse_clicks(ctx: &Ctx, chunk: &'static [u8], out: &mut ThreadOut) {
    let lo = ctx.base;
    let we = ctx.we;
    let start = chunk.as_ptr();
    let end = start.add(chunk.len());
    let mut sc = Scanner::new(start, end);
    let mut ls = start;
    let mut qs = 0u32;
    let mut fp: [*const u8; 4] = [std::ptr::null(); 4];
    let mut fl: [usize; 4] = [0; 4];
    let mut ring = ClickRing { ring: [PendingClick::default(); RING], rn: 0, rh: 0 };
    while ls < end {
        let (ok, nls) = next_line::<4>(&mut sc, ls, &mut qs, &mut fp, &mut fl);
        ls = nls;
        if !ok || fl[0] == 0 || fl[1] == 0 {
            continue;
        }
        let camp = ctx.campaigns.get(sl(fp[2], fl[2]));
        if camp == ABSENT {
            continue;
        }
        let ts = match parse_ts(sl(fp[3], fl[3])) {
            Some(t) if t >= lo && t < we => t,
            _ => continue,
        };
        let uhash = hash_ptr(fp[1], fl[1]);
        ring.push(PendingClick { uhash, uref: mkref(fp[1], fl[1]), kref: mkref(fp[0], fl[0]), ts: (ts - lo) as u32, camp }, ctx, out);
    }
    while ring.rn > 0 {
        ring.pop(ctx, out);
    }
}

// ---------------------------------------------------------------- output buffers

struct UserDaily {
    n: usize,
    uid: Vec<u8>,
    day: Vec<u8>,
    sessions: Vec<u8>,
    events: Vec<u8>,
    revenue: Vec<u8>,
    top_def: Vec<u8>,
    top_val: Vec<u8>,
}

impl UserDaily {
    fn new() -> Self {
        UserDaily { n: 0, uid: Vec::new(), day: Vec::new(), sessions: Vec::new(), events: Vec::new(), revenue: Vec::new(), top_def: Vec::new(), top_val: Vec::new() }
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
    stamp: Vec<u32>,
    sid: u32,
    camp_counts: Vec<[i64; 4]>,
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

unsafe fn process_person_bucket(ctx: &Ctx, cat_names: &[Vec<u8>], outs: &[ThreadOut], losers: &[&'static [AtomicU64]], b: usize, w: &mut PersonWork, ud: &mut UserDaily) {
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

    let base = ctx.base;
    let ndays = ctx.ndays;
    let organic = ctx.ncamp;
    let mut es = 0usize;
    let mut cs = 0usize;
    for p in 0..range {
        let ee = w.cnt[p] as usize;
        let ce = w.ccnt[p] as usize;
        if ee == es {
            cs = ce;
            continue;
        }
        sort_small(&mut w.ev[es..ee], |&x| x);
        sort_small(&mut w.ck[cs..ce], |c| c.key);
        let evs = &w.ev[es..ee];
        let clicks = &w.ck[cs..ce];
        es = ee;
        cs = ce;
        let name = deref(ctx.ids.prefs[lo_pid as usize + p].load(Relaxed));
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
            let start = (evs[k] >> 32) as u32;
            let mut prev = start;
            let mut rev = 0i64;
            let mut m = k;
            let day = (base + start as i64).div_euclid(DAY);
            if day != cur_day {
                if cur_day != i64::MIN {
                    flush_day(ud, name, cur_day, d_sessions, d_events, d_rev, d_purch, &mut w.cat_sum, cat_names);
                }
                cur_day = day;
                d_sessions = 0;
                d_events = 0;
                d_rev = 0;
                d_purch = false;
            }
            while m < n {
                let r = evs[m];
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
            let start_key = ((start as u64) << 32) | 0xFFFF_FFFF;
            while hi < clicks.len() && clicks[hi].key <= start_key {
                hi += 1;
            }
            let low_bound = (start as i64 - WEEK).max(0) as u64;
            while lo < hi && (clicks[lo].key >> 32) < low_bound {
                lo += 1;
            }
            let dayidx = (day - ctx.first_day) as usize;
            let camp = if hi > lo {
                let mut bi = hi - 1;
                let t = clicks[bi].key >> 32;
                let mut x = bi;
                while x > lo && (clicks[x - 1].key >> 32) == t {
                    x -= 1;
                    if deref(clicks[x].kref) < deref(clicks[bi].kref) {
                        bi = x;
                    }
                }
                (clicks[bi].key as u32) as usize
            } else {
                organic
            };
            let cc = w.camp_counts.get_unchecked_mut(camp * ndays + dayidx);
            cc[0] += 1;
            if rev > 0 {
                cc[1] += 1;
                cc[2] += rev;
            }
            if hi > lo {
                w.sid = w.sid.wrapping_add(1);
                if w.sid == 0 {
                    w.stamp.iter_mut().for_each(|s| *s = 0);
                    w.sid = 1;
                }
                let sid = w.sid;
                for c in &clicks[lo..hi] {
                    let cp = (c.key as u32) as usize;
                    if *w.stamp.get_unchecked(cp) != sid {
                        *w.stamp.get_unchecked_mut(cp) = sid;
                        w.camp_counts.get_unchecked_mut(cp * ndays + dayidx)[3] += 1;
                    }
                }
            }
            k = m;
        }
        flush_day(ud, name, cur_day, d_sessions, d_events, d_rev, d_purch, &mut w.cat_sum, cat_names);
    }
}

#[inline]
fn flush_day(ud: &mut UserDaily, name: &[u8], day: i64, sessions: i64, events: i64, rev: i64, purch: bool, cat_sum: &mut [u64], cat_names: &[Vec<u8>]) {
    ud.n += 1;
    put_str(&mut ud.uid, name);
    ud.day.extend_from_slice(&(day as i32).to_le_bytes());
    ud.sessions.extend_from_slice(&sessions.to_le_bytes());
    ud.events.extend_from_slice(&events.to_le_bytes());
    ud.revenue.extend_from_slice(&rev.to_le_bytes());
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
        put_str(&mut ud.top_val, &cat_names[best]);
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

    // products / campaigns
    let prows = read_small_csv(&data_dir.join("products.csv"));
    let mut cat_names: Vec<Vec<u8>> = prows.iter().map(|r| r.1.clone()).collect();
    cat_names.push(b"unknown".to_vec());
    cat_names.sort();
    cat_names.dedup();
    assert!(cat_names.len() <= 255, "too many categories");
    let mut products = Lookup::new(b"p");
    for (pid, cat) in &prows {
        let c = cat_names.binary_search(cat).unwrap() as u32;
        products.insert(pid, c);
    }
    let unknown_cat = cat_names.binary_search(&b"unknown".to_vec()).unwrap() as u32;
    let crows = read_small_csv(&data_dir.join("campaigns.csv"));
    let mut campaigns = Lookup::new(b"c_");
    for (i, (cid, _)) in crows.iter().enumerate() {
        campaigns.insert(cid, i as u32);
    }
    let ncamp = crows.len();
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
        let est_ids = link_bytes / 10 + est_rows / 25 + 4096;
        let ids = Ids::new(est_ids * 2 * grow);
        build_links(&ids, &link_chunks, nthreads);
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
                            keys.clear();
                            keys.resize(cap, 0);
                            vals.resize(cap, 0);
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
                        stamp: vec![0; ctx.ncamp + 1],
                        sid: 0,
                        camp_counts: vec![[0; 4]; (ctx.ncamp + 1) * ctx.ndays],
                        cnt: Vec::new(),
                        ccnt: Vec::new(),
                        ev: Vec::new(),
                        ck: Vec::new(),
                    };
                    let mut ud = UserDaily::new();
                    loop {
                        let j = next.fetch_add(1, Relaxed);
                        if j >= npb {
                            break;
                        }
                        unsafe {
                            process_person_bucket(ctx, cat_names, outs, losers, order[j], &mut w, &mut ud);
                        }
                    }
                    (ud, w.camp_counts)
                }));
            }
            for h in handles {
                results.push(h.join().unwrap());
            }
        });
        lap("sessions");

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
        std::thread::scope(|s| {
            for (t, (ud, _)) in results.iter().enumerate() {
                let ud_dir = &ud_dir;
                s.spawn(move || {
                    if ud.n == 0 && t != 0 {
                        return;
                    }
                    let bytes = parquet::write_user_daily(ud.n, &ud.uid, &ud.day, &ud.sessions, &ud.events, &ud.revenue, &ud.top_def, &ud.top_val);
                    fs::write(ud_dir.join(format!("part-{t:03}.parquet")), bytes).unwrap();
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
            let bytes = parquet::write_campaign_daily(n, &cid, &ch, &day, &s0, &s1, &s2, &s3);
            fs::write(cd_dir.join("part-000.parquet"), bytes).unwrap();
        });
        lap("written");
        std::process::exit(0);
    }
}
