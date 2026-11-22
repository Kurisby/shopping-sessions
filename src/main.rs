// Shopping Sessions pipeline: one native, multithreaded pass over the CSVs.
// Reads $DATA_DIR, writes Parquet to $OUTPUT_DIR/user_daily and $OUTPUT_DIR/campaign_daily.

use std::fs::{self, File};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

mod parquet;

const DAY: i64 = 86_400_000;
const WEEK: i64 = 7 * DAY;
const GAP: i64 = 1_800_000;
const INVALID_TS: u32 = u32::MAX;

// ---------------------------------------------------------------- files

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
}

fn map_file(path: &Path) -> &'static [u8] {
    let f = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let len = f.metadata().unwrap().len() as usize;
    if len == 0 {
        return &[];
    }
    unsafe {
        let p = mmap(std::ptr::null_mut(), len, 1, 2, f.as_raw_fd(), 0);
        if p as isize == -1 {
            panic!("mmap failed for {}", path.display());
        }
        std::slice::from_raw_parts(p, len)
    }
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
fn mkref(s: &[u8]) -> u64 {
    (s.as_ptr() as u64) | ((s.len().min(0xFFFF) as u64) << 48)
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

/// Injective for strings of equal length (every step is a bijection); random-like across lengths.
#[inline(always)]
fn hash_bytes(b: &[u8]) -> u64 {
    let n = b.len();
    let mut h = 0x243F6A8885A308D3u64 ^ (n as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let mut i = 0;
    while i + 8 <= n {
        let w = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        h = (h ^ w).wrapping_mul(0x9E3779B97F4A7C15).rotate_left(31);
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
        h = (h ^ w).wrapping_mul(0xD6E8FEB86659FD93).rotate_left(29);
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
            let kk = self.keys[i];
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

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((m + 9) % 12) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[inline]
fn is_leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

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
#[inline]
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

#[inline]
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

// ---------------------------------------------------------------- CSV dialect

/// Splits one line (no newline) into exactly N fields, stripping whole-field quotes.
/// Returns false for a malformed line.
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

// ---------------------------------------------------------------- lookups

struct Lookup {
    // "<prefix><canonical decimal>" fast path
    prefix: Vec<u8>,
    direct: Vec<u32>, // u32::MAX = absent
    other: U64Map<u32>, // hash -> value+1 for non-canonical keys
    has_other: bool,
}

const ABSENT: u32 = u32::MAX;

impl Lookup {
    fn new(prefix: &[u8]) -> Self {
        Lookup { prefix: prefix.to_vec(), direct: Vec::new(), other: U64Map::with_capacity(16), has_other: false }
    }
    fn canonical(&self, k: &[u8]) -> Option<usize> {
        let p = self.prefix.len();
        if k.len() <= p || k.len() > p + 9 || &k[..p] != &self.prefix[..] {
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
    #[inline]
    fn get(&self, k: &[u8]) -> u32 {
        // fast canonical path
        let p = self.prefix.len();
        if k.len() > p && k.len() <= p + 9 && k[..p] == self.prefix[..] {
            let digits = &k[p..];
            if !(digits[0] == b'0' && digits.len() > 1) {
                let mut x = 0usize;
                let mut ok = true;
                for &c in digits {
                    let d = c.wrapping_sub(b'0');
                    if d > 9 {
                        ok = false;
                        break;
                    }
                    x = x * 10 + d as usize;
                }
                if ok {
                    return if x < self.direct.len() { self.direct[x] } else { ABSENT };
                }
            }
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
    let data: &'static [u8] = Box::leak(fs::read(path).unwrap_or_default().into_boxed_slice());
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

// ---------------------------------------------------------------- linked IDs

#[derive(Clone, Copy, Default)]
struct Person {
    phash: u64,
    pref: u64,
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let p = parent[x as usize];
        parent[x as usize] = parent[p as usize];
        x = p;
    }
    x
}

fn process_links(chunks: &[&'static [u8]]) -> U64Map<Person> {
    let mut ids: U64Map<u32> = U64Map::with_capacity(1 << 16);
    let mut names: Vec<u64> = Vec::new();
    let mut parent: Vec<u32> = Vec::new();
    let mut size: Vec<u32> = Vec::new();
    let mut intern = |h: u64, r: u64, ids: &mut U64Map<u32>, names: &mut Vec<u64>, parent: &mut Vec<u32>, size: &mut Vec<u32>| -> u32 {
        let (s, ins) = ids.slot(h);
        if ins {
            let id = names.len() as u32;
            ids.vals[s] = id;
            names.push(r);
            parent.push(id);
            size.push(1);
            id
        } else {
            ids.vals[s]
        }
    };
    for &chunk in chunks {
        for_each_line(chunk, |line| {
            let mut f: [&'static [u8]; 2] = [&[]; 2];
            if !split_line(line, &mut f) || f[0].is_empty() || f[1].is_empty() {
                return;
            }
            let a = intern(hash_bytes(f[0]), mkref(f[0]), &mut ids, &mut names, &mut parent, &mut size);
            let b = intern(hash_bytes(f[1]), mkref(f[1]), &mut ids, &mut names, &mut parent, &mut size);
            let ra = find(&mut parent, a);
            let rb = find(&mut parent, b);
            if ra != rb {
                let (big, small) = if size[ra as usize] >= size[rb as usize] { (ra, rb) } else { (rb, ra) };
                parent[small as usize] = big;
                size[big as usize] += size[small as usize];
            }
        });
    }
    let n = names.len();
    let mut best: Vec<u32> = (0..n as u32).collect();
    for i in 0..n as u32 {
        let r = find(&mut parent, i);
        if deref(names[i as usize]) < deref(names[best[r as usize] as usize]) {
            best[r as usize] = i;
        }
    }
    let mut map: U64Map<Person> = U64Map::with_capacity(n.max(16));
    for i in 0..n {
        let r = parent[i] as usize; // fully compressed by the loop above? ensure
        let r = if parent[r] as usize == r { r } else { find(&mut parent, i as u32) as usize };
        if size[r] < 2 {
            continue;
        }
        let b = best[r] as usize;
        let rr = names[b];
        let (s, _) = map.slot(hash_bytes(deref(names[i])));
        map.vals[s] = Person { phash: hash_bytes(deref(rr)), pref: rr };
    }
    map
}

// ---------------------------------------------------------------- records

#[derive(Clone, Copy)]
#[repr(C)]
struct EvRec {
    ehash: u64,
    uhash: u64,
    uref: u64,
    ingest: i64,
    ts: u32, // relative to base; INVALID_TS when the copy fails validation
    pc: u32, // price << 8 | category
}

#[derive(Clone, Copy)]
#[repr(C)]
struct PRec {
    phash: u64,
    pref: u64,
    ts: u32,
    pc: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct CRec {
    h: u64, // user hash, then person hash
    kref: u64,
    ts: u32,
    camp: u32,
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
}

const EB_BITS: u32 = 9;
const NEB: usize = 1 << EB_BITS;
const PB_BITS: u32 = 9;
const NPB: usize = 1 << PB_BITS;

#[inline(always)]
fn ebucket(h: u64) -> usize {
    (h >> (64 - EB_BITS)) as usize
}
#[inline(always)]
fn pbucket(h: u64) -> usize {
    (h >> (64 - PB_BITS)) as usize
}

fn parse_events(ctx: &Ctx, chunk: &'static [u8], out: &mut [Vec<EvRec>]) {
    let ws = ctx.ws;
    let we = ctx.we;
    let base = ctx.base;
    for_each_line(chunk, |line| {
        let mut f: [&'static [u8]; 7] = [&[]; 7];
        if !split_line(line, &mut f) {
            return;
        }
        let eid = f[0];
        if eid.is_empty() {
            return;
        }
        let ehash = hash_bytes(eid);
        let ingest = parse_ts(f[3]).unwrap_or(i64::MAX);
        let mut rec = EvRec { ehash, uhash: 0, uref: 0, ingest, ts: INVALID_TS, pc: 0 };
        'v: {
            let uid = f[1];
            if uid.is_empty() || f[5].is_empty() {
                break 'v;
            }
            let ts = match parse_ts(f[2]) {
                Some(t) if t >= ws && t < we => t,
                _ => break 'v,
            };
            let price = match f[4] {
                b"view" | b"cart" => 0,
                b"purchase" => match parse_price(f[6]) {
                    Some(p) => p,
                    None => break 'v,
                },
                _ => break 'v,
            };
            if ingest == i64::MAX || ingest - ts > DAY {
                break 'v;
            }
            let cat = match ctx.products.get(f[5]) {
                ABSENT => ctx.unknown_cat,
                c => c,
            };
            rec.uhash = hash_bytes(uid);
            rec.uref = mkref(uid);
            rec.ts = (ts - base) as u32;
            rec.pc = (price << 8) | cat;
        }
        out[ebucket(ehash)].push(rec);
    });
}

fn parse_clicks(ctx: &Ctx, chunk: &'static [u8], out: &mut Vec<CRec>) {
    let lo = ctx.base;
    let we = ctx.we;
    for_each_line(chunk, |line| {
        let mut f: [&'static [u8]; 4] = [&[]; 4];
        if !split_line(line, &mut f) {
            return;
        }
        if f[0].is_empty() || f[1].is_empty() {
            return;
        }
        let camp = ctx.campaigns.get(f[2]);
        if camp == ABSENT {
            return;
        }
        let ts = match parse_ts(f[3]) {
            Some(t) if t >= lo && t < we => t,
            _ => return,
        };
        out.push(CRec { h: hash_bytes(f[1]), kref: mkref(f[0]), ts: (ts - lo) as u32, camp });
    });
}

// ---------------------------------------------------------------- output buffers

struct UserDaily {
    n: usize,
    uid: Vec<u8>,
    day: Vec<u8>,
    sessions: Vec<u8>,
    events: Vec<u8>,
    revenue: Vec<u8>,
    top_def: Vec<u8>, // one byte per row: 1 present, 0 null
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

struct PersonWork {
    cat_sum: Vec<u64>,
    stamp: Vec<u32>,
    sid: u32,
    camp_counts: Vec<[i64; 4]>, // [sessions, purchasing, revenue, assisted] per camp*ndays+day
}

fn process_person_bucket(ctx: &Ctx, cat_names: &[Vec<u8>], evs: &mut Vec<PRec>, cks: &mut Vec<CRec>, w: &mut PersonWork, ud: &mut UserDaily) {
    evs.sort_unstable_by_key(|r| (r.phash, r.ts));
    cks.sort_unstable_by_key(|r| (r.h, r.ts));
    let base = ctx.base;
    let ndays = ctx.ndays;
    let organic = ctx.ncamp;
    let n = evs.len();
    let mut ci = 0usize;
    let mut i = 0usize;
    while i < n {
        let ph = evs[i].phash;
        let mut j = i;
        while j < n && evs[j].phash == ph {
            j += 1;
        }
        // clicks of this person
        while ci < cks.len() && cks[ci].h < ph {
            ci += 1;
        }
        let cs = ci;
        while ci < cks.len() && cks[ci].h == ph {
            ci += 1;
        }
        let clicks = &cks[cs..ci];
        let name = deref(evs[i].pref);
        // walk sessions
        let mut lo = 0usize; // first click with ts >= start - 7d
        let mut hi = 0usize; // first click with ts > start
        let mut cur_day: i64 = i64::MIN;
        let mut d_sessions = 0i64;
        let mut d_events = 0i64;
        let mut d_rev = 0i64;
        let mut d_purch = false;
        let mut k = i;
        while k < j {
            let start = evs[k].ts;
            let mut prev = start;
            let mut cnt = 0i64;
            let mut rev = 0i64;
            let mut m = k;
            let abs_start = base + start as i64;
            let day = abs_start.div_euclid(DAY);
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
            while m < j && (evs[m].ts - prev) as i64 <= GAP {
                let r = evs[m];
                prev = r.ts;
                cnt += 1;
                let price = (r.pc >> 8) as i64;
                if price > 0 {
                    rev += price;
                    w.cat_sum[(r.pc & 0xFF) as usize] += price as u64;
                    d_purch = true;
                }
                m += 1;
            }
            d_sessions += 1;
            d_events += cnt;
            d_rev += rev;
            // attribution
            let st = start as i64 + WEEK; // shift clicks: click.ts (rel base) compared in same base
            // clicks ts are relative to base too.
            while hi < clicks.len() && clicks[hi].ts <= start {
                hi += 1;
            }
            let low_bound = start as i64 - WEEK;
            while lo < hi && (clicks[lo].ts as i64) < low_bound {
                lo += 1;
            }
            let _ = st;
            let dayidx = (day - ctx.first_day) as usize;
            let camp = if hi > lo {
                let mut b = hi - 1;
                let t = clicks[b].ts;
                let mut x = b;
                while x > lo && clicks[x - 1].ts == t {
                    x -= 1;
                    if deref(clicks[x].kref) < deref(clicks[b].kref) {
                        b = x;
                    }
                }
                clicks[b].camp as usize
            } else {
                organic
            };
            let cc = &mut w.camp_counts[camp * ndays + dayidx];
            cc[0] += 1;
            if rev > 0 {
                cc[1] += 1;
                cc[2] += rev;
            }
            // assists
            if hi > lo {
                w.sid = w.sid.wrapping_add(1);
                if w.sid == 0 {
                    w.stamp.iter_mut().for_each(|s| *s = 0);
                    w.sid = 1;
                }
                let sid = w.sid;
                for c in &clicks[lo..hi] {
                    let cp = c.camp as usize;
                    if w.stamp[cp] != sid {
                        w.stamp[cp] = sid;
                        w.camp_counts[cp * ndays + dayidx][3] += 1;
                    }
                }
            }
            k = m;
        }
        flush_day(ud, name, cur_day, d_sessions, d_events, d_rev, d_purch, &mut w.cat_sum, cat_names);
        i = j;
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

struct SyncPtr<T>(*mut T);
unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}

fn main() {
    let t0 = Instant::now();
    let verbose = std::env::var("SS_VERBOSE").is_ok();
    let args: Vec<String> = std::env::args().collect();
    let (data_dir, out_dir) = if args.len() >= 3 {
        (PathBuf::from(&args[args.len() - 2]), PathBuf::from(&args[args.len() - 1]))
    } else {
        (PathBuf::from(env_or("DATA_DIR", "/data")), PathBuf::from(env_or("OUTPUT_DIR", "/output")))
    };
    let nthreads: usize = std::env::var("NCPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |n| n.get()))
        .max(1);

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
    let ctx = Ctx { ws, we, base, products, campaigns, ncat: cat_names.len(), unknown_cat, ncamp, first_day, ndays };

    // files
    let ev_files: Vec<&'static [u8]> = csv_files(&data_dir.join("events")).iter().map(|p| map_file(p)).collect();
    let ck_files: Vec<&'static [u8]> = csv_files(&data_dir.join("clicks")).iter().map(|p| map_file(p)).collect();
    let ln_files: Vec<&'static [u8]> = csv_files(&data_dir.join("links")).iter().map(|p| map_file(p)).collect();
    let ev_bytes: usize = ev_files.iter().map(|f| f.len()).sum();
    let chunk_target = (ev_bytes / (nthreads * 16)).clamp(1 << 16, 8 << 20);
    let mut jobs: Vec<(u8, &'static [u8])> = Vec::new();
    {
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
    // big jobs first
    jobs.sort_by_key(|j| std::cmp::Reverse(j.1.len()));
    let mut link_chunks = Vec::new();
    for &f in &ln_files {
        chunks_of(f, usize::MAX / 4, &mut link_chunks);
    }
    if verbose {
        eprintln!("setup {:?}, {} jobs, ev bytes {}", t0.elapsed(), jobs.len(), ev_bytes);
    }

    // ---- phase 1: parse
    let est_rows = ev_bytes / 70 + 1024;
    let per_bucket = (est_rows / (nthreads * NEB)) * 5 / 4 + 16;
    let next = AtomicUsize::new(0);
    let mut ev_out: Vec<Vec<Vec<EvRec>>> = Vec::with_capacity(nthreads);
    let mut ck_out: Vec<Vec<CRec>> = Vec::with_capacity(nthreads);
    let mut persons: Option<U64Map<Person>> = None;
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..nthreads {
            let ctx = &ctx;
            let jobs = &jobs;
            let next = &next;
            let link_chunks = &link_chunks;
            handles.push(s.spawn(move || {
                let pm = if t == 0 { Some(process_links(link_chunks)) } else { None };
                let mut evb: Vec<Vec<EvRec>> = (0..NEB).map(|_| Vec::with_capacity(per_bucket)).collect();
                let mut ckb: Vec<CRec> = Vec::new();
                loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    if j >= jobs.len() {
                        break;
                    }
                    let (kind, chunk) = jobs[j];
                    if kind == 0 {
                        parse_events(ctx, chunk, &mut evb);
                    } else {
                        parse_clicks(ctx, chunk, &mut ckb);
                    }
                }
                (evb, ckb, pm)
            }));
        }
        for h in handles {
            let (e, c, p) = h.join().unwrap();
            ev_out.push(e);
            ck_out.push(c);
            if p.is_some() {
                persons = p;
            }
        }
    });
    let persons = persons.unwrap();
    if verbose {
        eprintln!("parsed {:?}", t0.elapsed());
    }

    // ---- phase 2: dedup events per bucket, map users to persons, partition by person
    let next = AtomicUsize::new(0);
    let ck_chunks: Vec<(usize, usize, usize)> = {
        let mut v = Vec::new();
        for (t, c) in ck_out.iter().enumerate() {
            let step = 1 << 16;
            let mut s = 0;
            while s < c.len() {
                v.push((t, s, (s + step).min(c.len())));
                s += step;
            }
        }
        v
    };
    let ntasks = NEB + ck_chunks.len();
    let mut p_ev: Vec<Vec<Vec<PRec>>> = Vec::with_capacity(nthreads);
    let mut p_ck: Vec<Vec<Vec<CRec>>> = Vec::with_capacity(nthreads);
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for _t in 0..nthreads {
            let next = &next;
            let ev_out = &ev_out;
            let ck_out = &ck_out;
            let ck_chunks = &ck_chunks;
            let persons = &persons;
            handles.push(s.spawn(move || {
                let mut pe: Vec<Vec<PRec>> = (0..NPB).map(|_| Vec::new()).collect();
                let mut pc: Vec<Vec<CRec>> = (0..NPB).map(|_| Vec::new()).collect();
                let mut keys: Vec<u64> = Vec::new();
                let mut idx: Vec<(u32, u32)> = Vec::new();
                loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    if j >= ntasks {
                        break;
                    }
                    if j < NEB {
                        let b = j;
                        let total: usize = ev_out.iter().map(|t| t[b].len()).sum();
                        let cap = (total * 2).next_power_of_two().max(16);
                        let mask = cap - 1;
                        keys.clear();
                        keys.resize(cap, 0);
                        idx.resize(cap, (0, 0));
                        for (t, tv) in ev_out.iter().enumerate() {
                            let v = &tv[b];
                            for (i, r) in v.iter().enumerate() {
                                let mut p = (r.ehash as usize) & mask;
                                loop {
                                    let k = keys[p];
                                    if k == 0 {
                                        keys[p] = r.ehash;
                                        idx[p] = (t as u32, i as u32);
                                        break;
                                    }
                                    if k == r.ehash {
                                        let (ot, oi) = idx[p];
                                        if r.ingest < ev_out[ot as usize][b][oi as usize].ingest {
                                            idx[p] = (t as u32, i as u32);
                                        }
                                        break;
                                    }
                                    p = (p + 1) & mask;
                                }
                            }
                        }
                        for p in 0..cap {
                            if keys[p] != 0 {
                                let (t, i) = idx[p];
                                let r = &ev_out[t as usize][b][i as usize];
                                if r.ts != INVALID_TS {
                                    let person = persons.get(r.uhash).unwrap_or(Person { phash: r.uhash, pref: r.uref });
                                    pe[pbucket(person.phash)].push(PRec { phash: person.phash, pref: person.pref, ts: r.ts, pc: r.pc });
                                }
                            }
                        }
                    } else {
                        let (t, a, z) = ck_chunks[j - NEB];
                        for r in &ck_out[t][a..z] {
                            let ph = match persons.get(r.h) {
                                Some(p) => p.phash,
                                None => r.h,
                            };
                            pc[pbucket(ph)].push(CRec { h: ph, kref: r.kref, ts: r.ts, camp: r.camp });
                        }
                    }
                }
                (pe, pc)
            }));
        }
        for h in handles {
            let (e, c) = h.join().unwrap();
            p_ev.push(e);
            p_ck.push(c);
        }
    });
    if verbose {
        eprintln!("deduped {:?}", t0.elapsed());
    }

    // ---- phase 3: per person bucket sessionize + attribute
    let mut order: Vec<usize> = (0..NPB).collect();
    let sizes: Vec<usize> = (0..NPB).map(|b| p_ev.iter().map(|t| t[b].len()).sum()).collect();
    order.sort_by_key(|&b| std::cmp::Reverse(sizes[b]));
    let next = AtomicUsize::new(0);
    let mut results: Vec<(UserDaily, Vec<[i64; 4]>)> = Vec::new();
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for _t in 0..nthreads {
            let next = &next;
            let p_ev = &p_ev;
            let p_ck = &p_ck;
            let order = &order;
            let ctx = &ctx;
            let cat_names = &cat_names;
            handles.push(s.spawn(move || {
                let mut w = PersonWork {
                    cat_sum: vec![0; ctx.ncat],
                    stamp: vec![0; ctx.ncamp + 1],
                    sid: 0,
                    camp_counts: vec![[0; 4]; (ctx.ncamp + 1) * ctx.ndays],
                };
                let mut ud = UserDaily::new();
                let mut evs: Vec<PRec> = Vec::new();
                let mut cks: Vec<CRec> = Vec::new();
                loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    if j >= NPB {
                        break;
                    }
                    let b = order[j];
                    evs.clear();
                    cks.clear();
                    for t in p_ev.iter() {
                        evs.extend_from_slice(&t[b]);
                    }
                    if evs.is_empty() {
                        continue;
                    }
                    for t in p_ck.iter() {
                        cks.extend_from_slice(&t[b]);
                    }
                    process_person_bucket(ctx, cat_names, &mut evs, &mut cks, &mut w, &mut ud);
                }
                (ud, w.camp_counts)
            }));
        }
        for h in handles {
            results.push(h.join().unwrap());
        }
    });
    if verbose {
        eprintln!("sessionized {:?}", t0.elapsed());
    }

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
        // campaign daily
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
    if verbose {
        eprintln!("done {:?}", t0.elapsed());
    }
    std::process::exit(0);
}
