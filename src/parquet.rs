// Minimal Parquet writer: one row group, uncompressed, v1 data pages.
// Columns are PLAIN, or dictionary-encoded (a PLAIN dictionary page plus an
// RLE_DICTIONARY data page of bit-packed indices), as pyarrow writes by default.

use std::io::Write;

struct Tw {
    buf: Vec<u8>,
    last: Vec<i16>,
}

const T_I32: u8 = 5;
const T_I64: u8 = 6;
const T_BIN: u8 = 8;
const T_LIST: u8 = 9;
const T_STRUCT: u8 = 12;

impl Tw {
    fn new() -> Self {
        Tw { buf: Vec::new(), last: vec![0] }
    }
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.buf.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.buf.push(v as u8);
    }
    fn zz(v: i64) -> u64 {
        ((v << 1) ^ (v >> 63)) as u64
    }
    fn field(&mut self, id: i16, ty: u8) {
        let last = *self.last.last().unwrap();
        let d = id - last;
        if d > 0 && d <= 15 {
            self.buf.push(((d as u8) << 4) | ty);
        } else {
            self.buf.push(ty);
            self.varint(Self::zz(id as i64));
        }
        *self.last.last_mut().unwrap() = id;
    }
    fn i32f(&mut self, id: i16, v: i32) {
        self.field(id, T_I32);
        self.varint(Self::zz(v as i64));
    }
    fn i64f(&mut self, id: i16, v: i64) {
        self.field(id, T_I64);
        self.varint(Self::zz(v));
    }
    fn strf(&mut self, id: i16, s: &[u8]) {
        self.field(id, T_BIN);
        self.varint(s.len() as u64);
        self.buf.extend_from_slice(s);
    }
    fn struct_begin(&mut self, id: i16) {
        self.field(id, T_STRUCT);
        self.last.push(0);
    }
    fn struct_end(&mut self) {
        self.buf.push(0);
        self.last.pop();
    }
    fn list_begin(&mut self, id: i16, elem: u8, n: usize) {
        self.field(id, T_LIST);
        if n < 15 {
            self.buf.push(((n as u8) << 4) | elem);
        } else {
            self.buf.push(0xF0 | elem);
            self.varint(n as u64);
        }
    }
    fn elem_struct_begin(&mut self) {
        self.last.push(0);
    }
    fn stop(&mut self) {
        self.buf.push(0);
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Str,
    Date,
    I64,
}

pub enum Data<'a> {
    /// PLAIN-encoded non-null values.
    Plain(&'a [u8]),
    /// PLAIN-encoded dictionary of `n` values, and one index per non-null value.
    Dict { dict: &'a [u8], n: usize, idx: &'a [u32] },
}

pub struct Col<'a> {
    pub name: &'a str,
    pub kind: Kind,
    pub optional: bool,
    pub defs: &'a [u8], // one byte per row when optional
    pub data: Data<'a>,
}

fn ptype(k: Kind) -> i32 {
    match k {
        Kind::Str => 6,
        Kind::Date => 1,
        Kind::I64 => 2,
    }
}

fn hdr_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn encode_defs(defs: &[u8]) -> Vec<u8> {
    // RLE/bit-packed hybrid, bit width 1, all bit-packed groups of 8.
    let n = defs.len();
    let groups = (n + 7) / 8;
    let mut out = Vec::with_capacity(groups + 16);
    hdr_varint(&mut out, ((groups as u64) << 1) | 1);
    for g in 0..groups {
        let mut b = 0u8;
        for k in 0..8 {
            let i = g * 8 + k;
            if i < n && defs[i] != 0 {
                b |= 1 << k;
            }
        }
        out.push(b);
    }
    out
}

/// Bit width byte, then one bit-packed run of `idx` (LSB first).
fn encode_indices(idx: &[u32], ndict: usize) -> Vec<u8> {
    let mut bw = 1u32;
    while (1u64 << bw) < ndict as u64 {
        bw += 1;
    }
    let n = idx.len();
    let groups = (n + 7) / 8;
    let mut out = Vec::with_capacity(groups * bw as usize + 16);
    out.push(bw as u8);
    if n == 0 {
        return out;
    }
    hdr_varint(&mut out, ((groups as u64) << 1) | 1);
    let mut acc: u64 = 0;
    let mut nb: u32 = 0;
    for g in 0..groups {
        for k in 0..8 {
            let i = g * 8 + k;
            let v = if i < n { idx[i] as u64 } else { 0 };
            acc |= v << nb;
            nb += bw;
            while nb >= 8 {
                out.push(acc as u8);
                acc >>= 8;
                nb -= 8;
            }
        }
    }
    // groups of 8 values always end on a byte boundary
    debug_assert!(nb == 0);
    out
}

fn page_header(kind: i32, size: usize, num_values: usize, dict: bool, def_enc: bool) -> Vec<u8> {
    let mut h = Tw::new();
    h.i32f(1, kind);
    h.i32f(2, size as i32);
    h.i32f(3, size as i32);
    if kind == 2 {
        h.struct_begin(7); // DictionaryPageHeader
        h.i32f(1, num_values as i32);
        h.i32f(2, 0); // PLAIN
        h.struct_end();
    } else {
        h.struct_begin(5); // DataPageHeader
        h.i32f(1, num_values as i32);
        h.i32f(2, if dict { 8 } else { 0 }); // RLE_DICTIONARY / PLAIN
        h.i32f(3, 3); // definition levels: RLE
        h.i32f(4, 3); // repetition levels: RLE
        h.struct_end();
    }
    let _ = def_enc;
    h.stop();
    h.buf
}

/// Writes the file to `w` without building it in memory first.
pub fn write(w: &mut impl Write, num_rows: usize, cols: &[Col]) -> std::io::Result<()> {
    struct Meta {
        offset: i64,      // first page of the chunk
        data_offset: i64, // the data page
        dict_offset: i64, // -1 when PLAIN
        size: i64,
    }
    let mut pos: i64 = 0;
    w.write_all(b"PAR1")?;
    pos += 4;
    let mut metas = Vec::new();
    if num_rows > 0 {
        for c in cols {
            let offset = pos;
            let defs = if c.optional { Some(encode_defs(c.defs)) } else { None };
            let defs_len = defs.as_ref().map_or(0, |d| 4 + d.len());
            let mut dict_offset = -1;
            let (body_owned, body_ref): (Vec<u8>, &[u8]) = match &c.data {
                Data::Plain(d) => (Vec::new(), d),
                Data::Dict { dict, n, idx } => {
                    let dh = page_header(2, dict.len(), *n, false, false);
                    dict_offset = pos;
                    w.write_all(&dh)?;
                    w.write_all(dict)?;
                    pos += (dh.len() + dict.len()) as i64;
                    (encode_indices(idx, *n), &[])
                }
            };
            let body: &[u8] = if body_owned.is_empty() { body_ref } else { &body_owned };
            let size = defs_len + body.len();
            let dict = matches!(c.data, Data::Dict { .. });
            let ph = page_header(0, size, num_rows, dict, c.optional);
            let data_offset = pos;
            w.write_all(&ph)?;
            if let Some(d) = &defs {
                w.write_all(&(d.len() as u32).to_le_bytes())?;
                w.write_all(d)?;
            }
            w.write_all(body)?;
            pos += (ph.len() + size) as i64;
            metas.push(Meta { offset, data_offset, dict_offset, size: pos - offset });
        }
    }
    // FileMetaData
    let mut m = Tw::new();
    m.i32f(1, 1);
    m.list_begin(2, T_STRUCT, cols.len() + 1);
    m.elem_struct_begin();
    m.strf(4, b"schema");
    m.i32f(5, cols.len() as i32);
    m.struct_end();
    for c in cols {
        m.elem_struct_begin();
        m.i32f(1, ptype(c.kind));
        m.i32f(3, if c.optional { 1 } else { 0 });
        m.strf(4, c.name.as_bytes());
        match c.kind {
            Kind::Str => {
                m.i32f(6, 0); // UTF8
                m.struct_begin(10);
                m.struct_begin(1); // STRING
                m.struct_end();
                m.struct_end();
            }
            Kind::Date => {
                m.i32f(6, 6); // DATE
                m.struct_begin(10);
                m.struct_begin(6); // DATE
                m.struct_end();
                m.struct_end();
            }
            Kind::I64 => {}
        }
        m.struct_end();
    }
    m.i64f(3, num_rows as i64);
    if num_rows > 0 {
        m.list_begin(4, T_STRUCT, 1);
        m.elem_struct_begin();
        m.list_begin(1, T_STRUCT, cols.len());
        let mut total = 0i64;
        for (c, meta) in cols.iter().zip(&metas) {
            total += meta.size;
            let dict = meta.dict_offset >= 0;
            m.elem_struct_begin();
            m.i64f(2, meta.offset);
            m.struct_begin(3);
            m.i32f(1, ptype(c.kind));
            if dict {
                m.list_begin(2, T_I32, 3);
                m.varint(Tw::zz(0)); // PLAIN
                m.varint(Tw::zz(3)); // RLE
                m.varint(Tw::zz(8)); // RLE_DICTIONARY
            } else {
                m.list_begin(2, T_I32, 2);
                m.varint(Tw::zz(0));
                m.varint(Tw::zz(3));
            }
            m.list_begin(3, T_BIN, 1);
            m.varint(c.name.len() as u64);
            m.buf.extend_from_slice(c.name.as_bytes());
            m.i32f(4, 0); // UNCOMPRESSED
            m.i64f(5, num_rows as i64);
            m.i64f(6, meta.size);
            m.i64f(7, meta.size);
            m.i64f(9, meta.data_offset);
            if dict {
                m.i64f(11, meta.dict_offset);
            }
            m.struct_end();
            m.struct_end();
        }
        m.i64f(2, total);
        m.i64f(3, num_rows as i64);
        m.struct_end();
    } else {
        m.list_begin(4, T_STRUCT, 0);
    }
    m.strf(6, b"shopping-sessions");
    m.stop();
    w.write_all(&m.buf)?;
    w.write_all(&(m.buf.len() as u32).to_le_bytes())?;
    w.write_all(b"PAR1")?;
    w.flush()
}

/// Dictionary-encodes integer values (first-appearance order): (PLAIN i64 dictionary, count, indices).
pub fn dict_i64(vals: &[u64]) -> (Vec<u8>, usize, Vec<u32>) {
    let mut cap = 1024usize;
    let mut keys: Vec<u64> = vec![u64::MAX; cap];
    let mut ids: Vec<u32> = vec![0; cap];
    let mut dict: Vec<u64> = Vec::new();
    let mut idx: Vec<u32> = Vec::with_capacity(vals.len());
    let hash = |v: u64| (v.wrapping_mul(0x9E3779B97F4A7C15) >> 20) as usize;
    for &v in vals {
        let mask = cap - 1;
        let mut p = hash(v) & mask;
        loop {
            let k = keys[p];
            if k == v {
                idx.push(ids[p]);
                break;
            }
            if k == u64::MAX {
                keys[p] = v;
                ids[p] = dict.len() as u32;
                idx.push(dict.len() as u32);
                dict.push(v);
                if dict.len() * 2 > cap {
                    cap *= 2;
                    keys = vec![u64::MAX; cap];
                    ids = vec![0; cap];
                    let mask = cap - 1;
                    for (i, &d) in dict.iter().enumerate() {
                        let mut q = hash(d) & mask;
                        while keys[q] != u64::MAX {
                            q = (q + 1) & mask;
                        }
                        keys[q] = d;
                        ids[q] = i as u32;
                    }
                }
                break;
            }
            p = (p + 1) & mask;
        }
    }
    let mut bytes = Vec::with_capacity(dict.len() * 8);
    for d in &dict {
        bytes.extend_from_slice(&(*d as i64).to_le_bytes());
    }
    (bytes, dict.len(), idx)
}
