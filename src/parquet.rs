// Minimal Parquet writer: one row group, one PLAIN data page (v1) per column, uncompressed.

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

pub struct Col<'a> {
    pub name: &'a str,
    pub kind: Kind,
    pub optional: bool,
    pub defs: &'a [u8], // one byte per row when optional
    pub data: &'a [u8], // PLAIN-encoded non-null values
}

fn ptype(k: Kind) -> i32 {
    match k {
        Kind::Str => 6,
        Kind::Date => 1,
        Kind::I64 => 2,
    }
}

fn encode_defs(defs: &[u8]) -> Vec<u8> {
    // RLE/bit-packed hybrid, bit width 1, all bit-packed groups of 8.
    let n = defs.len();
    let groups = (n + 7) / 8;
    let mut out = Vec::with_capacity(groups + 16);
    let mut hdr = ((groups as u64) << 1) | 1;
    while hdr >= 0x80 {
        out.push((hdr as u8) | 0x80);
        hdr >>= 7;
    }
    out.push(hdr as u8);
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

pub fn write(num_rows: usize, cols: &[Col]) -> Vec<u8> {
    let mut file: Vec<u8> = Vec::with_capacity(cols.iter().map(|c| c.data.len() + c.defs.len() / 8 + 64).sum::<usize>() + 1024);
    file.extend_from_slice(b"PAR1");
    struct Meta {
        offset: i64,
        size: i64,
    }
    let mut metas = Vec::new();
    if num_rows > 0 {
        for c in cols {
            let mut page: Vec<u8> = Vec::new();
            if c.optional {
                let enc = encode_defs(c.defs);
                page.extend_from_slice(&(enc.len() as u32).to_le_bytes());
                page.extend_from_slice(&enc);
            }
            page.extend_from_slice(c.data);
            let mut h = Tw::new();
            h.i32f(1, 0); // DATA_PAGE
            h.i32f(2, page.len() as i32);
            h.i32f(3, page.len() as i32);
            h.struct_begin(5);
            h.i32f(1, num_rows as i32);
            h.i32f(2, 0); // PLAIN
            h.i32f(3, 3); // RLE
            h.i32f(4, 3); // RLE
            h.struct_end();
            h.stop();
            let offset = file.len() as i64;
            file.extend_from_slice(&h.buf);
            file.extend_from_slice(&page);
            metas.push(Meta { offset, size: (h.buf.len() + page.len()) as i64 });
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
            m.elem_struct_begin();
            m.i64f(2, meta.offset);
            m.struct_begin(3);
            m.i32f(1, ptype(c.kind));
            m.list_begin(2, T_I32, 2);
            m.varint(Tw::zz(0));
            m.varint(Tw::zz(3));
            m.list_begin(3, T_BIN, 1);
            m.varint(c.name.len() as u64);
            m.buf.extend_from_slice(c.name.as_bytes());
            m.i32f(4, 0); // UNCOMPRESSED
            m.i64f(5, num_rows as i64);
            m.i64f(6, meta.size);
            m.i64f(7, meta.size);
            m.i64f(9, meta.offset);
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
    file.extend_from_slice(&m.buf);
    file.extend_from_slice(&(m.buf.len() as u32).to_le_bytes());
    file.extend_from_slice(b"PAR1");
    file
}

pub fn write_user_daily(n: usize, uid: &[u8], day: &[u8], sessions: &[u8], events: &[u8], revenue: &[u8], top_def: &[u8], top_val: &[u8]) -> Vec<u8> {
    write(
        n,
        &[
            Col { name: "user_id", kind: Kind::Str, optional: false, defs: &[], data: uid },
            Col { name: "day", kind: Kind::Date, optional: false, defs: &[], data: day },
            Col { name: "sessions", kind: Kind::I64, optional: false, defs: &[], data: sessions },
            Col { name: "events", kind: Kind::I64, optional: false, defs: &[], data: events },
            Col { name: "revenue_cents", kind: Kind::I64, optional: false, defs: &[], data: revenue },
            Col { name: "top_category", kind: Kind::Str, optional: true, defs: top_def, data: top_val },
        ],
    )
}

pub fn write_campaign_daily(n: usize, cid: &[u8], ch: &[u8], day: &[u8], s0: &[u8], s1: &[u8], s2: &[u8], s3: &[u8]) -> Vec<u8> {
    write(
        n,
        &[
            Col { name: "campaign_id", kind: Kind::Str, optional: false, defs: &[], data: cid },
            Col { name: "channel", kind: Kind::Str, optional: false, defs: &[], data: ch },
            Col { name: "day", kind: Kind::Date, optional: false, defs: &[], data: day },
            Col { name: "sessions", kind: Kind::I64, optional: false, defs: &[], data: s0 },
            Col { name: "purchasing_sessions", kind: Kind::I64, optional: false, defs: &[], data: s1 },
            Col { name: "revenue_cents", kind: Kind::I64, optional: false, defs: &[], data: s2 },
            Col { name: "assisted_sessions", kind: Kind::I64, optional: false, defs: &[], data: s3 },
        ],
    )
}
