// llkv_types/src/lib.rs
#![forbid(unsafe_code)]

use bitcode::{Decode, Encode};
use std::borrow::Cow;

/* ========== Core traits & helpers ========== */

pub trait Codec: 'static + Send + Sync {
    type Logical<'a>;
    const ENCODING: Encoding;
    const ORDER_PRESERVING: bool;

    fn encode_into(dst: &mut Vec<u8>, v: &Self::Logical<'_>);
    fn decode<'a>(src: &'a [u8]) -> Self::Logical<'a>;

    #[inline] fn encode_lower_bound(dst: &mut Vec<u8>, v: &Self::Logical<'_>) {
        Self::encode_into(dst, v)
    }
    #[inline] fn encode_upper_bound(dst: &mut Vec<u8>, v: &Self::Logical<'_>) {
        Self::encode_into(dst, v)
    }
}

/* sign/float transforms for order-preserving codecs */
#[inline] fn i_to_u<T,U>(x: T, sign: U) -> U where T: Into<U>, U: Copy + std::ops::BitXor<Output=U> { x.into() ^ sign }
#[inline] fn u_to_i<U,T>(u: U, sign: U) -> T where U: Copy + std::ops::BitXor<Output=U> + Into<T> { (u ^ sign).into() }
#[inline] fn f64_to_u64_sort(b: u64) -> u64 { if b & 0x8000_0000_0000_0000 == 0 { b ^ 0x8000_0000_0000_0000 } else { !b } }
#[inline] fn u64_sort_to_f64(u: u64) -> u64 { if u & 0x8000_0000_0000_0000 != 0 { u ^ 0x8000_0000_0000_0000 } else { !u } }
#[inline] fn f32_to_u32_sort(b: u32) -> u32 { if b & 0x8000_0000 == 0 { b ^ 0x8000_0000 } else { !b } }
#[inline] fn u32_sort_to_f32(u: u32) -> u32 { if u & 0x8000_0000 != 0 { u ^ 0x8000_0000 } else { !u } }

/* ========== Single registry declaration ========== */

/// This macro is the *only place* you list supported types.
/// Each row: (DataTypeVariant, EncodingVariant, RustType, CodecStruct)
macro_rules! for_each_type {
    ($m:ident) => {
        $m!(Bool,              BoolByte,               bool,        BoolByte);
        $m!(UInt8,             BeU8,                   u8,          BeU8);
        $m!(UInt16,            BeU16,                  u16,         BeU16);
        $m!(UInt32,            BeU32,                  u32,         BeU32);
        $m!(UInt64,            BeU64,                  u64,         BeU64);
        $m!(Int8,              SortI8,                 i8,          SortI8);
        $m!(Int16,             SortI16,                i16,         SortI16);
        $m!(Int32,             SortI32,                i32,         SortI32);
        $m!(Int64,             SortI64,                i64,         SortI64);
        $m!(Float32,           SortF32,                f32,         SortF32);
        $m!(Float64,           SortF64,                f64,         SortF64);
        $m!(Utf8,              Utf8Raw,                &'a str,     Utf8Raw);
        $m!(Binary,            BinaryRaw,              &'a [u8],    BinaryRaw);
        $m!(TimestampMicros,   TimestampMicrosSortI64, i64,         TimestampMicros);
    }
}

/* ========== Enums generated from registry ========== */

#[derive(Copy, Clone, Debug, PartialEq, Eq, Encode, Decode)]
#[repr(u16)]
pub enum DataType {
    Bool = 1,
    UInt8, UInt16, UInt32, UInt64,
    Int8, Int16, Int32, Int64,
    Float32, Float64,
    Utf8, Binary,
    TimestampMicros,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Encode, Decode)]
#[repr(u16)]
pub enum Encoding {
    BoolByte = 100,
    BeU8, BeU16, BeU32, BeU64,
    SortI8, SortI16, SortI32, SortI64,
    SortF32, SortF64,
    Utf8Raw, BinaryRaw,
    TimestampMicrosSortI64,
}

/* default_encoding() and names are auto-covered */
impl DataType {
    #[inline]
    pub fn default_encoding(self) -> Encoding {
        match self {
            $( DataType::$dt => Encoding::$enc, )*
        }
    }
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            $( DataType::$dt => stringify!($dt), )*
        }
    }
}
impl Encoding {
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            $( Encoding::$enc => stringify!($enc), )*
        }
    }
}

/* ========== Concrete Codec impls (small, inlined) ========== */

/* Unsigned BE */
pub struct BeU8;  impl Codec for BeU8  { type Logical<'a>=u8;  const ENCODING:Encoding=Encoding::BeU8;  const ORDER_PRESERVING:bool=true;
    #[inline] fn encode_into(d:&mut Vec<u8>, v:&u8){ d.push(*v); } #[inline] fn decode<'a>(s:&'a[u8])->u8{ s[0] } }
pub struct BeU16; impl Codec for BeU16 { type Logical<'a>=u16; const ENCODING:Encoding=Encoding::BeU16; const ORDER_PRESERVING:bool=true;
    #[inline] fn encode_into(d:&mut Vec<u8>, v:&u16){ d.extend_from_slice(&v.to_be_bytes()); }
    #[inline] fn decode<'a>(s:&'a[u8])->u16{ let mut a=[0u8;2]; a.copy_from_slice(&s[..2]); u16::from_be_bytes(a)} }
pub struct BeU32; impl Codec for BeU32 { type Logical<'a>=u32; const ENCODING:Encoding=Encoding::BeU32; const ORDER_PRESERVING:bool=true;
    #[inline] fn encode_into(d:&mut Vec<u8>, v:&u32){ d.extend_from_slice(&v.to_be_bytes()); }
    #[inline] fn decode<'a>(s:&'a[u8])->u32{ let mut a=[0u8;4]; a.copy_from_slice(&s[..4]); u32::from_be_bytes(a)} }
pub struct BeU64; impl Codec for BeU64 { type Logical<'a>=u64; const ENCODING:Encoding=Encoding::BeU64; const ORDER_PRESERVING:bool=true;
    #[inline] fn encode_into(d:&mut Vec<u8>, v:&u64){ d.extend_from_slice(&v.to_be_bytes()); }
    #[inline] fn decode<'a>(s:&'a[u8])->u64{ let mut a=[0u8;8]; a.copy_from_slice(&s[..8]); u64::from_be_bytes(a)} }

/* Signed -> sortable BE via sign flip */
pub struct SortI8;  impl Codec for SortI8  { type Logical<'a>=i8;  const ENCODING:Encoding=Encoding::SortI8;  const ORDER_PRESERVING:bool=true;
    #[inline] fn encode_into(d:&mut Vec<u8>, v:&i8){ d.push(i_to_u::<_,u8>(*v,0x80)); }
    #[inline] fn decode<'a>(s:&'a[u8])->i8 { u_to_i::<u8,i8>(s[0],0x80) } }
pub struct SortI16; impl Codec for SortI16 { type Logical<'a>=i16; const ENCODING:Encoding=Encoding::SortI16; const ORDER_PRESERVING:bool=true;
    #[inline] fn encode_into(d:&mut Vec<u8>, v:&i16){ let u=i_to_u::<_,u16>(*v,0x8000); d.extend_from_slice(&u.to_be_bytes()); }
    #[inline] fn decode<'a>(s:&'a[u8])->i16{ let mut a=[0u8;2]; a.copy_from_slice(&s[..2]); u_to_i::<u16,i16>(u16::from_be_bytes(a),0x8000)} }
pub struct SortI32; impl Codec for SortI32 { type Logical<'a>=i32; const ENCODING:Encoding=Encoding::SortI32; const ORDER_PRESERVING:bool=true;
    #[inline] fn encode_into(d:&mut Vec<u8>, v:&i32){ let u=i_to_u::<_,u32>(*v,0x8000_0000); d.extend_from_slice(&u.to_be_bytes()); }
    #[inline] fn decode<'a>(s:&'a[u8])->i32{ let mut a=[0u8;4]; a.copy_from_slice(&s[..4]); u_to_i::<u32,i32>(u32::from_be_bytes(a),0x8000_0000)} }
pub struct SortI64; impl Codec for SortI64 { type Logical<'a>=i64; const ENCODING:Encoding=Encoding::SortI64; const ORDER_PRESERVING:bool=true;
    #[inline] fn encode_into(d:&mut Vec<u8>, v:&i64){ let u=i_to_u::<_,u64>(*v,0x8000_0000_0000_0000); d.extend_from_slice(&u.to_be_bytes()); }
    #[inline] fn decode<'a>(s:&'a[u8])->i64{ let mut a=[0u8;8]; a.copy_from_slice(&s[..8]); u_to_i::<u64,i64>(u64::from_be_bytes(a),0x8000_0000_0000_0000)} }

/* Floats -> sortable BE */
pub struct SortF32; impl Codec for SortF32 { type Logical<'a>=f32; const ENCODING:Encoding=Encoding::SortF32; const ORDER_PRESERVING:bool=true;
    #[inline] fn encode_into(d:&mut Vec<u8>, v:&f32){ let u=f32_to_u32_sort(v.to_bits()); d.extend_from_slice(&u.to_be_bytes()); }
    #[inline] fn decode<'a>(s:&'a[u8])->f32{ let mut a=[0u8;4]; a.copy_from_slice(&s[..4]); f32::from_bits(u32_sort_to_f32(u32::from_be_bytes(a))) } }
pub struct SortF64; impl Codec for SortF64 { type Logical<'a>=f64; const ENCODING:Encoding=Encoding::SortF64; const ORDER_PRESERVING:bool=true;
    #[inline] fn encode_into(d:&mut Vec<u8>, v:&f64){ let u=f64_to_u64_sort(v.to_bits()); d.extend_from_slice(&u.to_be_bytes()); }
    #[inline] fn decode<'a>(s:&'a[u8])->f64{ let mut a=[0u8;8]; a.copy_from_slice(&s[..8]); f64::from_bits(u64_sort_to_f64(u64::from_be_bytes(a))) } }

/* Text/Binary */
pub struct Utf8Raw;   impl Codec for Utf8Raw { type Logical<'a>=&'a str; const ENCODING:Encoding=Encoding::Utf8Raw; const ORDER_PRESERVING:bool=true;
    #[inline] fn encode_into(d:&mut Vec<u8>, v:&&str){ d.extend_from_slice(v.as_bytes()); }
    #[inline] fn decode<'a>(s:&'a[u8])->&'a str{ std::str::from_utf8(s).expect("invalid UTF-8") } }
pub struct BinaryRaw; impl Codec for BinaryRaw { type Logical<'a>=&'a [u8]; const ENCODING:Encoding=Encoding::BinaryRaw; const ORDER_PRESERVING:bool=false;
    #[inline] fn encode_into(d:&mut Vec<u8>, v:&&[u8]){ d.extend_from_slice(v); }
    #[inline] fn decode<'a>(s:&'a[u8])->&'a [u8]{ s } }

/* Timestamp (micros) as sortable i64 */
pub struct TimestampMicros; impl Codec for TimestampMicros {
    type Logical<'a>=i64; const ENCODING:Encoding=Encoding::TimestampMicrosSortI64; const ORDER_PRESERVING:bool=true;
    #[inline] fn encode_into(d:&mut Vec<u8>, v:&i64){ SortI64::encode_into(d, v) }
    #[inline] fn decode<'a>(s:&'a[u8])->i64{ SortI64::decode(s) }
}

/* ========== Dynamic values + bounds helpers ========== */

#[derive(Clone, Debug)]
pub enum DynValue<'a> {
    Bool(bool),
    U8(u8), U16(u16), U32(u32), U64(u64),
    I8(i8), I16(i16), I32(i32), I64(i64),
    F32(f32), F64(f64),
    Str(&'a str),
    Bin(&'a [u8]),
    TsMicros(i64),
}

#[derive(Clone, Debug)]
pub struct EncodedBounds {
    pub lo: Cow<'static,[u8]>,
    pub hi: Cow<'static,[u8]>,
    pub inclusive_lo: bool,
    pub inclusive_hi: bool,
}

/* One place to maintain encode/decode/bounds for dynamic paths */
impl DataType {
    #[inline]
    pub fn encode_dyn<'a>(self, enc: Encoding, v: DynValue<'a>, out: &mut Vec<u8>) {
        match (self, enc, v) {
            (DataType::Bool, Encoding::BoolByte, DynValue::Bool(x)) => BoolByte::encode_into(out, &x),
            (DataType::UInt8, Encoding::BeU8, DynValue::U8(x)) => BeU8::encode_into(out, &x),
            (DataType::UInt16, Encoding::BeU16, DynValue::U16(x)) => BeU16::encode_into(out, &x),
            (DataType::UInt32, Encoding::BeU32, DynValue::U32(x)) => BeU32::encode_into(out, &x),
            (DataType::UInt64, Encoding::BeU64, DynValue::U64(x)) => BeU64::encode_into(out, &x),
            (DataType::Int8, Encoding::SortI8, DynValue::I8(x)) => SortI8::encode_into(out, &x),
            (DataType::Int16, Encoding::SortI16, DynValue::I16(x)) => SortI16::encode_into(out, &x),
            (DataType::Int32, Encoding::SortI32, DynValue::I32(x)) => SortI32::encode_into(out, &x),
            (DataType::Int64, Encoding::SortI64, DynValue::I64(x)) => SortI64::encode_into(out, &x),
            (DataType::Float32, Encoding::SortF32, DynValue::F32(x)) => SortF32::encode_into(out, &x),
            (DataType::Float64, Encoding::SortF64, DynValue::F64(x)) => SortF64::encode_into(out, &x),
            (DataType::Utf8, Encoding::Utf8Raw, DynValue::Str(s)) => Utf8Raw::encode_into(out, &s),
            (DataType::Binary, Encoding::BinaryRaw, DynValue::Bin(b)) => BinaryRaw::encode_into(out, &b),
            (DataType::TimestampMicros, Encoding::TimestampMicrosSortI64, DynValue::TsMicros(t)) => TimestampMicros::encode_into(out, &t),
            _ => panic!("DataType/Encoding/DynValue mismatch"),
        }
    }

    #[inline]
    pub fn decode_dyn<'a>(self, enc: Encoding, bytes: &'a [u8]) -> DynValue<'a> {
        match (self, enc) {
            (DataType::Bool, Encoding::BoolByte) => DynValue::Bool(BoolByte::decode(bytes)),
            (DataType::UInt8, Encoding::BeU8) => DynValue::U8(BeU8::decode(bytes)),
            (DataType::UInt16, Encoding::BeU16) => DynValue::U16(BeU16::decode(bytes)),
            (DataType::UInt32, Encoding::BeU32) => DynValue::U32(BeU32::decode(bytes)),
            (DataType::UInt64, Encoding::BeU64) => DynValue::U64(BeU64::decode(bytes)),
            (DataType::Int8, Encoding::SortI8) => DynValue::I8(SortI8::decode(bytes)),
            (DataType::Int16, Encoding::SortI16) => DynValue::I16(SortI16::decode(bytes)),
            (DataType::Int32, Encoding::SortI32) => DynValue::I32(SortI32::decode(bytes)),
            (DataType::Int64, Encoding::SortI64) => DynValue::I64(SortI64::decode(bytes)),
            (DataType::Float32, Encoding::SortF32) => DynValue::F32(SortF32::decode(bytes)),
            (DataType::Float64, Encoding::SortF64) => DynValue::F64(SortF64::decode(bytes)),
            (DataType::Utf8, Encoding::Utf8Raw) => DynValue::Str(Utf8Raw::decode(bytes)),
            (DataType::Binary, Encoding::BinaryRaw) => DynValue::Bin(BinaryRaw::decode(bytes)),
            (DataType::TimestampMicros, Encoding::TimestampMicrosSortI64) => DynValue::TsMicros(TimestampMicros::decode(bytes)),
            _ => panic!("DataType/Encoding mismatch"),
        }
    }

    /// Equality bounds for dynamic value.
    #[inline]
    pub fn eq_bounds_dyn<'a>(self, enc: Encoding, v: DynValue<'a>) -> EncodedBounds {
        let mut buf = Vec::new();
        self.encode_dyn(enc, v, &mut buf);
        EncodedBounds { lo: Cow::Owned(buf.clone()), hi: Cow::Owned(buf), inclusive_lo: true, inclusive_hi: true }
    }
}

/* UTF-8 prefix upper (for StartsWith) */
#[inline]
pub fn next_prefix_upper(prefix: &[u8]) -> Option<Vec<u8>> {
    if prefix.is_empty() { return None; }
    let mut up = prefix.to_vec();
    for i in (0..up.len()).rev() {
        if up[i] != 0xFF { up[i] = up[i].saturating_add(1); up.truncate(i+1); return Some(up); }
    }
    None
}

/* ========== Arrow-ish schema with better names ========== */

#[derive(Clone, Debug, PartialEq, Eq, Encode, Decode)]
pub struct Field {
    pub id: u32,
    pub name: String,
    pub data_type: DataType,
    pub encoding: Encoding,   // physical encoding actually used
    pub nullable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Encode, Decode, Default)]
pub struct Schema {
    pub table_id: u32,
    pub name: String,
    pub version: u32,
    pub fields: Vec<Field>,
}

impl Schema {
    #[inline] pub fn field(&self, id: u32) -> Option<&Field> { self.fields.iter().find(|f| f.id == id) }
}
