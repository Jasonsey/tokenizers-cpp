use std::collections::HashSet;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use bstr::ByteVec;
use rustc_hash::FxHashMap as HashMap;
use serde::{Deserialize, Serialize};
use super::core_bpe::CoreBPE;

#[derive(Serialize, Deserialize)]
struct TiktokenJson {
    name: String,
    mergeable_ranks: HashMap<String, usize>,
    special_tokens: HashMap<String, usize>,
    pat_str: Option<String>,
}

#[no_mangle]
extern "system" fn tiktoken_new_from_str(input_cstr: *const u8, len: usize) -> *mut CoreBPE {
    unsafe {
        let json = std::str::from_utf8(std::slice::from_raw_parts(input_cstr, len)).unwrap();
        let tiktoken_json: TiktokenJson = serde_json::from_str(json).expect("parse json failed");
        let mergeable_ranks: HashMap<Vec<u8>, usize> = tiktoken_json
            .mergeable_ranks
            .iter()
            .map(
                |(key, value)|
                (BASE64_STANDARD.decode(key).unwrap(), value.to_owned())
            )
            .collect();
        let core_bpe = CoreBPE::new(
            mergeable_ranks,
            tiktoken_json.special_tokens,
            tiktoken_json.pat_str.unwrap_or(
                r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#.to_string()
            ).as_str(),
        ).unwrap();
        Box::into_raw(Box::new(core_bpe))
    }
}

#[no_mangle]
extern "system" fn tiktoken_encode(
    handle: *mut CoreBPE,
    input_cstr: *const u8,
    input_cstr_len: usize,
    allowed_special: *const u8,
    allowed_special_len: usize,
    out_data: *mut *mut usize,
    out_len: *mut usize,
) {
    unsafe {
        let input_text = std::str::from_utf8(
            std::slice::from_raw_parts(input_cstr, input_cstr_len)
        ).unwrap();
        let allowed_special = std::str::from_utf8(
            std::slice::from_raw_parts(allowed_special, allowed_special_len)
        ).unwrap();
        let allowed_special: Vec<String> = serde_json::from_str(allowed_special)
            .expect("parse json failed");
        let allowed_special: HashSet<&str> = allowed_special
            .iter()
            .map(|x| x.as_str())
            .collect();
        let mut encode_ids: Vec<usize> = (*handle).encode(input_text, allowed_special);
        *out_data = encode_ids.as_mut_ptr();
        *out_len = encode_ids.len()
    }
}

#[no_mangle]
extern "system" fn tiktoken_decode(
    handle: *mut CoreBPE,
    input_ids: *const u32,
    input_len: usize,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) {
    unsafe {
        let input_data = std::slice::from_raw_parts(input_ids, input_len)
            .iter()
            .map(|x| *x as usize)
            .collect();
        let mut decode_str = (*handle).decode_bytes(input_data).into_string().expect("string decode error");
        *out_cstr = decode_str.as_mut_ptr();
        *out_len = decode_str.len();
    }
}

#[no_mangle]
extern "system" fn tiktoken_free(handle: *mut CoreBPE) {
    unsafe {
        drop(Box::from_raw(handle));
    }
}
