//! `compile_variant` tool — compile a single strategy variant to SPIR-V.
//!
//! Runs the full lex→parse→HIR→codegen pipeline with explicit hole assignments,
//! validates SPIR-V in-process via spirv-tools, and returns base64(spirv_bytes)
//! plus kernel metadata, required capabilities, and extensions.
//!
//! ## Base64 encoding (RFC 4648 §4 STANDARD alphabet)
//!
//! Hand-rolled ~30-line implementation — no new `base64` dependency.
//! Alphabet: `A-Za-z0-9+/` with `=` padding. Index 62 = `+`, index 63 = `/`.
//! This is the STANDARD alphabet, NOT the URL-safe (§5) variant.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::mcp::dispatch::McpToolError;

// ── Base64 alphabet (RFC 4648 §4 STANDARD) ───────────────────────────────────

/// RFC 4648 §4 STANDARD base64 alphabet.
///
/// Index 62 is `+` (0x2B), index 63 is `/` (0x2F).
/// This is explicitly NOT the URL-safe §5 alphabet (which uses `-` and `_`).
pub const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode `bytes` using the RFC 4648 §4 STANDARD base64 alphabet with `=` padding.
///
/// Output contains only characters in `[A-Za-z0-9+/=]`.
/// No line breaks, no whitespace, no BOM.
pub fn base64_encode(bytes: &[u8]) -> String {
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len().div_ceil(3) * 4);
    let a: &[u8; 64] = BASE64_ALPHABET;

    let mut i: usize = 0;
    let len: usize = bytes.len();
    while i + 2 < len {
        let b0: u32 = bytes[i] as u32;
        let b1: u32 = bytes[i + 1] as u32;
        let b2: u32 = bytes[i + 2] as u32;
        out.push(a[(b0 >> 2) as usize]);
        out.push(a[((b0 & 0x03) << 4 | (b1 >> 4)) as usize]);
        out.push(a[((b1 & 0x0F) << 2 | (b2 >> 6)) as usize]);
        out.push(a[(b2 & 0x3F) as usize]);
        i += 3;
    }
    if i < len {
        let b0: u32 = bytes[i] as u32;
        out.push(a[(b0 >> 2) as usize]);
        if i + 1 < len {
            let b1: u32 = bytes[i + 1] as u32;
            out.push(a[((b0 & 0x03) << 4 | (b1 >> 4)) as usize]);
            out.push(a[((b1 & 0x0F) << 2) as usize]);
            out.push(b'=');
        } else {
            out.push(a[((b0 & 0x03) << 4) as usize]);
            out.push(b'=');
            out.push(b'=');
        }
    }

    // SAFETY: we only push bytes from BASE64_ALPHABET (ASCII) and b'='.
    String::from_utf8(out).expect("base64_encode output is always valid ASCII")
}

/// Decode RFC 4648 §4 STANDARD base64.
///
/// Accepts only the standard alphabet (`[A-Za-z0-9+/=]`).
/// URL-safe input (containing `-` or `_`) returns `Err`.
/// Incorrect padding or invalid characters return `Err` with a description.
pub fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    // Decode table: 0xFF = invalid.
    let mut table: [u8; 256] = [0xFF_u8; 256];
    for (i, &b) in BASE64_ALPHABET.iter().enumerate() {
        table[b as usize] = i as u8;
    }

    let bytes: &[u8] = s.as_bytes();
    let len: usize = bytes.len();
    if !len.is_multiple_of(4) {
        return Err(format!("base64 length {len} is not a multiple of 4"));
    }

    let mut out: Vec<u8> = Vec::with_capacity(len / 4 * 3);

    let mut i: usize = 0;
    while i < len {
        let c0: u8 = bytes[i];
        let c1: u8 = bytes[i + 1];
        let c2: u8 = bytes[i + 2];
        let c3: u8 = bytes[i + 3];

        // Reject URL-safe characters
        if c0 == b'-' || c0 == b'_' || c1 == b'-' || c1 == b'_'
            || c2 == b'-' || c2 == b'_' || c3 == b'-' || c3 == b'_' {
            return Err(
                "base64 input contains URL-safe characters ('-' or '_'); expected standard alphabet".to_string()
            );
        }

        let v0: u8 = table[c0 as usize];
        let v1: u8 = table[c1 as usize];
        if v0 == 0xFF || v1 == 0xFF {
            return Err(format!("invalid base64 character at position {i}"));
        }

        out.push((v0 << 2) | (v1 >> 4));

        if c2 != b'=' {
            let v2: u8 = table[c2 as usize];
            if v2 == 0xFF {
                return Err(format!("invalid base64 character at position {}", i + 2));
            }
            out.push((v1 << 4) | (v2 >> 2));

            if c3 != b'=' {
                let v3: u8 = table[c3 as usize];
                if v3 == 0xFF {
                    return Err(format!("invalid base64 character at position {}", i + 3));
                }
                out.push((v2 << 6) | v3);
            }
        }

        i += 4;
    }
    Ok(out)
}

// ── SPIR-V capability / extension scanner ────────────────────────────────────

/// SPIR-V opcode for `OpCapability`.
const OP_CAPABILITY: u16 = 17;
/// SPIR-V opcode for `OpExtension`.
const OP_EXTENSION: u16 = 10;
/// SPIR-V opcode for `OpExtInstImport`.
const OP_EXT_INST_IMPORT: u16 = 11;

/// Scan a SPIR-V word stream for `OpCapability` and `OpExtension` instructions.
///
/// Returns `(capabilities, extensions)` in encounter order.
/// The first 5 words are the SPIR-V header (magic, version, generator, bound, schema)
/// and are skipped.
pub fn scan_caps_and_exts(words: &[u32]) -> (Vec<String>, Vec<String>) {
    let mut caps: Vec<String> = Vec::new();
    let mut exts: Vec<String> = Vec::new();

    if words.len() < 5 {
        return (caps, exts);
    }

    let mut pos: usize = 5; // skip SPIR-V header
    while pos < words.len() {
        let word: u32 = words[pos];
        let opcode: u16 = (word & 0xFFFF) as u16;
        let wc: u16 = ((word >> 16) & 0xFFFF) as u16;
        if wc == 0 {
            break; // malformed; stop
        }
        let end: usize = pos + wc as usize;
        if end > words.len() {
            break; // truncated instruction
        }

        match opcode {
            OP_CAPABILITY if wc == 2 => {
                let cap_id: u32 = words[pos + 1];
                // Map capability id to name using spirv crate.
                let name: String = spirv_capability_name(cap_id);
                caps.push(name);
            }
            OP_EXTENSION if wc >= 2 => {
                // Remaining words (pos+1 .. end) are a null-terminated UTF-8 string
                // packed as little-endian u32 words.
                let text: String = words_to_str(&words[pos + 1..end]);
                if !text.is_empty() {
                    exts.push(text);
                }
            }
            OP_EXT_INST_IMPORT if wc >= 3 => {
                // Second operand onwards is a null-terminated name string.
                let text: String = words_to_str(&words[pos + 2..end]);
                if !text.is_empty() {
                    // We only collect OpExtension, not OpExtInstImport, per spec.
                    // But we can use this in scanning caps/exts for extension sets.
                    let _ = text; // intentionally ignored per spec separation
                }
            }
            _ => {}
        }

        pos = end;
    }

    (caps, exts)
}

/// Convert SPIR-V null-terminated string words to a Rust `String`.
fn words_to_str(words: &[u32]) -> String {
    let mut bytes: Vec<u8> = Vec::with_capacity(words.len() * 4);
    for &w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    // Trim at first null byte.
    let end: usize = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Map a SPIR-V Capability enum value to its name string.
///
/// Uses `spirv::Capability::from_u32` when available, falls back to decimal.
fn spirv_capability_name(id: u32) -> String {
    use spirv::Capability;
    // spirv::Capability derives num_traits, but the crate exposes from_u32 via a
    // custom impl. We match the common values directly to avoid depending on num_traits.
    match Capability::from_u32(id) {
        Some(c) => format!("{c:?}"),
        None => format!("Unknown({id})"),
    }
}

/// Convert a byte slice to `&[u32]` (SPIR-V words) via safe byte reads.
///
/// Caller invariant: `bytes.len() % 4 == 0` (guaranteed by spirv-val before this call).
pub(crate) fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    let mut words: Vec<u32> = Vec::with_capacity(bytes.len() / 4);
    let mut i: usize = 0;
    while i + 3 < bytes.len() {
        words.push(u32::from_le_bytes([bytes[i], bytes[i+1], bytes[i+2], bytes[i+3]]));
        i += 4;
    }
    words
}

// ── Request / Response types ──────────────────────────────────────────────────

/// Request for the `compile_variant` tool.
#[derive(Debug, serde::Deserialize)]
pub struct CompileVariantRequest {
    /// Inline source text. Mutually exclusive with `path`.
    #[serde(default)]
    pub source: Option<String>,
    /// Path to an `.axc` source file. Mutually exclusive with `source`.
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// Explicit hole assignments (hole name → concrete value).
    /// Holes absent here retain the source's first candidate.
    #[serde(default)]
    pub assignments: BTreeMap<String, i64>,
}

/// Response from the `compile_variant` tool.
#[derive(Debug, serde::Serialize)]
pub struct CompileVariantResponse {
    /// Base64-encoded (RFC 4648 §4) SPIR-V binary.
    pub spirv_base64: String,
    /// Kernel metadata (entry point, binding plan, workgroup size).
    pub metadata: axc_runtime::KernelMetadata,
    /// SPIR-V capabilities declared by the compiled module.
    pub capabilities: Vec<String>,
    /// SPIR-V extension strings declared by the compiled module.
    pub extensions: Vec<String>,
    /// SPIR-V binary size in bytes.
    pub size_bytes: usize,
}

// ── Handler ───────────────────────────────────────────────────────────────────

/// Handle a `compile_variant` request.
pub(crate) fn handle(req: CompileVariantRequest) -> Result<CompileVariantResponse, McpToolError> {
    let source: String = crate::mcp::dispatch::resolve_source(&req.source, &req.path)?;
    compile_variant_str(&source, &req.assignments)
}

/// Shared helper: compile with assignments, validate, return response.
pub(crate) fn compile_variant_str(
    source: &str,
    assignments: &BTreeMap<String, i64>,
) -> Result<CompileVariantResponse, McpToolError> {
    let (spirv_bytes, metadata) = crate::compile_source_with_assignments(source, assignments)
        .map_err(McpToolError::Compile)?;

    // In-process spirv-val
    validate_spirv(&spirv_bytes)?;

    let words: Vec<u32> = bytes_to_words(&spirv_bytes);
    let (capabilities, extensions) = scan_caps_and_exts(&words);
    let size_bytes: usize = spirv_bytes.len();
    let spirv_base64: String = base64_encode(&spirv_bytes);

    Ok(CompileVariantResponse {
        spirv_base64,
        metadata,
        capabilities,
        extensions,
        size_bytes,
    })
}

/// Run in-process spirv-val on raw SPIR-V bytes.
///
/// Converts bytes to words first (SPIR-V is word-aligned); spirv-tools
/// `Validator::validate` takes `impl AsRef<[u32]>`.
pub(crate) fn validate_spirv(bytes: &[u8]) -> Result<(), McpToolError> {
    use spirv_tools::val::{self, Validator as _};
    if !bytes.len().is_multiple_of(4) {
        return Err(McpToolError::SpirvVal(format!(
            "SPIR-V byte length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    let words: Vec<u32> = bytes_to_words(bytes);
    let validator = val::create(None);
    validator.validate(words, None)
        .map_err(|e| McpToolError::SpirvVal(e.to_string()))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;

    // ── AT-1123: base64 RFC 4648 §4 standard alphabet test vectors ────────────

    #[test]
    fn at_1123_base64_rfc4648_vectors() {
        // RFC 4648 §10 test vectors
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");

        // AT-1123b: every output byte in [A-Za-z0-9+/=]
        let test_cases: &[&[u8]] = &[b"", b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar"];
        for &input in test_cases {
            let encoded: String = base64_encode(input);
            for b in encoded.bytes() {
                assert!(
                    b.is_ascii_alphanumeric() || b == b'+' || b == b'/' || b == b'=',
                    "base64 output byte {b:?} (0x{b:02X}) is not in [A-Za-z0-9+/=]; encoded={encoded:?}"
                );
                // Negative check: no URL-safe chars
                assert_ne!(b, b'-', "URL-safe '-' must not appear in standard base64 output");
                assert_ne!(b, b'_', "URL-safe '_' must not appear in standard base64 output");
            }
        }

        // AT-1123b: alphabet table assertions
        assert_eq!(BASE64_ALPHABET[62], b'+', "index 62 must be '+'");
        assert_eq!(BASE64_ALPHABET[63], b'/', "index 63 must be '/'");
    }

    // ── AT-1124: base64 round-trip with seeded random inputs ─────────────────

    #[test]
    fn at_1124_base64_roundtrip_random() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..1024_usize {
            let len: usize = rng.gen_range(0..=256);
            let input: Vec<u8> = (0..len).map(|_| rng.gen::<u8>()).collect();
            let encoded: String = base64_encode(&input);
            let decoded: Vec<u8> = base64_decode(&encoded)
                .expect("base64_decode must succeed on base64_encode output");
            assert_eq!(decoded, input, "base64 round-trip failed for len={len}");
        }
    }

    // ── AT-1126: scan_caps_and_exts ──────────────────────────────────────────

    #[test]
    fn at_1126_scan_caps_and_exts_order() {
        // Build a synthetic SPIR-V word stream:
        // Header (5 words) + OpCapability Shader (cap 1) + OpCapability StorageBuffer16BitAccess (cap 4433)
        // + OpExtension "SPV_KHR_16bit_storage"
        let mut words: Vec<u32> = vec![
            0x07230203, // magic
            0x00010500, // version 1.5
            0, 0, 0,    // generator, bound, schema
        ];
        // OpCapability Shader (opcode 17, wc 2) | capability 1 (Shader)
        words.push((2 << 16) | 17);
        words.push(1); // Shader
        // OpCapability StorageBuffer16BitAccess / StorageUniformBufferBlock16 (capability 4433)
        words.push((2 << 16) | 17);
        words.push(4433);
        // OpExtension "SPV_KHR_16bit_storage"
        let ext_str: &str = "SPV_KHR_16bit_storage";
        let ext_bytes: Vec<u8> = {
            let mut b: Vec<u8> = ext_str.as_bytes().to_vec();
            // Pad to 4-byte boundary with nulls
            while !b.len().is_multiple_of(4) { b.push(0); }
            b
        };
        let ext_word_count: u16 = 1 + (ext_bytes.len() / 4) as u16;
        words.push(((ext_word_count as u32) << 16) | 10); // OpExtension
        for chunk in ext_bytes.chunks(4) {
            words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        let (caps, exts) = scan_caps_and_exts(&words);
        assert_eq!(caps[0], "Shader", "first capability must be Shader");
        assert!(caps.len() >= 2, "must have at least 2 capabilities");
        assert_eq!(exts, vec!["SPV_KHR_16bit_storage"], "extension must match");
    }

    #[test]
    fn base64_decode_rejects_url_safe() {
        let err = base64_decode("Zm9v-Zm8=");
        assert!(err.is_err(), "URL-safe '-' must be rejected");
        let err2 = base64_decode("Zm9v_Zm8=");
        assert!(err2.is_err(), "URL-safe '_' must be rejected");
    }

    #[test]
    fn compile_empty_kernel_returns_spirv_magic() {
        let src = concat!(
            "@kernel @workgroup(64, 1, 1)\n",
            "fn empty() -> void { return; }\n",
        );
        let assignments: BTreeMap<String, i64> = BTreeMap::new();
        let resp = compile_variant_str(src, &assignments).expect("must compile");

        // Decode and check magic
        let decoded = base64_decode(&resp.spirv_base64).expect("must decode");
        assert_eq!(&decoded[0..4], &[0x03, 0x02, 0x23, 0x07], "SPIR-V magic mismatch");
        assert!(resp.capabilities.contains(&"Shader".to_string()), "Shader capability required");
        assert_eq!(resp.size_bytes, decoded.len(), "size_bytes must match decoded length");
    }
}
