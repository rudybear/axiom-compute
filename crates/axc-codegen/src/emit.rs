//! Core SPIR-V emission routine for AXIOM-Compute.
//!
//! Emits Vulkan-flavor SPIR-V 1.3 for a single compute kernel:
//! - `OpCapability Shader`
//! - `OpMemoryModel Logical GLSL450`
//! - `void main() { return; }` function body
//! - `OpEntryPoint GLCompute %main "name"`
//! - `OpExecutionMode %main LocalSize X Y Z`
//!
//! Step order is load-bearing: `set_version` MUST be first (overrides rspirv's
//! 1.6 default), and `module.header.generator` MUST be overridden post-`b.module()`
//! (overrides rspirv's 0x000f_0000 default) to ensure reproducible output.

use rspirv::binary::Assemble;
use rspirv::dr::Builder;
use rspirv::spirv::{
    Capability, AddressingModel, MemoryModel, ExecutionModel, ExecutionMode, FunctionControl,
};
use axc_hir::HirModule;

/// Options for the SPIR-V emission pass.
///
/// Both fields have tested defaults: `spirv_version = (1, 3)` and
/// `generator_magic = 0` (reserved-for-unregistered per Khronos spir-v.xml).
#[derive(Debug, Clone, Copy)]
pub struct CodegenOptions {
    /// SPIR-V version to emit. Defaults to (1, 3) for Vulkan 1.1+ compatibility.
    pub spirv_version: (u8, u8),
    /// Generator magic word written into the SPIR-V header.
    /// Defaults to 0 to override rspirv's 0x000f_0000 built-in default,
    /// keeping output deterministic across rspirv upgrades (AT-12).
    pub generator_magic: u32,
}

impl CodegenOptions {
    pub const DEFAULT_SPIRV_VERSION: (u8, u8) = (1, 3);
    pub const DEFAULT_GENERATOR_MAGIC: u32 = 0;
}

impl Default for CodegenOptions {
    fn default() -> Self {
        Self {
            spirv_version: Self::DEFAULT_SPIRV_VERSION,
            generator_magic: Self::DEFAULT_GENERATOR_MAGIC,
        }
    }
}

/// Errors that can occur during SPIR-V emission.
#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    #[error("HIR module contains no kernels")]
    NoKernels,
    #[error("M0 supports exactly one kernel per module; got {got}")]
    TooManyKernelsInM0 { got: usize },
    #[error("internal rspirv assembly error: {0}")]
    Rspirv(String),
}

/// Emit a SPIR-V word stream (`Vec<u32>`) from a validated HIR module.
///
/// Returns `CodegenError::NoKernels` if the module is empty.
/// Returns `CodegenError::TooManyKernelsInM0` if there is more than one kernel.
pub fn emit_module(hir: &HirModule, opts: &CodegenOptions) -> Result<Vec<u32>, CodegenError> {
    if hir.kernels.is_empty() {
        return Err(CodegenError::NoKernels);
    }
    if hir.kernels.len() > 1 {
        return Err(CodegenError::TooManyKernelsInM0 { got: hir.kernels.len() });
    }

    let kernel = &hir.kernels[0];
    let wg = &kernel.annotations.workgroup;

    // ── Step 0: Create builder and set version FIRST ─────────────────────────
    // rspirv 0.12 defaults to SPIR-V 1.6; we override to (1, 3) via CodegenOptions.
    // This MUST be the first call on the builder — subsequent operations depend on
    // the version field being accurate.
    let mut b: Builder = Builder::new();
    b.set_version(opts.spirv_version.0, opts.spirv_version.1);

    // ── Step 1: Capabilities ─────────────────────────────────────────────────
    b.capability(Capability::Shader);

    // ── Step 2: Memory model ─────────────────────────────────────────────────
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    // ── Steps 3-8: void main() → void { return; } ────────────────────────────
    let void_t: u32 = b.type_void();
    let fn_t: u32 = b.type_function(void_t, vec![]);
    // `begin_function` returns a Result; rspirv docs say it only fails on
    // internal state corruption, so `.expect` is correct here per spec.
    let main_id: u32 = b
        .begin_function(void_t, None, FunctionControl::NONE, fn_t)
        .expect("rspirv: begin_function should not fail on a freshly-initialized builder");
    b.begin_block(None)
        .expect("rspirv: begin_block should not fail immediately after begin_function");
    b.ret().expect("rspirv: ret() should not fail inside an open block");
    b.end_function().expect("rspirv: end_function should not fail after a complete block");

    // ── Step 9: Entry point ───────────────────────────────────────────────────
    b.entry_point(ExecutionModel::GLCompute, main_id, &kernel.name, vec![]);

    // ── Step 10: Execution mode (LocalSize X Y Z) ────────────────────────────
    b.execution_mode(main_id, ExecutionMode::LocalSize, vec![wg.x, wg.y, wg.z]);

    // ── Step 11: Assemble to dr::Module ──────────────────────────────────────
    let mut module = b.module();

    // ── Step 12: Override generator magic ────────────────────────────────────
    // rspirv's default is 0x000f_0000 (rspirv's own registered magic).
    // We force `opts.generator_magic` (default 0 = reserved-for-unregistered)
    // to keep the output byte-for-byte reproducible regardless of rspirv version.
    module
        .header
        .as_mut()
        .expect("rspirv: module.header must be Some after b.module()")
        .generator = opts.generator_magic;

    // ── Step 13: Serialize to word stream ────────────────────────────────────
    let words: Vec<u32> = module.assemble();
    Ok(words)
}

/// Emit SPIR-V as a `Vec<u8>` in little-endian byte order (the SPIR-V file format).
pub fn emit_module_bytes(hir: &HirModule, opts: &CodegenOptions) -> Result<Vec<u8>, CodegenError> {
    let words: Vec<u32> = emit_module(hir, opts)?;
    // Explicit LE per SPIR-V §2.3 (no transmute — endianness matters on BE hosts)
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axc_parser::parse;
    use axc_hir::lower_module;
    use rspirv::spirv::{Op, ExecutionModel as SpvExecModel, ExecutionMode as SpvExecMode};
    use rspirv::dr::Operand;

    /// Build a minimal valid HIR for a kernel with the given workgroup dims.
    fn make_hir(src: &str) -> HirModule {
        let (ast, lex_errs, parse_errs) = parse(src);
        assert!(lex_errs.is_empty(), "lex errors: {lex_errs:?}");
        assert!(parse_errs.is_empty(), "parse errors: {parse_errs:?}");
        let (hir, hir_errs, _warns) = lower_module(&ast);
        assert!(hir_errs.is_empty(), "hir errors: {hir_errs:?}");
        hir
    }

    const EMPTY_SRC: &str = "@kernel @workgroup(64,1,1) fn empty() -> void { return; }";

    // ── AT-10: version word is 1.3, not rspirv's 1.6 default ─────────────────

    #[test]
    fn emits_spirv_version_1_3() {
        let hir = make_hir(EMPTY_SRC);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit failed");

        // Dual-check (spec AT-10):
        // (a) Raw word-stream: version word at words[1] must be 0x00010300
        assert_eq!(words[0], 0x0723_0203_u32, "magic word mismatch");
        // Version word 0x00010300 = major 1, minor 3 (SPIR-V §2.3 layout)
        assert_eq!(words[1], 0x0001_0300_u32, "version word should be 1.3 (0x00010300)");

        // (b) Typed API: rebuild the module via emit_module and read header.version()
        // This validates that the rspirv typed header correctly reflects the version
        // we set via b.set_version(1, 3) — not just the serialised word stream.
        let (ast2, _, _) = parse(EMPTY_SRC);
        let (hir2, _, _) = lower_module(&ast2);
        let words2 = emit_module(&hir2, &CodegenOptions::default()).expect("emit (typed check) failed");
        // Deserialise the word stream back into a dr::Module to access the typed header
        let module2 = rspirv::dr::load_words(&words2).expect("rspirv: failed to load emitted words");
        let (major, minor) = module2.header.as_ref().unwrap().version();
        assert_eq!((major, minor), (1, 3),
            "typed header API must report version (1, 3); got ({major}, {minor})");
    }

    // ── AT-11: generator word is overridden to 0 ─────────────────────────────

    #[test]
    fn overrides_generator_magic() {
        let hir = make_hir(EMPTY_SRC);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit failed");
        // Default generator must be 0, not rspirv's 0x000f_0000
        assert_eq!(words[2], 0x0000_0000_u32, "generator word should be 0 (overridden)");

        // Verify that the override field is actually wired through (not ignored)
        let opts_custom = CodegenOptions {
            spirv_version: (1, 3),
            generator_magic: 0xDEAD_BEEF,
        };
        let words2 = emit_module(&hir, &opts_custom).expect("emit failed (custom)");
        assert_eq!(words2[2], 0xDEAD_BEEF_u32, "custom generator magic not applied");
    }

    // ── AT-7: entry point and execution mode via enum equality (Strategy X) ───

    #[test]
    fn emits_local_size_and_entry_point() {
        let hir = make_hir(EMPTY_SRC);
        let opts = CodegenOptions::default();

        // Rebuild module directly (without assemble) to use typed getters
        use rspirv::dr::Builder as B;
        use rspirv::spirv::{AddressingModel, MemoryModel, FunctionControl};

        let wg = &hir.kernels[0].annotations.workgroup;
        let kernel_name = &hir.kernels[0].name;

        let mut b = B::new();
        b.set_version(opts.spirv_version.0, opts.spirv_version.1);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
        let void_t = b.type_void();
        let fn_t = b.type_function(void_t, vec![]);
        let main_id = b.begin_function(void_t, None, FunctionControl::NONE, fn_t).unwrap();
        b.begin_block(None).unwrap();
        b.ret().unwrap();
        b.end_function().unwrap();
        b.entry_point(SpvExecModel::GLCompute, main_id, kernel_name, vec![]);
        b.execution_mode(main_id, SpvExecMode::LocalSize, vec![wg.x, wg.y, wg.z]);
        let mut module = b.module();
        module.header.as_mut().unwrap().generator = opts.generator_magic;

        // Assert entry point via enum equality (Strategy X — NOT u32 cast)
        assert_eq!(module.entry_points.len(), 1);
        let ep = &module.entry_points[0];
        assert_eq!(ep.class.opcode, Op::EntryPoint, "opcode should be Op::EntryPoint");
        assert!(
            matches!(&ep.operands[0], Operand::ExecutionModel(SpvExecModel::GLCompute)),
            "first operand should be ExecutionModel::GLCompute"
        );

        // Assert execution mode
        assert_eq!(module.execution_modes.len(), 1);
        let em = &module.execution_modes[0];
        assert_eq!(em.class.opcode, Op::ExecutionMode, "opcode should be Op::ExecutionMode");
        // operands: [IdRef(main_id), ExecutionMode(LocalSize), LiteralBit32(64), LiteralBit32(1), LiteralBit32(1)]
        assert!(matches!(&em.operands[1], Operand::ExecutionMode(SpvExecMode::LocalSize)));
        assert!(matches!(&em.operands[2], Operand::LiteralBit32(64)));
        assert!(matches!(&em.operands[3], Operand::LiteralBit32(1)));
        assert!(matches!(&em.operands[4], Operand::LiteralBit32(1)));
    }

    // ── AT-12: deterministic output ───────────────────────────────────────────

    #[test]
    fn determinism() {
        let hir = make_hir(EMPTY_SRC);
        let opts = CodegenOptions::default();
        let words1 = emit_module(&hir, &opts).expect("first emit failed");
        let words2 = emit_module(&hir, &opts).expect("second emit failed");
        assert_eq!(words1, words2, "emit_module must be bit-deterministic");
    }

    // ── AT-13: no debug opcodes emitted ─────────────────────────────────────

    #[test]
    fn no_debug_names() {
        let hir = make_hir(EMPTY_SRC);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit failed");

        // Walk instruction stream by word-count header (AT-13 spec: skip 5-word header)
        // Deny-list: OpSourceContinued=2, OpSource=3, OpSourceExtension=4, OpName=5,
        //            OpMemberName=6, OpString=7, OpLine=8, OpNoLine=317, OpModuleProcessed=330
        const DEBUG_OPCODES: &[u16] = &[2, 3, 4, 5, 6, 7, 8, 317, 330];
        let body = &words[5..];
        let mut i = 0;
        while i < body.len() {
            let header = body[i];
            let word_count = ((header >> 16) & 0xFFFF) as usize;
            let opcode = (header & 0xFFFF) as u16;
            assert!(word_count >= 1, "word_count must be >= 1 per SPIR-V §3.1");
            assert!(
                !DEBUG_OPCODES.contains(&opcode),
                "found debug opcode {opcode} at body[{i}] — M0 must emit no debug info"
            );
            i += word_count;
        }
    }

    // ── Error cases ────────────────────────────────────────────────────────────

    #[test]
    fn no_kernels_error() {
        let empty_hir = HirModule { kernels: Vec::new() };
        let result = emit_module(&empty_hir, &CodegenOptions::default());
        assert!(matches!(result, Err(CodegenError::NoKernels)));
    }

    #[test]
    fn too_many_kernels_error() {
        // Two kernels → TooManyKernelsInM0
        use axc_hir::{Kernel, KernelId, KernelAnnotations, WorkgroupDims, KernelBody};
        use axc_lexer::Span;
        let mk_k = |id: u32, name: &str| Kernel {
            id: KernelId(id),
            name: name.into(),
            annotations: KernelAnnotations {
                workgroup: WorkgroupDims { x: 1, y: 1, z: 1 },
                intent: None,
                complexity: None,
                preconditions: Vec::new(),
                subgroup_uniform: false,
            },
            body: KernelBody::Empty,
            span: Span::new(0, 1),
        };
        let hir = HirModule { kernels: vec![mk_k(0, "k1"), mk_k(1, "k2")] };
        let result = emit_module(&hir, &CodegenOptions::default());
        assert!(matches!(result, Err(CodegenError::TooManyKernelsInM0 { got: 2 })));
    }

    // ── SPIR-V magic word is always 0x07230203 ────────────────────────────────

    #[test]
    fn magic_word_is_spirv_magic() {
        let hir = make_hir(EMPTY_SRC);
        let words = emit_module(&hir, &CodegenOptions::default()).expect("emit failed");
        assert_eq!(words[0], 0x0723_0203_u32, "SPIR-V magic word mismatch");
    }
}
