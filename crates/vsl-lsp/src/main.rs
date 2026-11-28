use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use tokio::sync::RwLock;
use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::ls_types::{
    CompletionItem, CompletionItemKind, CompletionOptions, CompletionParams, CompletionResponse,
    CompletionTextEdit, Diagnostic as LspDiagnostic, DiagnosticSeverity,
    DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams,
    GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverContents, HoverParams,
    HoverProviderCapability, InitializeParams, InitializeResult, InitializedParams, Location,
    MarkupContent, MarkupKind, MessageType, OneOf, Position, PositionEncodingKind, Range,
    ReferenceParams, RenameParams, SemanticToken, SemanticTokenModifier, SemanticTokenType,
    SemanticTokens, SemanticTokensFullOptions, SemanticTokensLegend, SemanticTokensOptions,
    SemanticTokensParams, SemanticTokensResult, SemanticTokensServerCapabilities,
    ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind, TextEdit, Uri,
    WorkspaceEdit,
};
use tower_lsp_server::{Client, LanguageServer, LspService, Server};
use vsl_diagnostics::{Diagnostic, FileId, Severity, SourceFile, Span};
use vsl_driver::{CompileOptions, VslCompiler, pack_schema_from_pipeline};
use vsl_hir::{
    HirBlock, HirExpr, HirExprKind, HirFunction, HirModule, HirStmt, HirStruct, LayoutKind,
    PackSchema, ScalarKind, Type, TypeArena, TypeId, lower_unit_with_schema,
};
use vsl_mc::{
    BUILTIN_RESOURCES, BuiltinFieldType, BuiltinResourceId, CANONICAL_PASSES, CanonicalPass,
    HardwareCaps, HookStage, RenderTier, WORLD_UBO_LAYOUT,
};
use vsl_pack::{EvalConfig, evaluate_pack_vsl, validate_and_schedule};
use vsl_syntax::ast::ParamMode;
use vsl_syntax::parse_source;

const DEFAULT_ANALYSIS_CACHE_CAPACITY: usize = 64;

const TOK_FUNCTION: u32 = 0;
const TOK_STRUCT: u32 = 1;
const TOK_PARAMETER: u32 = 2;
const TOK_VARIABLE: u32 = 3;
const TOK_PROPERTY: u32 = 4;
const TOK_TYPE: u32 = 5;
const TOK_ENUM_MEMBER: u32 = 6;

const MOD_DECLARATION: u32 = 1 << 0;
const MOD_READONLY: u32 = 1 << 1;

fn semantic_token_types() -> Vec<SemanticTokenType> {
    vec![
        SemanticTokenType::FUNCTION,
        SemanticTokenType::STRUCT,
        SemanticTokenType::PARAMETER,
        SemanticTokenType::VARIABLE,
        SemanticTokenType::PROPERTY,
        SemanticTokenType::TYPE,
        SemanticTokenType::ENUM_MEMBER,
    ]
}

fn semantic_token_modifiers() -> Vec<SemanticTokenModifier> {
    vec![
        SemanticTokenModifier::DECLARATION,
        SemanticTokenModifier::READONLY,
    ]
}

#[derive(Debug, Clone)]
struct OpenDocument {
    uri: Uri,
    text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexedSymbolKind {
    Function,
    Struct,
    Field,
    Param,
    Local,
    Const,
    WorkgroupVar,
    Trait,
}

#[derive(Debug, Clone)]
struct IndexedSymbol {
    name: String,
    kind: IndexedSymbolKind,
    span: Span,
    def_span: Span,
    is_definition: bool,
    type_display: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RawSemanticToken {
    line: u32,
    start_char: u32,
    length: u32,
    token_type: u32,
    modifiers: u32,
}

#[derive(Debug, Clone)]
struct CachedAnalysis {
    source: SourceFile,
    hir: Option<HirModule>,
    pack_schema: Option<PackSchema>,
    symbols: Vec<IndexedSymbol>,
    tokens: Vec<RawSemanticToken>,
}

/// Bounded FIFO/LRU-style analysis cache keyed by normalized document URI.
#[derive(Debug)]
struct AnalysisCache {
    capacity: usize,
    entries: HashMap<String, CachedAnalysis>,
    order: VecDeque<String>,
}

impl AnalysisCache {
    fn new(capacity: usize) -> Self {
        let bounded = capacity.max(1);
        Self {
            capacity: bounded,
            entries: HashMap::with_capacity(bounded),
            order: VecDeque::with_capacity(bounded),
        }
    }

    fn get(&self, key: &str) -> Option<&CachedAnalysis> {
        self.entries.get(key)
    }

    fn insert(&mut self, key: String, value: CachedAnalysis) {
        if self.entries.contains_key(&key) {
            if let Some(pos) = self.order.iter().position(|k| k == &key) {
                self.order.remove(pos);
            }
            self.order.push_back(key.clone());
            self.entries.insert(key, value);
            return;
        }
        while self.entries.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            } else {
                break;
            }
        }
        self.order.push_back(key.clone());
        self.entries.insert(key, value);
    }

    fn remove(&mut self, key: &str) {
        if self.entries.remove(key).is_some()
            && let Some(pos) = self.order.iter().position(|k| k == key)
        {
            self.order.remove(pos);
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

#[derive(Debug)]
struct VslLanguageServer {
    client: Client,
    documents: RwLock<HashMap<String, OpenDocument>>,
    cache: RwLock<AnalysisCache>,
}

impl VslLanguageServer {
    fn new(client: Client) -> Self {
        Self {
            client,
            documents: RwLock::new(HashMap::new()),
            cache: RwLock::new(AnalysisCache::new(DEFAULT_ANALYSIS_CACHE_CAPACITY)),
        }
    }

    async fn publish_file_diagnostics(&self, uri: Uri, text: &str) {
        let norm_key = normalize_uri_key(uri.as_str());
        let open_snapshot: HashMap<String, OpenDocument> = {
            let mut docs = self.documents.write().await;
            docs.insert(
                norm_key.clone(),
                OpenDocument {
                    uri: uri.clone(),
                    text: text.to_owned(),
                },
            );
            docs.clone()
        };

        let (analysis, lsp_diags) = analyze_document(&uri, text, &open_snapshot);
        self.cache.write().await.insert(norm_key.clone(), analysis);
        self.client
            .publish_diagnostics(uri.clone(), lsp_diags, None)
            .await;

        // When `pack.vsl` changes, re-validate all open sibling `.vsl` files in the same directory
        // so updated `PackSchema` parameters, spec constants, and resource slots take effect live.
        if is_pack_vsl_uri(uri.as_str())
            && let Some(pack_dir) =
                uri_to_path(uri.as_str()).and_then(|p| p.parent().map(Path::to_path_buf))
        {
            let siblings: Vec<(String, OpenDocument)> = open_snapshot
                .iter()
                .filter(|(k, doc)| {
                    k.as_str() != norm_key.as_str()
                        && !is_pack_vsl_uri(doc.uri.as_str())
                        && uri_to_path(doc.uri.as_str())
                            .and_then(|p| p.parent().map(Path::to_path_buf))
                            .as_deref()
                            == Some(pack_dir.as_path())
                })
                .map(|(k, doc)| (k.clone(), doc.clone()))
                .collect();

            for (sib_key, sib_doc) in siblings {
                let (sib_analysis, sib_diags) =
                    analyze_document(&sib_doc.uri, &sib_doc.text, &open_snapshot);
                self.cache.write().await.insert(sib_key, sib_analysis);
                self.client
                    .publish_diagnostics(sib_doc.uri, sib_diags, None)
                    .await;
            }
        }
    }

    async fn ensure_cached(&self, uri: &Uri) -> Option<CachedAnalysis> {
        let norm_key = normalize_uri_key(uri.as_str());
        if let Some(cached) = self.cache.read().await.get(&norm_key).cloned() {
            return Some(cached);
        }
        let open_snapshot = self.documents.read().await.clone();
        let doc = open_snapshot.get(&norm_key)?;
        let (analysis, _) = analyze_document(&doc.uri, &doc.text, &open_snapshot);
        self.cache.write().await.insert(norm_key, analysis.clone());
        Some(analysis)
    }
}

/// Normalizes a URI or filesystem path string into a canonical lookup key across Windows and
/// POSIX: percent-decodes `%XX` escapes, converts backslashes to `/`, and lowercases Windows
/// drive letters (`file:///C:/...` and `file:///c%3A/...` both become `file:///c:/...`).
fn normalize_uri_key(uri: &str) -> String {
    let decoded = percent_decode(uri);
    let mut normalized = decoded.replace('\\', "/");

    if let Some(rest) = normalized.strip_prefix("file:///") {
        let bytes = rest.as_bytes();
        if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
            let drive = (bytes[0] as char).to_ascii_lowercase();
            normalized = format!("file:///{drive}{}", &rest[1..]);
        }
    } else {
        let bytes = normalized.as_bytes();
        if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
            let drive = (bytes[0] as char).to_ascii_lowercase();
            normalized = format!("{drive}{}", &normalized[1..]);
        }
    }

    normalized
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(hi), Some(lo)) = (hex_nibble(bytes[i + 1]), hex_nibble(bytes[i + 2]))
        {
            out.push((hi << 4) | lo);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| input.to_owned())
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Returns `true` only when the normalized URI basename is strictly `pack.vsl` (never `unpack.vsl`).
fn is_pack_vsl_uri(uri: &str) -> bool {
    let without_query = uri.split(['?', '#']).next().unwrap_or(uri);
    let norm = normalize_uri_key(without_query);
    norm == "pack.vsl" || norm.ends_with("/pack.vsl")
}

/// Converts a `file://` URI (or plain path) into an OS-independent [`PathBuf`].
fn uri_to_path(uri: &str) -> Option<PathBuf> {
    let without_query = uri.split(['?', '#']).next().unwrap_or(uri);
    let norm = normalize_uri_key(without_query);
    if let Some(rest) = norm.strip_prefix("file://") {
        let bytes = rest.as_bytes();
        // On Windows URIs (`file:///c:/foo`), strip the leading `/` before `<drive>:`.
        if bytes.len() >= 3
            && bytes[0] == b'/'
            && bytes[1].is_ascii_alphabetic()
            && bytes[2] == b':'
        {
            return Some(PathBuf::from(&rest[1..]));
        }
        return Some(PathBuf::from(rest));
    }
    if norm.contains("://") {
        return None;
    }
    Some(PathBuf::from(norm))
}

/// Discovers sibling `pack.vsl` and `.vsl` files in `doc_dir`, preferring in-memory editor buffers.
fn discover_workspace_context(
    target_key: &str,
    target_path: Option<&Path>,
    open_docs: &HashMap<String, OpenDocument>,
) -> (Option<PackSchema>, Vec<(String, String)>) {
    let Some(doc_dir) = target_path.and_then(Path::parent) else {
        return (None, Vec::new());
    };
    if !doc_dir.is_dir() {
        return (None, Vec::new());
    }

    let mut pack_schema = None;
    let pack_path = doc_dir.join("pack.vsl");
    let pack_text = find_open_text_for_path(&pack_path, open_docs)
        .or_else(|| fs::read_to_string(&pack_path).ok());
    if let Some(src) = pack_text {
        let parsed = parse_source(FileId(0), &src);
        if !parsed.has_errors() {
            let (ir, eval_diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
            if !eval_diags.has_errors() {
                pack_schema = Some(pack_schema_from_pipeline(&ir));
            }
        }
    }

    let mut sibling_paths = Vec::new();
    if let Ok(entries) = fs::read_dir(doc_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let is_vsl = path
                .extension()
                .and_then(std::ffi::OsStr::to_str)
                .is_some_and(|ext| ext.eq_ignore_ascii_case("vsl"));
            let is_pack = path
                .file_name()
                .and_then(std::ffi::OsStr::to_str)
                .is_some_and(|name| name.eq_ignore_ascii_case("pack.vsl"));
            if is_vsl && !is_pack {
                sibling_paths.push(path);
            }
        }
    }
    sibling_paths.sort();

    let mut siblings = Vec::new();
    for path in sibling_paths {
        let path_key = normalize_uri_key(&path.display().to_string());
        if target_key.ends_with(&path_key) || path_key.ends_with(target_key) {
            continue;
        }
        let content =
            find_open_text_for_path(&path, open_docs).or_else(|| fs::read_to_string(&path).ok());
        if let Some(src) = content {
            let label = path
                .file_name()
                .and_then(std::ffi::OsStr::to_str)
                .unwrap_or("sibling.vsl")
                .to_owned();
            siblings.push((label, src));
        }
    }

    (pack_schema, siblings)
}

fn find_open_text_for_path(
    path: &Path,
    open_docs: &HashMap<String, OpenDocument>,
) -> Option<String> {
    let target_norm = normalize_uri_key(&path.display().to_string());
    open_docs.values().find_map(|doc| {
        let doc_path = uri_to_path(doc.uri.as_str())?;
        let doc_norm = normalize_uri_key(&doc_path.display().to_string());
        (doc_norm == target_norm).then(|| doc.text.clone())
    })
}

fn analyze_document(
    uri: &Uri,
    text: &str,
    open_docs: &HashMap<String, OpenDocument>,
) -> (CachedAnalysis, Vec<LspDiagnostic>) {
    let file_id = FileId(0);
    let source_file = SourceFile::new(file_id, uri.as_str(), text);
    let parsed = parse_source(file_id, text);
    let mut all_diags = parsed.diagnostics;

    if is_pack_vsl_uri(uri.as_str()) {
        let (ir, eval_diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        all_diags.extend(eval_diags.into_vec());
        if let Err(sched_errors) = validate_and_schedule(&ir) {
            for msg in sched_errors {
                all_diags.push(Diagnostic::error("E0450", msg, Span::new(file_id, 0, 0)));
            }
        }
        let schema = pack_schema_from_pipeline(&ir);
        let lsp_diags = all_diags
            .iter()
            .map(|d| to_lsp_diagnostic(&source_file, d))
            .collect();
        let analysis = CachedAnalysis {
            source: source_file,
            hir: None,
            pack_schema: Some(schema),
            symbols: Vec::new(),
            tokens: Vec::new(),
        };
        return (analysis, lsp_diags);
    }

    let norm_key = normalize_uri_key(uri.as_str());
    let doc_path = uri_to_path(uri.as_str());
    let (pack_schema, sibling_files) =
        discover_workspace_context(&norm_key, doc_path.as_deref(), open_docs);

    // Lower current file with full capabilities across all tiers so editor hover/goto/rename
    // and diagnostics cover tier-gated functions without pruning.
    let (hir, hir_diags) = lower_unit_with_schema(
        &parsed.unit,
        RenderTier::Tier3PathTracing,
        HardwareCaps::all(),
        pack_schema.as_ref(),
    );
    all_diags.extend(hir_diags.into_vec());

    // When sibling `.vsl` files exist in the same pack directory, also run multi-file project
    // linking via `vsl-driver` to surface cross-module visibility/collision diagnostics (`E0500`–`E0503`).
    if !sibling_files.is_empty() && !all_diags.iter().any(Diagnostic::is_error) {
        let mut project_files = Vec::with_capacity(1 + sibling_files.len());
        let self_label = doc_path
            .as_deref()
            .and_then(Path::file_name)
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or("current.vsl")
            .to_owned();
        project_files.push((self_label, text.to_owned()));
        project_files.extend(sibling_files);

        let mut opts = CompileOptions::for_tier(RenderTier::Tier3PathTracing);
        opts.active_caps = HardwareCaps::all();
        opts.pack_schema.clone_from(&pack_schema);

        let mut compiler = VslCompiler::new();
        let proj_out = compiler.compile_project(&project_files, &opts);
        for diag in proj_out.diagnostics {
            if diag.primary_span.file == FileId(0) || diag.primary_span.file == FileId::SYNTHETIC {
                all_diags.push(diag);
            }
        }
    }

    let symbols = index_hir_symbols(&hir, &source_file);
    let tokens = build_semantic_tokens(&symbols, &source_file);
    let lsp_diags = all_diags
        .iter()
        .map(|d| to_lsp_diagnostic(&source_file, d))
        .collect();

    let analysis = CachedAnalysis {
        source: source_file,
        hir: Some(hir),
        pack_schema,
        symbols,
        tokens,
    };
    (analysis, lsp_diags)
}

fn format_scalar_kind(scalar: ScalarKind) -> &'static str {
    match scalar {
        ScalarKind::Bool => "bool",
        ScalarKind::I32 => "i32",
        ScalarKind::U32 => "u32",
        ScalarKind::I64 => "i64",
        ScalarKind::U64 => "u64",
        ScalarKind::F32 => "f32",
        ScalarKind::F16 => "f16",
    }
}

fn format_hir_type(types: &TypeArena, ty_id: TypeId) -> String {
    match types.get(ty_id) {
        Type::Void => "()".to_owned(),
        Type::Scalar(s) => format_scalar_kind(*s).to_owned(),
        Type::Vector { scalar, len } => match scalar {
            ScalarKind::F32 => format!("vec{len}"),
            ScalarKind::I32 => format!("ivec{len}"),
            ScalarKind::U32 => format!("uvec{len}"),
            ScalarKind::Bool => format!("bvec{len}"),
            ScalarKind::F16 => format!("vec{len}<f16>"),
            ScalarKind::I64 => format!("vec{len}<i64>"),
            ScalarKind::U64 => format!("vec{len}<u64>"),
        },
        Type::Matrix { scalar, cols, rows } => {
            if *scalar == ScalarKind::F32 && cols == rows {
                format!("mat{cols}")
            } else if *scalar == ScalarKind::F32 {
                format!("mat{cols}x{rows}")
            } else {
                format!("mat{cols}x{rows}<{}>", format_scalar_kind(*scalar))
            }
        }
        Type::Array { element, len } => format!("[{}; {len}]", format_hir_type(types, *element)),
        Type::Pointer {
            pointee,
            mutable,
            align,
        } => {
            let kw = if *mutable { "ptr_mut" } else { "ptr" };
            format!(
                "{kw}<{}, align = {align}>",
                format_hir_type(types, *pointee)
            )
        }
        Type::Struct { name, .. } => name.to_string(),
        Type::Texture2D {
            sample_scalar,
            unsized_array,
        } => {
            let suffix = if *unsized_array { "[]" } else { "" };
            format!("Texture2D<{}>{suffix}", format_scalar_kind(*sample_scalar))
        }
        Type::Texture2DArray { sample_scalar } => {
            format!("Texture2DArray<{}>", format_scalar_kind(*sample_scalar))
        }
        Type::Texture3D { sample_scalar } => {
            format!("Texture3D<{}>", format_scalar_kind(*sample_scalar))
        }
        Type::TextureCube { sample_scalar } => {
            format!("TextureCube<{}>", format_scalar_kind(*sample_scalar))
        }
        Type::RwImage2D(fmt) => format!("RWImage2D<{fmt:?}>"),
        Type::RwImage2DArray(fmt) => format!("RWImage2DArray<{fmt:?}>"),
        Type::RwImage3D(fmt) => format!("RWImage3D<{fmt:?}>"),
        Type::Sampler => "Sampler".to_owned(),
        Type::SamplerShadow => "SamplerShadow".to_owned(),
        Type::Buffer { element } => format!("Buffer<{}>", format_hir_type(types, *element)),
        Type::RwBuffer { element } => format!("RWBuffer<{}>", format_hir_type(types, *element)),
        Type::AccelStruct => "AccelStruct".to_owned(),
        Type::Texture1D { sample_scalar } => {
            format!("Texture1D<{}>", format_scalar_kind(*sample_scalar))
        }
        Type::Texture2DMS { sample_scalar } => {
            format!("Texture2DMS<{}>", format_scalar_kind(*sample_scalar))
        }
        Type::BufferView { sample_scalar } => {
            format!("BufferView<{}>", format_scalar_kind(*sample_scalar))
        }
        Type::SubpassInput { sample_scalar, ms } => {
            if *ms {
                format!("SubpassInputMS<{}>", format_scalar_kind(*sample_scalar))
            } else {
                format!("SubpassInput<{}>", format_scalar_kind(*sample_scalar))
            }
        }
        Type::RwImage1D(fmt) => format!("RWImage1D<{fmt:?}>"),
        Type::RwBufferView(fmt) => format!("RWBufferView<{fmt:?}>"),
        Type::RayQuery => "RayQuery".to_owned(),
        Type::HitObject => "HitObject".to_owned(),
        Type::CoopMatrix {
            scalar, rows, cols, ..
        } => {
            let s = match scalar {
                vsl_hir::CoopMatScalar::F16 => "f16",
                vsl_hir::CoopMatScalar::Bf16 => "bf16",
                vsl_hir::CoopMatScalar::F32 => "f32",
                vsl_hir::CoopMatScalar::I8 => "i8",
                vsl_hir::CoopMatScalar::U8 => "u8",
                vsl_hir::CoopMatScalar::I32 => "i32",
                vsl_hir::CoopMatScalar::U32 => "u32",
            };
            format!("CoopMat<{s}, {rows}, {cols}>")
        }
        Type::TypeParam(name) => name.to_string(),
    }
}

fn format_builtin_field_type(ty: BuiltinFieldType) -> String {
    match ty {
        BuiltinFieldType::Bool => "bool".to_owned(),
        BuiltinFieldType::I32 => "i32".to_owned(),
        BuiltinFieldType::U32 => "u32".to_owned(),
        BuiltinFieldType::I64 => "i64".to_owned(),
        BuiltinFieldType::U64 => "u64".to_owned(),
        BuiltinFieldType::F32 => "f32".to_owned(),
        BuiltinFieldType::Vec2 => "vec2".to_owned(),
        BuiltinFieldType::Vec3 => "vec3".to_owned(),
        BuiltinFieldType::Vec4 => "vec4".to_owned(),
        BuiltinFieldType::UVec2 => "uvec2".to_owned(),
        BuiltinFieldType::UVec3 => "uvec3".to_owned(),
        BuiltinFieldType::UVec4 => "uvec4".to_owned(),
        BuiltinFieldType::Mat4 => "mat4".to_owned(),
        BuiltinFieldType::RuntimeArrayStruct(name) => format!("{name}[]"),
    }
}

fn format_function_signature(hir: &HirModule, func: &HirFunction) -> String {
    let mut header = String::new();
    if let Some(stage) = func.stage {
        if let Some([x, y, z]) = func.workgroup_size {
            let _ = writeln!(header, "@stage({stage:?}({x}, {y}, {z}))");
        } else {
            let _ = writeln!(header, "@stage({stage:?})");
        }
    }
    if let Some(hook) = func.hook {
        let _ = writeln!(header, "@hook({})", hook.as_str());
    }
    let vis = if func.is_pub { "pub " } else { "" };
    let params: Vec<String> = func
        .params
        .iter()
        .map(|p| {
            let mode = match p.mode {
                ParamMode::In => "",
                ParamMode::InOut => "inout ",
                ParamMode::Out => "out ",
                ParamMode::Mut => "mut ",
            };
            let ty_str = p.ty.map_or_else(
                || "unknown".to_owned(),
                |id| format_hir_type(&hir.types, id),
            );
            format!("{mode}{}: {ty_str}", p.name)
        })
        .collect();
    let ret = func
        .return_ty
        .map(|id| format!(" -> {}", format_hir_type(&hir.types, id)))
        .unwrap_or_default();
    format!("{header}{vis}fn {}({}){ret}", func.name, params.join(", "))
}

fn format_struct_hover(hir: &HirModule, st: &HirStruct) -> String {
    let layout_name = match st.layout {
        LayoutKind::Scalar => "scalar",
        LayoutKind::Std430 => "std430",
        LayoutKind::Std140 => "std140",
    };
    let mut md = format!("```vsl\n#[layout({layout_name})]\nstruct {} {{\n", st.name);
    for (idx, field) in st.fields.iter().enumerate() {
        let ty_str = format_hir_type(&hir.types, field.ty);
        let offset_comment = st
            .computed_layout
            .as_ref()
            .and_then(|l| l.field_offsets.get(idx))
            .map(|(_, off, sz)| format!(" // offset {off}, size {sz}"))
            .unwrap_or_default();
        let _ = writeln!(md, "    {}: {ty_str},{offset_comment}", field.name);
    }
    md.push_str("}\n```");
    if let Some(ref layout) = st.computed_layout {
        let _ = write!(
            md,
            "\n\nSize: `{}` bytes, Alignment: `{}` bytes",
            layout.size, layout.align
        );
    }
    md
}

/// Narrow a declaration span to the exact byte span of `ident` inside `source_file` when possible.
fn find_ident_span_within(source: &SourceFile, enclosing: Span, ident: &str) -> Span {
    if ident.is_empty() || enclosing.file == FileId::SYNTHETIC {
        return enclosing;
    }
    let slice = source.slice(enclosing);
    let bytes = slice.as_bytes();
    let ident_bytes = ident.as_bytes();
    let is_ident_byte = |b: u8| b.is_ascii_alphanumeric() || b == b'_';

    let mut offset = 0usize;
    while offset + ident_bytes.len() <= bytes.len() {
        if &bytes[offset..offset + ident_bytes.len()] == ident_bytes {
            let left_ok = offset == 0 || !is_ident_byte(bytes[offset - 1]);
            let right_ok = offset + ident_bytes.len() == bytes.len()
                || !is_ident_byte(bytes[offset + ident_bytes.len()]);
            if left_ok && right_ok {
                let start = enclosing.start.saturating_add(offset as u32);
                let end = start.saturating_add(ident_bytes.len() as u32);
                return Span::new(enclosing.file, start, end);
            }
        }
        offset += 1;
    }
    enclosing
}

fn index_hir_symbols(hir: &HirModule, source: &SourceFile) -> Vec<IndexedSymbol> {
    let mut out = Vec::new();
    let mut defs: HashMap<String, (IndexedSymbolKind, Span, String)> = HashMap::new();

    for (_, st) in hir.structs.iter() {
        let name_span = find_ident_span_within(source, st.span, st.name.as_str());
        let ty_str = format!("struct {}", st.name);
        defs.insert(
            st.name.to_string(),
            (IndexedSymbolKind::Struct, name_span, ty_str.clone()),
        );
        out.push(IndexedSymbol {
            name: st.name.to_string(),
            kind: IndexedSymbolKind::Struct,
            span: name_span,
            def_span: name_span,
            is_definition: true,
            type_display: ty_str,
        });
        for field in &st.fields {
            let f_span = find_ident_span_within(source, field.span, field.name.as_str());
            let f_ty = format_hir_type(&hir.types, field.ty);
            out.push(IndexedSymbol {
                name: field.name.to_string(),
                kind: IndexedSymbolKind::Field,
                span: f_span,
                def_span: f_span,
                is_definition: true,
                type_display: f_ty,
            });
        }
    }

    for tr in &hir.traits {
        let t_span = find_ident_span_within(source, tr.span, tr.name.as_str());
        let ty_str = format!("trait {}", tr.name);
        defs.insert(
            tr.name.to_string(),
            (IndexedSymbolKind::Trait, t_span, ty_str.clone()),
        );
        out.push(IndexedSymbol {
            name: tr.name.to_string(),
            kind: IndexedSymbolKind::Trait,
            span: t_span,
            def_span: t_span,
            is_definition: true,
            type_display: ty_str,
        });
    }

    for c in &hir.consts {
        let c_span = find_ident_span_within(source, c.span, c.name.as_str());
        let ty_str = format_hir_type(&hir.types, c.ty);
        defs.insert(
            c.name.to_string(),
            (IndexedSymbolKind::Const, c_span, ty_str.clone()),
        );
        out.push(IndexedSymbol {
            name: c.name.to_string(),
            kind: IndexedSymbolKind::Const,
            span: c_span,
            def_span: c_span,
            is_definition: true,
            type_display: ty_str,
        });
    }

    for wg in &hir.workgroup_vars {
        let w_span = find_ident_span_within(source, wg.span, wg.name.as_str());
        let ty_str = format_hir_type(&hir.types, wg.ty);
        defs.insert(
            wg.name.to_string(),
            (IndexedSymbolKind::WorkgroupVar, w_span, ty_str.clone()),
        );
        out.push(IndexedSymbol {
            name: wg.name.to_string(),
            kind: IndexedSymbolKind::WorkgroupVar,
            span: w_span,
            def_span: w_span,
            is_definition: true,
            type_display: ty_str,
        });
    }

    for (_, func) in hir.functions.iter() {
        let fn_span = find_ident_span_within(source, func.span, func.name.as_str());
        let sig = format_function_signature(hir, func);
        defs.insert(
            func.name.to_string(),
            (IndexedSymbolKind::Function, fn_span, sig.clone()),
        );
        out.push(IndexedSymbol {
            name: func.name.to_string(),
            kind: IndexedSymbolKind::Function,
            span: fn_span,
            def_span: fn_span,
            is_definition: true,
            type_display: sig,
        });
    }

    for (_, func) in hir.functions.iter() {
        let mut local_defs = defs.clone();
        for param in &func.params {
            let p_span = find_ident_span_within(source, param.span, param.name.as_str());
            let p_ty = param.ty.map_or_else(
                || "unknown".to_owned(),
                |id| format_hir_type(&hir.types, id),
            );
            local_defs.insert(
                param.name.to_string(),
                (IndexedSymbolKind::Param, p_span, p_ty.clone()),
            );
            out.push(IndexedSymbol {
                name: param.name.to_string(),
                kind: IndexedSymbolKind::Param,
                span: p_span,
                def_span: p_span,
                is_definition: true,
                type_display: p_ty,
            });
        }
        if let Some(ref body) = func.body_hir {
            index_hir_block(hir, source, body, &mut local_defs, &mut out);
        }
    }

    out
}

fn index_hir_block(
    hir: &HirModule,
    source: &SourceFile,
    block: &HirBlock,
    scope: &mut HashMap<String, (IndexedSymbolKind, Span, String)>,
    out: &mut Vec<IndexedSymbol>,
) {
    let saved_scope = scope.clone();
    for stmt in &block.stmts {
        index_hir_stmt(hir, source, stmt, scope, out);
    }
    if let Some(ref tail) = block.tail_expr {
        index_hir_expr(hir, source, tail, scope, out);
    }
    *scope = saved_scope;
}

fn index_hir_stmt(
    hir: &HirModule,
    source: &SourceFile,
    stmt: &HirStmt,
    scope: &mut HashMap<String, (IndexedSymbolKind, Span, String)>,
    out: &mut Vec<IndexedSymbol>,
) {
    match stmt {
        HirStmt::Let {
            name,
            ty,
            init,
            span,
            ..
        } => {
            if let Some(init_expr) = init {
                index_hir_expr(hir, source, init_expr, scope, out);
            }
            let var_span = find_ident_span_within(source, *span, name.as_str());
            let ty_str = format_hir_type(&hir.types, *ty);
            scope.insert(
                name.to_string(),
                (IndexedSymbolKind::Local, var_span, ty_str.clone()),
            );
            out.push(IndexedSymbol {
                name: name.to_string(),
                kind: IndexedSymbolKind::Local,
                span: var_span,
                def_span: var_span,
                is_definition: true,
                type_display: ty_str,
            });
        }
        HirStmt::Return { value, .. } => {
            if let Some(expr) = value {
                index_hir_expr(hir, source, expr, scope, out);
            }
        }
        HirStmt::For {
            binding,
            binding_ty,
            iter,
            body,
            span,
            ..
        } => {
            index_hir_expr(hir, source, iter, scope, out);
            let saved = scope.clone();
            let b_span = find_ident_span_within(source, *span, binding.as_str());
            let ty_str = format_hir_type(&hir.types, *binding_ty);
            scope.insert(
                binding.to_string(),
                (IndexedSymbolKind::Local, b_span, ty_str.clone()),
            );
            out.push(IndexedSymbol {
                name: binding.to_string(),
                kind: IndexedSymbolKind::Local,
                span: b_span,
                def_span: b_span,
                is_definition: true,
                type_display: ty_str,
            });
            index_hir_block(hir, source, body, scope, out);
            *scope = saved;
        }
        HirStmt::While { cond, body, .. } => {
            index_hir_expr(hir, source, cond, scope, out);
            index_hir_block(hir, source, body, scope, out);
        }
        HirStmt::Loop { body, .. } => {
            index_hir_block(hir, source, body, scope, out);
        }
        HirStmt::Switch {
            selector,
            cases,
            default,
            ..
        } => {
            index_hir_expr(hir, source, selector, scope, out);
            for case in cases {
                index_hir_expr(hir, source, &case.value, scope, out);
                index_hir_block(hir, source, &case.body, scope, out);
            }
            if let Some(def_block) = default {
                index_hir_block(hir, source, def_block, scope, out);
            }
        }
        HirStmt::Expr { expr, .. } => index_hir_expr(hir, source, expr, scope, out),
        HirStmt::Break(_) | HirStmt::Continue(_) => {}
    }
}

fn index_hir_expr(
    hir: &HirModule,
    source: &SourceFile,
    expr: &HirExpr,
    scope: &mut HashMap<String, (IndexedSymbolKind, Span, String)>,
    out: &mut Vec<IndexedSymbol>,
) {
    match &expr.kind {
        HirExprKind::Path { path, .. } => {
            let short = path.rsplit("::").next().unwrap_or(path.as_str());
            if let Some((kind, def_span, ty_str)) = scope.get(short) {
                let ref_span = find_ident_span_within(source, expr.span, short);
                out.push(IndexedSymbol {
                    name: short.to_owned(),
                    kind: *kind,
                    span: ref_span,
                    def_span: *def_span,
                    is_definition: false,
                    type_display: ty_str.clone(),
                });
            }
        }
        HirExprKind::ArrayLit { elements } => {
            for elem in elements {
                index_hir_expr(hir, source, elem, scope, out);
            }
        }
        HirExprKind::StructLit {
            struct_name,
            fields,
        } => {
            if let Some((kind, def_span, ty_str)) = scope.get(struct_name.as_str()) {
                let ref_span = find_ident_span_within(source, expr.span, struct_name.as_str());
                out.push(IndexedSymbol {
                    name: struct_name.to_string(),
                    kind: *kind,
                    span: ref_span,
                    def_span: *def_span,
                    is_definition: false,
                    type_display: ty_str.clone(),
                });
            }
            for (_, val) in fields {
                index_hir_expr(hir, source, val, scope, out);
            }
        }
        HirExprKind::Unary { operand, .. } => {
            index_hir_expr(hir, source, operand, scope, out);
        }
        HirExprKind::Binary { lhs, rhs, .. } => {
            index_hir_expr(hir, source, lhs, scope, out);
            index_hir_expr(hir, source, rhs, scope, out);
        }
        HirExprKind::Call { callee, args, .. } => {
            index_hir_expr(hir, source, callee, scope, out);
            for arg in args {
                index_hir_expr(hir, source, &arg.value, scope, out);
            }
        }
        HirExprKind::MethodCall {
            receiver,
            method,
            resolved_fn,
            args,
            ..
        } => {
            index_hir_expr(hir, source, receiver, scope, out);
            if let Some(fn_id) = resolved_fn {
                let target_fn = &hir.functions[*fn_id];
                let def_span =
                    find_ident_span_within(source, target_fn.span, target_fn.name.as_str());
                let m_span = find_ident_span_within(source, expr.span, method.as_str());
                out.push(IndexedSymbol {
                    name: method.to_string(),
                    kind: IndexedSymbolKind::Function,
                    span: m_span,
                    def_span,
                    is_definition: false,
                    type_display: format_function_signature(hir, target_fn),
                });
            }
            for arg in args {
                index_hir_expr(hir, source, &arg.value, scope, out);
            }
        }
        HirExprKind::Field {
            receiver, field, ..
        } => {
            index_hir_expr(hir, source, receiver, scope, out);
            if let Type::Struct { name: st_name, .. } = hir.types.get(receiver.ty)
                && let Some(st) = hir.find_struct(st_name.as_str())
                && let Some(sf) = st.fields.iter().find(|f| f.name == *field)
            {
                let def_span = find_ident_span_within(source, sf.span, sf.name.as_str());
                let ref_span = find_ident_span_within(source, expr.span, field.as_str());
                out.push(IndexedSymbol {
                    name: field.to_string(),
                    kind: IndexedSymbolKind::Field,
                    span: ref_span,
                    def_span,
                    is_definition: false,
                    type_display: format_hir_type(&hir.types, sf.ty),
                });
            }
        }
        HirExprKind::Index { receiver, index } => {
            index_hir_expr(hir, source, receiver, scope, out);
            index_hir_expr(hir, source, index, scope, out);
        }
        HirExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            index_hir_expr(hir, source, cond, scope, out);
            index_hir_block(hir, source, then_branch, scope, out);
            if let Some(eb) = else_branch {
                index_hir_expr(hir, source, eb, scope, out);
            }
        }
        HirExprKind::Block(b) => index_hir_block(hir, source, b, scope, out),
        HirExprKind::Swizzle { base, .. } => index_hir_expr(hir, source, base, scope, out),
        HirExprKind::IntLit(_)
        | HirExprKind::FloatLit(_)
        | HirExprKind::BoolLit(_)
        | HirExprKind::StringLit(_)
        | HirExprKind::NoneLit => {}
    }
}

fn build_semantic_tokens(symbols: &[IndexedSymbol], source: &SourceFile) -> Vec<RawSemanticToken> {
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    let mut raw = Vec::with_capacity(symbols.len());

    for sym in symbols {
        if sym.span.file == FileId::SYNTHETIC || sym.span.end <= sym.span.start {
            continue;
        }
        let start = source.line_col_utf16(sym.span.start);
        let end = source.line_col_utf16(sym.span.end);
        if start.line != end.line || end.col <= start.col {
            continue;
        }
        if !seen.insert((start.line, start.col)) {
            continue;
        }
        let token_type = match sym.kind {
            IndexedSymbolKind::Function => TOK_FUNCTION,
            IndexedSymbolKind::Struct => TOK_STRUCT,
            IndexedSymbolKind::Field => TOK_PROPERTY,
            IndexedSymbolKind::Param => TOK_PARAMETER,
            IndexedSymbolKind::Local | IndexedSymbolKind::WorkgroupVar => TOK_VARIABLE,
            IndexedSymbolKind::Const => TOK_ENUM_MEMBER,
            IndexedSymbolKind::Trait => TOK_TYPE,
        };
        let mut modifiers = 0u32;
        if sym.is_definition {
            modifiers |= MOD_DECLARATION;
        }
        if sym.kind == IndexedSymbolKind::Const {
            modifiers |= MOD_READONLY;
        }
        raw.push(RawSemanticToken {
            line: start.line,
            start_char: start.col,
            length: end.col - start.col,
            token_type,
            modifiers,
        });
    }

    raw.sort_by_key(|t| (t.line, t.start_char));
    raw
}

fn encode_semantic_tokens(raw: &[RawSemanticToken]) -> Vec<SemanticToken> {
    let mut data = Vec::with_capacity(raw.len());
    let mut prev_line = 0u32;
    let mut prev_start = 0u32;
    for tok in raw {
        let delta_line = tok.line.saturating_sub(prev_line);
        let delta_start = if delta_line == 0 {
            tok.start_char.saturating_sub(prev_start)
        } else {
            tok.start_char
        };
        data.push(SemanticToken {
            delta_line,
            delta_start,
            length: tok.length,
            token_type: tok.token_type,
            token_modifiers_bitset: tok.modifiers,
        });
        prev_line = tok.line;
        prev_start = tok.start_char;
    }
    data
}

fn position_to_byte_offset(source: &SourceFile, pos: Position) -> Option<u32> {
    let mut cur_line = 0u32;
    let mut line_start = 0usize;
    for line in source.text.split_inclusive('\n') {
        if cur_line == pos.line {
            let trimmed = line.trim_end_matches(['\r', '\n']);
            let col_bytes = utf16_col_to_byte_offset(trimmed, pos.character);
            return Some((line_start + col_bytes) as u32);
        }
        line_start += line.len();
        cur_line += 1;
    }
    if cur_line == pos.line {
        return Some(source.text.len() as u32);
    }
    None
}

fn symbol_at_offset(symbols: &[IndexedSymbol], byte_offset: u32) -> Option<&IndexedSymbol> {
    // Prefer the narrowest span containing `byte_offset`.
    symbols
        .iter()
        .filter(|s| {
            s.span.start <= byte_offset && byte_offset <= s.span.end && s.span.start < s.span.end
        })
        .min_by_key(|s| s.span.end - s.span.start)
}

fn span_to_lsp_range(source: &SourceFile, span: Span) -> Range {
    let start = source.line_col_utf16(span.start);
    let end = source.line_col_utf16(span.end);
    Range {
        start: Position {
            line: start.line,
            character: start.col,
        },
        end: Position {
            line: end.line,
            character: end.col,
        },
    }
}

fn to_lsp_diagnostic(file: &SourceFile, diag: &Diagnostic) -> LspDiagnostic {
    let severity = match diag.severity {
        Severity::Error => DiagnosticSeverity::ERROR,
        Severity::Warning => DiagnosticSeverity::WARNING,
        Severity::Advice => DiagnosticSeverity::HINT,
    };

    LspDiagnostic {
        range: span_to_lsp_range(file, diag.primary_span),
        severity: Some(severity),
        source: Some("vsl2".to_owned()),
        message: format!("[{}] {}", diag.code, diag.message),
        ..LspDiagnostic::default()
    }
}

/// Converts an LSP UTF-16 character column into a UTF-8 byte offset within `line`.
fn utf16_col_to_byte_offset(line: &str, utf16_col: u32) -> usize {
    let mut utf16_seen = 0u32;
    for (byte_idx, ch) in line.char_indices() {
        if utf16_seen >= utf16_col {
            return byte_idx;
        }
        utf16_seen = utf16_seen.saturating_add(ch.len_utf16() as u32);
    }
    line.len()
}

fn byte_offset_to_utf16_col(line: &str, byte_offset: usize) -> u32 {
    let clamped = byte_offset.min(line.len());
    line[..clamped].chars().map(char::len_utf16).sum::<usize>() as u32
}

/// Extracts `(prefix_before_word, word)` at the given LSP UTF-16 `Position`.
fn word_with_prefix_at_position(text: &str, pos: Position) -> Option<(&str, &str)> {
    let line = text.lines().nth(pos.line as usize)?;
    let bytes = line.as_bytes();
    let idx = utf16_col_to_byte_offset(line, pos.character).min(bytes.len());
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';

    let mut start = idx;
    while start > 0 && is_ident(bytes[start - 1]) {
        start -= 1;
    }
    let mut end = idx;
    while end < bytes.len() && is_ident(bytes[end]) {
        end += 1;
    }
    if start < end {
        Some((&line[..start], &line[start..end]))
    } else {
        None
    }
}

/// Computes `(line_before_cursor, replace_range_for_qualified_token, replace_range_for_ident)`
/// at the given cursor `Position` so completions replace the typed prefix cleanly without
/// duplicating `mc::passes::` or `mc::targets::`.
fn completion_prefix_context(text: &str, pos: Position) -> (String, Range, Range) {
    let line = text.lines().nth(pos.line as usize).unwrap_or("");
    let bytes = line.as_bytes();
    let cursor_byte = utf16_col_to_byte_offset(line, pos.character).min(bytes.len());
    let cursor_utf16 = byte_offset_to_utf16_col(line, cursor_byte);

    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let is_path_char = |b: u8| is_ident(b) || b == b':';

    let mut ident_start = cursor_byte;
    while ident_start > 0 && is_ident(bytes[ident_start - 1]) {
        ident_start -= 1;
    }
    let mut path_start = cursor_byte;
    while path_start > 0 && is_path_char(bytes[path_start - 1]) {
        path_start -= 1;
    }

    let ident_range = Range {
        start: Position {
            line: pos.line,
            character: byte_offset_to_utf16_col(line, ident_start),
        },
        end: Position {
            line: pos.line,
            character: cursor_utf16,
        },
    };
    let path_range = Range {
        start: Position {
            line: pos.line,
            character: byte_offset_to_utf16_col(line, path_start),
        },
        end: Position {
            line: pos.line,
            character: cursor_utf16,
        },
    };

    (line[..cursor_byte].to_owned(), path_range, ident_range)
}

fn make_completion_item(
    label: String,
    new_text: String,
    replace_range: Range,
    kind: CompletionItemKind,
    detail: String,
) -> CompletionItem {
    CompletionItem {
        label,
        kind: Some(kind),
        detail: Some(detail),
        text_edit: Some(CompletionTextEdit::Edit(TextEdit {
            range: replace_range,
            new_text,
        })),
        ..CompletionItem::default()
    }
}

fn is_valid_vsl_ident(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

impl LanguageServer for VslLanguageServer {
    fn initialize(
        &self,
        _: InitializeParams,
    ) -> impl std::future::Future<Output = Result<InitializeResult>> {
        std::future::ready(Ok(InitializeResult {
            capabilities: ServerCapabilities {
                position_encoding: Some(PositionEncodingKind::UTF16),
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![":".to_owned(), ".".to_owned(), "@".to_owned()]),
                    ..CompletionOptions::default()
                }),
                definition_provider: Some(OneOf::Left(true)),
                references_provider: Some(OneOf::Left(true)),
                rename_provider: Some(OneOf::Left(true)),
                semantic_tokens_provider: Some(
                    SemanticTokensServerCapabilities::SemanticTokensOptions(
                        SemanticTokensOptions {
                            legend: SemanticTokensLegend {
                                token_types: semantic_token_types(),
                                token_modifiers: semantic_token_modifiers(),
                            },
                            full: Some(SemanticTokensFullOptions::Bool(true)),
                            range: None,
                            ..SemanticTokensOptions::default()
                        },
                    ),
                ),
                ..ServerCapabilities::default()
            },
            ..InitializeResult::default()
        }))
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "vsl-lsp initialized (VSL 2.0)")
            .await;
    }

    fn shutdown(&self) -> impl std::future::Future<Output = Result<()>> {
        std::future::ready(Ok(()))
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.publish_file_diagnostics(params.text_document.uri, &params.text_document.text)
            .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(change) = params.content_changes.into_iter().last() {
            self.publish_file_diagnostics(params.text_document.uri, &change.text)
                .await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let norm_key = normalize_uri_key(params.text_document.uri.as_str());
        self.documents.write().await.remove(&norm_key);
        self.cache.write().await.remove(&norm_key);
        self.client
            .publish_diagnostics(params.text_document.uri, Vec::new(), None)
            .await;
    }

    // These suffixes are VSL graph-builder method names, not filesystem extensions.
    #[allow(clippy::case_sensitive_file_extension_comparisons)]
    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        let Some(analysis) = self.ensure_cached(&uri).await else {
            return Ok(None);
        };
        let text = analysis.source.text.as_str();
        let Some((prefix, word)) = word_with_prefix_at_position(text, pos) else {
            return Ok(None);
        };
        let trimmed = prefix.trim_end();

        // Engine-contract hovers (`mc::passes::*`, `mc::targets::*`, `mc::world.*`, `@hook(...)`).
        let before_paren = trimmed.strip_suffix('(').map(str::trim_end);
        let is_pass_ctx = trimmed.ends_with("mc::passes::")
            || before_paren.is_some_and(|p| {
                p.ends_with(".after") || p.ends_with(".before") || p.ends_with(".replaces")
            });
        let is_target_ctx = trimmed.ends_with("mc::targets::") || word.starts_with("RES_");
        let is_hook_ctx =
            before_paren.is_some_and(|p| p.ends_with("@hook") || p.ends_with(".hook"));
        let is_world_field_ctx = trimmed.ends_with("mc::world.") || trimmed.ends_with("world.");

        let markdown = if is_pass_ctx && let Some(pass) = CanonicalPass::from_name(word) {
            format!(
                "**Hyaloclast Canonical Pass** `mc::passes::{}`\n\nRegistered in `vulkanshaders-core::RenderGraph`.",
                pass.as_str()
            )
        } else if is_target_ctx && let Some(res) = BuiltinResourceId::from_name(word) {
            format!(
                "**Hyaloclast Built-in Resource** `mc::targets::{word}`\n\n`ResourceId({})`",
                res.raw_id()
            )
        } else if is_hook_ctx && let Some(hook) = HookStage::from_name(word) {
            format!(
                "**VSL 2.0 Hook Stage** `@hook({})`\n\nLowering ABI: `{:?}`{}",
                hook.as_str(),
                hook.abi_route(),
                if hook.uses_alpha_over_blending() {
                    "\n\nReturns `vec4` (alpha-over composited across chains)."
                } else {
                    ""
                }
            )
        } else if is_world_field_ctx && let Some(field) = WORLD_UBO_LAYOUT.field(word) {
            format!(
                "**WorldUBO Field** `mc::world.{}`\n\nType: `{}` (`std140` offset `{}` bytes, size `{}` bytes)",
                field.name,
                format_builtin_field_type(field.ty),
                field.byte_offset,
                field.byte_size,
            )
        } else if let Some(ref hir) = analysis.hir {
            // HIR-backed hover for functions, structs, fields, params, locals, consts, and workgroup vars.
            if let Some(byte_off) = position_to_byte_offset(&analysis.source, pos)
                && let Some(sym) = symbol_at_offset(&analysis.symbols, byte_off)
            {
                match sym.kind {
                    IndexedSymbolKind::Function => {
                        if let Some(func) = hir.find_function(&sym.name) {
                            format!("```vsl\n{}\n```", format_function_signature(hir, func))
                        } else {
                            format!("```vsl\n{}\n```", sym.type_display)
                        }
                    }
                    IndexedSymbolKind::Struct => {
                        if let Some(st) = hir.find_struct(&sym.name) {
                            format_struct_hover(hir, st)
                        } else {
                            format!("```vsl\nstruct {}\n```", sym.name)
                        }
                    }
                    IndexedSymbolKind::Field => {
                        format!("```vsl\n(field) {}: {}\n```", sym.name, sym.type_display)
                    }
                    IndexedSymbolKind::Param => {
                        format!("```vsl\n(param) {}: {}\n```", sym.name, sym.type_display)
                    }
                    IndexedSymbolKind::Local => {
                        format!("```vsl\nlet {}: {}\n```", sym.name, sym.type_display)
                    }
                    IndexedSymbolKind::Const => {
                        format!("```vsl\nconst {}: {}\n```", sym.name, sym.type_display)
                    }
                    IndexedSymbolKind::WorkgroupVar => {
                        format!(
                            "```vsl\nworkgroup var {}: {}\n```",
                            sym.name, sym.type_display
                        )
                    }
                    IndexedSymbolKind::Trait => {
                        format!("```vsl\ntrait {}\n```", sym.name)
                    }
                }
            } else if let Some(func) = hir.find_function(word) {
                format!("```vsl\n{}\n```", format_function_signature(hir, func))
            } else if let Some(st) = hir.find_struct(word) {
                format_struct_hover(hir, st)
            } else {
                return Ok(None);
            }
        } else {
            return Ok(None);
        };

        Ok(Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: markdown,
            }),
            range: None,
        }))
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = params.text_document_position.text_document.uri;
        let pos = params.text_document_position.position;
        let analysis = self.ensure_cached(&uri).await;
        let text = analysis.as_ref().map_or("", |a| a.source.text.as_str());

        let (before_cursor, path_range, ident_range) = completion_prefix_context(text, pos);
        let trimmed = before_cursor.trim_end();
        let mut items = Vec::new();

        // 1. `@hook(...)` or `.hook(...)` stage completion.
        let in_hook_ctx = trimmed.rsplit_once('(').is_some_and(|(lhs, rhs)| {
            let lhs = lhs.trim_end();
            (lhs.ends_with("@hook") || lhs.ends_with(".hook")) && !rhs.contains(')')
        });
        if in_hook_ctx {
            for hook in HookStage::ALL {
                items.push(make_completion_item(
                    hook.as_str().to_owned(),
                    hook.as_str().to_owned(),
                    ident_range,
                    CompletionItemKind::ENUM_MEMBER,
                    format!("VSL 2.0 @hook stage ({:?})", hook.abi_route()),
                ));
            }
            return Ok(Some(CompletionResponse::Array(items)));
        }

        // 2. `mc::passes::*` completion (replacing full qualified prefix so no duplication occurs).
        if trimmed.contains("mc::passes::")
            && trimmed
                .rsplit("mc::passes::")
                .next()
                .is_some_and(|tail| tail.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'))
        {
            for pass in CANONICAL_PASSES {
                let full = format!("mc::passes::{}", pass.as_str());
                items.push(make_completion_item(
                    full.clone(),
                    full,
                    path_range,
                    CompletionItemKind::CONSTANT,
                    "Hyaloclast canonical RenderGraph pass".to_owned(),
                ));
            }
            return Ok(Some(CompletionResponse::Array(items)));
        }

        // 3. `mc::targets::*` completion.
        if trimmed.contains("mc::targets::")
            && trimmed
                .rsplit("mc::targets::")
                .next()
                .is_some_and(|tail| tail.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'))
        {
            for &(target, res) in BUILTIN_RESOURCES {
                let full = format!("mc::targets::{target}");
                items.push(make_completion_item(
                    full.clone(),
                    full,
                    path_range,
                    CompletionItemKind::CONSTANT,
                    format!("Hyaloclast RenderGraph ResourceId({})", res.raw_id()),
                ));
            }
            return Ok(Some(CompletionResponse::Array(items)));
        }

        // 4. `mc::world.<field>` member completion.
        if let Some((lhs, _)) = trimmed.rsplit_once('.') {
            let receiver = lhs
                .trim_end()
                .rsplit(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == ':'))
                .next()
                .unwrap_or("");
            if receiver == "mc::world" || receiver == "world" {
                for field in WORLD_UBO_LAYOUT.fields {
                    items.push(make_completion_item(
                        field.name.to_owned(),
                        field.name.to_owned(),
                        ident_range,
                        CompletionItemKind::FIELD,
                        format!(
                            "WorldUBO::{} ({}, offset {})",
                            field.name,
                            format_builtin_field_type(field.ty),
                            field.byte_offset
                        ),
                    ));
                }
                return Ok(Some(CompletionResponse::Array(items)));
            }

            // Struct field & method completion from HIR.
            if let Some(ref a) = analysis
                && let Some(ref hir) = a.hir
            {
                let mut seen_fields = HashSet::new();
                for (_, func) in hir.functions.iter() {
                    let ty_id = func.local_types.get(receiver).copied().or_else(|| {
                        func.params
                            .iter()
                            .find(|p| p.name == receiver)
                            .and_then(|p| p.ty)
                    });
                    if let Some(tid) = ty_id
                        && let Type::Struct { name: st_name, .. } = hir.types.get(tid)
                        && let Some(st) = hir.find_struct(st_name.as_str())
                    {
                        for field in &st.fields {
                            if seen_fields.insert(field.name.clone()) {
                                items.push(make_completion_item(
                                    field.name.to_string(),
                                    field.name.to_string(),
                                    ident_range,
                                    CompletionItemKind::FIELD,
                                    format_hir_type(&hir.types, field.ty),
                                ));
                            }
                        }
                        for imp in &hir.impls {
                            if imp.target_struct == st.name {
                                for &m_id in &imp.methods {
                                    let m = &hir.functions[m_id];
                                    if seen_fields.insert(m.name.clone()) {
                                        items.push(make_completion_item(
                                            m.name.to_string(),
                                            m.name.to_string(),
                                            ident_range,
                                            CompletionItemKind::METHOD,
                                            format_function_signature(hir, m),
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
                if !items.is_empty() {
                    return Ok(Some(CompletionResponse::Array(items)));
                }
            }
        }

        // 5. `mc::` namespace completion.
        if trimmed.contains("mc::")
            && trimmed
                .rsplit("mc::")
                .next()
                .is_some_and(|tail| tail.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'))
        {
            let mc_entries = [
                (
                    "mc::world",
                    CompletionItemKind::VARIABLE,
                    "WorldUBO (Set 0, binding 0, 544 B)",
                ),
                (
                    "mc::tlas",
                    CompletionItemKind::VARIABLE,
                    "Scene TLAS (Set 1, binding 0)",
                ),
                (
                    "mc::bindless",
                    CompletionItemKind::VARIABLE,
                    "COMBINED_IMAGE_SAMPLER[4096] (Set 2, binding 0)",
                ),
                (
                    "mc::instances",
                    CompletionItemKind::VARIABLE,
                    "InstanceMetaSSBO (Set 1, binding 1)",
                ),
                (
                    "mc::sharc",
                    CompletionItemKind::VARIABLE,
                    "SharcConfigUBO (Set 1, binding 2)",
                ),
                (
                    "mc::lights",
                    CompletionItemKind::VARIABLE,
                    "EmissiveLightSSBO (Set 1, binding 3)",
                ),
                (
                    "mc::reservoirs",
                    CompletionItemKind::VARIABLE,
                    "RestirReservoirSSBO (Set 1, binding 4)",
                ),
                (
                    "mc::param",
                    CompletionItemKind::FUNCTION,
                    "Load typed pack parameter via BDA",
                ),
                (
                    "mc::spec",
                    CompletionItemKind::FUNCTION,
                    "Load Vulkan specialization constant",
                ),
                (
                    "mc::reconstruct_world_pos",
                    CompletionItemKind::FUNCTION,
                    "Reconstruct world position from UV/depth",
                ),
                (
                    "mc::view_dir",
                    CompletionItemKind::FUNCTION,
                    "Compute normalized camera view direction",
                ),
                (
                    "mc::sample_sky",
                    CompletionItemKind::FUNCTION,
                    "Evaluate procedural gradient sky",
                ),
                (
                    "mc::debug_write",
                    CompletionItemKind::FUNCTION,
                    "Write RGB/RGBA value to debug storage image",
                ),
                (
                    "mc::passes::",
                    CompletionItemKind::MODULE,
                    "Hyaloclast canonical RenderGraph passes",
                ),
                (
                    "mc::targets::",
                    CompletionItemKind::MODULE,
                    "Hyaloclast built-in RenderGraph resources",
                ),
            ];
            for (name, kind, detail) in mc_entries {
                items.push(make_completion_item(
                    name.to_owned(),
                    name.to_owned(),
                    path_range,
                    kind,
                    detail.to_owned(),
                ));
            }
            return Ok(Some(CompletionResponse::Array(items)));
        }

        // 6. General completion: HIR symbols in current module + pack params/specs + `mc::*`.
        if let Some(ref a) = analysis {
            if let Some(ref hir) = a.hir {
                for (_, func) in hir.functions.iter() {
                    items.push(make_completion_item(
                        func.name.to_string(),
                        func.name.to_string(),
                        ident_range,
                        CompletionItemKind::FUNCTION,
                        format_function_signature(hir, func),
                    ));
                }
                for (_, st) in hir.structs.iter() {
                    items.push(make_completion_item(
                        st.name.to_string(),
                        st.name.to_string(),
                        ident_range,
                        CompletionItemKind::STRUCT,
                        format!("struct {} ({:?})", st.name, st.layout),
                    ));
                }
                for c in &hir.consts {
                    items.push(make_completion_item(
                        c.name.to_string(),
                        c.name.to_string(),
                        ident_range,
                        CompletionItemKind::CONSTANT,
                        format!("const {}: {}", c.name, format_hir_type(&hir.types, c.ty)),
                    ));
                }
                for wg in &hir.workgroup_vars {
                    items.push(make_completion_item(
                        wg.name.to_string(),
                        wg.name.to_string(),
                        ident_range,
                        CompletionItemKind::VARIABLE,
                        format!(
                            "workgroup var {}: {}",
                            wg.name,
                            format_hir_type(&hir.types, wg.ty)
                        ),
                    ));
                }
            }
            if let Some(ref schema) = a.pack_schema {
                for (name, ty) in &schema.param_types {
                    items.push(make_completion_item(
                        name.to_string(),
                        name.to_string(),
                        ident_range,
                        CompletionItemKind::PROPERTY,
                        format!("pack.vsl param ({ty:?})"),
                    ));
                }
                for (name, spec_id) in &schema.spec_ids {
                    items.push(make_completion_item(
                        name.to_string(),
                        name.to_string(),
                        ident_range,
                        CompletionItemKind::CONSTANT,
                        format!("pack.vsl spec constant (SpecId {spec_id})"),
                    ));
                }
            }
        }

        for pass in CANONICAL_PASSES {
            let full = format!("mc::passes::{}", pass.as_str());
            items.push(make_completion_item(
                full.clone(),
                full,
                path_range,
                CompletionItemKind::CONSTANT,
                "Hyaloclast canonical RenderGraph pass".to_owned(),
            ));
        }

        for &(target, res) in BUILTIN_RESOURCES {
            let full = format!("mc::targets::{target}");
            items.push(make_completion_item(
                full.clone(),
                full,
                path_range,
                CompletionItemKind::CONSTANT,
                format!("Hyaloclast RenderGraph ResourceId({})", res.raw_id()),
            ));
        }

        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        let Some(analysis) = self.ensure_cached(&uri).await else {
            return Ok(None);
        };
        let Some(byte_off) = position_to_byte_offset(&analysis.source, pos) else {
            return Ok(None);
        };
        let Some(sym) = symbol_at_offset(&analysis.symbols, byte_off) else {
            return Ok(None);
        };
        if sym.def_span.file == FileId::SYNTHETIC {
            return Ok(None);
        }
        let range = span_to_lsp_range(&analysis.source, sym.def_span);
        Ok(Some(GotoDefinitionResponse::Scalar(Location {
            uri,
            range,
        })))
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let uri = params.text_document_position.text_document.uri;
        let pos = params.text_document_position.position;
        let include_decl = params.context.include_declaration;
        let Some(analysis) = self.ensure_cached(&uri).await else {
            return Ok(None);
        };
        let Some(byte_off) = position_to_byte_offset(&analysis.source, pos) else {
            return Ok(None);
        };
        let Some(target) = symbol_at_offset(&analysis.symbols, byte_off) else {
            return Ok(None);
        };
        let target_def = target.def_span;

        let mut seen = HashSet::new();
        let mut locations = Vec::new();
        for sym in &analysis.symbols {
            if sym.def_span != target_def {
                continue;
            }
            if !include_decl && sym.is_definition {
                continue;
            }
            if seen.insert((sym.span.start, sym.span.end)) {
                locations.push(Location {
                    uri: uri.clone(),
                    range: span_to_lsp_range(&analysis.source, sym.span),
                });
            }
        }
        Ok(Some(locations))
    }

    async fn rename(&self, params: RenameParams) -> Result<Option<WorkspaceEdit>> {
        let uri = params.text_document_position.text_document.uri;
        let pos = params.text_document_position.position;
        if !is_valid_vsl_ident(&params.new_name) {
            return Ok(None);
        }
        let Some(analysis) = self.ensure_cached(&uri).await else {
            return Ok(None);
        };
        let Some(byte_off) = position_to_byte_offset(&analysis.source, pos) else {
            return Ok(None);
        };
        let Some(target) = symbol_at_offset(&analysis.symbols, byte_off) else {
            return Ok(None);
        };
        let target_def = target.def_span;

        let mut seen = HashSet::new();
        let mut edits = Vec::new();
        for sym in &analysis.symbols {
            if sym.def_span == target_def && seen.insert((sym.span.start, sym.span.end)) {
                edits.push(TextEdit {
                    range: span_to_lsp_range(&analysis.source, sym.span),
                    new_text: params.new_name.clone(),
                });
            }
        }
        if edits.is_empty() {
            return Ok(None);
        }
        let mut changes = HashMap::new();
        changes.insert(uri, edits);
        Ok(Some(WorkspaceEdit {
            changes: Some(changes),
            ..WorkspaceEdit::default()
        }))
    }

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        let uri = params.text_document.uri;
        let Some(analysis) = self.ensure_cached(&uri).await else {
            return Ok(None);
        };
        let data = encode_semantic_tokens(&analysis.tokens);
        Ok(Some(SemanticTokensResult::Tokens(SemanticTokens {
            result_id: None,
            data,
        })))
    }
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let (service, socket) = LspService::new(VslLanguageServer::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use vsl_hir::lower_unit;

    fn dummy_cached(name: &str) -> CachedAnalysis {
        CachedAnalysis {
            source: SourceFile::new(FileId(0), name, ""),
            hir: None,
            pack_schema: None,
            symbols: Vec::new(),
            tokens: Vec::new(),
        }
    }

    #[test]
    fn normalizes_windows_and_percent_encoded_uris() {
        assert_eq!(
            normalize_uri_key("file:///C:/Users/Dev/My%20Pack/pack.vsl"),
            "file:///c:/Users/Dev/My Pack/pack.vsl"
        );
        assert_eq!(
            normalize_uri_key("file:///c%3A/Users/Dev/pack.vsl"),
            "file:///c:/Users/Dev/pack.vsl"
        );
        assert_eq!(
            normalize_uri_key("C:\\Packs\\Sunlit\\reflections.vsl"),
            "c:/Packs/Sunlit/reflections.vsl"
        );
    }

    #[test]
    fn matches_pack_vsl_basename_strictly() {
        assert!(is_pack_vsl_uri("file:///example/pack/pack.vsl"));
        assert!(is_pack_vsl_uri("pack.vsl"));
        assert!(is_pack_vsl_uri("file:///C:/packs/pack.vsl?v=1"));
        assert!(is_pack_vsl_uri("file:///c%3A/packs/pack.vsl#frag"));
        assert!(is_pack_vsl_uri("C:\\packs\\pack.vsl"));
        assert!(!is_pack_vsl_uri("file:///example/pack/unpack.vsl"));
        assert!(!is_pack_vsl_uri("file:///example/pack/my_pack.vsl"));
    }

    #[test]
    fn analysis_cache_evicts_oldest_entries_at_capacity() {
        let mut cache = AnalysisCache::new(2);
        cache.insert("a".to_owned(), dummy_cached("a"));
        cache.insert("b".to_owned(), dummy_cached("b"));
        assert_eq!(cache.len(), 2);

        // Touching `a` refreshes its recency so `b` is evicted next.
        cache.insert("a".to_owned(), dummy_cached("a2"));
        cache.insert("c".to_owned(), dummy_cached("c"));
        assert_eq!(cache.len(), 2);
        assert!(cache.get("a").is_some());
        assert!(cache.get("c").is_some());
        assert!(cache.get("b").is_none());

        cache.remove("a");
        assert_eq!(cache.len(), 1);
        assert!(cache.get("a").is_none());
    }

    #[test]
    fn indexes_hir_symbols_for_goto_definition_references_and_completion_prefix() {
        let src = r"
module pack::demo;
pub struct Sample {
    pub weight: f32,
}
pub fn double_weight(s: Sample) -> f32 {
    let w = s.weight;
    w + w
}
";
        let file = SourceFile::new(FileId(0), "demo.vsl", src);
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let (hir, diags) = lower_unit(&parsed.unit, RenderTier::Tier2RayQuery, HardwareCaps::all());
        assert!(!diags.has_errors(), "{:?}", diags.as_slice());

        let symbols = index_hir_symbols(&hir, &file);
        let def_w = symbols
            .iter()
            .find(|s| s.name == "w" && s.is_definition)
            .expect("definition for local `w`");
        let refs_w: Vec<_> = symbols
            .iter()
            .filter(|s| s.def_span == def_w.def_span && !s.is_definition)
            .collect();
        assert_eq!(refs_w.len(), 2, "expected 2 references to `w` in `w + w`");

        let hit = symbol_at_offset(&symbols, refs_w[0].span.start).expect("symbol hit");
        assert_eq!(hit.def_span, def_w.def_span);

        // Completion prefix replacement avoids `mc::passes::mc::passes::` duplication.
        let line_src = "    .after(mc::passes::gb";
        let (_, path_range, _) = completion_prefix_context(
            line_src,
            Position {
                line: 0,
                character: line_src.len() as u32,
            },
        );
        assert_eq!(path_range.start.character, 11);
        assert_eq!(path_range.end.character, line_src.len() as u32);
    }
}
