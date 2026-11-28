use indexmap::{IndexMap, IndexSet};
use vsl_ir::{
    AtomicOrder, AtomicScope, BuiltinKind, CoopMatrixScalar, GlslStd450Op, InstData, InstId,
    IrDescriptorVar,
    IrFunction, IrInterlockMode, SubgroupClusteredOp, IrInterpolation, IrModule, IrPushConstantBlock, IrStorageClass,
    IrStructLayout,
    IrTessDomain, IrTessOrder, IrTessSpacing, IrTopology, IrType, IrWorkgroupVar, ScalarIrTy,
    SpecConstantOpKind, StructuredStmt,
};
use vsl_mc::{
    DescriptorAccess, DescriptorKind, HardwareCaps, ImageFormat, SampleScalar, ShaderStage,
    TextureDim, builtin_descriptor_layout,
};

/// SPIR-V 1.6 version word (`0x0001_0600`) required for Vulkan 1.3+ shader packs.
pub const SPIRV_VERSION_1_6: u32 = 0x0001_0600;

/// Unregistered generator magic word reserved for VSL 2.0 (`vsp-compiler`).
const VSL_GENERATOR_MAGIC: u32 = 0x5653_0002;

/// Deduplication key for structural SPIR-V types emitted into `types_globals`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TypeKey {
    Void,
    Bool,
    Int(u32, u32),
    Float(u32),
    Vector(u32, u32),
    Matrix(u32, u32),
    Array(u32, u32, Option<u32>),
    Pointer(u32, u32),
    Function(u32, Vec<u32>),
    Image {
        sampled_ty: u32,
        dim: u32,
        depth: u32,
        arrayed: u32,
        ms: u32,
        sampled: u32,
        format: u32,
    },
    Sampler,
    SampledImage(u32),
    RuntimeArray(u32, Option<u32>),
    AccelerationStructureKhr,
    RayQueryKhr,
    HitObject(bool),
    CoopMatrix {
        comp_ty: u32,
        scope_id: u32,
        rows_id: u32,
        cols_id: u32,
        use_id: u32,
    },
}

/// Deduplication key for SPIR-V constants emitted into `types_globals`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ConstKey {
    Bool(u32, bool),
    Bits32(u32, u32),
    Bits64(u32, u64),
    Null(u32),
}

/// Metadata for a module-level descriptor `OpVariable` deduplicated by `(set, binding)`.
#[derive(Debug, Clone)]
struct EmittedDescriptor {
    var_id: u32,
    ptr_ty: u32,
    storage_class: spirv::StorageClass,
    element_ty: u32,
    raw_image_ty: u32,
    kind: DescriptorKind,
    access: DescriptorAccess,
    is_unsized: bool,
    count: Option<u32>,
    coord_components: u32,
    grad_components: u32,
    size_components: u32,
    sampled_ty: u32,
    is_cube: bool,
    struct_layout: Option<IrStructLayout>,
}

impl EmittedDescriptor {
    /// Returns `true` for either runtime-sized or fixed-size descriptor arrays.
    const fn is_arrayed_descriptor(&self) -> bool {
        self.is_unsized || self.count.is_some()
    }
}

/// Deduplication key for module-level `StorageClass::Workgroup` variables.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct WorkgroupVarKey {
    name: String,
    ty: IrType,
}

/// Deduplication key for module-level `StorageClass::PushConstant` blocks.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PushConstantKey {
    total_size: u32,
    fields: Vec<(String, u32, u32, ScalarIrTy)>,
}

/// Metadata for a module-level `StorageClass::Workgroup` variable.
#[derive(Debug, Clone)]
struct EmittedWorkgroupVar {
    var_id: u32,
    ptr_ty: u32,
    elem_ty: u32,
    is_array: bool,
    var_ir_ty: IrType,
    elem_ir_ty: IrType,
}

/// Metadata for a module-level `BuiltIn` variable deduplicated by `(BuiltinKind, is_output)`.
#[derive(Debug, Clone, Copy)]
struct EmittedBuiltinVar {
    var_id: u32,
    value_ty: u32,
    scalar_ty: u32,
    vector_len: Option<u32>,
}

/// Metadata for a module-level raster stage `Input`/`Output` variable.
#[derive(Debug, Clone)]
struct EmittedStageIoVar {
    var_id: u32,
    val_ty: u32,
    is_output: bool,
}

#[derive(Debug, Clone, Copy)]
struct EmittedStageVar {
    var_id: u32,
    val_ty: u32,
}

/// Metadata for a function-local `StorageClass::Function` variable.
#[derive(Debug, Clone)]
struct EmittedLocalVar {
    var_id: u32,
    ptr_ty: u32,
    val_ty: u32,
    ir_ty: IrType,
}

/// Metadata for a module-level `StorageClass::PushConstant` block variable.
#[derive(Debug, Clone)]
struct EmittedPushConstant {
    var_id: u32,
    /// `(member_ty, scalar_ty, is_composite, is_f16)` for each struct member in `OpTypeStruct`.
    members: Vec<(u32, u32, bool, bool)>,
}

/// Pre-allocated SPIR-V signature metadata for any function in an [`IrModule`].
#[derive(Debug, Clone)]
struct EmittedFunctionInfo {
    fn_id: u32,
    fn_ty: u32,
    ret_ty: u32,
    /// `(spv_param_ty, is_ptr_param)` for each formal parameter.
    param_tys: Vec<(u32, bool)>,
}

/// Flat-arena SPIR-V 1.6 binary builder with zero per-instruction `Vec` allocations.
///
/// Sections are buffered into contiguous `Vec<u32>` slices and concatenated in canonical
/// SPIR-V logical layout order (`Capabilities` -> `Extensions` -> `MemoryModel` ->
/// `EntryPoints` -> `ExecutionModes` -> `DebugNames` -> `Decorations` -> `Types`/Constants
/// -> `Functions`).
#[derive(Debug)]
pub struct SpirvEmitter {
    next_id: u32,
    capabilities: Vec<u32>,
    capabilities_seen: IndexSet<u32>,
    extensions: Vec<u32>,
    extensions_seen: IndexSet<String>,
    ext_inst_imports: Vec<u32>,
    glsl_std_450_id: Option<u32>,
    debug_info_ext_id: Option<u32>,
    debug_printf_ext_id: Option<u32>,
    decorations_seen: IndexSet<(u32, u32)>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    execution_modes: Vec<u32>,
    debug_strings: Vec<u32>,
    string_cache: IndexMap<String, u32>,
    debug_names: Vec<u32>,
    decorations: Vec<u32>,
    types_globals: Vec<u32>,
    functions: Vec<u32>,
    type_cache: IndexMap<TypeKey, u32>,
    struct_type_cache: IndexMap<(IrStructLayout, bool, bool), u32>,
    const_cache: IndexMap<ConstKey, u32>,
    const_ids: IndexSet<u32>,
    spec_const_ids: IndexMap<u32, (u32, u32)>,
    bda_block_tys: IndexMap<(u32, u32, Option<u32>), u32>,
    /// Tracks emitted 64-bit integer types so `Int64` is declared even when no instruction scan sees one.
    uses_int64_type: bool,
}

impl Default for SpirvEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl SpirvEmitter {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            capabilities: Vec::new(),
            capabilities_seen: IndexSet::new(),
            extensions: Vec::new(),
            extensions_seen: IndexSet::new(),
            ext_inst_imports: Vec::new(),
            glsl_std_450_id: None,
            debug_info_ext_id: None,
            debug_printf_ext_id: None,
            decorations_seen: IndexSet::new(),
            memory_model: Vec::new(),
            entry_points: Vec::new(),
            execution_modes: Vec::new(),
            debug_strings: Vec::new(),
            string_cache: IndexMap::new(),
            debug_names: Vec::new(),
            decorations: Vec::new(),
            types_globals: Vec::new(),
            functions: Vec::new(),
            type_cache: IndexMap::new(),
            struct_type_cache: IndexMap::new(),
            const_cache: IndexMap::new(),
            const_ids: IndexSet::new(),
            spec_const_ids: IndexMap::new(),
            bda_block_tys: IndexMap::new(),
            uses_int64_type: false,
        }
    }

    pub const fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn emit_capability(&mut self, cap: spirv::Capability) {
        self.emit_capability_raw(cap as u32);
    }

    pub fn emit_capability_raw(&mut self, raw: u32) {
        if self.capabilities_seen.insert(raw) {
            push_inst(&mut self.capabilities, spirv::Op::Capability, &[raw]);
        }
    }

    pub fn emit_extension(&mut self, name: &str) {
        if self.extensions_seen.insert(name.to_owned()) {
            let words = encode_literal_string(name);
            push_inst(&mut self.extensions, spirv::Op::Extension, &words);
        }
    }

    pub fn emit_glsl_std_450_import(&mut self) -> u32 {
        if let Some(id) = self.glsl_std_450_id {
            return id;
        }
        let id = self.alloc_id();
        let mut ops = vec![id];
        ops.extend(encode_literal_string("GLSL.std.450"));
        push_inst(&mut self.ext_inst_imports, spirv::Op::ExtInstImport, &ops);
        self.glsl_std_450_id = Some(id);
        id
    }

    pub fn emit_debug_info_import(&mut self) -> u32 {
        if let Some(id) = self.debug_info_ext_id {
            return id;
        }
        self.emit_extension("SPV_KHR_non_semantic_info");
        let id = self.alloc_id();
        let mut ops = vec![id];
        ops.extend(encode_literal_string("NonSemantic.Shader.DebugInfo.100"));
        push_inst(&mut self.ext_inst_imports, spirv::Op::ExtInstImport, &ops);
        self.debug_info_ext_id = Some(id);
        id
    }

    pub fn emit_debug_printf_import(&mut self) -> u32 {
        if let Some(id) = self.debug_printf_ext_id {
            return id;
        }
        self.emit_extension("SPV_KHR_non_semantic_info");
        let id = self.alloc_id();
        let mut ops = vec![id];
        ops.extend(encode_literal_string("NonSemantic.DebugPrintf"));
        push_inst(&mut self.ext_inst_imports, spirv::Op::ExtInstImport, &ops);
        self.debug_printf_ext_id = Some(id);
        id
    }

    pub fn emit_debug_string(&mut self, text: &str) -> u32 {
        if let Some(&id) = self.string_cache.get(text) {
            return id;
        }
        let id = self.alloc_id();
        let mut ops = vec![id];
        ops.extend(encode_literal_string(text));
        push_inst(&mut self.debug_strings, spirv::Op::String, &ops);
        self.string_cache.insert(text.to_owned(), id);
        id
    }

    pub fn emit_types_ext_inst(
        &mut self,
        result_ty: u32,
        ext_id: u32,
        inst_op: u32,
        operands: &[u32],
    ) -> u32 {
        let res = self.alloc_id();
        let mut ops = Vec::with_capacity(4 + operands.len());
        ops.push(result_ty);
        ops.push(res);
        ops.push(ext_id);
        ops.push(inst_op);
        ops.extend_from_slice(operands);
        push_inst(&mut self.types_globals, spirv::Op::ExtInst, &ops);
        res
    }

    pub fn emit_memory_model(
        &mut self,
        addressing: spirv::AddressingModel,
        memory: spirv::MemoryModel,
    ) {
        self.memory_model.clear();
        push_inst(
            &mut self.memory_model,
            spirv::Op::MemoryModel,
            &[addressing as u32, memory as u32],
        );
    }

    pub fn emit_entry_point(
        &mut self,
        model: spirv::ExecutionModel,
        func_id: u32,
        name: &str,
        interface_ids: &[u32],
    ) {
        self.emit_entry_point_raw(model as u32, func_id, name, interface_ids);
    }

    pub fn emit_entry_point_raw(
        &mut self,
        model: u32,
        func_id: u32,
        name: &str,
        interface_ids: &[u32],
    ) {
        let name_words = encode_literal_string(name);
        let mut operands = Vec::with_capacity(2 + name_words.len() + interface_ids.len());
        operands.push(model);
        operands.push(func_id);
        operands.extend_from_slice(&name_words);
        operands.extend_from_slice(interface_ids);
        push_inst(&mut self.entry_points, spirv::Op::EntryPoint, &operands);
    }

    pub fn emit_execution_mode(
        &mut self,
        entry_id: u32,
        mode: spirv::ExecutionMode,
        literals: &[u32],
    ) {
        self.emit_execution_mode_raw(entry_id, mode as u32, literals);
    }

    pub fn emit_execution_mode_raw(
        &mut self,
        entry_id: u32,
        mode: u32,
        literals: &[u32],
    ) {
        let mut operands = Vec::with_capacity(2 + literals.len());
        operands.push(entry_id);
        operands.push(mode);
        operands.extend_from_slice(literals);
        push_inst(
            &mut self.execution_modes,
            spirv::Op::ExecutionMode,
            &operands,
        );
    }

    /// Emits `OpExecutionModeId` (e.g. `LocalSizeId` referencing specialization constant IDs).
    pub fn emit_execution_mode_id(
        &mut self,
        entry_id: u32,
        mode: spirv::ExecutionMode,
        id_operands: &[u32],
    ) {
        let mut operands = Vec::with_capacity(2 + id_operands.len());
        operands.push(entry_id);
        operands.push(mode as u32);
        operands.extend_from_slice(id_operands);
        push_inst(
            &mut self.execution_modes,
            spirv::Op::ExecutionModeId,
            &operands,
        );
    }

    /// Emits an `OpName` debug instruction into the debug-names section.
    pub fn emit_name(&mut self, target_id: u32, name: &str) {
        let name_words = encode_literal_string(name);
        let mut operands = Vec::with_capacity(1 + name_words.len());
        operands.push(target_id);
        operands.extend_from_slice(&name_words);
        push_inst(&mut self.debug_names, spirv::Op::Name, &operands);
    }

    /// Emits an `OpMemberName` debug instruction into the debug-names section.
    pub fn emit_member_name(&mut self, struct_id: u32, member: u32, name: &str) {
        let name_words = encode_literal_string(name);
        let mut operands = Vec::with_capacity(2 + name_words.len());
        operands.push(struct_id);
        operands.push(member);
        operands.extend_from_slice(&name_words);
        push_inst(&mut self.debug_names, spirv::Op::MemberName, &operands);
    }

    pub fn emit_decorate(
        &mut self,
        target_id: u32,
        decoration: spirv::Decoration,
        literals: &[u32],
    ) {
        self.emit_decorate_raw(target_id, decoration as u32, literals);
    }

    pub fn emit_decorate_raw(&mut self, target_id: u32, decoration: u32, literals: &[u32]) {
        let mut operands = Vec::with_capacity(2 + literals.len());
        operands.push(target_id);
        operands.push(decoration);
        operands.extend_from_slice(literals);
        push_inst(&mut self.decorations, spirv::Op::Decorate, &operands);
    }

    pub fn emit_decorate_once(
        &mut self,
        target_id: u32,
        decoration: spirv::Decoration,
        literals: &[u32],
    ) {
        if self.decorations_seen.insert((target_id, decoration as u32)) {
            self.emit_decorate(target_id, decoration, literals);
        }
    }

    /// Emits an `OpMemberDecorate` annotation targeting a specific struct member.
    pub fn emit_member_decorate(
        &mut self,
        struct_id: u32,
        member: u32,
        decoration: spirv::Decoration,
        literals: &[u32],
    ) {
        let mut operands = Vec::with_capacity(3 + literals.len());
        operands.push(struct_id);
        operands.push(member);
        operands.push(decoration as u32);
        operands.extend_from_slice(literals);
        push_inst(&mut self.decorations, spirv::Op::MemberDecorate, &operands);
    }

    pub fn emit_type_void(&mut self) -> u32 {
        if let Some(&id) = self.type_cache.get(&TypeKey::Void) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(&mut self.types_globals, spirv::Op::TypeVoid, &[id]);
        self.type_cache.insert(TypeKey::Void, id);
        id
    }

    pub fn emit_type_bool(&mut self) -> u32 {
        if let Some(&id) = self.type_cache.get(&TypeKey::Bool) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(&mut self.types_globals, spirv::Op::TypeBool, &[id]);
        self.type_cache.insert(TypeKey::Bool, id);
        id
    }

    pub fn emit_type_int(&mut self, width: u32, signedness: u32) -> u32 {
        let key = TypeKey::Int(width, signedness);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_globals,
            spirv::Op::TypeInt,
            &[id, width, signedness],
        );
        if width == 64 {
            self.uses_int64_type = true;
        }
        self.type_cache.insert(key, id);
        id
    }

    /// Returns `true` if `id` is a module-level constant or specialization constant (`O(1)`).
    pub fn is_const_id(&self, id: u32) -> bool {
        self.const_ids.contains(&id)
    }

    fn register_const_id(&mut self, id: u32) {
        self.const_ids.insert(id);
    }

    pub fn emit_type_float(&mut self, width: u32) -> u32 {
        if width == 16 {
            self.emit_capability(spirv::Capability::Float16);
        }
        let key = TypeKey::Float(width);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(&mut self.types_globals, spirv::Op::TypeFloat, &[id, width]);
        self.type_cache.insert(key, id);
        id
    }

    pub fn emit_type_vector(&mut self, component_ty: u32, component_count: u32) -> u32 {
        let key = TypeKey::Vector(component_ty, component_count);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_globals,
            spirv::Op::TypeVector,
            &[id, component_ty, component_count],
        );
        self.type_cache.insert(key, id);
        id
    }

    pub fn emit_type_matrix(&mut self, column_ty: u32, column_count: u32) -> u32 {
        let key = TypeKey::Matrix(column_ty, column_count);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_globals,
            spirv::Op::TypeMatrix,
            &[id, column_ty, column_count],
        );
        self.type_cache.insert(key, id);
        id
    }

    pub fn emit_type_array(&mut self, element_ty: u32, length_const_id: u32) -> u32 {
        self.emit_type_array_with_stride(element_ty, length_const_id, None)
    }

    fn emit_type_array_with_stride(
        &mut self,
        element_ty: u32,
        length_const_id: u32,
        stride: Option<u32>,
    ) -> u32 {
        let key = TypeKey::Array(element_ty, length_const_id, stride);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_globals,
            spirv::Op::TypeArray,
            &[id, element_ty, length_const_id],
        );
        if let Some(s) = stride {
            self.emit_decorate_once(id, spirv::Decoration::ArrayStride, &[s]);
        }
        self.type_cache.insert(key, id);
        id
    }

    /// Emits a nominal `OpTypeStruct` (not deduplicated because structs carry distinct layout decorations).
    pub fn emit_type_struct(&mut self, member_types: &[u32]) -> u32 {
        let id = self.alloc_id();
        let mut operands = Vec::with_capacity(1 + member_types.len());
        operands.push(id);
        operands.extend_from_slice(member_types);
        push_inst(&mut self.types_globals, spirv::Op::TypeStruct, &operands);
        id
    }

    /// Emits a SPIR-V type corresponding to a first-class [`IrType`].
    pub fn emit_ir_type(&mut self, ty: &IrType) -> u32 {
        self.emit_ir_type_with_layout(ty, false)
    }

    fn emit_ir_type_with_layout(&mut self, ty: &IrType, in_block: bool) -> u32 {
        match ty {
            IrType::Void => self.emit_type_void(),
            IrType::Scalar(scalar) => match scalar {
                ScalarIrTy::Bool => {
                    if in_block {
                        self.emit_type_int(32, 0)
                    } else {
                        self.emit_type_bool()
                    }
                }
                ScalarIrTy::I32 => self.emit_type_int(32, 1),
                ScalarIrTy::U32 => self.emit_type_int(32, 0),
                ScalarIrTy::I64 => self.emit_type_int(64, 1),
                ScalarIrTy::U64 => self.emit_type_int(64, 0),
                ScalarIrTy::F16 => {
                    if in_block {
                        self.emit_capability(spirv::Capability::StorageBuffer16BitAccess);
                        self.emit_extension("SPV_KHR_16bit_storage");
                    }
                    self.emit_type_float(16)
                }
                ScalarIrTy::F32 => self.emit_type_float(32),
            },
            IrType::Vector { scalar, len } => {
                let comp_ty = self.emit_ir_type_with_layout(&IrType::Scalar(*scalar), in_block);
                self.emit_type_vector(comp_ty, u32::from((*len).clamp(2, 4)))
            }
            IrType::Matrix { scalar, cols, rows } => {
                let comp_ty = self.emit_ir_type_with_layout(&IrType::Scalar(*scalar), in_block);
                let col_ty = self.emit_type_vector(comp_ty, u32::from((*rows).clamp(2, 4)));
                self.emit_type_matrix(col_ty, u32::from((*cols).clamp(2, 4)))
            }
            IrType::Array { element, len } => {
                let elem_ty = self.emit_ir_type_with_layout(element, in_block);
                let u32_ty = self.emit_type_int(32, 0);
                let len_const = self.emit_constant_32(u32_ty, (*len).max(1));
                let stride = in_block.then(|| ir_type_stride(element));
                self.emit_type_array_with_stride(elem_ty, len_const, stride)
            }
            IrType::RuntimeArray(element) => {
                let elem_ty = self.emit_ir_type_with_layout(element, true);
                let stride = ir_type_stride(element);
                self.emit_type_runtime_array_with_stride(elem_ty, Some(stride))
            }
            IrType::Struct(layout) => self.emit_struct_type(layout, false, in_block),
            IrType::Pointer {
                pointee, storage, ..
            } => {
                let sc = map_ir_storage_class(*storage);
                let is_block_sc = matches!(
                    storage,
                    IrStorageClass::Uniform
                        | IrStorageClass::StorageBuffer
                        | IrStorageClass::PhysicalStorageBuffer
                );
                let pointee_ty = if *storage == IrStorageClass::PhysicalStorageBuffer {
                    match pointee.as_ref() {
                        IrType::Void => self.emit_bda_block_struct(),
                        other => {
                            let elem_ty = self.emit_ir_type_with_layout(other, true);
                            let stride = ir_type_stride(other);
                            let mat_stride = matrix_col_stride(other);
                            self.emit_bda_block_struct_for(elem_ty, stride, mat_stride)
                        }
                    }
                } else if is_block_sc {
                    match pointee.as_ref() {
                        IrType::Struct(layout) => self.emit_struct_type(layout, true, true),
                        IrType::Void => self.emit_type_int(32, 0),
                        other => self.emit_ir_type_with_layout(other, true),
                    }
                } else if **pointee == IrType::Void {
                    self.emit_type_int(32, 0)
                } else {
                    self.emit_ir_type_with_layout(pointee, false)
                };
                self.emit_type_pointer(sc, pointee_ty)
            }
            IrType::CoopMatrix {
                component,
                scope,
                rows,
                cols,
                use_kind,
            } => {
                let comp_ty = match component {
                    CoopMatrixScalar::F16 => self.emit_type_float(16),
                    CoopMatrixScalar::Bf16 => {
                        self.emit_capability_raw(5116); // BFloat16TypeKHR
                        self.emit_extension("SPV_KHR_bfloat16");
                        let key = TypeKey::Float(0xBF16);
                        if let Some(&id) = self.type_cache.get(&key) {
                            id
                        } else {
                            let id = self.alloc_id();
                            push_inst(&mut self.types_globals, spirv::Op::TypeFloat, &[id, 16, 0]);
                            self.type_cache.insert(key, id);
                            id
                        }
                    }
                    CoopMatrixScalar::F32 => self.emit_type_float(32),
                    CoopMatrixScalar::I8 => {
                        self.emit_capability(spirv::Capability::Int8);
                        self.emit_type_int(8, 1)
                    }
                    CoopMatrixScalar::U8 => {
                        self.emit_capability(spirv::Capability::Int8);
                        self.emit_type_int(8, 0)
                    }
                    CoopMatrixScalar::I32 => self.emit_type_int(32, 1),
                    CoopMatrixScalar::U32 => self.emit_type_int(32, 0),
                };
                let u32_ty = self.emit_type_int(32, 0);
                let scope_id = self.emit_constant_32(u32_ty, map_atomic_scope(*scope));
                let rows_id = self.emit_constant_32(u32_ty, *rows);
                let cols_id = self.emit_constant_32(u32_ty, *cols);
                let use_id = self.emit_constant_32(u32_ty, *use_kind as u32);
                let key = TypeKey::CoopMatrix {
                    comp_ty,
                    scope_id,
                    rows_id,
                    cols_id,
                    use_id,
                };
                if let Some(&id) = self.type_cache.get(&key) {
                    return id;
                }
                let id = self.alloc_id();
                push_raw_inst(
                    &mut self.types_globals,
                    4456, // OpTypeCooperativeMatrixKHR
                    &[id, comp_ty, scope_id, rows_id, cols_id, use_id],
                );
                self.type_cache.insert(key, id);
                id
            }
        }
    }

    fn emit_struct_type(&mut self, layout: &IrStructLayout, as_block: bool, in_block: bool) -> u32 {
        let explicit = as_block || in_block;
        let cache_key = (layout.clone(), as_block, explicit);
        if let Some(&id) = self.struct_type_cache.get(&cache_key) {
            return id;
        }
        let u32_ty = self.emit_type_int(32, 0);
        let member_types: Vec<u32> = if layout.fields.is_empty() {
            vec![u32_ty]
        } else {
            layout
                .fields
                .iter()
                .map(|field| {
                    if explicit && let IrType::RuntimeArray(elem) = &field.ty {
                        let elem_ty = self.emit_ir_type_with_layout(elem, true);
                        let stride = if field.byte_size > 0 {
                            field.byte_size
                        } else {
                            ir_type_stride(elem)
                        };
                        self.emit_type_runtime_array_with_stride(elem_ty, Some(stride))
                    } else {
                        self.emit_ir_type_with_layout(&field.ty, explicit)
                    }
                })
                .collect()
        };

        let struct_ty = self.emit_type_struct(&member_types);
        if !layout.name.is_empty() {
            self.emit_name(struct_ty, layout.name.as_str());
        }
        if as_block {
            self.emit_decorate(struct_ty, spirv::Decoration::Block, &[]);
        }
        if layout.fields.is_empty() {
            if explicit {
                self.emit_member_decorate(struct_ty, 0, spirv::Decoration::Offset, &[0]);
            }
        } else {
            for (idx, field) in layout.fields.iter().enumerate() {
                let member = idx as u32;
                if !field.name.is_empty() {
                    self.emit_member_name(struct_ty, member, field.name.as_str());
                }
                if explicit {
                    self.emit_member_decorate(
                        struct_ty,
                        member,
                        spirv::Decoration::Offset,
                        &[field.byte_offset],
                    );
                    let mat_stride = matrix_col_stride(&field.ty).or_else(|| match &field.ty {
                        IrType::Array { element, .. } | IrType::RuntimeArray(element) => {
                            matrix_col_stride(element)
                        }
                        _ => None,
                    });
                    if let Some(col_stride) = mat_stride {
                        self.emit_member_decorate(
                            struct_ty,
                            member,
                            spirv::Decoration::ColMajor,
                            &[],
                        );
                        self.emit_member_decorate(
                            struct_ty,
                            member,
                            spirv::Decoration::MatrixStride,
                            &[col_stride],
                        );
                    }
                }
            }
        }
        self.struct_type_cache.insert(cache_key, struct_ty);
        struct_ty
    }

    /// Emits or reuses a `Block`-decorated BDA buffer struct (`{ RuntimeArray<T> }`)
    /// with the specified `ArrayStride`, member 0 `Offset 0`, and optional matrix layout
    /// decorations (`ColMajor` + `MatrixStride`) required by Vulkan `PhysicalStorageBuffer`.
    pub fn emit_bda_block_struct_for(
        &mut self,
        elem_ty: u32,
        stride: u32,
        matrix_stride: Option<u32>,
    ) -> u32 {
        let key = (elem_ty, stride, matrix_stride);
        if let Some(&id) = self.bda_block_tys.get(&key) {
            return id;
        }
        let arr_ty = self.emit_type_runtime_array_with_stride(elem_ty, Some(stride));
        let struct_ty = self.emit_type_struct(&[arr_ty]);
        self.emit_decorate(struct_ty, spirv::Decoration::Block, &[]);
        self.emit_member_decorate(struct_ty, 0, spirv::Decoration::Offset, &[0]);
        if let Some(col_stride) = matrix_stride {
            self.emit_member_decorate(struct_ty, 0, spirv::Decoration::ColMajor, &[]);
            self.emit_member_decorate(struct_ty, 0, spirv::Decoration::MatrixStride, &[col_stride]);
        }
        self.bda_block_tys.insert(key, struct_ty);
        struct_ty
    }

    /// Emits or reuses the default `u32` `Block`-decorated BDA buffer struct (`{ RuntimeArray<u32> }`).
    pub fn emit_bda_block_struct(&mut self) -> u32 {
        let u32_ty = self.emit_type_int(32, 0);
        self.emit_bda_block_struct_for(u32_ty, 4, None)
    }

    pub fn emit_type_pointer(
        &mut self,
        storage_class: spirv::StorageClass,
        pointee_ty: u32,
    ) -> u32 {
        self.emit_type_pointer_raw(storage_class as u32, pointee_ty)
    }

    pub fn emit_type_pointer_raw(&mut self, storage_class: u32, pointee_ty: u32) -> u32 {
        let key = TypeKey::Pointer(storage_class, pointee_ty);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_globals,
            spirv::Op::TypePointer,
            &[id, storage_class, pointee_ty],
        );
        self.type_cache.insert(key, id);
        id
    }

    pub fn emit_type_function(&mut self, ret_ty: u32, params: &[u32]) -> u32 {
        let key = TypeKey::Function(ret_ty, params.to_vec());
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        let mut operands = Vec::with_capacity(2 + params.len());
        operands.push(id);
        operands.push(ret_ty);
        operands.extend_from_slice(params);
        push_inst(&mut self.types_globals, spirv::Op::TypeFunction, &operands);
        self.type_cache.insert(key, id);
        id
    }

    #[allow(clippy::too_many_arguments)]
    pub fn emit_type_image(
        &mut self,
        sampled_ty: u32,
        dim: spirv::Dim,
        depth: u32,
        arrayed: u32,
        ms: u32,
        sampled: u32,
        format: spirv::ImageFormat,
    ) -> u32 {
        let key = TypeKey::Image {
            sampled_ty,
            dim: dim as u32,
            depth,
            arrayed,
            ms,
            sampled,
            format: format as u32,
        };
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_globals,
            spirv::Op::TypeImage,
            &[
                id,
                sampled_ty,
                dim as u32,
                depth,
                arrayed,
                ms,
                sampled,
                format as u32,
            ],
        );
        self.type_cache.insert(key, id);
        id
    }

    pub fn emit_type_sampler(&mut self) -> u32 {
        if let Some(&id) = self.type_cache.get(&TypeKey::Sampler) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(&mut self.types_globals, spirv::Op::TypeSampler, &[id]);
        self.type_cache.insert(TypeKey::Sampler, id);
        id
    }

    pub fn emit_type_sampled_image(&mut self, image_ty: u32) -> u32 {
        let key = TypeKey::SampledImage(image_ty);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_globals,
            spirv::Op::TypeSampledImage,
            &[id, image_ty],
        );
        self.type_cache.insert(key, id);
        id
    }

    pub fn emit_type_runtime_array(&mut self, element_ty: u32) -> u32 {
        self.emit_type_runtime_array_with_stride(element_ty, None)
    }

    fn emit_type_runtime_array_with_stride(&mut self, element_ty: u32, stride: Option<u32>) -> u32 {
        let key = TypeKey::RuntimeArray(element_ty, stride);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_globals,
            spirv::Op::TypeRuntimeArray,
            &[id, element_ty],
        );
        if let Some(s) = stride {
            self.emit_decorate_once(id, spirv::Decoration::ArrayStride, &[s]);
        }
        self.type_cache.insert(key, id);
        id
    }

    pub fn emit_type_acceleration_structure_khr(&mut self) -> u32 {
        if let Some(&id) = self.type_cache.get(&TypeKey::AccelerationStructureKhr) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_globals,
            spirv::Op::TypeAccelerationStructureKHR,
            &[id],
        );
        self.type_cache
            .insert(TypeKey::AccelerationStructureKhr, id);
        id
    }

    pub fn emit_type_ray_query_khr(&mut self) -> u32 {
        if let Some(&id) = self.type_cache.get(&TypeKey::RayQueryKhr) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(&mut self.types_globals, spirv::Op::TypeRayQueryKHR, &[id]);
        self.type_cache.insert(TypeKey::RayQueryKhr, id);
        id
    }

    pub fn emit_type_hit_object(&mut self, use_ext: bool) -> u32 {
        let key = TypeKey::HitObject(use_ext);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        let opcode: u16 = if use_ext { 5313 } else { 5281 };
        push_raw_inst(&mut self.types_globals, opcode, &[id]);
        self.type_cache.insert(key, id);
        id
    }

    pub fn emit_constant_bool(&mut self, bool_ty: u32, value: bool) -> u32 {
        let key = ConstKey::Bool(bool_ty, value);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        let op = if value {
            spirv::Op::ConstantTrue
        } else {
            spirv::Op::ConstantFalse
        };
        push_inst(&mut self.types_globals, op, &[bool_ty, id]);
        self.const_cache.insert(key, id);
        self.register_const_id(id);
        id
    }

    pub fn emit_constant_32(&mut self, ty_id: u32, raw_bits: u32) -> u32 {
        let key = ConstKey::Bits32(ty_id, raw_bits);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_globals,
            spirv::Op::Constant,
            &[ty_id, id, raw_bits],
        );
        self.const_cache.insert(key, id);
        self.register_const_id(id);
        id
    }

    /// Emits a 64-bit `OpConstant` as a little-endian `(low, high)` 32-bit word pair per SPIR-V spec.
    pub fn emit_constant_64(&mut self, ty_id: u32, raw_bits: u64) -> u32 {
        let key = ConstKey::Bits64(ty_id, raw_bits);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        let lo = raw_bits as u32;
        let hi = (raw_bits >> 32) as u32;
        push_inst(
            &mut self.types_globals,
            spirv::Op::Constant,
            &[ty_id, id, lo, hi],
        );
        self.const_cache.insert(key, id);
        self.register_const_id(id);
        id
    }

    /// Emits or reuses a deduplicated `OpConstantNull` in `types_globals`.
    pub fn emit_constant_null(&mut self, ty_id: u32) -> u32 {
        let key = ConstKey::Null(ty_id);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_globals,
            spirv::Op::ConstantNull,
            &[ty_id, id],
        );
        self.const_cache.insert(key, id);
        self.register_const_id(id);
        id
    }

    /// Emits or retrieves a deduplicated `OpSpecConstant` for `spec_id`.
    fn emit_spec_constant_u32(&mut self, spec_id: u32, default_val: u32) -> u32 {
        if let Some(&(existing, _)) = self.spec_const_ids.get(&spec_id) {
            return existing;
        }
        let u32_ty = self.emit_type_int(32, 0);
        let id = self.alloc_id();
        push_inst(
            &mut self.types_globals,
            spirv::Op::SpecConstant,
            &[u32_ty, id, default_val],
        );
        self.emit_decorate(id, spirv::Decoration::SpecId, &[spec_id]);
        self.register_const_id(id);
        self.spec_const_ids.insert(spec_id, (id, u32_ty));
        id
    }

    pub fn emit_variable(&mut self, ptr_ty: u32, storage_class: spirv::StorageClass) -> u32 {
        self.emit_variable_raw(ptr_ty, storage_class as u32)
    }

    pub fn emit_variable_raw(&mut self, ptr_ty: u32, storage_class: u32) -> u32 {
        let id = self.alloc_id();
        push_inst(
            &mut self.types_globals,
            spirv::Op::Variable,
            &[ptr_ty, id, storage_class],
        );
        id
    }

    pub fn emit_empty_void_function(&mut self, void_ty: u32, fn_ty: u32) -> u32 {
        let fn_id = self.alloc_id();
        let label_id = self.alloc_id();
        push_inst(
            &mut self.functions,
            spirv::Op::Function,
            &[void_ty, fn_id, spirv::FunctionControl::NONE.bits(), fn_ty],
        );
        push_inst(&mut self.functions, spirv::Op::Label, &[label_id]);
        push_inst(&mut self.functions, spirv::Op::Return, &[]);
        push_inst(&mut self.functions, spirv::Op::FunctionEnd, &[]);
        fn_id
    }

    /// Appends an instruction directly to the functions section.
    pub fn emit_fn_inst(&mut self, op: spirv::Op, operands: &[u32]) {
        push_inst(&mut self.functions, op, operands);
    }

    pub fn emit_fn_inst_raw(&mut self, opcode: u16, operands: &[u32]) {
        push_raw_inst(&mut self.functions, opcode, operands);
    }

    /// Assembles the final SPIR-V 1.6 word stream including the 5-word binary header.
    pub fn finish(mut self) -> Vec<u32> {
        if self.uses_int64_type
            && !self
                .capabilities_seen
                .contains(&(spirv::Capability::Int64 as u32))
        {
            self.emit_capability(spirv::Capability::Int64);
        }
        let total = 5
            + self.capabilities.len()
            + self.extensions.len()
            + self.ext_inst_imports.len()
            + self.memory_model.len()
            + self.entry_points.len()
            + self.execution_modes.len()
            + self.debug_strings.len()
            + self.debug_names.len()
            + self.decorations.len()
            + self.types_globals.len()
            + self.functions.len();

        let mut words = Vec::with_capacity(total);
        words.push(spirv::MAGIC_NUMBER);
        words.push(SPIRV_VERSION_1_6);
        words.push(VSL_GENERATOR_MAGIC);
        words.push(self.next_id);
        words.push(0); // Schema (reserved, must be 0)

        words.extend_from_slice(&self.capabilities);
        words.extend_from_slice(&self.extensions);
        words.extend_from_slice(&self.ext_inst_imports);
        words.extend_from_slice(&self.memory_model);
        words.extend_from_slice(&self.entry_points);
        words.extend_from_slice(&self.execution_modes);
        words.extend_from_slice(&self.debug_strings);
        words.extend_from_slice(&self.debug_names);
        words.extend_from_slice(&self.decorations);
        words.extend_from_slice(&self.types_globals);
        words.extend_from_slice(&self.functions);
        words
    }
}

const fn map_ir_storage_class(sc: IrStorageClass) -> spirv::StorageClass {
    match sc {
        IrStorageClass::Function => spirv::StorageClass::Function,
        IrStorageClass::Workgroup => spirv::StorageClass::Workgroup,
        IrStorageClass::Uniform => spirv::StorageClass::Uniform,
        IrStorageClass::StorageBuffer => spirv::StorageClass::StorageBuffer,
        IrStorageClass::PhysicalStorageBuffer => spirv::StorageClass::PhysicalStorageBuffer,
        IrStorageClass::Image => spirv::StorageClass::Image,
        IrStorageClass::RayPayloadKhr => spirv::StorageClass::RayPayloadKHR,
        IrStorageClass::IncomingRayPayloadKhr => spirv::StorageClass::IncomingRayPayloadKHR,
        IrStorageClass::HitAttributeKhr => spirv::StorageClass::HitAttributeKHR,
        IrStorageClass::CallableDataKhr => spirv::StorageClass::CallableDataKHR,
        IrStorageClass::IncomingCallableDataKhr => spirv::StorageClass::IncomingCallableDataKHR,
        IrStorageClass::ShaderRecordBufferKhr => spirv::StorageClass::ShaderRecordBufferKHR,
        IrStorageClass::TaskPayloadWorkgroupExt => spirv::StorageClass::TaskPayloadWorkgroupEXT,
    }
}

fn ir_type_stride(ty: &IrType) -> u32 {
    match ty {
        IrType::Scalar(s) => s.byte_size(),
        IrType::Vector { scalar, len: 3 } => scalar.byte_size().saturating_mul(4),
        IrType::Vector { scalar, len } => scalar.byte_size().saturating_mul(u32::from(*len)),
        IrType::Matrix { scalar, cols, rows } => {
            let col_stride = if *rows == 3 {
                scalar.byte_size().saturating_mul(4)
            } else {
                scalar.byte_size().saturating_mul(u32::from(*rows))
            };
            col_stride.saturating_mul(u32::from(*cols))
        }
        IrType::Array { element, len } => ir_type_stride(element).saturating_mul((*len).max(1)),
        IrType::Struct(layout) => {
            let align = layout.align.max(1);
            let aligned_size = layout.size.div_ceil(align).saturating_mul(align);
            aligned_size.max(align).max(4)
        }
        IrType::Pointer {
            storage: IrStorageClass::PhysicalStorageBuffer,
            ..
        } => 8,
        IrType::Void | IrType::RuntimeArray(_) | IrType::Pointer { .. } => 4,
    }
}

/// Returns the natural power-of-2 byte alignment of `ty` (never byte stride)
/// for SPIR-V `Aligned` memory access operands.
fn ir_type_align(ty: &IrType) -> u32 {
    match ty {
        IrType::Scalar(s) => s.byte_size().max(1).next_power_of_two(),
        IrType::Vector { scalar, len } => {
            let elem = scalar.byte_size().max(1);
            let comps = if *len == 3 { 4 } else { u32::from(*len).max(1) };
            elem.saturating_mul(comps).next_power_of_two()
        }
        IrType::Matrix { scalar, rows, .. } => {
            let elem = scalar.byte_size().max(1);
            let col_elems = if *rows == 3 {
                4
            } else {
                u32::from(*rows).max(1)
            };
            elem.saturating_mul(col_elems).next_power_of_two()
        }
        IrType::Array { element, .. } | IrType::RuntimeArray(element) => ir_type_align(element),
        IrType::Struct(layout) => layout.align.max(1).next_power_of_two(),
        IrType::Pointer {
            storage: IrStorageClass::PhysicalStorageBuffer,
            ..
        } => 8,
        IrType::Void | IrType::Pointer { .. } => 4,
    }
}

fn matrix_col_stride(ty: &IrType) -> Option<u32> {
    match ty {
        IrType::Matrix { scalar, rows, .. } => {
            let col_elems = if *rows == 3 { 4 } else { u32::from(*rows) };
            Some(scalar.byte_size().saturating_mul(col_elems))
        }
        IrType::Array { element, .. } | IrType::RuntimeArray(element) => matrix_col_stride(element),
        _ => None,
    }
}

fn ir_type_contains_f16(ty: &IrType) -> bool {
    match ty {
        IrType::Scalar(ScalarIrTy::F16)
        | IrType::Vector {
            scalar: ScalarIrTy::F16,
            ..
        }
        | IrType::Matrix {
            scalar: ScalarIrTy::F16,
            ..
        } => true,
        IrType::Array { element, .. }
        | IrType::RuntimeArray(element)
        | IrType::Pointer {
            pointee: element, ..
        } => ir_type_contains_f16(element),
        IrType::Struct(layout) => layout.fields.iter().any(|f| ir_type_contains_f16(&f.ty)),
        _ => false,
    }
}

/// Emits a valid SPIR-V 1.6 module for an [`IrModule`], configuring Vulkan Memory Model,
/// `PhysicalStorageBuffer64`, deduplicated descriptor and workgroup/push-constant globals,
/// capability/extension headers, helper functions, and structured entry-point bodies.
pub fn emit_module_spirv(module: &IrModule) -> Vec<u32> {
    let mut emitter = SpirvEmitter::new();
    emitter.emit_capability(spirv::Capability::Shader);

    let mut caps = HardwareCaps::empty();
    let mut uses_bda = false;
    let mut uses_bda_mem_op = false;
    let mut uses_any_atomic = false;
    let mut uses_device_atomic = false;
    let mut uses_int64_atomic = false;
    let mut uses_float_atomic_add = false;
    let mut uses_float_atomic_min_max = false;
    let mut uses_subgroups = false;
    let mut uses_subgroup_vote = false;
    let mut uses_subgroup_arith = false;
    let mut uses_subgroup_shuffle = false;
    let mut uses_subgroup_quad = false;
    let mut uses_subgroup_basic = false;
    let mut uses_quad_control = false;
    let mut uses_maximal_reconvergence = false;
    let mut uses_int64 = false;
    let mut uses_f16 = false;
    let mut uses_f16_storage = false;
    let mut uses_f16_uniform_storage = false;
    let mut uses_f16_stage_io = false;
    let mut uses_runtime_descriptor_array = false;
    let mut uses_non_uniform = false;
    let mut uses_accel_struct = false;
    let mut uses_ray_query = false;
    let mut uses_image_query = false;
    let mut uses_demote = false;
    let mut uses_storage_image_extended_formats = false;
    let mut uses_storage_image_read_without_format = false;
    let mut uses_storage_image_write_without_format = false;

    for func in &module.functions {
        caps |= func.required_caps;
        uses_bda |= func.uses_bda;
        uses_subgroups |= func.uses_subgroups;
        uses_quad_control |= func.uses_quad_control;
        if func.maximally_reconverges
            || func.uses_quad_control
            || func
                .required_caps
                .contains(HardwareCaps::MAXIMAL_RECONVERGENCE)
        {
            uses_maximal_reconvergence = true;
        }
        if ir_type_contains_f16(&func.return_ty) {
            uses_f16 = true;
        }
        for p in &func.params {
            if ir_type_contains_f16(&p.ty) {
                uses_f16 = true;
            }
        }
        for lv in &func.local_vars {
            if ir_type_contains_f16(&lv.ty) {
                uses_f16 = true;
            }
        }
        for wg in &func.workgroup_vars {
            if ir_type_contains_f16(&wg.effective_type()) {
                uses_f16 = true;
            }
        }
        for io in &func.stage_io {
            if ir_type_contains_f16(&io.ty) {
                uses_f16 = true;
                uses_f16_stage_io = true;
            }
        }
        for desc in &func.descriptors {
            if desc.is_unsized {
                uses_runtime_descriptor_array = true;
            }
            if desc.kind == DescriptorKind::AccelerationStructure {
                uses_accel_struct = true;
            }
            if matches!(
                desc.image_format,
                Some(ImageFormat::Rg16f | ImageFormat::R16f | ImageFormat::Rg8 | ImageFormat::R8)
            ) {
                uses_storage_image_extended_formats = true;
            }
            if let Some(layout) = &desc.struct_layout
                && layout.fields.iter().any(|f| ir_type_contains_f16(&f.ty))
            {
                uses_f16 = true;
                uses_f16_storage = true;
                if desc.kind == DescriptorKind::UniformBuffer {
                    uses_f16_uniform_storage = true;
                }
            }
        }
        for (_, inst) in func.insts.iter() {
            if inst.non_uniform {
                uses_non_uniform = true;
            }
            if ir_type_contains_f16(&inst.ir_ty) {
                uses_f16 = true;
            }
            if inst.ir_ty.is_64bit() {
                uses_int64 = true;
            }
            match &inst.data {
                InstData::ConstF16(_)
                | InstData::FConvert {
                    target: ScalarIrTy::F16,
                    ..
                } => {
                    uses_f16 = true;
                }
                InstData::ConstI64(_)
                | InstData::ConstU64(_)
                | InstData::IAdd64(_, _)
                | InstData::IMul64(_, _)
                | InstData::CastI64ToU64(_)
                | InstData::CastToI64(_)
                | InstData::CastToU64(_) => {
                    uses_int64 = true;
                }
                InstData::ConvertPtrToU(_)
                | InstData::ConvertUToPtr { .. }
                | InstData::BdaOffset { .. } => {
                    uses_bda = true;
                    uses_int64 = true;
                }
                InstData::BdaLoad { .. } | InstData::BdaStore { .. } => {
                    uses_bda = true;
                    uses_bda_mem_op = true;
                    uses_int64 = true;
                    if ir_type_contains_f16(&inst.ir_ty) {
                        uses_f16_storage = true;
                    }
                }
                InstData::AtomicAdd { value, scope, .. } => {
                    uses_any_atomic = true;
                    if *scope == AtomicScope::Device {
                        uses_device_atomic = true;
                    }
                    if inst.ir_ty.is_64bit() || func.insts[*value].ir_ty.is_64bit() {
                        uses_int64 = true;
                        uses_int64_atomic = true;
                    }
                    if inst.ir_ty.is_float() || func.insts[*value].ir_ty.is_float() {
                        uses_float_atomic_add = true;
                    }
                }
                InstData::AtomicMin { value, scope, .. }
                | InstData::AtomicMax { value, scope, .. } => {
                    uses_any_atomic = true;
                    if *scope == AtomicScope::Device {
                        uses_device_atomic = true;
                    }
                    if inst.ir_ty.is_64bit() || func.insts[*value].ir_ty.is_64bit() {
                        uses_int64 = true;
                        uses_int64_atomic = true;
                    }
                    if inst.ir_ty.is_float() || func.insts[*value].ir_ty.is_float() {
                        uses_float_atomic_min_max = true;
                    }
                }
                InstData::MemoryBarrier {
                    scope: AtomicScope::Device,
                    ..
                } => {
                    uses_device_atomic = true;
                }
                InstData::AtomicCas {
                    expected, scope, ..
                } => {
                    uses_any_atomic = true;
                    if *scope == AtomicScope::Device {
                        uses_device_atomic = true;
                    }
                    if inst.ir_ty.is_64bit() || func.insts[*expected].ir_ty.is_64bit() {
                        uses_int64 = true;
                        uses_int64_atomic = true;
                    }
                }
                InstData::ImageSample {
                    descriptor_index, ..
                } => {
                    if let Some(desc) = func.descriptors.get(*descriptor_index as usize)
                        && desc.kind == DescriptorKind::StorageImage
                        && desc.image_format.is_none()
                        && desc.access != DescriptorAccess::WriteOnly
                    {
                        uses_storage_image_read_without_format = true;
                    }
                }
                InstData::ImageStore {
                    descriptor_index, ..
                } => {
                    if let Some(desc) = func.descriptors.get(*descriptor_index as usize)
                        && desc.kind == DescriptorKind::StorageImage
                        && desc.image_format.is_none()
                    {
                        uses_storage_image_write_without_format = true;
                    }
                }
                InstData::ResourceAccessChain {
                    descriptor_index,
                    index,
                } => {
                    if let Some(desc) = func.descriptors.get(*descriptor_index as usize)
                        && (desc.is_unsized || desc.count.is_some())
                    {
                        let is_dynamic = !is_const_inst_data(&func.insts[*index].data);
                        let is_non_uniform = inst.non_uniform
                            || func.insts[*index].non_uniform
                            || matches!(func.insts[*index].data, InstData::NonUniform(_));
                        if is_non_uniform {
                            uses_non_uniform = true;
                        }
                        match desc.kind {
                            DescriptorKind::SampledImage | DescriptorKind::CombinedImageSampler => {
                                if is_dynamic {
                                    emitter.emit_capability(
                                        spirv::Capability::SampledImageArrayDynamicIndexing,
                                    );
                                }
                                if is_non_uniform {
                                    emitter.emit_capability(
                                        spirv::Capability::SampledImageArrayNonUniformIndexing,
                                    );
                                }
                            }
                            DescriptorKind::StorageImage => {
                                if is_dynamic {
                                    emitter.emit_capability(
                                        spirv::Capability::StorageImageArrayDynamicIndexing,
                                    );
                                }
                                if is_non_uniform {
                                    emitter.emit_capability(
                                        spirv::Capability::StorageImageArrayNonUniformIndexing,
                                    );
                                }
                            }
                            DescriptorKind::StorageBuffer => {
                                if is_dynamic {
                                    emitter.emit_capability(
                                        spirv::Capability::StorageBufferArrayDynamicIndexing,
                                    );
                                }
                                if is_non_uniform {
                                    emitter.emit_capability(
                                        spirv::Capability::StorageBufferArrayNonUniformIndexing,
                                    );
                                }
                            }
                            DescriptorKind::UniformBuffer => {
                                if is_dynamic {
                                    emitter.emit_capability(
                                        spirv::Capability::UniformBufferArrayDynamicIndexing,
                                    );
                                }
                                if is_non_uniform {
                                    emitter.emit_capability(
                                        spirv::Capability::UniformBufferArrayNonUniformIndexing,
                                    );
                                }
                            }
                            DescriptorKind::Sampler
                            | DescriptorKind::AccelerationStructure
                            | DescriptorKind::InputAttachment => {}
                        }
                    }
                }
                InstData::ImageQuerySize { .. } | InstData::ImageQueryLevels { .. } => {
                    uses_image_query = true;
                }
                InstData::SubgroupBallot(_)
                | InstData::SubgroupBallotBitCount(_)
                | InstData::SubgroupBallotExclusiveBitCount(_)
                | InstData::SubgroupBallotInclusiveBitCount(_)
                | InstData::SubgroupBroadcastFirst(_)
                | InstData::SubgroupBroadcast { .. }
                | InstData::SubgroupBallotBitExtract { .. }
                | InstData::SubgroupBallotFindLsb(_)
                | InstData::SubgroupBallotFindMsb(_) => {
                    uses_subgroups = true;
                }
                InstData::SubgroupAdd(_)
                | InstData::SubgroupMin(_)
                | InstData::SubgroupMax(_)
                | InstData::SubgroupExclusiveAdd(_)
                | InstData::SubgroupInclusiveAdd(_)
                | InstData::SubgroupMul(_)
                | InstData::SubgroupExclusiveMul(_)
                | InstData::SubgroupInclusiveMul(_)
                | InstData::SubgroupExclusiveMin(_)
                | InstData::SubgroupInclusiveMin(_)
                | InstData::SubgroupExclusiveMax(_)
                | InstData::SubgroupInclusiveMax(_)
                | InstData::SubgroupBitAnd(_)
                | InstData::SubgroupBitOr(_)
                | InstData::SubgroupBitXor(_)
                | InstData::SubgroupLogicalAnd(_)
                | InstData::SubgroupLogicalOr(_)
                | InstData::SubgroupLogicalXor(_)
                | InstData::SubgroupClustered { .. } => {
                    uses_subgroup_arith = true;
                }
                InstData::SubgroupShuffle { .. }
                | InstData::SubgroupShuffleXor { .. }
                | InstData::SubgroupShuffleUp { .. }
                | InstData::SubgroupShuffleDown { .. } => {
                    uses_subgroup_shuffle = true;
                }
                InstData::SubgroupAny(_) | InstData::SubgroupAll(_) => {
                    uses_subgroup_vote = true;
                }
                InstData::SubgroupElect => {
                    uses_subgroup_basic = true;
                }
                InstData::QuadSwapHorizontal(_)
                | InstData::QuadSwapVertical(_)
                | InstData::QuadSwapDiagonal(_)
                | InstData::QuadBroadcast { .. } => {
                    uses_subgroup_quad = true;
                }
                InstData::LoadBuiltin {
                    kind: BuiltinKind::SubgroupLocalInvocationId | BuiltinKind::SubgroupSize,
                    ..
                } => {
                    uses_subgroup_basic = true;
                }
                InstData::DemoteToHelperInvocation => {
                    uses_demote = true;
                }
                InstData::RayQueryInitialize { .. }
                | InstData::RayQueryProceed(_)
                | InstData::RayQueryCommittedType(_)
                | InstData::RayQueryCandidateType(_)
                | InstData::RayQueryConfirmIntersection(_)
                | InstData::RayQueryTerminate(_)
                | InstData::RayQueryGenerateIntersection { .. }
                | InstData::RayQueryCommittedT(_)
                | InstData::RayQueryCandidateT(_)
                | InstData::RayQueryCommittedInstanceId(_)
                | InstData::RayQueryCommittedInstanceCustomIndex(_)
                | InstData::RayQueryCommittedPrimitiveIndex(_)
                | InstData::RayQueryCommittedGeometryIndex(_)
                | InstData::RayQueryCommittedObjectToWorld(_)
                | InstData::RayQueryCommittedWorldToObject(_)
                | InstData::RayQueryCommittedFrontFace(_)
                | InstData::RayQueryCommittedBarycentrics(_) => {
                    uses_ray_query = true;
                }
                _ => {}
            }
        }
    }

    let needs_int64 =
        uses_int64 || uses_bda || uses_int64_atomic || caps.contains(HardwareCaps::INT64_ATOMICS);
    if needs_int64 {
        emitter.emit_capability(spirv::Capability::Int64);
    }
    if uses_bda {
        emitter.emit_capability(spirv::Capability::PhysicalStorageBufferAddresses);
        emitter.emit_capability(spirv::Capability::VulkanMemoryModel);
        if uses_device_atomic || uses_any_atomic || uses_bda_mem_op {
            emitter.emit_capability(spirv::Capability::VulkanMemoryModelDeviceScope);
        }
        emitter.emit_extension("SPV_KHR_physical_storage_buffer");
        emitter.emit_extension("SPV_KHR_vulkan_memory_model");
    }
    if uses_storage_image_extended_formats {
        emitter.emit_capability(spirv::Capability::StorageImageExtendedFormats);
    }
    if uses_storage_image_read_without_format {
        emitter.emit_capability(spirv::Capability::StorageImageReadWithoutFormat);
    }
    if uses_storage_image_write_without_format {
        emitter.emit_capability(spirv::Capability::StorageImageWriteWithoutFormat);
    }
    if uses_image_query {
        emitter.emit_capability(spirv::Capability::ImageQuery);
    }
    if uses_runtime_descriptor_array {
        emitter.emit_capability(spirv::Capability::RuntimeDescriptorArray);
        emitter.emit_extension("SPV_EXT_descriptor_indexing");
    }
    if uses_non_uniform {
        emitter.emit_capability(spirv::Capability::ShaderNonUniform);
        emitter.emit_extension("SPV_EXT_descriptor_indexing");
    }
    if uses_f16 || caps.contains(HardwareCaps::FLOAT16) {
        emitter.emit_capability(spirv::Capability::Float16);
    }
    if uses_f16_storage {
        emitter.emit_capability(spirv::Capability::StorageBuffer16BitAccess);
        emitter.emit_extension("SPV_KHR_16bit_storage");
    }
    if uses_f16_uniform_storage {
        emitter.emit_capability(spirv::Capability::UniformAndStorageBuffer16BitAccess);
        emitter.emit_extension("SPV_KHR_16bit_storage");
    }
    if uses_f16_stage_io {
        emitter.emit_capability(spirv::Capability::StorageInputOutput16);
        emitter.emit_extension("SPV_KHR_16bit_storage");
    }
    if uses_int64_atomic || caps.contains(HardwareCaps::INT64_ATOMICS) {
        emitter.emit_capability(spirv::Capability::Int64Atomics);
    }
    if uses_float_atomic_add || caps.contains(HardwareCaps::FLOAT_ATOMICS) {
        emitter.emit_capability(spirv::Capability::AtomicFloat32AddEXT);
        emitter.emit_extension("SPV_EXT_shader_atomic_float_add");
    }
    if uses_float_atomic_min_max {
        emitter.emit_capability(spirv::Capability::AtomicFloat32MinMaxEXT);
        emitter.emit_extension("SPV_EXT_shader_atomic_float_min_max");
    }
    let uses_rt_pipeline = caps.contains(HardwareCaps::RAY_TRACING_PIPELINE)
        || module
            .functions
            .iter()
            .any(|f| f.stage.is_some_and(ShaderStage::is_ray_tracing));
    let uses_mesh_pipeline = caps.contains(HardwareCaps::MESH_SHADER)
        || module
            .functions
            .iter()
            .any(|f| f.stage.is_some_and(ShaderStage::is_mesh_pipeline));
    let uses_tessellation = caps.contains(HardwareCaps::TESSELLATION)
        || module.functions.iter().any(|f| {
            matches!(
                f.stage,
                Some(ShaderStage::TessControl | ShaderStage::TessEval)
            )
        });
    let uses_geometry = caps.contains(HardwareCaps::GEOMETRY_SHADER)
        || module
            .functions
            .iter()
            .any(|f| f.stage == Some(ShaderStage::Geometry));

    if uses_rt_pipeline {
        emitter.emit_capability_raw(4479); // RayTracingKHR
        emitter.emit_extension("SPV_KHR_ray_tracing");
    }
    if uses_mesh_pipeline {
        emitter.emit_capability_raw(5283); // MeshShadingEXT
        emitter.emit_extension("SPV_EXT_mesh_shader");
    }
    if uses_tessellation {
        emitter.emit_capability(spirv::Capability::Tessellation);
    }
    if uses_geometry {
        emitter.emit_capability(spirv::Capability::Geometry);
    }
    let uses_ser_ops = module.functions.iter().any(|f| {
        f.insts.iter().any(|(_, inst)| {
            matches!(
                inst.data,
                InstData::HitObjectTraceRay { .. }
                    | InstData::HitObjectRecordEmpty(_)
                    | InstData::HitObjectIsHit(_)
                    | InstData::HitObjectGetRayTMax(_)
                    | InstData::HitObjectGetSbtRecordIndex(_)
                    | InstData::HitObjectExecuteShader { .. }
                    | InstData::ReorderThreadWithHint { .. }
                    | InstData::ReorderThreadWithHitObject { .. }
            )
        })
    });
    let uses_ser_ext = caps.contains(HardwareCaps::SHADER_INVOCATION_REORDER_EXT);
    let uses_ser_nv =
        !uses_ser_ext && (uses_ser_ops || caps.contains(HardwareCaps::SHADER_INVOCATION_REORDER));
    if uses_ser_ext {
        emitter.emit_capability_raw(5388); // ShaderInvocationReorderEXT
        emitter.emit_extension("SPV_EXT_shader_invocation_reorder");
    } else if uses_ser_nv {
        emitter.emit_capability_raw(5383); // ShaderInvocationReorderNV
        emitter.emit_extension("SPV_NV_shader_invocation_reorder");
    }
    if caps.contains(HardwareCaps::OPACITY_MICROMAP) {
        emitter.emit_capability_raw(5381); // RayTracingOpacityMicromapEXT
        emitter.emit_extension("SPV_EXT_opacity_micromap");
    }
    let uses_rq_pos_fetch = module.functions.iter().any(|f| {
        f.insts
            .iter()
            .any(|(_, inst)| matches!(inst.data, InstData::RayQueryTriangleVertexPositions { .. }))
    });
    let uses_rt_pos_fetch = module.functions.iter().any(|f| {
        f.insts.iter().any(|(_, inst)| {
            matches!(
                inst.data,
                InstData::LoadBuiltin {
                    kind: BuiltinKind::HitTriangleVertexPositionsKhr,
                    ..
                }
            )
        })
    });
    if uses_rq_pos_fetch {
        emitter.emit_capability_raw(5391); // RayQueryPositionFetchKHR
        emitter.emit_extension("SPV_KHR_ray_tracing_position_fetch");
    }
    if uses_rt_pos_fetch
        || (caps.contains(HardwareCaps::RAY_TRACING_POSITION_FETCH) && uses_rt_pipeline)
    {
        emitter.emit_capability_raw(5336); // RayTracingPositionFetchKHR
        emitter.emit_extension("SPV_KHR_ray_tracing_position_fetch");
    }
    if uses_ray_query
        || uses_rq_pos_fetch
        || caps.contains(HardwareCaps::RAY_QUERY)
        || (uses_accel_struct && !uses_rt_pipeline)
    {
        emitter.emit_capability(spirv::Capability::RayQueryKHR);
        emitter.emit_extension("SPV_KHR_ray_query");
    }
    if uses_subgroup_basic {
        emitter.emit_capability(spirv::Capability::GroupNonUniform);
    }
    if uses_subgroups || caps.contains(HardwareCaps::MAXIMAL_RECONVERGENCE) {
        emitter.emit_capability(spirv::Capability::GroupNonUniformBallot);
    }
    if uses_maximal_reconvergence {
        emitter.emit_extension("SPV_KHR_maximal_reconvergence");
    }
    if uses_subgroup_vote {
        emitter.emit_capability(spirv::Capability::GroupNonUniformVote);
    }
    if uses_subgroup_arith {
        emitter.emit_capability(spirv::Capability::GroupNonUniformArithmetic);
    }
    if uses_subgroup_shuffle {
        emitter.emit_capability(spirv::Capability::GroupNonUniformShuffle);
    }
    if uses_subgroup_quad {
        emitter.emit_capability(spirv::Capability::GroupNonUniformQuad);
    }
    if uses_quad_control || caps.contains(HardwareCaps::QUAD_CONTROL) {
        emitter.emit_capability(spirv::Capability::QuadControlKHR);
        emitter.emit_extension("SPV_KHR_quad_control");
    }
    if uses_demote {
        emitter.emit_capability(spirv::Capability::DemoteToHelperInvocation);
        emitter.emit_extension("SPV_EXT_demote_to_helper_invocation");
    }

    let uses_coop_matrix = caps.contains(HardwareCaps::COOPERATIVE_MATRIX)
        || module.functions.iter().any(|f| {
            f.insts.iter().any(|(_, inst)| {
                matches!(
                    inst.data,
                    InstData::CoopMatrixLoad { .. }
                        | InstData::CoopMatrixStore { .. }
                        | InstData::CoopMatrixMulAdd { .. }
                        | InstData::CoopMatrixLength { .. }
                ) || matches!(inst.ir_ty, IrType::CoopMatrix { .. })
            })
        });
    if uses_coop_matrix {
        emitter.emit_capability_raw(6022); // CooperativeMatrixKHR
        emitter.emit_extension("SPV_KHR_cooperative_matrix");
        emitter.emit_capability(spirv::Capability::VulkanMemoryModel);
        emitter.emit_extension("SPV_KHR_vulkan_memory_model");
    }

    if uses_bda {
        emitter.emit_memory_model(
            spirv::AddressingModel::PhysicalStorageBuffer64,
            spirv::MemoryModel::Vulkan,
        );
    } else if uses_coop_matrix {
        emitter.emit_memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::Vulkan);
    } else {
        emitter.emit_memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    }

    let void_ty = emitter.emit_type_void();
    let entry_fn_ty = emitter.emit_type_function(void_ty, &[]);

    let module_debug_ctx: Option<(u32, u32, u32, u32)> = module
        .functions
        .iter()
        .find_map(|f| f.debug_source.as_ref())
        .map(|dbg| {
            let ext_id = emitter.emit_debug_info_import();
            let u32_ty = emitter.emit_type_int(32, 0);
            let file_str = emitter.emit_debug_string(dbg.file_name.as_str());
            let info_none = emitter.emit_types_ext_inst(void_ty, ext_id, 0, &[]); // DebugInfoNone
            let debug_source = if let Some(src_text) = &dbg.source_text {
                let text_str = emitter.emit_debug_string(src_text.as_str());
                emitter.emit_types_ext_inst(void_ty, ext_id, 35, &[file_str, text_str])
            // DebugSource
            } else {
                emitter.emit_types_ext_inst(void_ty, ext_id, 35, &[file_str])
            };
            let ver_100 = emitter.emit_constant_32(u32_ty, 100);
            let dwarf_4 = emitter.emit_constant_32(u32_ty, 4);
            let lang_hlsl = emitter.emit_constant_32(u32_ty, 5);
            let compile_unit = emitter.emit_types_ext_inst(
                void_ty,
                ext_id,
                1, // DebugCompilationUnit
                &[ver_100, dwarf_4, debug_source, lang_hlsl],
            );
            let flags_0 = emitter.emit_constant_32(u32_ty, 0);
            let debug_fn_ty = emitter.emit_types_ext_inst(
                void_ty,
                ext_id,
                8, // DebugTypeFunction
                &[flags_0, info_none],
            );
            (ext_id, debug_source, compile_unit, debug_fn_ty)
        });

    // Pass 1: Pre-allocate function IDs and signatures for all helpers and entry points so
    // `OpFunctionCall` can reference any callee in `module.functions` regardless of order.
    let mut emitted_functions: IndexMap<String, EmittedFunctionInfo> = IndexMap::new();
    for func in &module.functions {
        let fn_id = emitter.alloc_id();
        let (ret_ty, param_tys, fn_ty) = if func.stage.is_some() {
            (void_ty, Vec::new(), entry_fn_ty)
        } else {
            let ret_ty = emitter.emit_ir_type(&func.return_ty);
            let mut param_tys = Vec::with_capacity(func.params.len());
            let mut raw_params = Vec::with_capacity(func.params.len());
            for p in &func.params {
                let val_ty = emitter.emit_ir_type(&p.ty);
                if p.is_inout || p.is_out {
                    let ptr_ty = emitter.emit_type_pointer(spirv::StorageClass::Function, val_ty);
                    param_tys.push((ptr_ty, true));
                    raw_params.push(ptr_ty);
                } else {
                    param_tys.push((val_ty, false));
                    raw_params.push(val_ty);
                }
            }
            let fn_ty = emitter.emit_type_function(ret_ty, &raw_params);
            (ret_ty, param_tys, fn_ty)
        };
        emitted_functions.insert(
            func.name.to_string(),
            EmittedFunctionInfo {
                fn_id,
                fn_ty,
                ret_ty,
                param_tys,
            },
        );
    }

    // Deduplicate module-level descriptors by `(set, binding)`, workgroup variables by
    // `(name, effective_type)`, push-constant blocks by `(total_size, fields)`, and
    // built-in variables by `(BuiltinKind, is_output)`.
    let mut descriptor_vars: IndexMap<(u32, u32), EmittedDescriptor> = IndexMap::new();
    let mut workgroup_vars_map: IndexMap<WorkgroupVarKey, EmittedWorkgroupVar> = IndexMap::new();
    let mut push_constant_blocks: IndexMap<PushConstantKey, EmittedPushConstant> = IndexMap::new();
    let mut builtin_vars_map: IndexMap<(BuiltinKind, bool), EmittedBuiltinVar> = IndexMap::new();
    let mut func_direct_globals: IndexMap<String, Vec<u32>> = IndexMap::new();
    let mut func_callees: IndexMap<String, Vec<String>> = IndexMap::new();
    let mut pending_entry_points: Vec<(u32, u32, String)> = Vec::new();

    // Emit non-entry helper functions first, then entry points.
    let ordered_indices: Vec<usize> = module
        .functions
        .iter()
        .enumerate()
        .filter(|(_, f)| f.stage.is_none())
        .map(|(i, _)| i)
        .chain(
            module
                .functions
                .iter()
                .enumerate()
                .filter(|(_, f)| f.stage.is_some())
                .map(|(i, _)| i),
        )
        .collect();

    // Pre-scan all functions to determine OpTypeImage `Depth` operand (0 = color, 1 = depth-only,
    // 2 = both dref and non-dref) per descriptor `(set, binding)`.
    let mut descriptor_depth_flags: IndexMap<(u32, u32), (bool, bool)> = IndexMap::new();
    for f in &module.functions {
        for (_, inst) in f.insts.iter() {
            match &inst.data {
                InstData::ImageSampleDrefExplicitLod {
                    descriptor_index, ..
                } => {
                    if let Some(d) = f.descriptors.get(*descriptor_index as usize) {
                        let entry = descriptor_depth_flags
                            .entry((d.set, d.binding))
                            .or_insert((false, false));
                        entry.0 = true;
                    }
                }
                InstData::ImageSample {
                    descriptor_index, ..
                }
                | InstData::ImageSampleImplicitLod {
                    descriptor_index, ..
                }
                | InstData::ImageSampleGrad {
                    descriptor_index, ..
                }
                | InstData::ImageGather {
                    descriptor_index, ..
                } => {
                    if let Some(d) = f.descriptors.get(*descriptor_index as usize) {
                        let entry = descriptor_depth_flags
                            .entry((d.set, d.binding))
                            .or_insert((false, false));
                        entry.1 = true;
                    }
                }
                _ => {}
            }
        }
    }

    for func_idx in ordered_indices {
        let func = &module.functions[func_idx];
        let fn_info = emitted_functions[func.name.as_str()].clone();
        let mut interface_ids = Vec::with_capacity(
            func.descriptors.len() + func.workgroup_vars.len() + func.stage_io.len() + 2,
        );
        let mut interface_seen: IndexSet<u32> = IndexSet::new();
        let mut func_descriptors = Vec::with_capacity(func.descriptors.len());
        let mut func_workgroup_vars = Vec::with_capacity(func.workgroup_vars.len());
        let mut func_stage_io = Vec::with_capacity(func.stage_io.len());

        for desc in &func.descriptors {
            let key = (desc.set, desc.binding);
            let desc_var = if let Some(existing) = descriptor_vars.get(&key) {
                existing.clone()
            } else {
                let depth_operand = match descriptor_depth_flags.get(&key).copied() {
                    Some((true, true)) => 2,
                    Some((true, false)) => 1,
                    _ => 0,
                };
                let created = emit_descriptor_var(&mut emitter, desc, depth_operand);
                descriptor_vars.insert(key, created.clone());
                created
            };
            if interface_seen.insert(desc_var.var_id) {
                interface_ids.push(desc_var.var_id);
            }
            func_descriptors.push(desc_var);
        }
        for &(desc_idx, att_idx) in &func.stage_exec_meta.input_attachment_indices {
            if let Some(desc_var) = func_descriptors.get(desc_idx as usize) {
                emitter.emit_decorate(
                    desc_var.var_id,
                    spirv::Decoration::InputAttachmentIndex,
                    &[att_idx],
                );
            }
        }

        if func.stage.is_none() || func.stage.is_some_and(ShaderStage::allows_workgroup_vars) {
            for wg in &func.workgroup_vars {
                let wg_key = WorkgroupVarKey {
                    name: wg.name.to_string(),
                    ty: wg.effective_type(),
                };
                let wg_var = if let Some(existing) = workgroup_vars_map.get(&wg_key) {
                    existing.clone()
                } else {
                    let created = emit_workgroup_var(&mut emitter, wg);
                    workgroup_vars_map.insert(wg_key, created.clone());
                    created
                };
                if interface_seen.insert(wg_var.var_id) {
                    interface_ids.push(wg_var.var_id);
                }
                func_workgroup_vars.push(wg_var);
            }
        }

        for io in &func.stage_io {
            let sc = if io.is_output {
                spirv::StorageClass::Output
            } else {
                spirv::StorageClass::Input
            };
            let val_ty = emitter.emit_ir_type(&io.ty);
            let ptr_ty = emitter.emit_type_pointer(sc, val_ty);
            let var_id = emitter.emit_variable(ptr_ty, sc);
            if !io.name.is_empty() {
                emitter.emit_name(var_id, io.name.as_str());
            }
            emitter.emit_decorate(var_id, spirv::Decoration::Location, &[io.location]);
            match io.interpolation {
                IrInterpolation::Smooth => {}
                IrInterpolation::Flat => {
                    emitter.emit_decorate(var_id, spirv::Decoration::Flat, &[]);
                }
                IrInterpolation::NoPerspective => {
                    emitter.emit_decorate(var_id, spirv::Decoration::NoPerspective, &[]);
                }
                IrInterpolation::Centroid => {
                    emitter.emit_decorate(var_id, spirv::Decoration::Centroid, &[]);
                }
            }
            if io.per_primitive {
                emitter.emit_decorate_raw(var_id, 5271, &[]); // PerPrimitiveEXT
            }
            if interface_seen.insert(var_id) {
                interface_ids.push(var_id);
            }
            func_stage_io.push(EmittedStageIoVar {
                var_id,
                val_ty,
                is_output: io.is_output,
            });
        }

        let mut func_stage_vars: Vec<EmittedStageVar> = Vec::with_capacity(func.stage_vars.len());
        for sv in &func.stage_vars {
            let sc_raw = map_ir_storage_class_raw(sv.storage);
            let val_ty = if sv.storage == IrStorageClass::ShaderRecordBufferKhr {
                match &sv.ty {
                    IrType::Struct(layout) => emitter.emit_struct_type(layout, true, true),
                    other => emitter.emit_ir_type_with_layout(other, true),
                }
            } else {
                emitter.emit_ir_type(&sv.ty)
            };
            let ptr_ty = emitter.emit_type_pointer_raw(sc_raw, val_ty);
            let var_id = emitter.emit_variable_raw(ptr_ty, sc_raw);
            if !sv.name.is_empty() {
                emitter.emit_name(var_id, sv.name.as_str());
            }
            if let Some(loc) = sv.location {
                emitter.emit_decorate(var_id, spirv::Decoration::Location, &[loc]);
            }
            if interface_seen.insert(var_id) {
                interface_ids.push(var_id);
            }
            func_stage_vars.push(EmittedStageVar { var_id, val_ty });
        }

        let mut func_push_constant: Option<EmittedPushConstant> = None;
        if let Some(pc) = &func.push_constants {
            let pc_key = PushConstantKey {
                total_size: pc.total_size,
                fields: pc
                    .fields
                    .iter()
                    .map(|f| (f.name.to_string(), f.byte_offset, f.byte_size, f.scalar))
                    .collect(),
            };
            let emitted_pc = if let Some(existing) = push_constant_blocks.get(&pc_key) {
                existing.clone()
            } else {
                let created = emit_push_constant_block(&mut emitter, pc);
                push_constant_blocks.insert(pc_key, created.clone());
                created
            };
            if interface_seen.insert(emitted_pc.var_id) {
                interface_ids.push(emitted_pc.var_id);
            }
            func_push_constant = Some(emitted_pc);
        }

        let mut func_builtins: IndexMap<(BuiltinKind, bool), EmittedBuiltinVar> = IndexMap::new();
        let mut writes_frag_depth = false;
        for (_, inst) in func.insts.iter() {
            let bi_entry = match inst.data {
                InstData::LoadBuiltin { kind, .. } => Some((kind, false)),
                InstData::StoreBuiltin { kind, .. } => {
                    if kind == BuiltinKind::FragDepth {
                        writes_frag_depth = true;
                    }
                    Some((kind, true))
                }
                _ => None,
            };
            if let Some((kind, is_output)) = bi_entry
                && !func_builtins.contains_key(&(kind, is_output))
            {
                let emitted_bi = if let Some(&existing) = builtin_vars_map.get(&(kind, is_output)) {
                    existing
                } else {
                    let created = emit_builtin_var(&mut emitter, kind, is_output);
                    builtin_vars_map.insert((kind, is_output), created);
                    created
                };
                func_builtins.insert((kind, is_output), emitted_bi);
                if interface_seen.insert(emitted_bi.var_id) {
                    interface_ids.push(emitted_bi.var_id);
                }
            }
        }

        // Only separate sampled images need the fallback sampler; combined image samplers
        // (including the Set 2 bindless array) carry their own sampler and emit no extra binding.
        let needs_default_sampler = func.insts.iter().any(|(_, inst)| match inst.data {
            InstData::ImageSample {
                descriptor_index,
                is_fetch,
                ..
            } => func_descriptors
                .get(descriptor_index as usize)
                .is_some_and(|desc| {
                    desc.kind == DescriptorKind::SampledImage && (!is_fetch || desc.is_cube)
                }),
            InstData::ImageSampleImplicitLod {
                descriptor_index, ..
            }
            | InstData::ImageSampleGrad {
                descriptor_index, ..
            }
            | InstData::ImageSampleDrefExplicitLod {
                descriptor_index, ..
            }
            | InstData::ImageGather {
                descriptor_index, ..
            } => func_descriptors
                .get(descriptor_index as usize)
                .is_some_and(|desc| desc.kind == DescriptorKind::SampledImage),
            _ => false,
        });
        let existing_sampler_var = func_descriptors
            .iter()
            .find(|d| d.kind == DescriptorKind::Sampler && !d.is_arrayed_descriptor())
            .map(|d| d.var_id);
        let default_sampler_var = if needs_default_sampler {
            if let Some(var_id) = existing_sampler_var {
                Some(var_id)
            } else {
                let mut used_set0: IndexSet<u32> = module
                    .functions
                    .iter()
                    .flat_map(|f| f.descriptors.iter())
                    .filter(|d| d.set == 0)
                    .map(|d| d.binding)
                    .collect();
                for &(set, binding) in descriptor_vars.keys() {
                    if set == 0 {
                        used_set0.insert(binding);
                    }
                }
                let sampler_binding = (15..=31)
                    .find(|b| !used_set0.contains(b))
                    .or_else(|| (1..15).find(|b| !used_set0.contains(b)))
                    .unwrap_or(15);
                let sampler_desc = IrDescriptorVar {
                    name: "vsl_default_sampler".into(),
                    set: 0,
                    binding: sampler_binding,
                    kind: DescriptorKind::Sampler,
                    dim: None,
                    sample_scalar: None,
                    image_format: None,
                    access: DescriptorAccess::ReadOnly,
                    is_unsized: false,
                    count: None,
                    struct_layout: None,
                };
                let key = (0, sampler_binding);
                let sampler_var = if let Some(existing) = descriptor_vars.get(&key) {
                    existing.clone()
                } else {
                    let created = emit_descriptor_var(&mut emitter, &sampler_desc, 0);
                    descriptor_vars.insert(key, created.clone());
                    created
                };
                if interface_seen.insert(sampler_var.var_id) {
                    interface_ids.push(sampler_var.var_id);
                }
                Some(sampler_var.var_id)
            }
        } else {
            existing_sampler_var
        };

        let fn_debug_ctx =
            module_debug_ctx.map(|(ext_id, debug_source, compile_unit, debug_fn_ty)| {
                let u32_ty = emitter.emit_type_int(32, 0);
                let name_str = emitter.emit_debug_string(func.name.as_str());
                let line_1 = emitter.emit_constant_32(u32_ty, 1);
                let col_1 = emitter.emit_constant_32(u32_ty, 1);
                let flags_0 = emitter.emit_constant_32(u32_ty, 0);
                let debug_fn = emitter.emit_types_ext_inst(
                    void_ty,
                    ext_id,
                    20, // DebugFunction
                    &[
                        name_str,
                        debug_fn_ty,
                        debug_source,
                        line_1,
                        col_1,
                        compile_unit,
                        name_str,
                        flags_0,
                        line_1,
                    ],
                );
                (ext_id, debug_source, debug_fn)
            });

        let fn_id = emit_function_body(
            &mut emitter,
            func,
            &fn_info,
            &emitted_functions,
            &func_descriptors,
            &func_workgroup_vars,
            &func_stage_io,
            &func_stage_vars,
            &func_builtins,
            func_push_constant.as_ref(),
            default_sampler_var,
            void_ty,
            uses_bda,
            fn_debug_ctx,
        );
        emitter.emit_name(fn_id, func.name.as_str());

        let callees: Vec<String> = func
            .insts
            .iter()
            .filter_map(|(_, inst)| match &inst.data {
                InstData::FunctionCall { callee, .. } => Some(callee.to_string()),
                _ => None,
            })
            .collect();
        func_direct_globals.insert(func.name.to_string(), interface_ids);
        func_callees.insert(func.name.to_string(), callees);

        if let Some(stage) = func.stage {
            let exec_model_raw: u32 = match stage {
                ShaderStage::Compute => spirv::ExecutionModel::GLCompute as u32,
                ShaderStage::Vertex => spirv::ExecutionModel::Vertex as u32,
                ShaderStage::Fragment => spirv::ExecutionModel::Fragment as u32,
                ShaderStage::RayGen => 5313,       // RayGenerationKHR
                ShaderStage::Intersection => 5314, // IntersectionKHR
                ShaderStage::AnyHit => 5315,       // AnyHitKHR
                ShaderStage::ClosestHit => 5316,   // ClosestHitKHR
                ShaderStage::Miss => 5317,         // MissKHR
                ShaderStage::Callable => 5318,     // CallableKHR
                ShaderStage::Task => 5364,         // TaskEXT
                ShaderStage::Mesh => 5365,         // MeshEXT
                ShaderStage::TessControl => spirv::ExecutionModel::TessellationControl as u32,
                ShaderStage::TessEval => spirv::ExecutionModel::TessellationEvaluation as u32,
                ShaderStage::Geometry => spirv::ExecutionModel::Geometry as u32,
            };
            pending_entry_points.push((exec_model_raw, fn_id, func.name.to_string()));
            if stage == ShaderStage::Fragment {
                emitter.emit_execution_mode(fn_id, spirv::ExecutionMode::OriginUpperLeft, &[]);
                if writes_frag_depth {
                    emitter.emit_execution_mode(fn_id, spirv::ExecutionMode::DepthReplacing, &[]);
                }
                if func.stage_exec_meta.early_fragment_tests {
                    emitter.emit_execution_mode(
                        fn_id,
                        spirv::ExecutionMode::EarlyFragmentTests,
                        &[],
                    );
                }
                if let Some(interlock) = func.stage_exec_meta.interlock_mode {
                    emitter.emit_extension("SPV_EXT_fragment_shader_interlock");
                    let (cap_raw, mode_raw) = match interlock {
                        IrInterlockMode::PixelOrdered => (5378, 5366),
                        IrInterlockMode::PixelUnordered => (5378, 5367),
                        IrInterlockMode::SampleOrdered => (5379, 5368),
                        IrInterlockMode::SampleUnordered => (5379, 5369),
                    };
                    emitter.emit_capability_raw(cap_raw);
                    emitter.emit_execution_mode_raw(fn_id, mode_raw, &[]);
                }
            }
            let meta = &func.stage_exec_meta;
            if stage == ShaderStage::Mesh {
                if let Some(v) = meta.max_vertices {
                    emitter.emit_execution_mode_raw(fn_id, 26, &[v]); // OutputVertices
                }
                if let Some(p) = meta.max_primitives {
                    emitter.emit_execution_mode_raw(fn_id, 5270, &[p]); // OutputPrimitivesEXT
                }
                if let Some(topo) = meta.output_topology {
                    let mode_raw = match topo {
                        IrTopology::Points => 27, // OutputPoints
                        IrTopology::Lines | IrTopology::LineStrip => 5269, // OutputLinesEXT
                        IrTopology::Triangles | IrTopology::TriangleStrip => 5298, // OutputTrianglesEXT
                    };
                    emitter.emit_execution_mode_raw(fn_id, mode_raw, &[]);
                }
            } else if stage == ShaderStage::Geometry {
                if let Some(inv) = meta.invocations {
                    emitter.emit_execution_mode_raw(fn_id, 0, &[inv]); // Invocations
                }
                if let Some(v) = meta.max_vertices {
                    emitter.emit_execution_mode_raw(fn_id, 26, &[v]); // OutputVertices
                }
                if let Some(in_topo) = meta.input_topology {
                    let mode_raw = match in_topo {
                        IrTopology::Points => 19, // InputPoints
                        IrTopology::Lines | IrTopology::LineStrip => 20, // InputLines
                        IrTopology::Triangles | IrTopology::TriangleStrip => 22, // Triangles
                    };
                    emitter.emit_execution_mode_raw(fn_id, mode_raw, &[]);
                }
                if let Some(out_topo) = meta.output_topology {
                    let mode_raw = match out_topo {
                        IrTopology::Points => 27, // OutputPoints
                        IrTopology::Lines | IrTopology::LineStrip => 28, // OutputLineStrip
                        IrTopology::Triangles | IrTopology::TriangleStrip => 29, // OutputTriangleStrip
                    };
                    emitter.emit_execution_mode_raw(fn_id, mode_raw, &[]);
                }
            } else if stage == ShaderStage::TessControl {
                if let Some(v) = meta.max_vertices {
                    emitter.emit_execution_mode_raw(fn_id, 26, &[v]); // OutputVertices
                }
            } else if stage == ShaderStage::TessEval {
                if let Some(dom) = meta.tess_domain {
                    let mode_raw = match dom {
                        IrTessDomain::Triangles => 22,
                        IrTessDomain::Quads => 24,
                        IrTessDomain::Isolines => 25,
                    };
                    emitter.emit_execution_mode_raw(fn_id, mode_raw, &[]);
                }
                if let Some(sp) = meta.tess_spacing {
                    let mode_raw = match sp {
                        IrTessSpacing::Equal => 1,
                        IrTessSpacing::FractionalEven => 2,
                        IrTessSpacing::FractionalOdd => 3,
                    };
                    emitter.emit_execution_mode_raw(fn_id, mode_raw, &[]);
                }
                if let Some(ord) = meta.tess_order {
                    let mode_raw = match ord {
                        IrTessOrder::Cw => 4,
                        IrTessOrder::Ccw => 5,
                    };
                    emitter.emit_execution_mode_raw(fn_id, mode_raw, &[]);
                }
                if meta.point_mode {
                    emitter.emit_execution_mode_raw(fn_id, 10, &[]); // PointMode
                }
            }
            if let Some([sx, sy, sz]) = func.local_size_spec_ids {
                let defaults = func.workgroup_size.unwrap_or([1, 1, 1]);
                let id_x = emitter.emit_spec_constant_u32(sx, defaults[0].max(1));
                let id_y = emitter.emit_spec_constant_u32(sy, defaults[1].max(1));
                let id_z = emitter.emit_spec_constant_u32(sz, defaults[2].max(1));
                emitter.emit_execution_mode_id(
                    fn_id,
                    spirv::ExecutionMode::LocalSizeId,
                    &[id_x, id_y, id_z],
                );
            } else if let Some([x, y, z]) = func.workgroup_size {
                emitter.emit_execution_mode(fn_id, spirv::ExecutionMode::LocalSize, &[x, y, z]);
            }
            let emit_quad_derivatives = func.uses_quad_control
                && matches!(stage, ShaderStage::Fragment | ShaderStage::Compute);
            if func
                .required_caps
                .contains(HardwareCaps::MAXIMAL_RECONVERGENCE)
                || func.maximally_reconverges
                || emit_quad_derivatives
            {
                emitter.emit_execution_mode(
                    fn_id,
                    spirv::ExecutionMode::MaximallyReconvergesKHR,
                    &[],
                );
            }
            if emit_quad_derivatives {
                emitter.emit_execution_mode(fn_id, spirv::ExecutionMode::QuadDerivativesKHR, &[]);
            }
        }
    }

    // Compute transitive OpEntryPoint interface closure across FunctionCall edges so every
    // global OpVariable referenced by reachable helper functions is declared on the entry point.
    for (exec_model, fn_id, entry_name) in pending_entry_points {
        let mut resolved_interface: Vec<u32> = Vec::new();
        let mut resolved_seen: IndexSet<u32> = IndexSet::new();
        let mut visited_fns: IndexSet<String> = IndexSet::new();
        let mut worklist: Vec<String> = vec![entry_name.clone()];
        while let Some(curr) = worklist.pop() {
            if !visited_fns.insert(curr.clone()) {
                continue;
            }
            if let Some(globals) = func_direct_globals.get(&curr) {
                for &gid in globals {
                    if resolved_seen.insert(gid) {
                        resolved_interface.push(gid);
                    }
                }
            }
            if let Some(callees) = func_callees.get(&curr) {
                for callee in callees {
                    if !visited_fns.contains(callee) {
                        worklist.push(callee.clone());
                    }
                }
            }
        }
        emitter.emit_entry_point_raw(exec_model, fn_id, &entry_name, &resolved_interface);
    }

    emitter.finish()
}

fn map_image_format(fmt: Option<ImageFormat>) -> spirv::ImageFormat {
    match fmt {
        None => spirv::ImageFormat::Unknown,
        Some(ImageFormat::Rgba16f) => spirv::ImageFormat::Rgba16f,
        Some(ImageFormat::Rg16f) => spirv::ImageFormat::Rg16f,
        Some(ImageFormat::R16f) => spirv::ImageFormat::R16f,
        Some(ImageFormat::R32f) => spirv::ImageFormat::R32f,
        Some(ImageFormat::Rgba8) => spirv::ImageFormat::Rgba8,
        Some(ImageFormat::Rg8) => spirv::ImageFormat::Rg8,
        Some(ImageFormat::R8) => spirv::ImageFormat::R8,
        Some(ImageFormat::R32ui) => spirv::ImageFormat::R32ui,
        Some(ImageFormat::R32i) => spirv::ImageFormat::R32i,
        Some(ImageFormat::Rgba32f) => spirv::ImageFormat::Rgba32f,
    }
}

/// Returns `(spv_dim, arrayed, coord_components, grad_components, query_size_components)`.
fn map_texture_dim(dim: Option<TextureDim>) -> (spirv::Dim, u32, u32, u32, u32) {
    match dim.unwrap_or(TextureDim::Dim2D) {
        TextureDim::Dim1D => (spirv::Dim::Dim1D, 0, 1, 1, 1),
        TextureDim::Dim2D | TextureDim::Dim2DMS => (spirv::Dim::Dim2D, 0, 2, 2, 2),
        TextureDim::Dim2DArray => (spirv::Dim::Dim2D, 1, 3, 2, 3),
        TextureDim::Dim3D => (spirv::Dim::Dim3D, 0, 3, 3, 3),
        TextureDim::DimCube => (spirv::Dim::DimCube, 0, 3, 3, 2),
        TextureDim::DimBuffer => (spirv::Dim::DimBuffer, 0, 1, 1, 1),
        TextureDim::SubpassData => (spirv::Dim::SubpassData, 0, 2, 2, 2),
    }
}

fn map_sample_scalar(emitter: &mut SpirvEmitter, scalar: Option<SampleScalar>) -> u32 {
    match scalar.unwrap_or(SampleScalar::F32) {
        SampleScalar::F32 => emitter.emit_type_float(32),
        SampleScalar::I32 => emitter.emit_type_int(32, 1),
        SampleScalar::U32 => emitter.emit_type_int(32, 0),
    }
}

fn resolve_descriptor_struct_layout(desc: &IrDescriptorVar) -> Option<IrStructLayout> {
    if let Some(layout) = &desc.struct_layout {
        return Some(layout.clone());
    }
    let builtin = builtin_descriptor_layout(desc.set, desc.binding)?;
    let matches_kind = match desc.kind {
        DescriptorKind::UniformBuffer => builtin.name.ends_with("UBO"),
        DescriptorKind::StorageBuffer => builtin.name.ends_with("SSBO"),
        _ => false,
    };
    if !matches_kind {
        return None;
    }
    match IrType::from_builtin_struct(builtin) {
        IrType::Struct(layout) => Some(layout),
        _ => None,
    }
}

fn emit_descriptor_var(
    emitter: &mut SpirvEmitter,
    desc: &IrDescriptorVar,
    depth_operand: u32,
) -> EmittedDescriptor {
    let u32_ty = emitter.emit_type_int(32, 0);
    let (spv_dim, arrayed, coord_components, grad_components, size_components) =
        map_texture_dim(desc.dim);
    let resolved_layout = resolve_descriptor_struct_layout(desc);

    let (storage_class, base_ty, raw_image_ty, sampled_ty) = match desc.kind {
        DescriptorKind::UniformBuffer => {
            let block_ty = if let Some(layout) = &resolved_layout {
                emitter.emit_struct_type(layout, true, true)
            } else {
                let block = emitter.emit_type_struct(&[u32_ty, u32_ty]);
                emitter.emit_decorate(block, spirv::Decoration::Block, &[]);
                emitter.emit_member_decorate(block, 0, spirv::Decoration::Offset, &[0]);
                emitter.emit_member_decorate(block, 1, spirv::Decoration::Offset, &[4]);
                block
            };
            (spirv::StorageClass::Uniform, block_ty, 0, u32_ty)
        }
        DescriptorKind::StorageBuffer => {
            let block_ty = if let Some(layout) = &resolved_layout {
                emitter.emit_struct_type(layout, true, true)
            } else {
                let arr_ty = emitter.emit_type_runtime_array_with_stride(u32_ty, Some(4));
                let block = emitter.emit_type_struct(&[arr_ty]);
                emitter.emit_decorate(block, spirv::Decoration::Block, &[]);
                emitter.emit_member_decorate(block, 0, spirv::Decoration::Offset, &[0]);
                block
            };
            (spirv::StorageClass::StorageBuffer, block_ty, 0, u32_ty)
        }
        DescriptorKind::SampledImage => {
            if desc.dim == Some(TextureDim::Dim1D) {
                emitter.emit_capability(spirv::Capability::Sampled1D);
            } else if desc.dim == Some(TextureDim::DimBuffer) {
                emitter.emit_capability(spirv::Capability::SampledBuffer);
            }
            let sampled_ty = map_sample_scalar(emitter, desc.sample_scalar);
            let ms = u32::from(desc.dim == Some(TextureDim::Dim2DMS));
            let img_ty = emitter.emit_type_image(
                sampled_ty,
                spv_dim,
                depth_operand,
                arrayed,
                ms,
                1,
                spirv::ImageFormat::Unknown,
            );
            (
                spirv::StorageClass::UniformConstant,
                img_ty,
                img_ty,
                sampled_ty,
            )
        }
        DescriptorKind::InputAttachment => {
            emitter.emit_capability(spirv::Capability::InputAttachment);
            let sampled_ty = map_sample_scalar(emitter, desc.sample_scalar);
            let img_ty = emitter.emit_type_image(
                sampled_ty,
                spirv::Dim::DimSubpassData,
                0,
                0,
                0,
                2,
                spirv::ImageFormat::Unknown,
            );
            (
                spirv::StorageClass::UniformConstant,
                img_ty,
                img_ty,
                sampled_ty,
            )
        }
        DescriptorKind::StorageImage => {
            if desc.dim == Some(TextureDim::Dim1D) {
                emitter.emit_capability(spirv::Capability::Image1D);
            } else if desc.dim == Some(TextureDim::DimBuffer) {
                emitter.emit_capability(spirv::Capability::ImageBuffer);
            }
            let sampled_ty = match desc.image_format {
                Some(ImageFormat::R32ui) => u32_ty,
                Some(ImageFormat::R32i) => emitter.emit_type_int(32, 1),
                _ => map_sample_scalar(emitter, desc.sample_scalar),
            };
            let img_ty = emitter.emit_type_image(
                sampled_ty,
                spv_dim,
                0,
                arrayed,
                0,
                2,
                map_image_format(desc.image_format),
            );
            (
                spirv::StorageClass::UniformConstant,
                img_ty,
                img_ty,
                sampled_ty,
            )
        }
        DescriptorKind::CombinedImageSampler => {
            let sampled_ty = map_sample_scalar(emitter, desc.sample_scalar);
            let img_ty = emitter.emit_type_image(
                sampled_ty,
                spv_dim,
                depth_operand,
                arrayed,
                0,
                1,
                spirv::ImageFormat::Unknown,
            );
            let sampled_img_ty = emitter.emit_type_sampled_image(img_ty);
            (
                spirv::StorageClass::UniformConstant,
                sampled_img_ty,
                img_ty,
                sampled_ty,
            )
        }
        DescriptorKind::Sampler => {
            let sampler_ty = emitter.emit_type_sampler();
            (spirv::StorageClass::UniformConstant, sampler_ty, 0, u32_ty)
        }
        DescriptorKind::AccelerationStructure => {
            let tlas_ty = emitter.emit_type_acceleration_structure_khr();
            (spirv::StorageClass::UniformConstant, tlas_ty, 0, u32_ty)
        }
    };

    let pointee_ty = if desc.is_unsized {
        emitter.emit_type_runtime_array(base_ty)
    } else if let Some(count) = desc.count {
        // Fixed descriptor arrays are reflected by rspirv as a binding with `count` elements.
        let len_const = emitter.emit_constant_32(u32_ty, count);
        emitter.emit_type_array(base_ty, len_const)
    } else {
        base_ty
    };
    let ptr_ty = emitter.emit_type_pointer(storage_class, pointee_ty);
    let var_id = emitter.emit_variable(ptr_ty, storage_class);

    emitter.emit_name(var_id, desc.name.as_str());
    emitter.emit_decorate(var_id, spirv::Decoration::DescriptorSet, &[desc.set]);
    emitter.emit_decorate(var_id, spirv::Decoration::Binding, &[desc.binding]);
    if desc.access == DescriptorAccess::ReadOnly
        && matches!(
            desc.kind,
            DescriptorKind::StorageBuffer | DescriptorKind::StorageImage
        )
    {
        emitter.emit_decorate(var_id, spirv::Decoration::NonWritable, &[]);
    }
    if desc.access == DescriptorAccess::WriteOnly
        && matches!(
            desc.kind,
            DescriptorKind::StorageBuffer | DescriptorKind::StorageImage
        )
    {
        emitter.emit_decorate(var_id, spirv::Decoration::NonReadable, &[]);
    }

    EmittedDescriptor {
        var_id,
        ptr_ty,
        storage_class,
        element_ty: base_ty,
        raw_image_ty,
        kind: desc.kind,
        access: desc.access,
        is_unsized: desc.is_unsized,
        count: desc.count,
        coord_components,
        grad_components,
        size_components,
        sampled_ty,
        is_cube: desc.dim == Some(TextureDim::DimCube),
        struct_layout: resolved_layout,
    }
}

fn emit_workgroup_var(emitter: &mut SpirvEmitter, wg: &IrWorkgroupVar) -> EmittedWorkgroupVar {
    let var_ir_ty = wg.effective_type();
    let var_ty = emitter.emit_ir_type(&var_ir_ty);
    let (elem_ty, is_array, elem_ir_ty) = match &var_ir_ty {
        IrType::Array { element, .. } => (
            emitter.emit_ir_type(element),
            true,
            element.as_ref().clone(),
        ),
        other => (var_ty, false, other.clone()),
    };
    let ptr_ty = emitter.emit_type_pointer(spirv::StorageClass::Workgroup, var_ty);
    let var_id = emitter.emit_variable(ptr_ty, spirv::StorageClass::Workgroup);
    emitter.emit_name(var_id, wg.name.as_str());
    EmittedWorkgroupVar {
        var_id,
        ptr_ty,
        elem_ty,
        is_array,
        var_ir_ty,
        elem_ir_ty,
    }
}

fn emit_builtin_var(
    emitter: &mut SpirvEmitter,
    kind: BuiltinKind,
    is_output: bool,
) -> EmittedBuiltinVar {
    let u32_ty = emitter.emit_type_int(32, 0);
    let f32_ty = emitter.emit_type_float(32);
    let bool_ty = emitter.emit_type_bool();
    let (value_ty, scalar_ty, vector_len, spv_builtin, name) = match kind {
        BuiltinKind::GlobalInvocationId => (
            emitter.emit_type_vector(u32_ty, 3),
            u32_ty,
            Some(3),
            spirv::BuiltIn::GlobalInvocationId as u32,
            "gl_GlobalInvocationID",
        ),
        BuiltinKind::LocalInvocationId => (
            emitter.emit_type_vector(u32_ty, 3),
            u32_ty,
            Some(3),
            spirv::BuiltIn::LocalInvocationId as u32,
            "gl_LocalInvocationID",
        ),
        BuiltinKind::LocalInvocationIndex => (
            u32_ty,
            u32_ty,
            None,
            spirv::BuiltIn::LocalInvocationIndex as u32,
            "gl_LocalInvocationIndex",
        ),
        BuiltinKind::WorkgroupId => (
            emitter.emit_type_vector(u32_ty, 3),
            u32_ty,
            Some(3),
            spirv::BuiltIn::WorkgroupId as u32,
            "gl_WorkGroupID",
        ),
        BuiltinKind::NumWorkgroups => (
            emitter.emit_type_vector(u32_ty, 3),
            u32_ty,
            Some(3),
            spirv::BuiltIn::NumWorkgroups as u32,
            "gl_NumWorkGroups",
        ),
        BuiltinKind::VertexIndex => (
            u32_ty,
            u32_ty,
            None,
            spirv::BuiltIn::VertexIndex as u32,
            "gl_VertexIndex",
        ),
        BuiltinKind::InstanceIndex => (
            u32_ty,
            u32_ty,
            None,
            spirv::BuiltIn::InstanceIndex as u32,
            "gl_InstanceIndex",
        ),
        BuiltinKind::FragCoord => (
            emitter.emit_type_vector(f32_ty, 4),
            f32_ty,
            Some(4),
            spirv::BuiltIn::FragCoord as u32,
            "gl_FragCoord",
        ),
        BuiltinKind::SubgroupLocalInvocationId => {
            emitter.emit_capability(spirv::Capability::GroupNonUniform);
            (
                u32_ty,
                u32_ty,
                None,
                spirv::BuiltIn::SubgroupLocalInvocationId as u32,
                "gl_SubgroupInvocationID",
            )
        }
        BuiltinKind::SubgroupSize => {
            emitter.emit_capability(spirv::Capability::GroupNonUniform);
            (
                u32_ty,
                u32_ty,
                None,
                spirv::BuiltIn::SubgroupSize as u32,
                "gl_SubgroupSize",
            )
        }
        BuiltinKind::Position => (
            emitter.emit_type_vector(f32_ty, 4),
            f32_ty,
            Some(4),
            spirv::BuiltIn::Position as u32,
            "gl_Position",
        ),
        BuiltinKind::FragDepth => (
            f32_ty,
            f32_ty,
            None,
            spirv::BuiltIn::FragDepth as u32,
            "gl_FragDepth",
        ),
        BuiltinKind::FrontFacing => (
            bool_ty,
            bool_ty,
            None,
            spirv::BuiltIn::FrontFacing as u32,
            "gl_FrontFacing",
        ),
        BuiltinKind::LaunchIdKhr => (
            emitter.emit_type_vector(u32_ty, 3),
            u32_ty,
            Some(3),
            5319,
            "gl_LaunchIDEXT",
        ),
        BuiltinKind::LaunchSizeKhr => (
            emitter.emit_type_vector(u32_ty, 3),
            u32_ty,
            Some(3),
            5320,
            "gl_LaunchSizeEXT",
        ),
        BuiltinKind::WorldRayOriginKhr => (
            emitter.emit_type_vector(f32_ty, 3),
            f32_ty,
            Some(3),
            5321,
            "gl_WorldRayOriginEXT",
        ),
        BuiltinKind::WorldRayDirectionKhr => (
            emitter.emit_type_vector(f32_ty, 3),
            f32_ty,
            Some(3),
            5322,
            "gl_WorldRayDirectionEXT",
        ),
        BuiltinKind::ObjectRayOriginKhr => (
            emitter.emit_type_vector(f32_ty, 3),
            f32_ty,
            Some(3),
            5323,
            "gl_ObjectRayOriginEXT",
        ),
        BuiltinKind::ObjectRayDirectionKhr => (
            emitter.emit_type_vector(f32_ty, 3),
            f32_ty,
            Some(3),
            5324,
            "gl_ObjectRayDirectionEXT",
        ),
        BuiltinKind::RayTminKhr => (f32_ty, f32_ty, None, 5325, "gl_RayTminEXT"),
        BuiltinKind::RayTmaxKhr => (f32_ty, f32_ty, None, 5326, "gl_RayTmaxEXT"),
        BuiltinKind::InstanceCustomIndexKhr => {
            (u32_ty, u32_ty, None, 5327, "gl_InstanceCustomIndexEXT")
        }
        BuiltinKind::ObjectToWorldKhr => {
            let col_ty = emitter.emit_type_vector(f32_ty, 3);
            let mat_ty = emitter.emit_type_matrix(col_ty, 4);
            (mat_ty, f32_ty, None, 5330, "gl_ObjectToWorldEXT")
        }
        BuiltinKind::WorldToObjectKhr => {
            let col_ty = emitter.emit_type_vector(f32_ty, 3);
            let mat_ty = emitter.emit_type_matrix(col_ty, 4);
            (mat_ty, f32_ty, None, 5331, "gl_WorldToObjectEXT")
        }
        BuiltinKind::HitKindKhr => (u32_ty, u32_ty, None, 5333, "gl_HitKindEXT"),
        BuiltinKind::IncomingRayFlagsKhr => (u32_ty, u32_ty, None, 5351, "gl_IncomingRayFlagsEXT"),
        BuiltinKind::RayGeometryIndexKhr => (u32_ty, u32_ty, None, 5352, "gl_RayGeometryIndexEXT"),
        BuiltinKind::PrimitivePointIndicesExt => {
            (u32_ty, u32_ty, None, 5294, "gl_PrimitivePointIndicesEXT")
        }
        BuiltinKind::PrimitiveLineIndicesExt => (
            emitter.emit_type_vector(u32_ty, 2),
            u32_ty,
            Some(2),
            5295,
            "gl_PrimitiveLineIndicesEXT",
        ),
        BuiltinKind::PrimitiveTriangleIndicesExt => (
            emitter.emit_type_vector(u32_ty, 3),
            u32_ty,
            Some(3),
            5296,
            "gl_PrimitiveTriangleIndicesEXT",
        ),
        BuiltinKind::CullPrimitiveExt => (bool_ty, bool_ty, None, 5299, "gl_CullPrimitiveEXT"),
        BuiltinKind::InvocationId => (u32_ty, u32_ty, None, 8, "gl_InvocationID"),
        BuiltinKind::PatchVertices => (u32_ty, u32_ty, None, 14, "gl_PatchVerticesIn"),
        BuiltinKind::TessLevelOuter => {
            let len_const = emitter.emit_constant_32(u32_ty, 4);
            let arr_ty = emitter.emit_type_array(f32_ty, len_const);
            (arr_ty, f32_ty, None, 11, "gl_TessLevelOuter")
        }
        BuiltinKind::TessLevelInner => {
            let len_const = emitter.emit_constant_32(u32_ty, 2);
            let arr_ty = emitter.emit_type_array(f32_ty, len_const);
            (arr_ty, f32_ty, None, 12, "gl_TessLevelInner")
        }
        BuiltinKind::TessCoord => (
            emitter.emit_type_vector(f32_ty, 3),
            f32_ty,
            Some(3),
            13,
            "gl_TessCoord",
        ),
        BuiltinKind::PrimitiveId => (u32_ty, u32_ty, None, 7, "gl_PrimitiveID"),
        BuiltinKind::Layer => (u32_ty, u32_ty, None, 9, "gl_Layer"),
        BuiltinKind::HitTriangleVertexPositionsKhr => {
            let v3_ty = emitter.emit_type_vector(f32_ty, 3);
            let len_const = emitter.emit_constant_32(u32_ty, 3);
            let arr_ty = emitter.emit_type_array(v3_ty, len_const);
            (arr_ty, f32_ty, None, 5335, "gl_HitTriangleVertexPositionsEXT")
        }
        BuiltinKind::SampleId => {
            emitter.emit_capability(spirv::Capability::SampleRateShading);
            (u32_ty, u32_ty, None, 18, "gl_SampleID")
        }
        BuiltinKind::SamplePosition => {
            emitter.emit_capability(spirv::Capability::SampleRateShading);
            (
                emitter.emit_type_vector(f32_ty, 2),
                f32_ty,
                Some(2),
                19,
                "gl_SamplePosition",
            )
        }
        BuiltinKind::SampleMask => {
            let len_const = emitter.emit_constant_32(u32_ty, 1);
            let arr_ty = emitter.emit_type_array(u32_ty, len_const);
            (arr_ty, u32_ty, None, 20, "gl_SampleMask")
        }
        BuiltinKind::PrimitiveShadingRateKhr => {
            emitter.emit_capability_raw(4422); // FragmentShadingRateKHR
            emitter.emit_extension("SPV_KHR_fragment_shading_rate");
            (u32_ty, u32_ty, None, 4432, "gl_PrimitiveShadingRateEXT")
        }
        BuiltinKind::ShadingRateKhr => {
            emitter.emit_capability_raw(4422); // FragmentShadingRateKHR
            emitter.emit_extension("SPV_KHR_fragment_shading_rate");
            (u32_ty, u32_ty, None, 4444, "gl_ShadingRateEXT")
        }
        BuiltinKind::ViewIndex => {
            emitter.emit_capability_raw(4439); // MultiView
            (u32_ty, u32_ty, None, 4440, "gl_ViewIndex")
        }
        BuiltinKind::ClipDistance => {
            emitter.emit_capability(spirv::Capability::ClipDistance);
            let len_const = emitter.emit_constant_32(u32_ty, 1);
            let arr_ty = emitter.emit_type_array(f32_ty, len_const);
            (arr_ty, f32_ty, None, 3, "gl_ClipDistance")
        }
        BuiltinKind::CullDistance => {
            emitter.emit_capability(spirv::Capability::CullDistance);
            let len_const = emitter.emit_constant_32(u32_ty, 1);
            let arr_ty = emitter.emit_type_array(f32_ty, len_const);
            (arr_ty, f32_ty, None, 4, "gl_CullDistance")
        }
    };
    let sc = if is_output {
        spirv::StorageClass::Output
    } else {
        spirv::StorageClass::Input
    };
    let ptr_ty = emitter.emit_type_pointer(sc, value_ty);
    let var_id = emitter.emit_variable(ptr_ty, sc);
    emitter.emit_name(var_id, name);
    emitter.emit_decorate(var_id, spirv::Decoration::BuiltIn, &[spv_builtin]);
    EmittedBuiltinVar {
        var_id,
        value_ty,
        scalar_ty,
        vector_len,
    }
}

fn emit_push_constant_block(
    emitter: &mut SpirvEmitter,
    pc: &IrPushConstantBlock,
) -> EmittedPushConstant {
    let u32_ty = emitter.emit_type_int(32, 0);
    let members: Vec<(u32, u32, bool, bool)> = if pc.fields.is_empty() {
        vec![(u32_ty, u32_ty, false, false)]
    } else {
        pc.fields
            .iter()
            .map(|field| {
                let (scalar_ty, elem_bytes, is_f16) = match field.scalar {
                    ScalarIrTy::F32 => (emitter.emit_type_float(32), 4u32, false),
                    ScalarIrTy::I32 => (emitter.emit_type_int(32, 1), 4u32, false),
                    ScalarIrTy::U32 | ScalarIrTy::Bool => (u32_ty, 4u32, false),
                    ScalarIrTy::F16 => {
                        emitter.emit_capability(spirv::Capability::StoragePushConstant16);
                        emitter.emit_extension("SPV_KHR_16bit_storage");
                        (emitter.emit_type_float(16), 2u32, true)
                    }
                    ScalarIrTy::I64 => (emitter.emit_type_int(64, 1), 8u32, false),
                    ScalarIrTy::U64 => (emitter.emit_type_int(64, 0), 8u32, false),
                };
                let count = field.byte_size / elem_bytes;
                match count {
                    0 | 1 => (scalar_ty, scalar_ty, false, is_f16),
                    2..=4 => (
                        emitter.emit_type_vector(scalar_ty, count),
                        scalar_ty,
                        true,
                        is_f16,
                    ),
                    n => {
                        let len_const = emitter.emit_constant_32(u32_ty, n);
                        let arr_ty = emitter.emit_type_array_with_stride(
                            scalar_ty,
                            len_const,
                            Some(elem_bytes),
                        );
                        (arr_ty, scalar_ty, true, is_f16)
                    }
                }
            })
            .collect()
    };
    let member_types: Vec<u32> = members.iter().map(|&(ty, _, _, _)| ty).collect();

    let struct_ty = emitter.emit_type_struct(&member_types);
    emitter.emit_name(struct_ty, pc.name.as_str());
    emitter.emit_decorate(struct_ty, spirv::Decoration::Block, &[]);

    if pc.fields.is_empty() {
        emitter.emit_member_decorate(struct_ty, 0, spirv::Decoration::Offset, &[0]);
    } else {
        for (idx, field) in pc.fields.iter().enumerate() {
            let member = idx as u32;
            emitter.emit_member_name(struct_ty, member, field.name.as_str());
            emitter.emit_member_decorate(
                struct_ty,
                member,
                spirv::Decoration::Offset,
                &[field.byte_offset],
            );
        }
    }

    let ptr_ty = emitter.emit_type_pointer(spirv::StorageClass::PushConstant, struct_ty);
    let var_id = emitter.emit_variable(ptr_ty, spirv::StorageClass::PushConstant);
    emitter.emit_name(var_id, pc.name.as_str());
    EmittedPushConstant { var_id, members }
}

struct LoopFrame {
    merge_label: u32,
    continue_label: u32,
    merge_phis: Vec<InstId>,
    next_break_tag: u32,
    break_preds: Vec<(u32, u32)>,
}

struct FnLowerCtx<'a> {
    emitter: &'a mut SpirvEmitter,
    func: &'a IrFunction,
    fn_ret_ty: u32,
    void_ty: u32,
    emitted_functions: &'a IndexMap<String, EmittedFunctionInfo>,
    descriptors: &'a [EmittedDescriptor],
    workgroup_vars: &'a [EmittedWorkgroupVar],
    stage_io_vars: &'a [EmittedStageIoVar],
    stage_vars: &'a [EmittedStageVar],
    local_vars: Vec<EmittedLocalVar>,
    /// `(param_spv_id, param_spv_ty, is_ptr_param, value_ir_ty)` for formal parameters.
    param_ids: Vec<(u32, u32, bool, IrType)>,
    builtins: &'a IndexMap<(BuiltinKind, bool), EmittedBuiltinVar>,
    push_constant: Option<&'a EmittedPushConstant>,
    default_sampler_var: Option<u32>,
    ray_query_vars: IndexMap<u32, u32>,
    hit_object_vars: IndexMap<u32, u32>,
    use_ser_ext: bool,
    debug_ctx: Option<(u32, u32)>,
    uses_bda: bool,
    inst_ids: IndexMap<InstId, u32>,
    preallocated_ids: IndexMap<InstId, (u32, u32)>,
    inst_types: IndexMap<InstId, u32>,
    inst_pointee_types: IndexMap<InstId, u32>,
    inst_ptr_storage: IndexMap<InstId, spirv::StorageClass>,
    inst_ptr_ir_pointee: IndexMap<InstId, IrType>,
    bda_ptr_insts: IndexSet<InstId>,
    workgroup_ptr_insts: IndexSet<InstId>,
    phi_target_types: IndexMap<InstId, u32>,
    phi_coerced_inputs: IndexMap<(InstId, u32), u32>,
    decorated_non_uniform: IndexSet<u32>,
    loop_stack: Vec<LoopFrame>,
    current_label: u32,
    entry_label: u32,
    /// `(predecessor_tag, spirv_label_id)` pairs feeding the current merge/header block.
    current_block_preds: Vec<(u32, u32)>,
}

impl FnLowerCtx<'_> {
    fn bool_ty(&mut self) -> u32 {
        self.emitter.emit_type_bool()
    }

    fn i32_ty(&mut self) -> u32 {
        self.emitter.emit_type_int(32, 1)
    }

    fn u32_ty(&mut self) -> u32 {
        self.emitter.emit_type_int(32, 0)
    }

    fn f16_ty(&mut self) -> u32 {
        self.emitter.emit_type_float(16)
    }

    fn f32_ty(&mut self) -> u32 {
        self.emitter.emit_type_float(32)
    }

    fn i64_ty(&mut self) -> u32 {
        self.emitter.emit_type_int(64, 1)
    }

    fn u64_ty(&mut self) -> u32 {
        self.emitter.emit_type_int(64, 0)
    }

    /// Reads the cached signed 64-bit integer type ID without emitting `OpTypeInt 64` or `Int64`.
    fn cached_i64_ty(&self) -> Option<u32> {
        self.emitter.type_cache.get(&TypeKey::Int(64, 1)).copied()
    }

    /// Reads the cached unsigned 64-bit integer type ID without emitting `OpTypeInt 64` or `Int64`.
    fn cached_u64_ty(&self) -> Option<u32> {
        self.emitter.type_cache.get(&TypeKey::Int(64, 0)).copied()
    }

    fn bda_storage_class(&self) -> spirv::StorageClass {
        if self.uses_bda {
            spirv::StorageClass::PhysicalStorageBuffer
        } else {
            spirv::StorageClass::StorageBuffer
        }
    }

    fn bda_ptr_ty_for(&mut self, elem_ty: u32, stride: u32, matrix_stride: Option<u32>) -> u32 {
        let block_ty = self
            .emitter
            .emit_bda_block_struct_for(elem_ty, stride, matrix_stride);
        let sc = self.bda_storage_class();
        self.emitter.emit_type_pointer(sc, block_ty)
    }

    fn bda_ptr_ty(&mut self) -> u32 {
        let u32_ty = self.u32_ty();
        self.bda_ptr_ty_for(u32_ty, 4, None)
    }

    fn is_scalar_value_ty(&self, ty: u32) -> bool {
        [
            TypeKey::Bool,
            TypeKey::Int(32, 1),
            TypeKey::Int(32, 0),
            TypeKey::Float(16),
            TypeKey::Float(32),
            TypeKey::Int(64, 1),
            TypeKey::Int(64, 0),
        ]
        .iter()
        .any(|key| self.emitter.type_cache.get(key) == Some(&ty))
    }

    /// Returns `(component_spv_ty, vector_len)` if `ty` is a cached SPIR-V vector type.
    fn vector_component_info(&self, ty: u32) -> Option<(u32, u32)> {
        for comp_key in [
            TypeKey::Float(32),
            TypeKey::Float(16),
            TypeKey::Int(32, 1),
            TypeKey::Int(32, 0),
            TypeKey::Bool,
            TypeKey::Int(64, 1),
            TypeKey::Int(64, 0),
        ] {
            if let Some(&comp_ty) = self.emitter.type_cache.get(&comp_key) {
                for len in 2u32..=4 {
                    if self.emitter.type_cache.get(&TypeKey::Vector(comp_ty, len)) == Some(&ty) {
                        return Some((comp_ty, len));
                    }
                }
            }
        }
        None
    }

    fn is_float_spv_ty(&self, ty: u32) -> bool {
        let scalar = self.vector_component_info(ty).map_or(ty, |(c, _)| c);
        self.emitter.type_cache.get(&TypeKey::Float(32)) == Some(&scalar)
            || self.emitter.type_cache.get(&TypeKey::Float(16)) == Some(&scalar)
    }

    fn is_signed_int_spv_ty(&self, ty: u32) -> bool {
        let scalar = self.vector_component_info(ty).map_or(ty, |(c, _)| c);
        self.emitter.type_cache.get(&TypeKey::Int(32, 1)) == Some(&scalar)
            || self.emitter.type_cache.get(&TypeKey::Int(64, 1)) == Some(&scalar)
    }

    fn is_unsigned_int_spv_ty(&self, ty: u32) -> bool {
        let scalar = self.vector_component_info(ty).map_or(ty, |(c, _)| c);
        self.emitter.type_cache.get(&TypeKey::Int(32, 0)) == Some(&scalar)
            || self.emitter.type_cache.get(&TypeKey::Int(64, 0)) == Some(&scalar)
    }

    fn emit_zero_const(&mut self, dst_ty: u32) -> u32 {
        if let Some((comp_ty, len)) = self.vector_component_info(dst_ty) {
            let zero = self.emit_zero_const(comp_ty);
            let res = self.emitter.alloc_id();
            let mut ops = Vec::with_capacity(2 + len as usize);
            ops.push(dst_ty);
            ops.push(res);
            for _ in 0..len {
                ops.push(zero);
            }
            self.emitter
                .emit_fn_inst(spirv::Op::CompositeConstruct, &ops);
            return res;
        }
        let bool_ty = self.bool_ty();
        let i32_ty = self.i32_ty();
        let u32_ty = self.u32_ty();
        let f32_ty = self.f32_ty();
        let i64_cached = self.cached_i64_ty();
        let u64_cached = self.cached_u64_ty();
        let f16_cached = self.emitter.type_cache.get(&TypeKey::Float(16)).copied();
        if dst_ty == bool_ty {
            self.emitter.emit_constant_bool(bool_ty, false)
        } else if dst_ty == i32_ty
            || dst_ty == u32_ty
            || dst_ty == f32_ty
            || f16_cached == Some(dst_ty)
        {
            self.emitter.emit_constant_32(dst_ty, 0)
        } else if i64_cached == Some(dst_ty) || u64_cached == Some(dst_ty) {
            self.emitter.emit_constant_64(dst_ty, 0)
        } else {
            let id = self.emitter.alloc_id();
            self.emitter.emit_fn_inst(spirv::Op::Undef, &[dst_ty, id]);
            id
        }
    }

    /// Ensures `inst` yields a SPIR-V `OpTypeBool` value for branches, logical ops, and selects.
    fn ensure_bool(&mut self, inst: InstId) -> u32 {
        let val = self.ensure_inst(inst);
        let bool_ty = self.bool_ty();
        let src_ty = self.inst_types.get(&inst).copied();
        if src_ty == Some(bool_ty) {
            return val;
        }
        if val == 0 {
            return self.emitter.emit_constant_bool(bool_ty, false);
        }
        if let Some(st) = src_ty {
            return self.coerce_to_ty(val, st, bool_ty);
        }
        self.emitter.emit_constant_bool(bool_ty, false)
    }

    /// Safely coerces `spv_id` of type `src_ty` to `dst_ty` without ever emitting illegal
    /// `OpBitcast` on opaque image/sampler types or mismatched bit widths.
    fn coerce_to_ty(&mut self, spv_id: u32, src_ty: u32, dst_ty: u32) -> u32 {
        if src_ty == dst_ty {
            return spv_id;
        }
        if spv_id == 0 {
            return self.emit_zero_const(dst_ty);
        }
        let bool_ty = self.bool_ty();
        let i32_ty = self.i32_ty();
        let u32_ty = self.u32_ty();
        let f32_ty = self.f32_ty();
        let i64_cached = self.cached_i64_ty();
        let u64_cached = self.cached_u64_ty();
        let f16_cached = self.emitter.type_cache.get(&TypeKey::Float(16)).copied();

        // Vector -> vector coercion of matching length (component-wise).
        if let (Some((src_c, src_n)), Some((dst_c, dst_n))) = (
            self.vector_component_info(src_ty),
            self.vector_component_info(dst_ty),
        ) {
            let mut elems = Vec::with_capacity(dst_n as usize);
            for i in 0..dst_n {
                if i < src_n {
                    let c_id = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::CompositeExtract, &[src_c, c_id, spv_id, i]);
                    elems.push(self.coerce_to_ty(c_id, src_c, dst_c));
                } else {
                    elems.push(self.emit_zero_const(dst_c));
                }
            }
            let res = self.emitter.alloc_id();
            let mut ops = Vec::with_capacity(2 + elems.len());
            ops.push(dst_ty);
            ops.push(res);
            ops.extend_from_slice(&elems);
            self.emitter
                .emit_fn_inst(spirv::Op::CompositeConstruct, &ops);
            return res;
        }

        // Scalar -> vector splat coercion.
        if let Some((dst_c, dst_n)) = self.vector_component_info(dst_ty)
            && self.is_scalar_value_ty(src_ty)
        {
            let elem = self.coerce_to_ty(spv_id, src_ty, dst_c);
            let res = self.emitter.alloc_id();
            let mut ops = Vec::with_capacity(2 + dst_n as usize);
            ops.push(dst_ty);
            ops.push(res);
            for _ in 0..dst_n {
                ops.push(elem);
            }
            self.emitter
                .emit_fn_inst(spirv::Op::CompositeConstruct, &ops);
            return res;
        }

        // Vector -> scalar coercion (extract element 0, e.g. legacy barycentrics -> u32).
        if let Some((comp_ty, _)) = self.vector_component_info(src_ty)
            && self.is_scalar_value_ty(dst_ty)
        {
            let elem0 = self.emitter.alloc_id();
            self.emitter
                .emit_fn_inst(spirv::Op::CompositeExtract, &[comp_ty, elem0, spv_id, 0]);
            return self.coerce_to_ty(elem0, comp_ty, dst_ty);
        }

        if let Some(f16_ty) = f16_cached {
            if (src_ty == f16_ty && dst_ty == f32_ty) || (src_ty == f32_ty && dst_ty == f16_ty) {
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::FConvert, &[dst_ty, res, spv_id]);
                return res;
            }
            if src_ty == f16_ty {
                let widened = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::FConvert, &[f32_ty, widened, spv_id]);
                return self.coerce_to_ty(widened, f32_ty, dst_ty);
            }
            if dst_ty == f16_ty {
                let as_f32 = self.coerce_to_ty(spv_id, src_ty, f32_ty);
                let narrowed = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::FConvert, &[f16_ty, narrowed, as_f32]);
                return narrowed;
            }
        }

        let src_is_i64 = i64_cached == Some(src_ty);
        let src_is_u64 = u64_cached == Some(src_ty);
        let dst_is_i64 = i64_cached == Some(dst_ty);
        let dst_is_u64 = u64_cached == Some(dst_ty);
        if (src_ty == i32_ty && dst_ty == u32_ty)
            || (src_ty == u32_ty && dst_ty == i32_ty)
            || (src_is_i64 && dst_is_u64)
            || (src_is_u64 && dst_is_i64)
        {
            let res = self.emitter.alloc_id();
            self.emitter
                .emit_fn_inst(spirv::Op::Bitcast, &[dst_ty, res, spv_id]);
            return res;
        }
        if (src_is_u64 || src_is_i64) && dst_ty == u32_ty {
            let res = self.emitter.alloc_id();
            self.emitter
                .emit_fn_inst(spirv::Op::UConvert, &[u32_ty, res, spv_id]);
            return res;
        }
        if (src_is_u64 || src_is_i64) && dst_ty == i32_ty {
            let res = self.emitter.alloc_id();
            self.emitter
                .emit_fn_inst(spirv::Op::SConvert, &[i32_ty, res, spv_id]);
            return res;
        }
        if (src_ty == u32_ty || src_ty == i32_ty) && dst_is_u64 {
            let res = self.emitter.alloc_id();
            self.emitter
                .emit_fn_inst(spirv::Op::UConvert, &[dst_ty, res, spv_id]);
            return res;
        }
        if (src_ty == u32_ty || src_ty == i32_ty) && dst_is_i64 {
            let res = self.emitter.alloc_id();
            self.emitter
                .emit_fn_inst(spirv::Op::SConvert, &[dst_ty, res, spv_id]);
            return res;
        }
        if (src_ty == u32_ty || src_is_u64) && dst_ty == f32_ty {
            let res = self.emitter.alloc_id();
            self.emitter
                .emit_fn_inst(spirv::Op::ConvertUToF, &[f32_ty, res, spv_id]);
            return res;
        }
        if (src_ty == i32_ty || src_is_i64) && dst_ty == f32_ty {
            let res = self.emitter.alloc_id();
            self.emitter
                .emit_fn_inst(spirv::Op::ConvertSToF, &[f32_ty, res, spv_id]);
            return res;
        }
        if src_ty == f32_ty && (dst_ty == u32_ty || dst_is_u64) {
            let res = self.emitter.alloc_id();
            self.emitter
                .emit_fn_inst(spirv::Op::ConvertFToU, &[dst_ty, res, spv_id]);
            return res;
        }
        if src_ty == f32_ty && (dst_ty == i32_ty || dst_is_i64) {
            let res = self.emitter.alloc_id();
            self.emitter
                .emit_fn_inst(spirv::Op::ConvertFToS, &[dst_ty, res, spv_id]);
            return res;
        }
        if src_ty == bool_ty {
            let (one, zero) = if dst_ty == u32_ty || dst_ty == i32_ty {
                (
                    self.emitter.emit_constant_32(dst_ty, 1),
                    self.emitter.emit_constant_32(dst_ty, 0),
                )
            } else if dst_ty == f32_ty {
                (
                    self.emitter.emit_constant_32(f32_ty, 1.0_f32.to_bits()),
                    self.emitter.emit_constant_32(f32_ty, 0.0_f32.to_bits()),
                )
            } else if dst_is_u64 || dst_is_i64 {
                (
                    self.emitter.emit_constant_64(dst_ty, 1),
                    self.emitter.emit_constant_64(dst_ty, 0),
                )
            } else {
                return self.emit_zero_const(dst_ty);
            };
            let res = self.emitter.alloc_id();
            self.emitter
                .emit_fn_inst(spirv::Op::Select, &[dst_ty, res, spv_id, one, zero]);
            return res;
        }
        if dst_ty == bool_ty {
            if src_ty == i32_ty || src_ty == u32_ty {
                let zero = self.emitter.emit_constant_32(src_ty, 0);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::INotEqual, &[bool_ty, res, spv_id, zero]);
                return res;
            }
            if src_ty == f32_ty {
                let zero = self.emitter.emit_constant_32(f32_ty, 0.0_f32.to_bits());
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::FOrdNotEqual, &[bool_ty, res, spv_id, zero]);
                return res;
            }
            if src_is_i64 || src_is_u64 {
                let zero = self.emitter.emit_constant_64(src_ty, 0);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::INotEqual, &[bool_ty, res, spv_id, zero]);
                return res;
            }
        }
        self.emit_zero_const(dst_ty)
    }

    /// Determines `(elem_ty, stride, elem_align, matrix_col_stride)` for a BDA load in `O(1)`
    /// from `inst.ir_ty` or pointee `ir_ty`.
    fn bda_load_elem_ty(&mut self, load_id: InstId, ptr: InstId) -> (u32, u32, u32, Option<u32>) {
        let ir_ty = self.func.insts[load_id].ir_ty.clone();
        if ir_ty != IrType::Void {
            let spv_ty = self.emitter.emit_ir_type_with_layout(&ir_ty, true);
            return (
                spv_ty,
                ir_type_stride(&ir_ty),
                ir_type_align(&ir_ty),
                matrix_col_stride(&ir_ty),
            );
        }
        if let IrType::Pointer { pointee, .. } = &self.func.insts[ptr].ir_ty
            && **pointee != IrType::Void
        {
            let p_clone = pointee.as_ref().clone();
            let spv_ty = self.emitter.emit_ir_type_with_layout(&p_clone, true);
            return (
                spv_ty,
                ir_type_stride(&p_clone),
                ir_type_align(&p_clone),
                matrix_col_stride(&p_clone),
            );
        }
        (self.u32_ty(), 4, 4, None)
    }

    /// Infers a non-opaque value SPIR-V type for `inst` in `O(1)` without emitting function instructions.
    fn peek_value_ty(&mut self, inst: InstId) -> Option<u32> {
        if let Some(&ty) = self.inst_types.get(&inst) {
            return (ty != self.void_ty).then_some(ty);
        }
        let ir_ty = self.func.insts[inst].ir_ty.clone();
        if ir_ty != IrType::Void && !matches!(ir_ty, IrType::Pointer { .. }) {
            return Some(self.emitter.emit_ir_type(&ir_ty));
        }
        let data = self.func.insts[inst].data.clone();
        match &data {
            InstData::ConstBool(_)
            | InstData::ICmpEq(_, _)
            | InstData::ICmpNe(_, _)
            | InstData::ICmpLt(_, _)
            | InstData::ICmpLe(_, _)
            | InstData::ICmpGt(_, _)
            | InstData::ICmpGe(_, _)
            | InstData::UCmpLt(_, _)
            | InstData::FCmpEq(_, _)
            | InstData::FCmpNe(_, _)
            | InstData::FCmpLt(_, _)
            | InstData::FCmpLe(_, _)
            | InstData::FCmpGt(_, _)
            | InstData::FCmpGe(_, _)
            | InstData::LogicalAnd(_, _)
            | InstData::LogicalOr(_, _)
            | InstData::LogicalNot(_)
            | InstData::IsNan(_)
            | InstData::IsInf(_)
            | InstData::SubgroupAny(_)
            | InstData::SubgroupAll(_)
            | InstData::SubgroupElect
            | InstData::SubgroupBallotBitExtract { .. }
            | InstData::SubgroupLogicalAnd(_)
            | InstData::SubgroupLogicalOr(_)
            | InstData::SubgroupLogicalXor(_)
            | InstData::RayQueryProceed(_)
            | InstData::RayQueryCommittedFrontFace(_) => Some(self.bool_ty()),
            InstData::ConstI32(_) | InstData::CastToI32(_) => Some(self.i32_ty()),
            InstData::ConstU32(_)
            | InstData::CastToU32(_)
            | InstData::AtomicAdd { .. }
            | InstData::AtomicMin { .. }
            | InstData::AtomicMax { .. }
            | InstData::AtomicCas { .. }
            | InstData::SubgroupBallotBitCount(_)
            | InstData::SubgroupBallotExclusiveBitCount(_)
            | InstData::SubgroupBallotInclusiveBitCount(_)
            | InstData::SubgroupBallotFindLsb(_)
            | InstData::SubgroupBallotFindMsb(_)
            | InstData::BitCount(_)
            | InstData::ImageQueryLevels { .. }
            | InstData::RayQueryCommittedType(_)
            | InstData::RayQueryCandidateType(_)
            | InstData::RayQueryCommittedInstanceId(_)
            | InstData::RayQueryCommittedInstanceCustomIndex(_)
            | InstData::RayQueryCommittedPrimitiveIndex(_)
            | InstData::RayQueryCommittedGeometryIndex(_) => Some(self.u32_ty()),
            InstData::SpecConstant { scalar, .. } | InstData::DescriptorLoad { scalar, .. } => {
                Some(self.emitter.emit_ir_type(&IrType::Scalar(*scalar)))
            }
            InstData::GlslStd450 { op, .. } => Some(match op {
                GlslStd450Op::SAbs
                | GlslStd450Op::SMin
                | GlslStd450Op::SMax
                | GlslStd450Op::SClamp
                | GlslStd450Op::FindSMsb
                | GlslStd450Op::FindILsb => self.i32_ty(),
                GlslStd450Op::UMin
                | GlslStd450Op::UMax
                | GlslStd450Op::UClamp
                | GlslStd450Op::PackHalf2x16
                | GlslStd450Op::PackUnorm4x8
                | GlslStd450Op::FindUMsb => self.u32_ty(),
                GlslStd450Op::UnpackHalf2x16 => {
                    let f = self.f32_ty();
                    self.emitter.emit_type_vector(f, 2)
                }
                GlslStd450Op::UnpackUnorm4x8 => {
                    let f = self.f32_ty();
                    self.emitter.emit_type_vector(f, 4)
                }
                _ => self.f32_ty(),
            }),
            InstData::LoadPushConstant { field_index } => {
                if let Some(pc) = self.push_constant
                    && !pc.members.is_empty()
                {
                    let clamped = (*field_index as usize).min(pc.members.len() - 1);
                    let (_, scalar_ty, _, is_f16) = pc.members[clamped];
                    Some(if is_f16 { self.f32_ty() } else { scalar_ty })
                } else {
                    Some(self.u32_ty())
                }
            }
            InstData::LoadBuiltin { kind, .. } => Some(match kind {
                BuiltinKind::FragCoord | BuiltinKind::Position | BuiltinKind::FragDepth => {
                    self.f32_ty()
                }
                BuiltinKind::FrontFacing => self.bool_ty(),
                _ => self.u32_ty(),
            }),
            InstData::ConstF16(_) => Some(self.f16_ty()),
            InstData::CastToF32(_)
            | InstData::ConstF32(_)
            | InstData::FAdd32(_, _)
            | InstData::FSub32(_, _)
            | InstData::FMul32(_, _)
            | InstData::FDiv32(_, _)
            | InstData::ImageSample { .. }
            | InstData::ImageSampleImplicitLod { .. }
            | InstData::ImageSampleGrad { .. }
            | InstData::ImageSampleDrefExplicitLod { .. }
            | InstData::ImageGather { .. }
            | InstData::RayQueryCommittedT(_)
            | InstData::RayQueryCandidateT(_) => Some(self.f32_ty()),
            InstData::RayQueryCommittedBarycentrics(_) => {
                let f = self.f32_ty();
                Some(self.emitter.emit_type_vector(f, 2))
            }
            InstData::ConstI64(_) | InstData::CastToI64(_) => Some(self.i64_ty()),
            InstData::ConstU64(_)
            | InstData::CastToU64(_)
            | InstData::CastI64ToU64(_)
            | InstData::ConvertPtrToU(_) => Some(self.u64_ty()),
            InstData::WorkgroupLoad { var_index, .. } => self
                .workgroup_vars
                .get(*var_index as usize)
                .map(|wg| wg.elem_ty),
            InstData::IAdd32(a, _)
            | InstData::ISub32(a, _)
            | InstData::IMul32(a, _)
            | InstData::IDiv32(a, _)
            | InstData::IRem32(a, _)
            | InstData::BitAnd32(a, _)
            | InstData::BitOr32(a, _)
            | InstData::BitXor32(a, _)
            | InstData::BitNot32(a)
            | InstData::Shl32(a, _)
            | InstData::Shr32(a, _) => {
                let lhs_ty = self.peek_value_ty(*a);
                let i32_ty = self.i32_ty();
                if lhs_ty == Some(i32_ty) {
                    Some(i32_ty)
                } else {
                    Some(self.u32_ty())
                }
            }
            InstData::IAdd64(a, _) | InstData::IMul64(a, _) => {
                let lhs_ty = self.peek_value_ty(*a);
                let i64_ty = self.i64_ty();
                if lhs_ty == Some(i64_ty) {
                    Some(i64_ty)
                } else {
                    Some(self.u64_ty())
                }
            }
            InstData::BdaLoad { ptr, .. } => Some(self.bda_load_elem_ty(inst, *ptr).0),
            InstData::Select {
                on_true, on_false, ..
            } => self
                .peek_value_ty(*on_true)
                .or_else(|| self.peek_value_ty(*on_false)),
            InstData::NonUniform(inner) => self.peek_value_ty(*inner),
            _ => None,
        }
    }

    /// Determines and caches the target non-opaque SPIR-V type for a `Phi` node before predecessor
    /// branches are emitted so all incoming operands can be coerced in their predecessor blocks.
    fn infer_phi_ty(&mut self, phi_id: InstId) -> u32 {
        if let Some(&ty) = self.phi_target_types.get(&phi_id) {
            return ty;
        }
        let phi_ir_ty = self.func.insts[phi_id].ir_ty.clone();
        if phi_ir_ty != IrType::Void && !matches!(phi_ir_ty, IrType::Pointer { .. }) {
            let ty = self.emitter.emit_ir_type(&phi_ir_ty);
            self.phi_target_types.insert(phi_id, ty);
            return ty;
        }
        let incoming = if let InstData::Phi { incoming } = &self.func.insts[phi_id].data {
            incoming.clone()
        } else {
            Vec::new()
        };
        let mut chosen = None;
        for (val_inst, _) in incoming {
            if let Some(ty) = self.peek_value_ty(val_inst) {
                chosen = Some(ty);
                break;
            }
        }
        let ty = chosen.unwrap_or_else(|| self.u32_ty());
        self.phi_target_types.insert(phi_id, ty);
        ty
    }

    /// Ensures an address operand has unsigned 64-bit integer SPIR-V type (`u64_ty`)
    /// before feeding `OpConvertUToPtr`.
    fn ensure_u64_addr(&mut self, addr_inst: InstId) -> u32 {
        let raw_id = self.ensure_inst(addr_inst);
        let u64_ty = self.u64_ty();
        let src_ty = self.inst_types.get(&addr_inst).copied().unwrap_or(u64_ty);
        self.coerce_to_ty(raw_id, src_ty, u64_ty)
    }

    /// Ensures `ptr` is materialized as a BDA block pointer for `elem_ty` with `stride`
    /// and optional matrix column stride decorations.
    fn ensure_bda_block_ptr(
        &mut self,
        ptr: InstId,
        elem_ty: u32,
        stride: u32,
        matrix_stride: Option<u32>,
    ) -> u32 {
        let raw_ptr_id = self.ensure_inst(ptr);
        let target_ptr_ty = self.bda_ptr_ty_for(elem_ty, stride, matrix_stride);
        if self.inst_types.get(&ptr).copied() == Some(target_ptr_ty) {
            return raw_ptr_id;
        }
        if self.bda_ptr_insts.contains(&ptr) {
            let casted = self.emitter.alloc_id();
            self.emitter
                .emit_fn_inst(spirv::Op::Bitcast, &[target_ptr_ty, casted, raw_ptr_id]);
            return casted;
        }
        let addr_u64 = self.ensure_u64_addr(ptr);
        let promoted = self.emitter.alloc_id();
        self.emitter.emit_fn_inst(
            spirv::Op::ConvertUToPtr,
            &[target_ptr_ty, promoted, addr_u64],
        );
        promoted
    }

    /// Indexes a BDA block pointer (`*BdaBlock<T>`) to element 0 (`*T`) for scalar/vector load/store/atomic.
    fn bda_elem_ptr_for(&mut self, block_ptr_id: u32, elem_ty: u32) -> u32 {
        let u32_ty = self.u32_ty();
        let zero = self.emitter.emit_constant_32(u32_ty, 0);
        let sc = self.bda_storage_class();
        let elem_ptr_ty = self.emitter.emit_type_pointer(sc, elem_ty);
        let elem_ptr = self.emitter.alloc_id();
        self.emitter.emit_fn_inst(
            spirv::Op::AccessChain,
            &[elem_ptr_ty, elem_ptr, block_ptr_id, zero, zero],
        );
        elem_ptr
    }

    /// Determines `(elem_ty, stride, elem_align, matrix_col_stride)` for a BDA store
    /// from the operand's IR or SPIR-V type.
    fn bda_store_elem_ty(&mut self, val_inst: InstId, val_ty: u32) -> (u32, u32, u32, Option<u32>) {
        let ir_ty = self.func.insts[val_inst].ir_ty.clone();
        if ir_ty != IrType::Void && !matches!(ir_ty, IrType::Pointer { .. }) {
            let spv_ty = self.emitter.emit_ir_type_with_layout(&ir_ty, true);
            return (
                spv_ty,
                ir_type_stride(&ir_ty),
                ir_type_align(&ir_ty),
                matrix_col_stride(&ir_ty),
            );
        }
        let u64_cached = self.cached_u64_ty();
        let i64_cached = self.cached_i64_ty();
        let f32_ty = self.f32_ty();
        let i32_ty = self.i32_ty();
        let f16_cached = self.emitter.type_cache.get(&TypeKey::Float(16)).copied();
        if u64_cached == Some(val_ty) || i64_cached == Some(val_ty) {
            (val_ty, 8, 8, None)
        } else if f16_cached == Some(val_ty) {
            self.emitter
                .emit_capability(spirv::Capability::StorageBuffer16BitAccess);
            self.emitter.emit_extension("SPV_KHR_16bit_storage");
            (val_ty, 2, 2, None)
        } else if val_ty == f32_ty || val_ty == i32_ty {
            (val_ty, 4, 4, None)
        } else if let Some((comp_ty, len)) = self.vector_component_info(val_ty) {
            let comp_bytes = if u64_cached == Some(comp_ty) || i64_cached == Some(comp_ty) {
                8
            } else if f16_cached == Some(comp_ty) {
                2
            } else {
                4
            };
            let stride = if len == 3 {
                comp_bytes * 4
            } else {
                comp_bytes * len
            };
            (val_ty, stride, stride.next_power_of_two(), None)
        } else {
            (self.u32_ty(), 4, 4, None)
        }
    }

    /// Constructs a SPIR-V image coordinate vector of `coord_components` elements of `target_elem_ty`.
    fn emit_coord_vec(
        &mut self,
        raw_coord_id: u32,
        coord_ty: u32,
        target_elem_ty: u32,
        coord_components: u32,
    ) -> u32 {
        let coord_vec_ty = self
            .emitter
            .emit_type_vector(target_elem_ty, coord_components);
        if coord_ty == coord_vec_ty {
            return raw_coord_id;
        }
        let coord_vec_id = self.emitter.alloc_id();
        let mut coord_ops = Vec::with_capacity(2 + coord_components as usize);
        coord_ops.push(coord_vec_ty);
        coord_ops.push(coord_vec_id);
        if let Some((comp_ty, src_len)) = self.vector_component_info(coord_ty) {
            for i in 0..coord_components {
                if i < src_len {
                    let raw_c = self.emitter.alloc_id();
                    self.emitter.emit_fn_inst(
                        spirv::Op::CompositeExtract,
                        &[comp_ty, raw_c, raw_coord_id, i],
                    );
                    coord_ops.push(self.coerce_to_ty(raw_c, comp_ty, target_elem_ty));
                } else {
                    coord_ops.push(self.emit_zero_const(target_elem_ty));
                }
            }
        } else {
            let coord_scalar_id = self.coerce_to_ty(raw_coord_id, coord_ty, target_elem_ty);
            for _ in 0..coord_components {
                coord_ops.push(coord_scalar_id);
            }
        }
        self.emitter
            .emit_fn_inst(spirv::Op::CompositeConstruct, &coord_ops);
        coord_vec_id
    }

    /// Constructs a `vec3<f32>` from a `vec3` (pass-through), other vector, or scalar splat.
    fn emit_vec3_f32(&mut self, raw_id: u32, src_ty: u32) -> u32 {
        let f32_ty = self.f32_ty();
        self.emit_coord_vec(raw_id, src_ty, f32_ty, 3)
    }

    /// Finalizes a 4-component image texel result according to `func.insts[id].ir_ty`.
    ///
    /// Returns full `vec4`/`ivec4`/`uvec4` when `ir_ty` is a 4-element vector, shuffles
    /// for 2- or 3-element vectors, and only extracts component `0` when `ir_ty` is scalar
    /// (or untyped `Void` for legacy IR compatibility).
    fn finalize_texel_result(
        &mut self,
        id: InstId,
        texel_vec_id: u32,
        sampled_ty: u32,
    ) -> (u32, u32) {
        let texel_vec_ty = self.emitter.emit_type_vector(sampled_ty, 4);
        match self.func.insts[id].ir_ty.clone() {
            IrType::Vector { scalar, len: 4 } => {
                let target_ty = self
                    .emitter
                    .emit_ir_type(&IrType::Vector { scalar, len: 4 });
                let coerced = self.coerce_to_ty(texel_vec_id, texel_vec_ty, target_ty);
                (coerced, target_ty)
            }
            IrType::Vector { scalar, len } if (2..=3).contains(&len) => {
                let sub_vec_ty = self.emitter.emit_type_vector(sampled_ty, u32::from(len));
                let sub_id = self.emitter.alloc_id();
                let mut ops = Vec::with_capacity(4 + usize::from(len));
                ops.push(sub_vec_ty);
                ops.push(sub_id);
                ops.push(texel_vec_id);
                ops.push(texel_vec_id);
                for c in 0..u32::from(len) {
                    ops.push(c);
                }
                self.emitter.emit_fn_inst(spirv::Op::VectorShuffle, &ops);
                let target_ty = self.emitter.emit_ir_type(&IrType::Vector { scalar, len });
                let coerced = self.coerce_to_ty(sub_id, sub_vec_ty, target_ty);
                (coerced, target_ty)
            }
            IrType::Scalar(scalar) => {
                let scalar_id = self.emitter.alloc_id();
                self.emitter.emit_fn_inst(
                    spirv::Op::CompositeExtract,
                    &[sampled_ty, scalar_id, texel_vec_id, 0],
                );
                let target_ty = self.emitter.emit_ir_type(&IrType::Scalar(scalar));
                let coerced = self.coerce_to_ty(scalar_id, sampled_ty, target_ty);
                (coerced, target_ty)
            }
            _ => {
                let f32_ty = self.f32_ty();
                let scalar_id = self.emitter.alloc_id();
                self.emitter.emit_fn_inst(
                    spirv::Op::CompositeExtract,
                    &[sampled_ty, scalar_id, texel_vec_id, 0],
                );
                let res_f32 = self.coerce_to_ty(scalar_id, sampled_ty, f32_ty);
                (res_f32, f32_ty)
            }
        }
    }

    /// Loads or builds a `SampledImage` handle for sampling/gathering from `desc`.
    fn load_sampled_image(
        &mut self,
        desc: &EmittedDescriptor,
        image_inst: Option<InstId>,
        preloaded_img: Option<u32>,
        sampler_descriptor_index: Option<u32>,
    ) -> Option<u32> {
        match desc.kind {
            DescriptorKind::CombinedImageSampler => {
                if let Some(loaded) = preloaded_img {
                    Some(loaded)
                } else if desc.is_arrayed_descriptor() {
                    None
                } else {
                    let loaded = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::Load, &[desc.element_ty, loaded, desc.var_id]);
                    Some(loaded)
                }
            }
            DescriptorKind::SampledImage => {
                let sampler_var = sampler_descriptor_index
                    .and_then(|idx| self.descriptors.get(idx as usize))
                    .filter(|d| d.kind == DescriptorKind::Sampler)
                    .map(|d| d.var_id)
                    .or(self.default_sampler_var)?;
                let img_id = if let Some(loaded) = preloaded_img {
                    loaded
                } else if desc.is_arrayed_descriptor() {
                    return None;
                } else {
                    let loaded = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::Load, &[desc.element_ty, loaded, desc.var_id]);
                    loaded
                };
                let sampler_ty = self.emitter.emit_type_sampler();
                let sampler_val = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::Load, &[sampler_ty, sampler_val, sampler_var]);
                let sampled_img_ty = self.emitter.emit_type_sampled_image(desc.element_ty);
                let sampled_img_id = self.emitter.alloc_id();
                self.emitter.emit_fn_inst(
                    spirv::Op::SampledImage,
                    &[sampled_img_ty, sampled_img_id, img_id, sampler_val],
                );
                if image_inst.is_some_and(|ii| self.func.insts[ii].non_uniform) {
                    self.emitter
                        .emit_decorate(sampled_img_id, spirv::Decoration::NonUniform, &[]);
                }
                Some(sampled_img_id)
            }
            _ => None,
        }
    }

    /// Loads a raw `OpTypeImage` handle from `desc` (extracting via `OpImage` for combined samplers).
    fn load_raw_image(&mut self, desc: &EmittedDescriptor) -> Option<u32> {
        if desc.raw_image_ty == 0 || desc.is_arrayed_descriptor() {
            return None;
        }
        let loaded = self.emitter.alloc_id();
        self.emitter
            .emit_fn_inst(spirv::Op::Load, &[desc.element_ty, loaded, desc.var_id]);
        if desc.kind == DescriptorKind::CombinedImageSampler {
            let raw_img = self.emitter.alloc_id();
            self.emitter
                .emit_fn_inst(spirv::Op::Image, &[desc.raw_image_ty, raw_img, loaded]);
            Some(raw_img)
        } else {
            Some(loaded)
        }
    }

    fn emit_workgroup_ptr(&mut self, wg: &EmittedWorkgroupVar, index: Option<InstId>) -> u32 {
        if wg.is_array {
            let u32_ty = self.u32_ty();
            let i32_ty = self.i32_ty();
            let idx_id = if let Some(idx_inst) = index {
                let raw_idx = self.ensure_inst(idx_inst);
                let src_ty = self.inst_types.get(&idx_inst).copied().unwrap_or(u32_ty);
                if src_ty == u32_ty || src_ty == i32_ty {
                    raw_idx
                } else {
                    self.coerce_to_ty(raw_idx, src_ty, u32_ty)
                }
            } else {
                self.emitter.emit_constant_32(u32_ty, 0)
            };
            let elem_ptr_ty = self
                .emitter
                .emit_type_pointer(spirv::StorageClass::Workgroup, wg.elem_ty);
            let ptr = self.emitter.alloc_id();
            self.emitter.emit_fn_inst(
                spirv::Op::AccessChain,
                &[elem_ptr_ty, ptr, wg.var_id, idx_id],
            );
            ptr
        } else {
            wg.var_id
        }
    }

    /// Walks an [`IrType`] through a sequence of index instructions to compute the resulting pointee type.
    fn walk_ir_type_indices(&self, base_ty: &IrType, indices: &[InstId]) -> Option<IrType> {
        let mut cur = base_ty.clone();
        for &idx_inst in indices {
            cur = match cur {
                IrType::Struct(layout) => {
                    let member_idx = match self.func.insts[idx_inst].data {
                        InstData::ConstU32(v) => v as usize,
                        InstData::ConstI32(v) => usize::try_from(v).unwrap_or(0),
                        InstData::ConstU64(v) => v as usize,
                        InstData::ConstI64(v) => usize::try_from(v).unwrap_or(0),
                        _ => 0,
                    };
                    layout.fields.get(member_idx)?.ty.clone()
                }
                IrType::Array { element, .. } | IrType::RuntimeArray(element) => *element,
                IrType::Matrix { scalar, rows, .. } => IrType::Vector { scalar, len: rows },
                IrType::Vector { scalar, .. } => IrType::Scalar(scalar),
                _ => return None,
            };
        }
        Some(cur)
    }

    /// Walks an [`IrType`] through constant composite indices (`CompositeExtract`).
    fn walk_ir_type_const_indices(base_ty: &IrType, indices: &[u32]) -> Option<IrType> {
        let mut cur = base_ty.clone();
        for &idx in indices {
            cur = match cur {
                IrType::Struct(layout) => layout.fields.get(idx as usize)?.ty.clone(),
                IrType::Array { element, .. } | IrType::RuntimeArray(element) => *element,
                IrType::Matrix { scalar, rows, .. } => IrType::Vector { scalar, len: rows },
                IrType::Vector { scalar, .. } => IrType::Scalar(scalar),
                _ => return None,
            };
        }
        Some(cur)
    }

    fn ensure_inst(&mut self, id: InstId) -> u32 {
        if let Some(&spv_id) = self.inst_ids.get(&id) {
            return spv_id;
        }
        let data = self.func.insts[id].data.clone();
        let mut is_bda_ptr = false;
        let mut is_workgroup_ptr = false;
        let mut is_global_var = false;
        let (spv_id, res_ty) = match data {
            InstData::ConstBool(v) => {
                let ty = self.bool_ty();
                (self.emitter.emit_constant_bool(ty, v), ty)
            }
            InstData::ConstI32(v) => {
                let ty = self.i32_ty();
                (self.emitter.emit_constant_32(ty, v.cast_unsigned()), ty)
            }
            InstData::ConstU32(v) => {
                let ty = self.u32_ty();
                (self.emitter.emit_constant_32(ty, v), ty)
            }
            InstData::ConstF32(bits) => {
                let ty = self.f32_ty();
                (self.emitter.emit_constant_32(ty, bits), ty)
            }
            InstData::ConstF16(bits) => {
                let ty = self.f16_ty();
                (self.emitter.emit_constant_32(ty, u32::from(bits)), ty)
            }
            InstData::ConstI64(v) => {
                let ty = self.i64_ty();
                (self.emitter.emit_constant_64(ty, v.cast_unsigned()), ty)
            }
            InstData::ConstU64(v) => {
                let ty = self.u64_ty();
                (self.emitter.emit_constant_64(ty, v), ty)
            }
            InstData::IAdd32(a, b) => self.emit_binop_32_with_id(spirv::Op::IAdd, a, b, Some(id)),
            InstData::ISub32(a, b) => self.emit_binop_32_with_id(spirv::Op::ISub, a, b, Some(id)),
            InstData::IMul32(a, b) => self.emit_binop_32_with_id(spirv::Op::IMul, a, b, Some(id)),
            InstData::IDiv32(a, b) => self.emit_signed_or_unsigned_binop_32_with_id(
                spirv::Op::SDiv,
                spirv::Op::UDiv,
                a,
                b,
                Some(id),
            ),
            InstData::IRem32(a, b) => self.emit_signed_or_unsigned_binop_32_with_id(
                spirv::Op::SRem,
                spirv::Op::UMod,
                a,
                b,
                Some(id),
            ),
            InstData::BitAnd32(a, b) => {
                self.emit_binop_32_with_id(spirv::Op::BitwiseAnd, a, b, Some(id))
            }
            InstData::BitOr32(a, b) => {
                self.emit_binop_32_with_id(spirv::Op::BitwiseOr, a, b, Some(id))
            }
            InstData::BitXor32(a, b) => {
                self.emit_binop_32_with_id(spirv::Op::BitwiseXor, a, b, Some(id))
            }
            InstData::BitNot32(a) => {
                let raw_val = self.ensure_inst(a);
                let u32_ty = self.u32_ty();
                let src_ty = self.inst_types.get(&a).copied().unwrap_or(u32_ty);
                let ir_ty = self.func.insts[id].ir_ty.clone();
                let ty = if ir_ty.is_64bit() {
                    self.emitter.emit_ir_type(&ir_ty)
                } else if self.is_signed_int_spv_ty(src_ty) || self.is_unsigned_int_spv_ty(src_ty) {
                    src_ty
                } else {
                    u32_ty
                };
                let val = self.coerce_to_ty(raw_val, src_ty, ty);
                let res = self.take_preallocated_id(Some(id), ty);
                self.emitter.emit_fn_inst(spirv::Op::Not, &[ty, res, val]);
                (res, ty)
            }
            InstData::Shl32(a, b) => {
                self.emit_binop_32_with_id(spirv::Op::ShiftLeftLogical, a, b, Some(id))
            }
            InstData::Shr32(a, b) => self.emit_signed_or_unsigned_binop_32_with_id(
                spirv::Op::ShiftRightArithmetic,
                spirv::Op::ShiftRightLogical,
                a,
                b,
                Some(id),
            ),
            InstData::FAdd32(a, b) => {
                self.emit_float_binop_with_id(spirv::Op::FAdd, a, b, Some(id))
            }
            InstData::FSub32(a, b) => {
                self.emit_float_binop_with_id(spirv::Op::FSub, a, b, Some(id))
            }
            InstData::FMul32(a, b) => {
                self.emit_float_binop_with_id(spirv::Op::FMul, a, b, Some(id))
            }
            InstData::FDiv32(a, b) => {
                self.emit_float_binop_with_id(spirv::Op::FDiv, a, b, Some(id))
            }
            InstData::ICmpEq(a, b) => self.emit_cmp_op(spirv::Op::IEqual, a, b),
            InstData::ICmpNe(a, b) => self.emit_cmp_op(spirv::Op::INotEqual, a, b),
            InstData::ICmpLt(a, b) => {
                self.emit_int_cmp_op(spirv::Op::SLessThan, spirv::Op::ULessThan, a, b)
            }
            InstData::ICmpLe(a, b) => {
                self.emit_int_cmp_op(spirv::Op::SLessThanEqual, spirv::Op::ULessThanEqual, a, b)
            }
            InstData::ICmpGt(a, b) => {
                self.emit_int_cmp_op(spirv::Op::SGreaterThan, spirv::Op::UGreaterThan, a, b)
            }
            InstData::ICmpGe(a, b) => self.emit_int_cmp_op(
                spirv::Op::SGreaterThanEqual,
                spirv::Op::UGreaterThanEqual,
                a,
                b,
            ),
            InstData::FCmpEq(a, b) => self.emit_fcmp_op(spirv::Op::FOrdEqual, a, b),
            InstData::FCmpNe(a, b) => self.emit_fcmp_op(spirv::Op::FUnordNotEqual, a, b),
            InstData::FCmpLt(a, b) => self.emit_fcmp_op(spirv::Op::FOrdLessThan, a, b),
            InstData::FCmpLe(a, b) => self.emit_fcmp_op(spirv::Op::FOrdLessThanEqual, a, b),
            InstData::FCmpGt(a, b) => self.emit_fcmp_op(spirv::Op::FOrdGreaterThan, a, b),
            InstData::FCmpGe(a, b) => self.emit_fcmp_op(spirv::Op::FOrdGreaterThanEqual, a, b),
            InstData::UCmpLt(a, b) => {
                let raw_lhs = self.ensure_inst(a);
                let raw_rhs = self.ensure_inst(b);
                let u32_ty = self.u32_ty();
                let u64_cached = self.cached_u64_ty();
                let i64_cached = self.cached_i64_ty();
                let lhs_ty = self.inst_types.get(&a).copied().unwrap_or(u32_ty);
                let rhs_ty = self.inst_types.get(&b).copied().unwrap_or(u32_ty);
                let (cmp_ty, res_ty) = if let Some((_, len)) = self
                    .vector_component_info(lhs_ty)
                    .or_else(|| self.vector_component_info(rhs_ty))
                {
                    let vec_u32 = self.emitter.emit_type_vector(u32_ty, len);
                    let bool_ty = self.bool_ty();
                    let vec_bool = self.emitter.emit_type_vector(bool_ty, len);
                    (vec_u32, vec_bool)
                } else if u64_cached == Some(lhs_ty)
                    || i64_cached == Some(lhs_ty)
                    || u64_cached == Some(rhs_ty)
                    || i64_cached == Some(rhs_ty)
                {
                    (self.u64_ty(), self.bool_ty())
                } else {
                    (u32_ty, self.bool_ty())
                };
                let lhs = self.coerce_to_ty(raw_lhs, lhs_ty, cmp_ty);
                let rhs = self.coerce_to_ty(raw_rhs, rhs_ty, cmp_ty);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::ULessThan, &[res_ty, res, lhs, rhs]);
                (res, res_ty)
            }
            InstData::LogicalAnd(a, b) => {
                let raw_lhs = self.ensure_inst(a);
                let raw_rhs = self.ensure_inst(b);
                let bool_ty = self.bool_ty();
                let lhs_ty = self.inst_types.get(&a).copied().unwrap_or(bool_ty);
                let rhs_ty = self.inst_types.get(&b).copied().unwrap_or(bool_ty);
                let ty = if let Some((_, len)) = self
                    .vector_component_info(lhs_ty)
                    .or_else(|| self.vector_component_info(rhs_ty))
                {
                    self.emitter.emit_type_vector(bool_ty, len)
                } else {
                    bool_ty
                };
                let lhs = self.coerce_to_ty(raw_lhs, lhs_ty, ty);
                let rhs = self.coerce_to_ty(raw_rhs, rhs_ty, ty);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::LogicalAnd, &[ty, res, lhs, rhs]);
                (res, ty)
            }
            InstData::LogicalOr(a, b) => {
                let raw_lhs = self.ensure_inst(a);
                let raw_rhs = self.ensure_inst(b);
                let bool_ty = self.bool_ty();
                let lhs_ty = self.inst_types.get(&a).copied().unwrap_or(bool_ty);
                let rhs_ty = self.inst_types.get(&b).copied().unwrap_or(bool_ty);
                let ty = if let Some((_, len)) = self
                    .vector_component_info(lhs_ty)
                    .or_else(|| self.vector_component_info(rhs_ty))
                {
                    self.emitter.emit_type_vector(bool_ty, len)
                } else {
                    bool_ty
                };
                let lhs = self.coerce_to_ty(raw_lhs, lhs_ty, ty);
                let rhs = self.coerce_to_ty(raw_rhs, rhs_ty, ty);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::LogicalOr, &[ty, res, lhs, rhs]);
                (res, ty)
            }
            InstData::LogicalNot(a) => {
                let raw = self.ensure_inst(a);
                let bool_ty = self.bool_ty();
                let src_ty = self.inst_types.get(&a).copied().unwrap_or(bool_ty);
                let ty = if let Some((_, len)) = self.vector_component_info(src_ty) {
                    self.emitter.emit_type_vector(bool_ty, len)
                } else {
                    bool_ty
                };
                let val = self.coerce_to_ty(raw, src_ty, ty);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::LogicalNot, &[ty, res, val]);
                (res, ty)
            }
            InstData::Select {
                cond,
                on_true,
                on_false,
            } => {
                let raw_cond = self.ensure_inst(cond);
                let raw_true = self.ensure_inst(on_true);
                let raw_false = self.ensure_inst(on_false);
                let u32_ty = self.u32_ty();
                let bool_ty = self.bool_ty();
                let cond_src_ty = self.inst_types.get(&cond).copied().unwrap_or(bool_ty);
                let true_ty = self.inst_types.get(&on_true).copied().unwrap_or(u32_ty);
                let false_ty = self.inst_types.get(&on_false).copied().unwrap_or(u32_ty);
                let ty = if true_ty != self.void_ty {
                    true_ty
                } else if false_ty != self.void_ty {
                    false_ty
                } else {
                    u32_ty
                };
                let cond_id = if let Some((c_ty, _)) = self.vector_component_info(cond_src_ty)
                    && c_ty == bool_ty
                {
                    raw_cond
                } else {
                    self.coerce_to_ty(raw_cond, cond_src_ty, bool_ty)
                };
                let true_id = self.coerce_to_ty(raw_true, true_ty, ty);
                let false_id = self.coerce_to_ty(raw_false, false_ty, ty);
                let res = self.take_preallocated_id(Some(id), ty);
                self.emitter
                    .emit_fn_inst(spirv::Op::Select, &[ty, res, cond_id, true_id, false_id]);
                (res, ty)
            }
            InstData::CastI64ToU64(inner) => {
                let u64_ty = self.u64_ty();
                let coerced = if let InstData::ConstI64(v) = self.func.insts[inner].data {
                    self.emitter.emit_constant_64(u64_ty, v.cast_unsigned())
                } else {
                    let inner_id = self.ensure_inst(inner);
                    let src_ty = self.inst_types.get(&inner).copied().unwrap_or(u64_ty);
                    self.coerce_to_ty(inner_id, src_ty, u64_ty)
                };
                let res = if let Some(prealloc) = self.take_preallocated_exact(id, u64_ty) {
                    self.emitter
                        .emit_fn_inst(spirv::Op::CopyObject, &[u64_ty, prealloc, coerced]);
                    prealloc
                } else {
                    coerced
                };
                (res, u64_ty)
            }
            InstData::CastToU32(val) => {
                let val_id = self.ensure_inst(val);
                let u32_ty = self.u32_ty();
                let src_ty = self.inst_types.get(&val).copied().unwrap_or(u32_ty);
                let coerced = self.coerce_to_ty(val_id, src_ty, u32_ty);
                let res = if let Some(prealloc) = self.take_preallocated_exact(id, u32_ty) {
                    self.emitter
                        .emit_fn_inst(spirv::Op::CopyObject, &[u32_ty, prealloc, coerced]);
                    prealloc
                } else if self.func.insts[id].non_uniform && self.emitter.is_const_id(coerced) {
                    let copy = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::CopyObject, &[u32_ty, copy, coerced]);
                    copy
                } else {
                    coerced
                };
                (res, u32_ty)
            }
            InstData::CastToI32(val) => {
                let val_id = self.ensure_inst(val);
                let i32_ty = self.i32_ty();
                let src_ty = self.inst_types.get(&val).copied().unwrap_or(i32_ty);
                (self.coerce_to_ty(val_id, src_ty, i32_ty), i32_ty)
            }
            InstData::CastToF32(val) => {
                let val_id = self.ensure_inst(val);
                let f32_ty = self.f32_ty();
                let src_ty = self.inst_types.get(&val).copied().unwrap_or(f32_ty);
                (self.coerce_to_ty(val_id, src_ty, f32_ty), f32_ty)
            }
            InstData::CastToI64(val) => {
                let val_id = self.ensure_inst(val);
                let i64_ty = self.i64_ty();
                let src_ty = self.inst_types.get(&val).copied().unwrap_or(i64_ty);
                (self.coerce_to_ty(val_id, src_ty, i64_ty), i64_ty)
            }
            InstData::CastToU64(val) => {
                let val_id = self.ensure_inst(val);
                let u64_ty = self.u64_ty();
                let src_ty = self.inst_types.get(&val).copied().unwrap_or(u64_ty);
                (self.coerce_to_ty(val_id, src_ty, u64_ty), u64_ty)
            }
            InstData::FConvert { value, target } => {
                let raw = self.ensure_inst(value);
                let scalar_dst = match target {
                    ScalarIrTy::F16 => self.f16_ty(),
                    _ => self.f32_ty(),
                };
                let src_ty = self.inst_types.get(&value).copied().unwrap_or(scalar_dst);
                let dst_ty = if let Some((_, len)) = self.vector_component_info(src_ty) {
                    self.emitter.emit_type_vector(scalar_dst, len)
                } else {
                    scalar_dst
                };
                if src_ty == dst_ty {
                    (raw, dst_ty)
                } else {
                    let res = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::FConvert, &[dst_ty, res, raw]);
                    (res, dst_ty)
                }
            }
            InstData::SpecConstant {
                spec_id,
                scalar,
                default_bits,
            } => {
                if let Some(&(existing, emitted_ty)) = self.emitter.spec_const_ids.get(&spec_id) {
                    let target_ty = self.emitter.emit_ir_type(&IrType::Scalar(scalar));
                    if emitted_ty == target_ty {
                        (existing, emitted_ty)
                    } else {
                        let coerced = self.coerce_to_ty(existing, emitted_ty, target_ty);
                        (coerced, target_ty)
                    }
                } else {
                    let spv_id = self.emitter.alloc_id();
                    let ty = match scalar {
                        ScalarIrTy::Bool => {
                            let bty = self.bool_ty();
                            let op = if default_bits != 0 {
                                spirv::Op::SpecConstantTrue
                            } else {
                                spirv::Op::SpecConstantFalse
                            };
                            push_inst(&mut self.emitter.types_globals, op, &[bty, spv_id]);
                            bty
                        }
                        ScalarIrTy::I32 => {
                            let ity = self.i32_ty();
                            push_inst(
                                &mut self.emitter.types_globals,
                                spirv::Op::SpecConstant,
                                &[ity, spv_id, default_bits],
                            );
                            ity
                        }
                        ScalarIrTy::F16 => {
                            let fty = self.f16_ty();
                            push_inst(
                                &mut self.emitter.types_globals,
                                spirv::Op::SpecConstant,
                                &[fty, spv_id, default_bits & 0xFFFF],
                            );
                            fty
                        }
                        ScalarIrTy::F32 => {
                            let fty = self.f32_ty();
                            push_inst(
                                &mut self.emitter.types_globals,
                                spirv::Op::SpecConstant,
                                &[fty, spv_id, default_bits],
                            );
                            fty
                        }
                        ScalarIrTy::I64 => {
                            let ty64 = self.i64_ty();
                            let extended = i64::from(default_bits.cast_signed()).cast_unsigned();
                            let lo = extended as u32;
                            let hi = (extended >> 32) as u32;
                            push_inst(
                                &mut self.emitter.types_globals,
                                spirv::Op::SpecConstant,
                                &[ty64, spv_id, lo, hi],
                            );
                            ty64
                        }
                        ScalarIrTy::U64 => {
                            let ty64 = self.u64_ty();
                            push_inst(
                                &mut self.emitter.types_globals,
                                spirv::Op::SpecConstant,
                                &[ty64, spv_id, default_bits, 0],
                            );
                            ty64
                        }
                        ScalarIrTy::U32 => {
                            let uty = self.u32_ty();
                            push_inst(
                                &mut self.emitter.types_globals,
                                spirv::Op::SpecConstant,
                                &[uty, spv_id, default_bits],
                            );
                            uty
                        }
                    };
                    self.emitter
                        .emit_decorate(spv_id, spirv::Decoration::SpecId, &[spec_id]);
                    self.emitter.register_const_id(spv_id);
                    self.emitter.spec_const_ids.insert(spec_id, (spv_id, ty));
                    (spv_id, ty)
                }
            }
            InstData::SpecConstantOp { op, operands } => {
                let mut spv_ops = Vec::with_capacity(3 + operands.len());
                let mut op_ids = Vec::with_capacity(operands.len());
                for op_inst in &operands {
                    op_ids.push(self.ensure_inst(*op_inst));
                }
                let spv_opcode = match op {
                    SpecConstantOpKind::IAdd => spirv::Op::IAdd,
                    SpecConstantOpKind::ISub => spirv::Op::ISub,
                    SpecConstantOpKind::IMul => spirv::Op::IMul,
                    SpecConstantOpKind::UDiv => spirv::Op::UDiv,
                    SpecConstantOpKind::SDiv => spirv::Op::SDiv,
                    SpecConstantOpKind::BitwiseAnd => spirv::Op::BitwiseAnd,
                    SpecConstantOpKind::BitwiseOr => spirv::Op::BitwiseOr,
                    SpecConstantOpKind::BitwiseXor => spirv::Op::BitwiseXor,
                    SpecConstantOpKind::ShiftLeftLogical => spirv::Op::ShiftLeftLogical,
                    SpecConstantOpKind::ShiftRightLogical => spirv::Op::ShiftRightLogical,
                    SpecConstantOpKind::Select => spirv::Op::Select,
                    SpecConstantOpKind::IEqual => spirv::Op::IEqual,
                    SpecConstantOpKind::INotEqual => spirv::Op::INotEqual,
                    SpecConstantOpKind::ULessThan => spirv::Op::ULessThan,
                    SpecConstantOpKind::SLessThan => spirv::Op::SLessThan,
                };
                let res_ty = match op {
                    SpecConstantOpKind::IEqual
                    | SpecConstantOpKind::INotEqual
                    | SpecConstantOpKind::ULessThan
                    | SpecConstantOpKind::SLessThan => self.bool_ty(),
                    _ if self.func.insts[id].ir_ty != IrType::Void => {
                        let ir_ty = self.func.insts[id].ir_ty.clone();
                        self.emitter.emit_ir_type(&ir_ty)
                    }
                    SpecConstantOpKind::Select => operands
                        .get(1)
                        .and_then(|o| self.inst_types.get(o).copied())
                        .unwrap_or_else(|| self.u32_ty()),
                    _ => operands
                        .first()
                        .and_then(|o| self.inst_types.get(o).copied())
                        .unwrap_or_else(|| self.u32_ty()),
                };
                let spv_id = self.emitter.alloc_id();
                spv_ops.push(res_ty);
                spv_ops.push(spv_id);
                spv_ops.push(spv_opcode as u32);
                spv_ops.extend_from_slice(&op_ids);
                push_inst(
                    &mut self.emitter.types_globals,
                    spirv::Op::SpecConstantOp,
                    &spv_ops,
                );
                self.emitter.register_const_id(spv_id);
                (spv_id, res_ty)
            }
            InstData::GlslStd450 { op, ref args } => {
                let args_cloned = args.clone();
                self.emit_glsl_std_450_op(op, &args_cloned)
            }
            InstData::VectorTimesScalar { vec, scalar } => {
                let vec_id = self.ensure_inst(vec);
                let raw_scalar = self.ensure_inst(scalar);
                let f32_ty = self.f32_ty();
                let vec_ty = self
                    .inst_types
                    .get(&vec)
                    .copied()
                    .unwrap_or_else(|| self.emitter.emit_type_vector(f32_ty, 3));
                let comp_ty = self
                    .vector_component_info(vec_ty)
                    .map_or(f32_ty, |(c, _)| c);
                let scalar_src_ty = self.inst_types.get(&scalar).copied().unwrap_or(comp_ty);
                let scalar_id = self.coerce_to_ty(raw_scalar, scalar_src_ty, comp_ty);
                let res = self.emitter.alloc_id();
                self.emitter.emit_fn_inst(
                    spirv::Op::VectorTimesScalar,
                    &[vec_ty, res, vec_id, scalar_id],
                );
                (res, vec_ty)
            }
            InstData::MatrixTimesVector { mat, vec } => {
                let mat_id = self.ensure_inst(mat);
                let vec_id = self.ensure_inst(vec);
                let ir_ty = self.func.insts[id].ir_ty.clone();
                let res_ty = if ir_ty != IrType::Void {
                    self.emitter.emit_ir_type(&ir_ty)
                } else if let IrType::Matrix { scalar, rows, .. } = self.func.insts[mat].ir_ty {
                    self.emitter
                        .emit_ir_type(&IrType::Vector { scalar, len: rows })
                } else {
                    self.inst_types.get(&vec).copied().unwrap_or_else(|| {
                        let f = self.f32_ty();
                        self.emitter.emit_type_vector(f, 4)
                    })
                };
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::MatrixTimesVector, &[res_ty, res, mat_id, vec_id]);
                (res, res_ty)
            }
            InstData::MatrixTimesMatrix { lhs, rhs } => {
                let lhs_id = self.ensure_inst(lhs);
                let rhs_id = self.ensure_inst(rhs);
                let ir_ty = self.func.insts[id].ir_ty.clone();
                let res_ty = if ir_ty != IrType::Void {
                    self.emitter.emit_ir_type(&ir_ty)
                } else {
                    self.inst_types.get(&lhs).copied().unwrap_or_else(|| {
                        let f = self.f32_ty();
                        let v4 = self.emitter.emit_type_vector(f, 4);
                        self.emitter.emit_type_matrix(v4, 4)
                    })
                };
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::MatrixTimesMatrix, &[res_ty, res, lhs_id, rhs_id]);
                (res, res_ty)
            }
            InstData::VectorShuffle {
                lhs,
                rhs,
                components,
            } => {
                let lhs_id = self.ensure_inst(lhs);
                let rhs_id = self.ensure_inst(rhs);
                let f32_ty = self.f32_ty();
                let lhs_ty = self.inst_types.get(&lhs).copied().unwrap_or(f32_ty);
                let rhs_ty = self.inst_types.get(&rhs).copied().unwrap_or(lhs_ty);
                let lhs_vec_info = self.vector_component_info(lhs_ty);
                let rhs_vec_info = self.vector_component_info(rhs_ty);
                let (comp_ty, lhs_len) = lhs_vec_info.unwrap_or((lhs_ty, 1));
                if components.is_empty() {
                    (self.emit_zero_const(comp_ty), comp_ty)
                } else if components.len() == 1 {
                    let idx = components[0];
                    if idx < lhs_len {
                        if lhs_vec_info.is_some() {
                            let res = self.emitter.alloc_id();
                            self.emitter.emit_fn_inst(
                                spirv::Op::CompositeExtract,
                                &[comp_ty, res, lhs_id, idx],
                            );
                            (res, comp_ty)
                        } else {
                            (lhs_id, comp_ty)
                        }
                    } else if let Some((rhs_comp_ty, rhs_len)) = rhs_vec_info {
                        let local_idx = (idx - lhs_len).min(rhs_len.saturating_sub(1));
                        let extracted = self.emitter.alloc_id();
                        self.emitter.emit_fn_inst(
                            spirv::Op::CompositeExtract,
                            &[rhs_comp_ty, extracted, rhs_id, local_idx],
                        );
                        (self.coerce_to_ty(extracted, rhs_comp_ty, comp_ty), comp_ty)
                    } else {
                        (self.coerce_to_ty(rhs_id, rhs_ty, comp_ty), comp_ty)
                    }
                } else {
                    let (lhs_vec_id, lhs_vec_len, rhs_idx_shift) = if lhs_vec_info.is_some() {
                        (lhs_id, lhs_len, 0u32)
                    } else {
                        let v2_ty = self.emitter.emit_type_vector(comp_ty, 2);
                        let splat = self.coerce_to_ty(lhs_id, lhs_ty, v2_ty);
                        (splat, 2, 1u32)
                    };
                    let (rhs_vec_id, rhs_vec_len) =
                        if let Some((rhs_comp_ty, rhs_len)) = rhs_vec_info {
                            if rhs_comp_ty == comp_ty {
                                (rhs_id, rhs_len)
                            } else {
                                let target_rhs_ty = self.emitter.emit_type_vector(comp_ty, rhs_len);
                                (self.coerce_to_ty(rhs_id, rhs_ty, target_rhs_ty), rhs_len)
                            }
                        } else {
                            let v2_ty = self.emitter.emit_type_vector(comp_ty, 2);
                            (self.coerce_to_ty(rhs_id, rhs_ty, v2_ty), 2)
                        };
                    let total_lanes = lhs_vec_len.saturating_add(rhs_vec_len);
                    let out_len = components.len().clamp(2, 4);
                    let res_ty = self.emitter.emit_type_vector(comp_ty, out_len as u32);
                    let res = self.emitter.alloc_id();
                    let mut ops = Vec::with_capacity(4 + out_len);
                    ops.push(res_ty);
                    ops.push(res);
                    ops.push(lhs_vec_id);
                    ops.push(rhs_vec_id);
                    for &c in components.iter().take(out_len) {
                        let mapped = if c < lhs_len {
                            c
                        } else {
                            c.saturating_add(rhs_idx_shift)
                        };
                        ops.push(mapped.min(total_lanes.saturating_sub(1)));
                    }
                    self.emitter.emit_fn_inst(spirv::Op::VectorShuffle, &ops);
                    (res, res_ty)
                }
            }
            InstData::CompositeConstruct { elems } => {
                let mut elem_ids = Vec::with_capacity(elems.len());
                for &e in &elems {
                    elem_ids.push(self.ensure_inst(e));
                }
                let ir_ty = self.func.insts[id].ir_ty.clone();
                let res_ty = if ir_ty != IrType::Void {
                    self.emitter.emit_ir_type(&ir_ty)
                } else if let Some(&first) = elems.first() {
                    let first_ty = self
                        .inst_types
                        .get(&first)
                        .copied()
                        .unwrap_or_else(|| self.f32_ty());
                    self.emitter
                        .emit_type_vector(first_ty, elems.len().clamp(2, 4) as u32)
                } else {
                    self.u32_ty()
                };
                let res = self.emitter.alloc_id();
                let mut ops = Vec::with_capacity(2 + elem_ids.len());
                ops.push(res_ty);
                ops.push(res);
                ops.extend_from_slice(&elem_ids);
                self.emitter
                    .emit_fn_inst(spirv::Op::CompositeConstruct, &ops);
                (res, res_ty)
            }
            InstData::CompositeExtract { composite, indices } => {
                let comp_id = self.ensure_inst(composite);
                let ir_ty = self.func.insts[id].ir_ty.clone();
                let res_ty = if ir_ty != IrType::Void {
                    self.emitter.emit_ir_type(&ir_ty)
                } else if let Some(walked) =
                    Self::walk_ir_type_const_indices(&self.func.insts[composite].ir_ty, &indices)
                {
                    self.emitter.emit_ir_type(&walked)
                } else {
                    let comp_ty = self
                        .inst_types
                        .get(&composite)
                        .copied()
                        .unwrap_or_else(|| self.f32_ty());
                    self.vector_component_info(comp_ty)
                        .map_or(comp_ty, |(c, _)| c)
                };
                let res = self.emitter.alloc_id();
                let mut ops = Vec::with_capacity(3 + indices.len());
                ops.push(res_ty);
                ops.push(res);
                ops.push(comp_id);
                ops.extend_from_slice(&indices);
                self.emitter.emit_fn_inst(spirv::Op::CompositeExtract, &ops);
                (res, res_ty)
            }
            InstData::CompositeInsert {
                composite,
                value,
                indices,
            } => {
                let comp_id = self.ensure_inst(composite);
                let val_id = self.ensure_inst(value);
                let comp_ty = self.inst_types.get(&composite).copied().unwrap_or_else(|| {
                    let ir_ty = self.func.insts[id].ir_ty.clone();
                    self.emitter.emit_ir_type(&ir_ty)
                });
                let res = self.emitter.alloc_id();
                // SPIR-V OpCompositeInsert operand order: Result Type, Result <id>, Object, Composite, Indexes...
                let mut ops = Vec::with_capacity(4 + indices.len());
                ops.push(comp_ty);
                ops.push(res);
                ops.push(val_id);
                ops.push(comp_id);
                ops.extend_from_slice(&indices);
                self.emitter.emit_fn_inst(spirv::Op::CompositeInsert, &ops);
                (res, comp_ty)
            }
            InstData::LocalAddr { local_index } => {
                if let Some(lv) = self.local_vars.get(local_index as usize).cloned() {
                    self.inst_pointee_types.insert(id, lv.val_ty);
                    self.inst_ptr_storage
                        .insert(id, spirv::StorageClass::Function);
                    self.inst_ptr_ir_pointee.insert(id, lv.ir_ty);
                    is_global_var = true;
                    (lv.var_id, lv.ptr_ty)
                } else {
                    (0, self.void_ty)
                }
            }
            InstData::LocalLoad { ptr } => {
                let ptr_id = self.ensure_inst(ptr);
                if ptr_id == 0 {
                    (0, self.void_ty)
                } else {
                    let val_ty = self
                        .inst_pointee_types
                        .get(&ptr)
                        .copied()
                        .unwrap_or_else(|| {
                            let ir_ty = self.func.insts[id].ir_ty.clone();
                            if ir_ty != IrType::Void {
                                self.emitter.emit_ir_type(&ir_ty)
                            } else {
                                self.u32_ty()
                            }
                        });
                    let res = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::Load, &[val_ty, res, ptr_id]);
                    (res, val_ty)
                }
            }
            InstData::LocalStore { ptr, value } => {
                let ptr_id = self.ensure_inst(ptr);
                let raw_val = self.ensure_inst(value);
                if ptr_id != 0 {
                    let val_src_ty = self
                        .inst_types
                        .get(&value)
                        .copied()
                        .unwrap_or_else(|| self.u32_ty());
                    let pointee_ty = self
                        .inst_pointee_types
                        .get(&ptr)
                        .copied()
                        .unwrap_or(val_src_ty);
                    let val_id = self.coerce_to_ty(raw_val, val_src_ty, pointee_ty);
                    self.emitter
                        .emit_fn_inst(spirv::Op::Store, &[ptr_id, val_id]);
                }
                (0, self.void_ty)
            }
            InstData::AccessChain { base_ptr, indices } => {
                let base_id = self.ensure_inst(base_ptr);
                if base_id == 0 {
                    (0, self.void_ty)
                } else {
                    let sc = self
                        .inst_ptr_storage
                        .get(&base_ptr)
                        .copied()
                        .unwrap_or_else(|| {
                            if let IrType::Pointer { storage, .. } = self.func.insts[base_ptr].ir_ty
                            {
                                map_ir_storage_class(storage)
                            } else {
                                spirv::StorageClass::Function
                            }
                        });
                    let in_block = matches!(
                        sc,
                        spirv::StorageClass::Uniform
                            | spirv::StorageClass::StorageBuffer
                            | spirv::StorageClass::PhysicalStorageBuffer
                    );
                    let u32_ty = self.u32_ty();
                    let i32_ty = self.i32_ty();
                    let mut idx_ids = Vec::with_capacity(indices.len());
                    for &idx_inst in &indices {
                        let raw_idx = self.ensure_inst(idx_inst);
                        let src_ty = self.inst_types.get(&idx_inst).copied().unwrap_or(u32_ty);
                        if src_ty == u32_ty || src_ty == i32_ty {
                            idx_ids.push(raw_idx);
                        } else {
                            idx_ids.push(self.coerce_to_ty(raw_idx, src_ty, u32_ty));
                        }
                    }
                    let walked_ir = self
                        .inst_ptr_ir_pointee
                        .get(&base_ptr)
                        .cloned()
                        .and_then(|base_ir| self.walk_ir_type_indices(&base_ir, &indices));
                    let res_pointee_ir = match &self.func.insts[id].ir_ty {
                        IrType::Pointer { pointee, .. } if **pointee != IrType::Void => {
                            Some(pointee.as_ref().clone())
                        }
                        _ => walked_ir,
                    };
                    let res_pointee_ty = if let Some(ref p_ir) = res_pointee_ir {
                        self.emitter.emit_ir_type_with_layout(p_ir, in_block)
                    } else {
                        u32_ty
                    };
                    let res_ptr_ty = self.emitter.emit_type_pointer(sc, res_pointee_ty);
                    let res = self.emitter.alloc_id();
                    let mut ops = Vec::with_capacity(3 + idx_ids.len());
                    ops.push(res_ptr_ty);
                    ops.push(res);
                    ops.push(base_id);
                    ops.extend_from_slice(&idx_ids);
                    self.emitter.emit_fn_inst(spirv::Op::AccessChain, &ops);
                    self.inst_pointee_types.insert(id, res_pointee_ty);
                    self.inst_ptr_storage.insert(id, sc);
                    if let Some(p_ir) = res_pointee_ir {
                        self.inst_ptr_ir_pointee.insert(id, p_ir);
                    }
                    if sc == spirv::StorageClass::Workgroup {
                        is_workgroup_ptr = true;
                    } else if sc == spirv::StorageClass::PhysicalStorageBuffer {
                        is_bda_ptr = true;
                    }
                    (res, res_ptr_ty)
                }
            }
            InstData::DescriptorAccessChain {
                descriptor_index,
                indices,
            } => {
                if let Some(desc) = self.descriptors.get(descriptor_index as usize).cloned() {
                    let u32_ty = self.u32_ty();
                    let i32_ty = self.i32_ty();
                    let mut idx_ids = Vec::with_capacity(indices.len());
                    for &idx_inst in &indices {
                        let raw_idx = self.ensure_inst(idx_inst);
                        let src_ty = self.inst_types.get(&idx_inst).copied().unwrap_or(u32_ty);
                        if src_ty == u32_ty || src_ty == i32_ty {
                            idx_ids.push(raw_idx);
                        } else {
                            idx_ids.push(self.coerce_to_ty(raw_idx, src_ty, u32_ty));
                        }
                    }
                    let in_block = matches!(
                        desc.storage_class,
                        spirv::StorageClass::Uniform | spirv::StorageClass::StorageBuffer
                    );
                    let walked_ir = desc.struct_layout.as_ref().and_then(|layout| {
                        self.walk_ir_type_indices(&IrType::Struct(layout.clone()), &indices)
                    });
                    let res_pointee_ir = match &self.func.insts[id].ir_ty {
                        IrType::Pointer { pointee, .. } if **pointee != IrType::Void => {
                            Some(pointee.as_ref().clone())
                        }
                        _ => walked_ir,
                    };
                    let res_pointee_ty = if let Some(ref p_ir) = res_pointee_ir {
                        self.emitter.emit_ir_type_with_layout(p_ir, in_block)
                    } else if in_block {
                        u32_ty
                    } else {
                        desc.element_ty
                    };
                    let res_ptr_ty = self
                        .emitter
                        .emit_type_pointer(desc.storage_class, res_pointee_ty);
                    let res = self.emitter.alloc_id();
                    let mut ops = Vec::with_capacity(3 + idx_ids.len());
                    ops.push(res_ptr_ty);
                    ops.push(res);
                    ops.push(desc.var_id);
                    ops.extend_from_slice(&idx_ids);
                    self.emitter.emit_fn_inst(spirv::Op::AccessChain, &ops);
                    self.inst_pointee_types.insert(id, res_pointee_ty);
                    self.inst_ptr_storage.insert(id, desc.storage_class);
                    if let Some(p_ir) = res_pointee_ir {
                        self.inst_ptr_ir_pointee.insert(id, p_ir);
                    }
                    (res, res_ptr_ty)
                } else {
                    (0, self.void_ty)
                }
            }
            InstData::DescriptorStore { ptr, value } => {
                let ptr_id = self.ensure_inst(ptr);
                let raw_val = self.ensure_inst(value);
                if ptr_id != 0 {
                    let val_src_ty = self
                        .inst_types
                        .get(&value)
                        .copied()
                        .unwrap_or_else(|| self.u32_ty());
                    let pointee_ty = self
                        .inst_pointee_types
                        .get(&ptr)
                        .copied()
                        .unwrap_or(val_src_ty);
                    let val_id = self.coerce_to_ty(raw_val, val_src_ty, pointee_ty);
                    self.emitter
                        .emit_fn_inst(spirv::Op::Store, &[ptr_id, val_id]);
                }
                (0, self.void_ty)
            }
            InstData::Param { index } => {
                if let Some((param_id, spv_ty, is_ptr, val_ir_ty)) =
                    self.param_ids.get(index as usize).cloned()
                {
                    if is_ptr {
                        let pointee_ty = self.emitter.emit_ir_type(&val_ir_ty);
                        self.inst_pointee_types.insert(id, pointee_ty);
                        self.inst_ptr_storage
                            .insert(id, spirv::StorageClass::Function);
                        self.inst_ptr_ir_pointee.insert(id, val_ir_ty);
                    }
                    (param_id, spv_ty)
                } else {
                    (0, self.void_ty)
                }
            }
            InstData::FunctionCall { callee, args } => {
                let mut raw_args = Vec::with_capacity(args.len());
                for &arg in &args {
                    raw_args.push(self.ensure_inst(arg));
                }
                if let Some(callee_info) = self.emitted_functions.get(callee.as_str()).cloned() {
                    let mut call_ops = Vec::with_capacity(3 + args.len());
                    let res = self.emitter.alloc_id();
                    call_ops.push(callee_info.ret_ty);
                    call_ops.push(res);
                    call_ops.push(callee_info.fn_id);
                    for (idx, (&arg_inst, &raw_arg)) in args.iter().zip(raw_args.iter()).enumerate()
                    {
                        if let Some(&(expected_ty, is_ptr)) = callee_info.param_tys.get(idx) {
                            if is_ptr {
                                let src_ty = self
                                    .inst_types
                                    .get(&arg_inst)
                                    .copied()
                                    .unwrap_or(expected_ty);
                                if src_ty != expected_ty && self.bda_ptr_insts.contains(&arg_inst) {
                                    let casted = self.emitter.alloc_id();
                                    self.emitter.emit_fn_inst(
                                        spirv::Op::Bitcast,
                                        &[expected_ty, casted, raw_arg],
                                    );
                                    call_ops.push(casted);
                                } else {
                                    call_ops.push(raw_arg);
                                }
                            } else {
                                let src_ty = self
                                    .inst_types
                                    .get(&arg_inst)
                                    .copied()
                                    .unwrap_or(expected_ty);
                                call_ops.push(self.coerce_to_ty(raw_arg, src_ty, expected_ty));
                            }
                        } else {
                            call_ops.push(raw_arg);
                        }
                    }
                    self.emitter
                        .emit_fn_inst(spirv::Op::FunctionCall, &call_ops);
                    if callee_info.ret_ty == self.void_ty {
                        (0, self.void_ty)
                    } else {
                        (res, callee_info.ret_ty)
                    }
                } else {
                    (0, self.void_ty)
                }
            }
            InstData::IAdd64(a, b) => self.emit_binop_64_with_id(spirv::Op::IAdd, a, b, Some(id)),
            InstData::IMul64(a, b) => self.emit_binop_64_with_id(spirv::Op::IMul, a, b, Some(id)),
            InstData::ConvertPtrToU(ptr) => {
                let raw_id = self.ensure_inst(ptr);
                let actual_ptr_id = if self.bda_ptr_insts.contains(&ptr) {
                    raw_id
                } else {
                    let ptr_ty = self.bda_ptr_ty();
                    let addr_u64 = self.ensure_u64_addr(ptr);
                    let promoted = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::ConvertUToPtr, &[ptr_ty, promoted, addr_u64]);
                    promoted
                };
                let ty = self.u64_ty();
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::ConvertPtrToU, &[ty, res, actual_ptr_id]);
                (res, ty)
            }
            InstData::ConvertUToPtr { addr_u64, .. } => {
                let addr_id = self.ensure_u64_addr(addr_u64);
                let ty = if let IrType::Pointer { pointee, .. } = &self.func.insts[id].ir_ty
                    && **pointee != IrType::Void
                {
                    let p_clone = pointee.as_ref().clone();
                    let spv_ty = self.emitter.emit_ir_type_with_layout(&p_clone, true);
                    self.bda_ptr_ty_for(
                        spv_ty,
                        ir_type_stride(&p_clone),
                        matrix_col_stride(&p_clone),
                    )
                } else {
                    self.bda_ptr_ty()
                };
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::ConvertUToPtr, &[ty, res, addr_id]);
                is_bda_ptr = true;
                self.inst_ptr_storage.insert(id, self.bda_storage_class());
                (res, ty)
            }
            InstData::BdaLoad { ptr, access } => {
                let (elem_ty, stride, elem_align, mat_col_stride) = self.bda_load_elem_ty(id, ptr);
                let block_ptr_id = self.ensure_bda_block_ptr(ptr, elem_ty, stride, mat_col_stride);
                let elem_ptr = self.bda_elem_ptr_for(block_ptr_id, elem_ty);
                let res = self.emitter.alloc_id();
                let align = access.align.max(elem_align).next_power_of_two();
                if self.uses_bda {
                    let u32_ty = self.u32_ty();
                    let mem_access = spirv::MemoryAccess::ALIGNED
                        | spirv::MemoryAccess::MAKE_POINTER_VISIBLE
                        | spirv::MemoryAccess::NON_PRIVATE_POINTER;
                    let device_scope = self
                        .emitter
                        .emit_constant_32(u32_ty, spirv::Scope::Device as u32);
                    self.emitter.emit_fn_inst(
                        spirv::Op::Load,
                        &[
                            elem_ty,
                            res,
                            elem_ptr,
                            mem_access.bits(),
                            align,
                            device_scope,
                        ],
                    );
                } else {
                    self.emitter.emit_fn_inst(
                        spirv::Op::Load,
                        &[
                            elem_ty,
                            res,
                            elem_ptr,
                            spirv::MemoryAccess::ALIGNED.bits(),
                            align,
                        ],
                    );
                }
                (res, elem_ty)
            }
            InstData::BdaStore { ptr, value, access } => {
                let raw_val_id = self.ensure_inst(value);
                let val_ty = self
                    .inst_types
                    .get(&value)
                    .copied()
                    .unwrap_or_else(|| self.u32_ty());
                let (elem_ty, stride, elem_align, mat_col_stride) =
                    self.bda_store_elem_ty(value, val_ty);
                let val_id = self.coerce_to_ty(raw_val_id, val_ty, elem_ty);
                let block_ptr_id = self.ensure_bda_block_ptr(ptr, elem_ty, stride, mat_col_stride);
                let elem_ptr = self.bda_elem_ptr_for(block_ptr_id, elem_ty);
                let align = access.align.max(elem_align).next_power_of_two();
                if self.uses_bda {
                    let u32_ty = self.u32_ty();
                    let mem_access = spirv::MemoryAccess::ALIGNED
                        | spirv::MemoryAccess::MAKE_POINTER_AVAILABLE
                        | spirv::MemoryAccess::NON_PRIVATE_POINTER;
                    let device_scope = self
                        .emitter
                        .emit_constant_32(u32_ty, spirv::Scope::Device as u32);
                    self.emitter.emit_fn_inst(
                        spirv::Op::Store,
                        &[elem_ptr, val_id, mem_access.bits(), align, device_scope],
                    );
                } else {
                    self.emitter.emit_fn_inst(
                        spirv::Op::Store,
                        &[elem_ptr, val_id, spirv::MemoryAccess::ALIGNED.bits(), align],
                    );
                }
                (0, self.void_ty)
            }
            InstData::ImageSample {
                descriptor_index,
                sampler_descriptor_index,
                image_inst,
                coord,
                lod,
                is_fetch,
            } => {
                let preloaded_img = image_inst.map(|img_inst| self.ensure_inst(img_inst));
                let raw_coord_id = self.ensure_inst(coord);
                let raw_lod_id = self.ensure_inst(lod);
                let f32_ty = self.f32_ty();
                let i32_ty = self.i32_ty();
                let coord_ty = self.inst_types.get(&coord).copied().unwrap_or(i32_ty);
                let lod_ty = self.inst_types.get(&lod).copied().unwrap_or(i32_ty);

                if let Some(desc) = self.descriptors.get(descriptor_index as usize).cloned()
                    && (!desc.is_arrayed_descriptor()
                        || preloaded_img.is_some_and(|img_id| img_id != 0))
                    && preloaded_img != Some(0)
                {
                    match desc.kind {
                        DescriptorKind::SampledImage | DescriptorKind::CombinedImageSampler
                            if is_fetch && !desc.is_cube =>
                        {
                            let loaded_handle = if let Some(loaded) = preloaded_img {
                                loaded
                            } else {
                                let loaded = self.emitter.alloc_id();
                                self.emitter.emit_fn_inst(
                                    spirv::Op::Load,
                                    &[desc.element_ty, loaded, desc.var_id],
                                );
                                loaded
                            };
                            let img_id = if desc.kind == DescriptorKind::CombinedImageSampler {
                                let raw_img = self.emitter.alloc_id();
                                self.emitter.emit_fn_inst(
                                    spirv::Op::Image,
                                    &[desc.raw_image_ty, raw_img, loaded_handle],
                                );
                                if image_inst.is_some_and(|ii| self.func.insts[ii].non_uniform) {
                                    self.emitter.emit_decorate(
                                        raw_img,
                                        spirv::Decoration::NonUniform,
                                        &[],
                                    );
                                }
                                raw_img
                            } else {
                                loaded_handle
                            };
                            let coord_vec_id = self.emit_coord_vec(
                                raw_coord_id,
                                coord_ty,
                                i32_ty,
                                desc.coord_components,
                            );
                            let lod_i32 = self.coerce_to_ty(raw_lod_id, lod_ty, i32_ty);
                            let texel_vec_ty = self.emitter.emit_type_vector(desc.sampled_ty, 4);
                            let texel_vec_id = self.emitter.alloc_id();
                            self.emitter.emit_fn_inst(
                                spirv::Op::ImageFetch,
                                &[
                                    texel_vec_ty,
                                    texel_vec_id,
                                    img_id,
                                    coord_vec_id,
                                    spirv::ImageOperands::LOD.bits(),
                                    lod_i32,
                                ],
                            );
                            self.finalize_texel_result(id, texel_vec_id, desc.sampled_ty)
                        }
                        DescriptorKind::SampledImage | DescriptorKind::CombinedImageSampler => {
                            if let Some(sampled_img_id) =
                                self.load_sampled_image(&desc, image_inst, preloaded_img, sampler_descriptor_index)
                            {
                                let coord_vec_id = self.emit_coord_vec(
                                    raw_coord_id,
                                    coord_ty,
                                    f32_ty,
                                    desc.coord_components,
                                );
                                let lod_f32 = self.coerce_to_ty(raw_lod_id, lod_ty, f32_ty);
                                let texel_vec_ty =
                                    self.emitter.emit_type_vector(desc.sampled_ty, 4);
                                let texel_vec_id = self.emitter.alloc_id();
                                self.emitter.emit_fn_inst(
                                    spirv::Op::ImageSampleExplicitLod,
                                    &[
                                        texel_vec_ty,
                                        texel_vec_id,
                                        sampled_img_id,
                                        coord_vec_id,
                                        spirv::ImageOperands::LOD.bits(),
                                        lod_f32,
                                    ],
                                );
                                self.finalize_texel_result(id, texel_vec_id, desc.sampled_ty)
                            } else {
                                (
                                    self.emitter.emit_constant_32(f32_ty, 0.0_f32.to_bits()),
                                    f32_ty,
                                )
                            }
                        }
                        DescriptorKind::StorageImage
                            if desc.access != DescriptorAccess::WriteOnly =>
                        {
                            let img_id = if let Some(loaded) = preloaded_img {
                                loaded
                            } else {
                                let loaded = self.emitter.alloc_id();
                                self.emitter.emit_fn_inst(
                                    spirv::Op::Load,
                                    &[desc.element_ty, loaded, desc.var_id],
                                );
                                loaded
                            };
                            let coord_vec_id = self.emit_coord_vec(
                                raw_coord_id,
                                coord_ty,
                                i32_ty,
                                desc.coord_components,
                            );
                            let texel_vec_ty = self.emitter.emit_type_vector(desc.sampled_ty, 4);
                            let texel_vec_id = self.emitter.alloc_id();
                            self.emitter.emit_fn_inst(
                                spirv::Op::ImageRead,
                                &[texel_vec_ty, texel_vec_id, img_id, coord_vec_id],
                            );
                            self.finalize_texel_result(id, texel_vec_id, desc.sampled_ty)
                        }
                        _ => (
                            self.emitter.emit_constant_32(f32_ty, 0.0_f32.to_bits()),
                            f32_ty,
                        ),
                    }
                } else {
                    (
                        self.emitter.emit_constant_32(f32_ty, 0.0_f32.to_bits()),
                        f32_ty,
                    )
                }
            }
            InstData::ImageSampleImplicitLod {
                descriptor_index,
                sampler_descriptor_index,
                image_inst,
                coord,
            } => {
                let preloaded_img = image_inst.map(|img_inst| self.ensure_inst(img_inst));
                let raw_coord_id = self.ensure_inst(coord);
                let f32_ty = self.f32_ty();
                let coord_ty = self.inst_types.get(&coord).copied().unwrap_or(f32_ty);
                if let Some(desc) = self.descriptors.get(descriptor_index as usize).cloned()
                    && let Some(sampled_img_id) =
                        self.load_sampled_image(&desc, image_inst, preloaded_img, sampler_descriptor_index)
                {
                    let coord_vec_id =
                        self.emit_coord_vec(raw_coord_id, coord_ty, f32_ty, desc.coord_components);
                    let texel_vec_ty = self.emitter.emit_type_vector(desc.sampled_ty, 4);
                    let texel_vec_id = self.emitter.alloc_id();
                    self.emitter.emit_fn_inst(
                        spirv::Op::ImageSampleImplicitLod,
                        &[texel_vec_ty, texel_vec_id, sampled_img_id, coord_vec_id],
                    );
                    self.finalize_texel_result(id, texel_vec_id, desc.sampled_ty)
                } else {
                    (
                        self.emitter.emit_constant_32(f32_ty, 0.0_f32.to_bits()),
                        f32_ty,
                    )
                }
            }
            InstData::ImageSampleGrad {
                descriptor_index,
                sampler_descriptor_index,
                image_inst,
                coord,
                ddx,
                ddy,
            } => {
                let preloaded_img = image_inst.map(|img_inst| self.ensure_inst(img_inst));
                let raw_coord_id = self.ensure_inst(coord);
                let raw_ddx = self.ensure_inst(ddx);
                let raw_ddy = self.ensure_inst(ddy);
                let f32_ty = self.f32_ty();
                let coord_ty = self.inst_types.get(&coord).copied().unwrap_or(f32_ty);
                let ddx_ty = self.inst_types.get(&ddx).copied().unwrap_or(f32_ty);
                let ddy_ty = self.inst_types.get(&ddy).copied().unwrap_or(f32_ty);
                if let Some(desc) = self.descriptors.get(descriptor_index as usize).cloned()
                    && let Some(sampled_img_id) =
                        self.load_sampled_image(&desc, image_inst, preloaded_img, sampler_descriptor_index)
                {
                    let coord_vec_id =
                        self.emit_coord_vec(raw_coord_id, coord_ty, f32_ty, desc.coord_components);
                    let ddx_vec =
                        self.emit_coord_vec(raw_ddx, ddx_ty, f32_ty, desc.grad_components);
                    let ddy_vec =
                        self.emit_coord_vec(raw_ddy, ddy_ty, f32_ty, desc.grad_components);
                    let texel_vec_ty = self.emitter.emit_type_vector(desc.sampled_ty, 4);
                    let texel_vec_id = self.emitter.alloc_id();
                    self.emitter.emit_fn_inst(
                        spirv::Op::ImageSampleExplicitLod,
                        &[
                            texel_vec_ty,
                            texel_vec_id,
                            sampled_img_id,
                            coord_vec_id,
                            spirv::ImageOperands::GRAD.bits(),
                            ddx_vec,
                            ddy_vec,
                        ],
                    );
                    self.finalize_texel_result(id, texel_vec_id, desc.sampled_ty)
                } else {
                    (
                        self.emitter.emit_constant_32(f32_ty, 0.0_f32.to_bits()),
                        f32_ty,
                    )
                }
            }
            InstData::ImageSampleDrefExplicitLod {
                descriptor_index,
                sampler_descriptor_index,
                image_inst,
                coord,
                dref,
                lod,
            } => {
                let preloaded_img = image_inst.map(|img_inst| self.ensure_inst(img_inst));
                let raw_coord_id = self.ensure_inst(coord);
                let raw_dref = self.ensure_inst(dref);
                let raw_lod = self.ensure_inst(lod);
                let f32_ty = self.f32_ty();
                let coord_ty = self.inst_types.get(&coord).copied().unwrap_or(f32_ty);
                let dref_ty = self.inst_types.get(&dref).copied().unwrap_or(f32_ty);
                let lod_ty = self.inst_types.get(&lod).copied().unwrap_or(f32_ty);
                if let Some(desc) = self.descriptors.get(descriptor_index as usize).cloned()
                    && let Some(sampled_img_id) =
                        self.load_sampled_image(&desc, image_inst, preloaded_img, sampler_descriptor_index)
                {
                    let coord_vec_id =
                        self.emit_coord_vec(raw_coord_id, coord_ty, f32_ty, desc.coord_components);
                    let dref_f32 = self.coerce_to_ty(raw_dref, dref_ty, f32_ty);
                    let lod_f32 = self.coerce_to_ty(raw_lod, lod_ty, f32_ty);
                    let res = self.emitter.alloc_id();
                    self.emitter.emit_fn_inst(
                        spirv::Op::ImageSampleDrefExplicitLod,
                        &[
                            f32_ty,
                            res,
                            sampled_img_id,
                            coord_vec_id,
                            dref_f32,
                            spirv::ImageOperands::LOD.bits(),
                            lod_f32,
                        ],
                    );
                    (res, f32_ty)
                } else {
                    (
                        self.emitter.emit_constant_32(f32_ty, 0.0_f32.to_bits()),
                        f32_ty,
                    )
                }
            }
            InstData::ImageGather {
                descriptor_index,
                sampler_descriptor_index,
                image_inst,
                coord,
                component,
            } => {
                let preloaded_img = image_inst.map(|img_inst| self.ensure_inst(img_inst));
                let raw_coord_id = self.ensure_inst(coord);
                let f32_ty = self.f32_ty();
                let u32_ty = self.u32_ty();
                let coord_ty = self.inst_types.get(&coord).copied().unwrap_or(f32_ty);
                if let Some(desc) = self.descriptors.get(descriptor_index as usize).cloned()
                    && let Some(sampled_img_id) =
                        self.load_sampled_image(&desc, image_inst, preloaded_img, sampler_descriptor_index)
                {
                    let coord_vec_id =
                        self.emit_coord_vec(raw_coord_id, coord_ty, f32_ty, desc.coord_components);
                    let comp_const = self.emitter.emit_constant_32(u32_ty, component.min(3));
                    let texel_vec_ty = self.emitter.emit_type_vector(desc.sampled_ty, 4);
                    let texel_vec_id = self.emitter.alloc_id();
                    self.emitter.emit_fn_inst(
                        spirv::Op::ImageGather,
                        &[
                            texel_vec_ty,
                            texel_vec_id,
                            sampled_img_id,
                            coord_vec_id,
                            comp_const,
                        ],
                    );
                    self.finalize_texel_result(id, texel_vec_id, desc.sampled_ty)
                } else {
                    (
                        self.emitter.emit_constant_32(f32_ty, 0.0_f32.to_bits()),
                        f32_ty,
                    )
                }
            }
            InstData::ImageQuerySize {
                descriptor_index,
                lod,
            } => {
                self.emitter.emit_capability(spirv::Capability::ImageQuery);
                let raw_lod = lod.map(|l| self.ensure_inst(l));
                let u32_ty = self.u32_ty();
                let i32_ty = self.i32_ty();
                if let Some(desc) = self.descriptors.get(descriptor_index as usize).cloned()
                    && let Some(img_id) = self.load_raw_image(&desc)
                {
                    let elem_ty = if self.func.insts[id].ir_ty.is_signed_int() {
                        i32_ty
                    } else {
                        u32_ty
                    };
                    let res_ty = self.emitter.emit_type_vector(elem_ty, desc.size_components);
                    let res = self.emitter.alloc_id();
                    if desc.kind == DescriptorKind::StorageImage {
                        self.emitter
                            .emit_fn_inst(spirv::Op::ImageQuerySize, &[res_ty, res, img_id]);
                    } else {
                        let lod_id = if let (Some(l_inst), Some(l_raw)) = (lod, raw_lod) {
                            let l_ty = self.inst_types.get(&l_inst).copied().unwrap_or(i32_ty);
                            self.coerce_to_ty(l_raw, l_ty, i32_ty)
                        } else {
                            self.emitter.emit_constant_32(i32_ty, 0)
                        };
                        self.emitter.emit_fn_inst(
                            spirv::Op::ImageQuerySizeLod,
                            &[res_ty, res, img_id, lod_id],
                        );
                    }
                    (res, res_ty)
                } else {
                    let v2 = self.emitter.emit_type_vector(u32_ty, 2);
                    (self.emit_zero_const(v2), v2)
                }
            }
            InstData::ImageQueryLevels { descriptor_index } => {
                self.emitter.emit_capability(spirv::Capability::ImageQuery);
                let u32_ty = self.u32_ty();
                if let Some(desc) = self.descriptors.get(descriptor_index as usize).cloned()
                    && let Some(img_id) = self.load_raw_image(&desc)
                {
                    let res = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::ImageQueryLevels, &[u32_ty, res, img_id]);
                    (res, u32_ty)
                } else {
                    (self.emitter.emit_constant_32(u32_ty, 1), u32_ty)
                }
            }
            InstData::ImageTexelPointer {
                descriptor_index,
                coord,
                sample,
            } => {
                let raw_coord = self.ensure_inst(coord);
                let raw_sample = self.ensure_inst(sample);
                let i32_ty = self.i32_ty();
                if let Some(desc) = self.descriptors.get(descriptor_index as usize).cloned()
                    && desc.kind == DescriptorKind::StorageImage
                    && !desc.is_arrayed_descriptor()
                {
                    let coord_ty = self.inst_types.get(&coord).copied().unwrap_or(i32_ty);
                    let coord_vec =
                        self.emit_coord_vec(raw_coord, coord_ty, i32_ty, desc.coord_components);
                    let sample_ty = self.inst_types.get(&sample).copied().unwrap_or(i32_ty);
                    let sample_i32 = self.coerce_to_ty(raw_sample, sample_ty, i32_ty);
                    let texel_ptr_ty = self
                        .emitter
                        .emit_type_pointer(spirv::StorageClass::Image, desc.sampled_ty);
                    let res = self.emitter.alloc_id();
                    self.emitter.emit_fn_inst(
                        spirv::Op::ImageTexelPointer,
                        &[texel_ptr_ty, res, desc.var_id, coord_vec, sample_i32],
                    );
                    self.inst_pointee_types.insert(id, desc.sampled_ty);
                    self.inst_ptr_storage.insert(id, spirv::StorageClass::Image);
                    (res, texel_ptr_ty)
                } else {
                    (0, self.void_ty)
                }
            }
            InstData::ImageStore {
                descriptor_index,
                coord,
                value,
            } => {
                let raw_coord_id = self.ensure_inst(coord);
                let raw_val_id = self.ensure_inst(value);
                if let Some(desc) = self.descriptors.get(descriptor_index as usize).cloned()
                    && desc.kind == DescriptorKind::StorageImage
                    && !desc.is_arrayed_descriptor()
                {
                    let img_id = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::Load, &[desc.element_ty, img_id, desc.var_id]);

                    let i32_ty = self.i32_ty();
                    let u32_ty = self.u32_ty();
                    let coord_ty = self.inst_types.get(&coord).copied().unwrap_or(i32_ty);
                    let coord_target_ty = if coord_ty == u32_ty { u32_ty } else { i32_ty };
                    let coord_vec_id = self.emit_coord_vec(
                        raw_coord_id,
                        coord_ty,
                        coord_target_ty,
                        desc.coord_components,
                    );

                    let val_ty = self.inst_types.get(&value).copied().unwrap_or(u32_ty);
                    let texel_vec_ty = self.emitter.emit_type_vector(desc.sampled_ty, 4);
                    let texel_vec_id = if val_ty == texel_vec_ty {
                        raw_val_id
                    } else if self.vector_component_info(val_ty).is_some() {
                        self.coerce_to_ty(raw_val_id, val_ty, texel_vec_ty)
                    } else {
                        let texel_scalar_id =
                            self.coerce_to_ty(raw_val_id, val_ty, desc.sampled_ty);
                        let splat_id = self.emitter.alloc_id();
                        self.emitter.emit_fn_inst(
                            spirv::Op::CompositeConstruct,
                            &[
                                texel_vec_ty,
                                splat_id,
                                texel_scalar_id,
                                texel_scalar_id,
                                texel_scalar_id,
                                texel_scalar_id,
                            ],
                        );
                        splat_id
                    };

                    self.emitter
                        .emit_fn_inst(spirv::Op::ImageWrite, &[img_id, coord_vec_id, texel_vec_id]);
                }
                (0, self.void_ty)
            }
            InstData::WorkgroupAddr { var_index, index } => {
                if let Some(wg) = self.workgroup_vars.get(var_index as usize).cloned() {
                    let wants_whole_array = index.is_none()
                        && wg.is_array
                        && matches!(
                            self.func.insts[id].ir_ty,
                            IrType::Pointer { ref pointee, .. }
                                if matches!(pointee.as_ref(), IrType::Array { .. })
                        );
                    if wants_whole_array {
                        self.inst_pointee_types
                            .insert(id, self.emitter.emit_ir_type(&wg.var_ir_ty));
                        self.inst_ptr_storage
                            .insert(id, spirv::StorageClass::Workgroup);
                        self.inst_ptr_ir_pointee.insert(id, wg.var_ir_ty);
                        is_workgroup_ptr = true;
                        is_global_var = true;
                        (wg.var_id, wg.ptr_ty)
                    } else {
                        let ptr_id = self.emit_workgroup_ptr(&wg, index);
                        let ptr_ty = self
                            .emitter
                            .emit_type_pointer(spirv::StorageClass::Workgroup, wg.elem_ty);
                        self.inst_pointee_types.insert(id, wg.elem_ty);
                        self.inst_ptr_storage
                            .insert(id, spirv::StorageClass::Workgroup);
                        self.inst_ptr_ir_pointee.insert(id, wg.elem_ir_ty);
                        is_workgroup_ptr = true;
                        is_global_var = !wg.is_array;
                        (ptr_id, ptr_ty)
                    }
                } else {
                    (0, self.void_ty)
                }
            }
            InstData::WorkgroupLoad { var_index, index } => {
                if let Some(wg) = self.workgroup_vars.get(var_index as usize).cloned() {
                    let ptr_id = self.emit_workgroup_ptr(&wg, index);
                    let res = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::Load, &[wg.elem_ty, res, ptr_id]);
                    (res, wg.elem_ty)
                } else {
                    (0, self.void_ty)
                }
            }
            InstData::WorkgroupStore {
                var_index,
                index,
                value,
            } => {
                let raw_val_id = self.ensure_inst(value);
                if let Some(wg) = self.workgroup_vars.get(var_index as usize).cloned() {
                    let ptr_id = self.emit_workgroup_ptr(&wg, index);
                    let val_ty = self.inst_types.get(&value).copied().unwrap_or(wg.elem_ty);
                    let val_id = self.coerce_to_ty(raw_val_id, val_ty, wg.elem_ty);
                    self.emitter
                        .emit_fn_inst(spirv::Op::Store, &[ptr_id, val_id]);
                }
                (0, self.void_ty)
            }
            InstData::NonUniform(inner) => {
                let inner_id = self.ensure_inst(inner);
                let ty = self
                    .inst_types
                    .get(&inner)
                    .copied()
                    .unwrap_or_else(|| self.u32_ty());
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::CopyObject, &[ty, res, inner_id]);
                (res, ty)
            }
            InstData::ResourceAccessChain {
                descriptor_index,
                index,
            } => {
                let idx_id = self.ensure_inst(index);
                if let Some(desc) = self.descriptors.get(descriptor_index as usize).cloned() {
                    if desc.kind == DescriptorKind::StorageBuffer {
                        let u32_ty = self.u32_ty();
                        let (member_idx, elem_ty, elem_ir_ty) = if let Some(layout) =
                            &desc.struct_layout
                        {
                            if let Some((m_idx, field)) = layout
                                .fields
                                .iter()
                                .enumerate()
                                .find(|(_, f)| matches!(f.ty, IrType::RuntimeArray(_)))
                            {
                                let IrType::RuntimeArray(inner) = &field.ty else {
                                    unreachable!()
                                };
                                let spv_elem = self.emitter.emit_ir_type_with_layout(inner, true);
                                (m_idx as u32, spv_elem, Some(inner.as_ref().clone()))
                            } else {
                                (0, u32_ty, None)
                            }
                        } else {
                            (0, u32_ty, None)
                        };
                        let member_const = self.emitter.emit_constant_32(u32_ty, member_idx);
                        let zero = self.emitter.emit_constant_32(u32_ty, 0);
                        let elem_ptr_ty =
                            self.emitter.emit_type_pointer(desc.storage_class, elem_ty);
                        let res = self.emitter.alloc_id();
                        if desc.is_arrayed_descriptor() {
                            self.emitter.emit_fn_inst(
                                spirv::Op::AccessChain,
                                &[elem_ptr_ty, res, desc.var_id, idx_id, member_const, zero],
                            );
                        } else {
                            self.emitter.emit_fn_inst(
                                spirv::Op::AccessChain,
                                &[elem_ptr_ty, res, desc.var_id, member_const, idx_id],
                            );
                        }
                        self.inst_pointee_types.insert(id, elem_ty);
                        self.inst_ptr_storage.insert(id, desc.storage_class);
                        if let Some(ir) = elem_ir_ty {
                            self.inst_ptr_ir_pointee.insert(id, ir);
                        }
                        (res, elem_ptr_ty)
                    } else if desc.is_arrayed_descriptor() {
                        // Both runtime arrays and fixed descriptor arrays are indexed with a
                        // single element index in `UniformConstant`.
                        let elem_ptr_ty = self
                            .emitter
                            .emit_type_pointer(desc.storage_class, desc.element_ty);
                        let res = self.emitter.alloc_id();
                        self.emitter.emit_fn_inst(
                            spirv::Op::AccessChain,
                            &[elem_ptr_ty, res, desc.var_id, idx_id],
                        );
                        self.inst_pointee_types.insert(id, desc.element_ty);
                        self.inst_ptr_storage.insert(id, desc.storage_class);
                        (res, elem_ptr_ty)
                    } else if desc.kind == DescriptorKind::UniformBuffer {
                        let u32_ty = self.u32_ty();
                        let zero = self.emitter.emit_constant_32(u32_ty, 0);
                        let first_ty = desc
                            .struct_layout
                            .as_ref()
                            .and_then(|l| l.fields.first())
                            .map_or(u32_ty, |f| {
                                self.emitter.emit_ir_type_with_layout(&f.ty, true)
                            });
                        let elem_ptr_ty =
                            self.emitter.emit_type_pointer(desc.storage_class, first_ty);
                        let res = self.emitter.alloc_id();
                        self.emitter.emit_fn_inst(
                            spirv::Op::AccessChain,
                            &[elem_ptr_ty, res, desc.var_id, zero],
                        );
                        self.inst_pointee_types.insert(id, first_ty);
                        self.inst_ptr_storage.insert(id, desc.storage_class);
                        (res, elem_ptr_ty)
                    } else {
                        // Non-arrayed single image/sampler/acceleration structure: return var_id directly.
                        self.inst_pointee_types.insert(id, desc.element_ty);
                        self.inst_ptr_storage.insert(id, desc.storage_class);
                        is_global_var = true;
                        (desc.var_id, desc.ptr_ty)
                    }
                } else {
                    (0, self.void_ty)
                }
            }
            InstData::ResourceLoad { ptr } => {
                let ptr_id = self.ensure_inst(ptr);
                if ptr_id == 0 {
                    (0, self.void_ty)
                } else {
                    let ty = self
                        .inst_pointee_types
                        .get(&ptr)
                        .copied()
                        .unwrap_or_else(|| self.u32_ty());
                    let res = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::Load, &[ty, res, ptr_id]);
                    (res, ty)
                }
            }
            InstData::AtomicAdd {
                ptr,
                value,
                scope,
                order,
            } => self.emit_atomic_op(id, AtomicOpKind::Add, ptr, value, scope, order),
            InstData::AtomicMin {
                ptr,
                value,
                scope,
                order,
            } => self.emit_atomic_op(id, AtomicOpKind::Min, ptr, value, scope, order),
            InstData::AtomicMax {
                ptr,
                value,
                scope,
                order,
            } => self.emit_atomic_op(id, AtomicOpKind::Max, ptr, value, scope, order),
            InstData::AtomicCas {
                ptr,
                expected,
                desired,
                scope,
                order,
            } => self.emit_atomic_cas(id, ptr, expected, desired, scope, order),
            InstData::WorkgroupBarrier => {
                let u32_ty = self.u32_ty();
                let scope_id = self
                    .emitter
                    .emit_constant_32(u32_ty, spirv::Scope::Workgroup as u32);
                let sem = spirv::MemorySemantics::ACQUIRE_RELEASE
                    | spirv::MemorySemantics::WORKGROUP_MEMORY;
                let sem_id = self.emitter.emit_constant_32(u32_ty, sem.bits());
                self.emitter
                    .emit_fn_inst(spirv::Op::ControlBarrier, &[scope_id, scope_id, sem_id]);
                (0, self.void_ty)
            }
            InstData::MemoryBarrier { scope, order } => {
                if self.uses_bda && scope == AtomicScope::Device {
                    self.emitter
                        .emit_capability(spirv::Capability::VulkanMemoryModelDeviceScope);
                }
                let u32_ty = self.u32_ty();
                let scope_id = self
                    .emitter
                    .emit_constant_32(u32_ty, map_atomic_scope(scope));
                let ord = match order {
                    AtomicOrder::Relaxed | AtomicOrder::AcqRel => {
                        spirv::MemorySemantics::ACQUIRE_RELEASE
                    }
                    AtomicOrder::Acquire => spirv::MemorySemantics::ACQUIRE,
                    AtomicOrder::Release => spirv::MemorySemantics::RELEASE,
                };
                let storage = match scope {
                    AtomicScope::Workgroup => spirv::MemorySemantics::WORKGROUP_MEMORY,
                    AtomicScope::Device | AtomicScope::Subgroup => {
                        spirv::MemorySemantics::UNIFORM_MEMORY
                            | spirv::MemorySemantics::WORKGROUP_MEMORY
                            | spirv::MemorySemantics::IMAGE_MEMORY
                    }
                };
                let sem_id = self
                    .emitter
                    .emit_constant_32(u32_ty, (ord | storage).bits());
                self.emitter
                    .emit_fn_inst(spirv::Op::MemoryBarrier, &[scope_id, sem_id]);
                (0, self.void_ty)
            }
            InstData::SubgroupBallot(pred) => {
                self.emitter
                    .emit_capability(spirv::Capability::GroupNonUniformBallot);
                let pred_id = self.ensure_bool(pred);
                let u32_ty = self.u32_ty();
                let uvec4_ty = self.emitter.emit_type_vector(u32_ty, 4);
                let scope_id = self
                    .emitter
                    .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
                let res = self.emitter.alloc_id();
                self.emitter.emit_fn_inst(
                    spirv::Op::GroupNonUniformBallot,
                    &[uvec4_ty, res, scope_id, pred_id],
                );
                (res, uvec4_ty)
            }
            InstData::SubgroupBallotBitCount(ballot) => {
                self.emit_subgroup_ballot_bit_count(ballot, spirv::GroupOperation::Reduce)
            }
            InstData::SubgroupBallotExclusiveBitCount(ballot) => {
                self.emit_subgroup_ballot_bit_count(ballot, spirv::GroupOperation::ExclusiveScan)
            }
            InstData::SubgroupBallotInclusiveBitCount(ballot) => {
                self.emit_subgroup_ballot_bit_count(ballot, spirv::GroupOperation::InclusiveScan)
            }
            InstData::SubgroupBroadcastFirst(val) => {
                self.emitter
                    .emit_capability(spirv::Capability::GroupNonUniformBallot);
                let raw = self.ensure_inst(val);
                let u32_ty = self.u32_ty();
                let val_ty = self.inst_types.get(&val).copied().unwrap_or(u32_ty);
                let scope = self
                    .emitter
                    .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
                let res = self.emitter.alloc_id();
                self.emitter.emit_fn_inst(
                    spirv::Op::GroupNonUniformBroadcastFirst,
                    &[val_ty, res, scope, raw],
                );
                (res, val_ty)
            }
            InstData::SubgroupAdd(val) => {
                self.emit_subgroup_arith(val, SubgroupArithKind::Add, spirv::GroupOperation::Reduce)
            }
            InstData::SubgroupExclusiveAdd(val) => self.emit_subgroup_arith(
                val,
                SubgroupArithKind::Add,
                spirv::GroupOperation::ExclusiveScan,
            ),
            InstData::SubgroupInclusiveAdd(val) => self.emit_subgroup_arith(
                val,
                SubgroupArithKind::Add,
                spirv::GroupOperation::InclusiveScan,
            ),
            InstData::SubgroupMin(val) => {
                self.emit_subgroup_arith(val, SubgroupArithKind::Min, spirv::GroupOperation::Reduce)
            }
            InstData::SubgroupMax(val) => {
                self.emit_subgroup_arith(val, SubgroupArithKind::Max, spirv::GroupOperation::Reduce)
            }
            InstData::SubgroupShuffle { value, lane } => {
                self.emitter
                    .emit_capability(spirv::Capability::GroupNonUniformShuffle);
                let raw_val = self.ensure_inst(value);
                let raw_lane = self.ensure_inst(lane);
                let u32_ty = self.u32_ty();
                let val_ty = self.inst_types.get(&value).copied().unwrap_or(u32_ty);
                let lane_ty = self.inst_types.get(&lane).copied().unwrap_or(u32_ty);
                let lane_u32 = self.coerce_to_ty(raw_lane, lane_ty, u32_ty);
                let scope = self
                    .emitter
                    .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
                let res = self.emitter.alloc_id();
                self.emitter.emit_fn_inst(
                    spirv::Op::GroupNonUniformShuffle,
                    &[val_ty, res, scope, raw_val, lane_u32],
                );
                (res, val_ty)
            }
            InstData::SubgroupShuffleXor { value, mask } => {
                self.emitter
                    .emit_capability(spirv::Capability::GroupNonUniformShuffle);
                let raw_val = self.ensure_inst(value);
                let raw_mask = self.ensure_inst(mask);
                let u32_ty = self.u32_ty();
                let val_ty = self.inst_types.get(&value).copied().unwrap_or(u32_ty);
                let mask_ty = self.inst_types.get(&mask).copied().unwrap_or(u32_ty);
                let mask_u32 = self.coerce_to_ty(raw_mask, mask_ty, u32_ty);
                let scope = self
                    .emitter
                    .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
                let res = self.emitter.alloc_id();
                self.emitter.emit_fn_inst(
                    spirv::Op::GroupNonUniformShuffleXor,
                    &[val_ty, res, scope, raw_val, mask_u32],
                );
                (res, val_ty)
            }
            InstData::SubgroupAny(pred) => {
                self.emitter
                    .emit_capability(spirv::Capability::GroupNonUniformVote);
                let pred_id = self.ensure_bool(pred);
                let u32_ty = self.u32_ty();
                let bool_ty = self.bool_ty();
                let scope = self
                    .emitter
                    .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
                let res = self.emitter.alloc_id();
                self.emitter.emit_fn_inst(
                    spirv::Op::GroupNonUniformAny,
                    &[bool_ty, res, scope, pred_id],
                );
                (res, bool_ty)
            }
            InstData::SubgroupAll(pred) => {
                self.emitter
                    .emit_capability(spirv::Capability::GroupNonUniformVote);
                let pred_id = self.ensure_bool(pred);
                let u32_ty = self.u32_ty();
                let bool_ty = self.bool_ty();
                let scope = self
                    .emitter
                    .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
                let res = self.emitter.alloc_id();
                self.emitter.emit_fn_inst(
                    spirv::Op::GroupNonUniformAll,
                    &[bool_ty, res, scope, pred_id],
                );
                (res, bool_ty)
            }
            InstData::SubgroupElect => {
                self.emitter
                    .emit_capability(spirv::Capability::GroupNonUniform);
                let u32_ty = self.u32_ty();
                let bool_ty = self.bool_ty();
                let scope = self
                    .emitter
                    .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::GroupNonUniformElect, &[bool_ty, res, scope]);
                (res, bool_ty)
            }
            InstData::QuadSwapHorizontal(val) => self.emit_quad_swap(val, 0),
            InstData::QuadSwapVertical(val) => self.emit_quad_swap(val, 1),
            InstData::SubgroupBroadcast { value, lane } => {
                self.emitter
                    .emit_capability(spirv::Capability::GroupNonUniformBallot);
                let raw_val = self.ensure_inst(value);
                let raw_lane = self.ensure_inst(lane);
                let u32_ty = self.u32_ty();
                let val_ty = self.inst_types.get(&value).copied().unwrap_or(u32_ty);
                let lane_ty = self.inst_types.get(&lane).copied().unwrap_or(u32_ty);
                let lane_u32 = self.coerce_to_ty(raw_lane, lane_ty, u32_ty);
                let scope = self
                    .emitter
                    .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst_raw(337, &[val_ty, res, scope, raw_val, lane_u32]); // OpGroupNonUniformBroadcast
                (res, val_ty)
            }
            InstData::SubgroupBallotBitExtract { ballot, index } => {
                self.emitter
                    .emit_capability(spirv::Capability::GroupNonUniformBallot);
                let raw_ballot = self.ensure_inst(ballot);
                let raw_idx = self.ensure_inst(index);
                let u32_ty = self.u32_ty();
                let bool_ty = self.bool_ty();
                let idx_ty = self.inst_types.get(&index).copied().unwrap_or(u32_ty);
                let idx_u32 = self.coerce_to_ty(raw_idx, idx_ty, u32_ty);
                let scope = self
                    .emitter
                    .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst_raw(341, &[bool_ty, res, scope, raw_ballot, idx_u32]); // OpGroupNonUniformBallotBitExtract
                (res, bool_ty)
            }
            InstData::SubgroupBallotFindLsb(ballot) => {
                self.emitter
                    .emit_capability(spirv::Capability::GroupNonUniformBallot);
                let raw_ballot = self.ensure_inst(ballot);
                let u32_ty = self.u32_ty();
                let scope = self
                    .emitter
                    .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst_raw(343, &[u32_ty, res, scope, raw_ballot]); // OpGroupNonUniformBallotFindLSB
                (res, u32_ty)
            }
            InstData::SubgroupBallotFindMsb(ballot) => {
                self.emitter
                    .emit_capability(spirv::Capability::GroupNonUniformBallot);
                let raw_ballot = self.ensure_inst(ballot);
                let u32_ty = self.u32_ty();
                let scope = self
                    .emitter
                    .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst_raw(344, &[u32_ty, res, scope, raw_ballot]); // OpGroupNonUniformBallotFindMSB
                (res, u32_ty)
            }
            InstData::SubgroupShuffleUp { value, delta } => {
                self.emitter.emit_capability_raw(66); // GroupNonUniformShuffleRelative
                let raw_val = self.ensure_inst(value);
                let raw_delta = self.ensure_inst(delta);
                let u32_ty = self.u32_ty();
                let val_ty = self.inst_types.get(&value).copied().unwrap_or(u32_ty);
                let delta_ty = self.inst_types.get(&delta).copied().unwrap_or(u32_ty);
                let delta_u32 = self.coerce_to_ty(raw_delta, delta_ty, u32_ty);
                let scope = self
                    .emitter
                    .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst_raw(347, &[val_ty, res, scope, raw_val, delta_u32]); // OpGroupNonUniformShuffleUp
                (res, val_ty)
            }
            InstData::SubgroupShuffleDown { value, delta } => {
                self.emitter.emit_capability_raw(66); // GroupNonUniformShuffleRelative
                let raw_val = self.ensure_inst(value);
                let raw_delta = self.ensure_inst(delta);
                let u32_ty = self.u32_ty();
                let val_ty = self.inst_types.get(&value).copied().unwrap_or(u32_ty);
                let delta_ty = self.inst_types.get(&delta).copied().unwrap_or(u32_ty);
                let delta_u32 = self.coerce_to_ty(raw_delta, delta_ty, u32_ty);
                let scope = self
                    .emitter
                    .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst_raw(348, &[val_ty, res, scope, raw_val, delta_u32]); // OpGroupNonUniformShuffleDown
                (res, val_ty)
            }
            InstData::SubgroupMul(val) => {
                self.emit_subgroup_arith(val, SubgroupArithKind::Mul, spirv::GroupOperation::Reduce)
            }
            InstData::SubgroupExclusiveMul(val) => self.emit_subgroup_arith(
                val,
                SubgroupArithKind::Mul,
                spirv::GroupOperation::ExclusiveScan,
            ),
            InstData::SubgroupInclusiveMul(val) => self.emit_subgroup_arith(
                val,
                SubgroupArithKind::Mul,
                spirv::GroupOperation::InclusiveScan,
            ),
            InstData::SubgroupExclusiveMin(val) => self.emit_subgroup_arith(
                val,
                SubgroupArithKind::Min,
                spirv::GroupOperation::ExclusiveScan,
            ),
            InstData::SubgroupInclusiveMin(val) => self.emit_subgroup_arith(
                val,
                SubgroupArithKind::Min,
                spirv::GroupOperation::InclusiveScan,
            ),
            InstData::SubgroupExclusiveMax(val) => self.emit_subgroup_arith(
                val,
                SubgroupArithKind::Max,
                spirv::GroupOperation::ExclusiveScan,
            ),
            InstData::SubgroupInclusiveMax(val) => self.emit_subgroup_arith(
                val,
                SubgroupArithKind::Max,
                spirv::GroupOperation::InclusiveScan,
            ),
            InstData::SubgroupBitAnd(val) => {
                self.emit_subgroup_bitwise(val, 359, spirv::GroupOperation::Reduce, None)
            }
            InstData::SubgroupBitOr(val) => {
                self.emit_subgroup_bitwise(val, 360, spirv::GroupOperation::Reduce, None)
            }
            InstData::SubgroupBitXor(val) => {
                self.emit_subgroup_bitwise(val, 361, spirv::GroupOperation::Reduce, None)
            }
            InstData::SubgroupLogicalAnd(val) => self.emit_subgroup_logical(val, 362),
            InstData::SubgroupLogicalOr(val) => self.emit_subgroup_logical(val, 363),
            InstData::SubgroupLogicalXor(val) => self.emit_subgroup_logical(val, 364),
            InstData::SubgroupClustered {
                op,
                value,
                cluster_size,
            } => {
                self.emitter.emit_capability_raw(67); // GroupNonUniformClustered
                let raw_cs = self.ensure_inst(cluster_size);
                let u32_ty = self.u32_ty();
                let cs_ty = self.inst_types.get(&cluster_size).copied().unwrap_or(u32_ty);
                let cs_u32 = self.coerce_to_ty(raw_cs, cs_ty, u32_ty);
                match op {
                    SubgroupClusteredOp::Add => self.emit_subgroup_arith_clustered(
                        value,
                        SubgroupArithKind::Add,
                        cs_u32,
                    ),
                    SubgroupClusteredOp::Mul => self.emit_subgroup_arith_clustered(
                        value,
                        SubgroupArithKind::Mul,
                        cs_u32,
                    ),
                    SubgroupClusteredOp::Min => self.emit_subgroup_arith_clustered(
                        value,
                        SubgroupArithKind::Min,
                        cs_u32,
                    ),
                    SubgroupClusteredOp::Max => self.emit_subgroup_arith_clustered(
                        value,
                        SubgroupArithKind::Max,
                        cs_u32,
                    ),
                    SubgroupClusteredOp::BitAnd => self.emit_subgroup_bitwise(
                        value,
                        359,
                        spirv::GroupOperation::ClusteredReduce,
                        Some(cs_u32),
                    ),
                    SubgroupClusteredOp::BitOr => self.emit_subgroup_bitwise(
                        value,
                        360,
                        spirv::GroupOperation::ClusteredReduce,
                        Some(cs_u32),
                    ),
                    SubgroupClusteredOp::BitXor => self.emit_subgroup_bitwise(
                        value,
                        361,
                        spirv::GroupOperation::ClusteredReduce,
                        Some(cs_u32),
                    ),
                }
            }
            InstData::QuadSwapDiagonal(val) => self.emit_quad_swap(val, 2),
            InstData::QuadBroadcast { value, lane } => {
                self.emitter
                    .emit_capability(spirv::Capability::GroupNonUniformQuad);
                let raw_val = self.ensure_inst(value);
                let raw_lane = self.ensure_inst(lane);
                let u32_ty = self.u32_ty();
                let val_ty = self.inst_types.get(&value).copied().unwrap_or(u32_ty);
                let lane_ty = self.inst_types.get(&lane).copied().unwrap_or(u32_ty);
                let lane_u32 = self.coerce_to_ty(raw_lane, lane_ty, u32_ty);
                let scope = self
                    .emitter
                    .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst_raw(365, &[val_ty, res, scope, raw_val, lane_u32]); // OpGroupNonUniformQuadBroadcast
                (res, val_ty)
            }
            InstData::BitFieldExtract {
                base,
                offset,
                count,
                signed,
            } => {
                let raw_base = self.ensure_inst(base);
                let raw_off = self.ensure_inst(offset);
                let raw_cnt = self.ensure_inst(count);
                let i32_ty = self.i32_ty();
                let u32_ty = self.u32_ty();
                let base_ty = self.inst_types.get(&base).copied().unwrap_or(if signed {
                    i32_ty
                } else {
                    u32_ty
                });
                let off_ty = self.inst_types.get(&offset).copied().unwrap_or(u32_ty);
                let cnt_ty = self.inst_types.get(&count).copied().unwrap_or(u32_ty);
                let off_id = self.coerce_to_ty(raw_off, off_ty, u32_ty);
                let cnt_id = self.coerce_to_ty(raw_cnt, cnt_ty, u32_ty);
                let op = if signed {
                    spirv::Op::BitFieldSExtract
                } else {
                    spirv::Op::BitFieldUExtract
                };
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(op, &[base_ty, res, raw_base, off_id, cnt_id]);
                (res, base_ty)
            }
            InstData::BitFieldInsert {
                base,
                insert,
                offset,
                count,
            } => {
                let raw_base = self.ensure_inst(base);
                let raw_ins = self.ensure_inst(insert);
                let raw_off = self.ensure_inst(offset);
                let raw_cnt = self.ensure_inst(count);
                let u32_ty = self.u32_ty();
                let base_ty = self.inst_types.get(&base).copied().unwrap_or(u32_ty);
                let ins_ty = self.inst_types.get(&insert).copied().unwrap_or(base_ty);
                let ins_id = self.coerce_to_ty(raw_ins, ins_ty, base_ty);
                let off_ty = self.inst_types.get(&offset).copied().unwrap_or(u32_ty);
                let cnt_ty = self.inst_types.get(&count).copied().unwrap_or(u32_ty);
                let off_id = self.coerce_to_ty(raw_off, off_ty, u32_ty);
                let cnt_id = self.coerce_to_ty(raw_cnt, cnt_ty, u32_ty);
                let res = self.emitter.alloc_id();
                self.emitter.emit_fn_inst(
                    spirv::Op::BitFieldInsert,
                    &[base_ty, res, raw_base, ins_id, off_id, cnt_id],
                );
                (res, base_ty)
            }
            InstData::BitCount(val) => {
                let raw = self.ensure_inst(val);
                let u32_ty = self.u32_ty();
                let src_ty = self.inst_types.get(&val).copied().unwrap_or(u32_ty);
                let ty = if self.is_signed_int_spv_ty(src_ty) || self.is_unsigned_int_spv_ty(src_ty)
                {
                    src_ty
                } else {
                    u32_ty
                };
                let v = self.coerce_to_ty(raw, src_ty, ty);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::BitCount, &[ty, res, v]);
                (res, ty)
            }
            InstData::Bitcast { value, target } => {
                let raw = self.ensure_inst(value);
                let dst_ty = self.emitter.emit_ir_type(&target);
                let src_ty = self.inst_types.get(&value).copied().unwrap_or(dst_ty);
                if src_ty == dst_ty {
                    (raw, dst_ty)
                } else {
                    let res = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::Bitcast, &[dst_ty, res, raw]);
                    (res, dst_ty)
                }
            }
            InstData::IsNan(val) => {
                let raw = self.ensure_inst(val);
                let f32_ty = self.f32_ty();
                let bool_ty = self.bool_ty();
                let src_ty = self.inst_types.get(&val).copied().unwrap_or(f32_ty);
                let res_ty = if let Some((_, len)) = self.vector_component_info(src_ty) {
                    self.emitter.emit_type_vector(bool_ty, len)
                } else {
                    bool_ty
                };
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::IsNan, &[res_ty, res, raw]);
                (res, res_ty)
            }
            InstData::IsInf(val) => {
                let raw = self.ensure_inst(val);
                let f32_ty = self.f32_ty();
                let bool_ty = self.bool_ty();
                let src_ty = self.inst_types.get(&val).copied().unwrap_or(f32_ty);
                let res_ty = if let Some((_, len)) = self.vector_component_info(src_ty) {
                    self.emitter.emit_type_vector(bool_ty, len)
                } else {
                    bool_ty
                };
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::IsInf, &[res_ty, res, raw]);
                (res, res_ty)
            }
            InstData::DPdx(val) => self.emit_derivative_op(spirv::Op::DPdx, val),
            InstData::DPdy(val) => self.emit_derivative_op(spirv::Op::DPdy, val),
            InstData::Fwidth(val) => self.emit_derivative_op(spirv::Op::Fwidth, val),
            InstData::DescriptorLoad {
                descriptor_index,
                word_index,
                scalar,
            } => {
                let u32_ty = self.u32_ty();
                let target_ty = self.emitter.emit_ir_type(&IrType::Scalar(scalar));
                if let Some(desc) = self.descriptors.get(descriptor_index as usize).cloned()
                    && !desc.is_arrayed_descriptor()
                {
                    if let Some(layout) = &desc.struct_layout {
                        let byte_off = word_index.saturating_mul(4);
                        if let Some((member_idx, field)) =
                            layout.fields.iter().enumerate().find(|(_, f)| {
                                byte_off >= f.byte_offset
                                    && byte_off < f.byte_offset.saturating_add(f.byte_size)
                            })
                        {
                            let mut chain_indices =
                                vec![self.emitter.emit_constant_32(u32_ty, member_idx as u32)];
                            let mut cur_ir_ty = field.ty.clone();
                            let mut rel_off = byte_off.saturating_sub(field.byte_offset);
                            loop {
                                match cur_ir_ty.clone() {
                                    IrType::Struct(sub_layout) => {
                                        if let Some((sub_idx, sub_field)) =
                                            sub_layout.fields.iter().enumerate().find(|(_, f)| {
                                                rel_off >= f.byte_offset
                                                    && rel_off
                                                        < f.byte_offset.saturating_add(f.byte_size)
                                            })
                                        {
                                            chain_indices.push(
                                                self.emitter
                                                    .emit_constant_32(u32_ty, sub_idx as u32),
                                            );
                                            rel_off = rel_off.saturating_sub(sub_field.byte_offset);
                                            cur_ir_ty = sub_field.ty.clone();
                                        } else {
                                            break;
                                        }
                                    }
                                    IrType::Array { element, len } => {
                                        let elem_stride = ir_type_stride(&element).max(1);
                                        let elem_idx =
                                            (rel_off / elem_stride).min(len.saturating_sub(1));
                                        chain_indices
                                            .push(self.emitter.emit_constant_32(u32_ty, elem_idx));
                                        rel_off %= elem_stride;
                                        cur_ir_ty = *element;
                                    }
                                    IrType::RuntimeArray(element) => {
                                        let elem_stride = ir_type_stride(&element).max(1);
                                        let elem_idx = rel_off / elem_stride;
                                        chain_indices
                                            .push(self.emitter.emit_constant_32(u32_ty, elem_idx));
                                        rel_off %= elem_stride;
                                        cur_ir_ty = *element;
                                    }
                                    IrType::Matrix { scalar, cols, rows } => {
                                        let col_elems = if rows == 3 { 4 } else { u32::from(rows) };
                                        let col_stride =
                                            scalar.byte_size().saturating_mul(col_elems).max(1);
                                        let col_idx = (rel_off / col_stride)
                                            .min(u32::from(cols.saturating_sub(1)));
                                        chain_indices
                                            .push(self.emitter.emit_constant_32(u32_ty, col_idx));
                                        rel_off %= col_stride;
                                        cur_ir_ty = IrType::Vector { scalar, len: rows };
                                    }
                                    IrType::Vector { scalar, len } => {
                                        let elem_size = scalar.byte_size().max(1);
                                        let lane_idx = (rel_off / elem_size)
                                            .min(u32::from(len.saturating_sub(1)));
                                        chain_indices
                                            .push(self.emitter.emit_constant_32(u32_ty, lane_idx));
                                        cur_ir_ty = IrType::Scalar(scalar);
                                        break;
                                    }
                                    _ => break,
                                }
                            }
                            let leaf_ty = self.emitter.emit_ir_type_with_layout(&cur_ir_ty, true);
                            let ptr_ty =
                                self.emitter.emit_type_pointer(desc.storage_class, leaf_ty);
                            let ptr_id = self.emitter.alloc_id();
                            let mut ac_ops = Vec::with_capacity(3 + chain_indices.len());
                            ac_ops.push(ptr_ty);
                            ac_ops.push(ptr_id);
                            ac_ops.push(desc.var_id);
                            ac_ops.extend_from_slice(&chain_indices);
                            self.emitter.emit_fn_inst(spirv::Op::AccessChain, &ac_ops);
                            let loaded = self.emitter.alloc_id();
                            self.emitter
                                .emit_fn_inst(spirv::Op::Load, &[leaf_ty, loaded, ptr_id]);
                            let val = self.coerce_to_ty(loaded, leaf_ty, target_ty);
                            (val, target_ty)
                        } else {
                            (self.emit_zero_const(target_ty), target_ty)
                        }
                    } else {
                        let zero = self.emitter.emit_constant_32(u32_ty, 0);
                        let raw_u32 = if desc.kind == DescriptorKind::StorageBuffer {
                            let idx_const = self.emitter.emit_constant_32(u32_ty, word_index);
                            let ptr_ty = self
                                .emitter
                                .emit_type_pointer(spirv::StorageClass::StorageBuffer, u32_ty);
                            let ptr_id = self.emitter.alloc_id();
                            self.emitter.emit_fn_inst(
                                spirv::Op::AccessChain,
                                &[ptr_ty, ptr_id, desc.var_id, zero, idx_const],
                            );
                            let loaded = self.emitter.alloc_id();
                            self.emitter
                                .emit_fn_inst(spirv::Op::Load, &[u32_ty, loaded, ptr_id]);
                            loaded
                        } else if desc.kind == DescriptorKind::UniformBuffer {
                            let idx_const = self.emitter.emit_constant_32(u32_ty, word_index);
                            let ptr_ty = self
                                .emitter
                                .emit_type_pointer(spirv::StorageClass::Uniform, u32_ty);
                            let ptr_id = self.emitter.alloc_id();
                            self.emitter.emit_fn_inst(
                                spirv::Op::AccessChain,
                                &[ptr_ty, ptr_id, desc.var_id, idx_const],
                            );
                            let loaded = self.emitter.alloc_id();
                            self.emitter
                                .emit_fn_inst(spirv::Op::Load, &[u32_ty, loaded, ptr_id]);
                            loaded
                        } else {
                            self.emitter.emit_constant_32(u32_ty, 0)
                        };
                        let val = if target_ty == u32_ty {
                            raw_u32
                        } else if target_ty == self.f32_ty() || target_ty == self.i32_ty() {
                            let casted = self.emitter.alloc_id();
                            self.emitter
                                .emit_fn_inst(spirv::Op::Bitcast, &[target_ty, casted, raw_u32]);
                            casted
                        } else if (self.cached_u64_ty() == Some(target_ty)
                            || self.cached_i64_ty() == Some(target_ty))
                            && matches!(
                                desc.kind,
                                DescriptorKind::StorageBuffer | DescriptorKind::UniformBuffer
                            )
                        {
                            let u64_ty = self.u64_ty();
                            let hi_idx = self.emitter.emit_constant_32(u32_ty, word_index + 1);
                            let ptr_ty = self.emitter.emit_type_pointer(desc.storage_class, u32_ty);
                            let hi_ptr = self.emitter.alloc_id();
                            let hi_operands = if desc.kind == DescriptorKind::StorageBuffer {
                                vec![ptr_ty, hi_ptr, desc.var_id, zero, hi_idx]
                            } else {
                                vec![ptr_ty, hi_ptr, desc.var_id, hi_idx]
                            };
                            self.emitter
                                .emit_fn_inst(spirv::Op::AccessChain, &hi_operands);
                            let raw_hi = self.emitter.alloc_id();
                            self.emitter
                                .emit_fn_inst(spirv::Op::Load, &[u32_ty, raw_hi, hi_ptr]);
                            let lo_64 = self.coerce_to_ty(raw_u32, u32_ty, u64_ty);
                            let hi_64 = self.coerce_to_ty(raw_hi, u32_ty, u64_ty);
                            let shift32 = self.emitter.emit_constant_32(u32_ty, 32);
                            let hi_shifted = self.emitter.alloc_id();
                            self.emitter.emit_fn_inst(
                                spirv::Op::ShiftLeftLogical,
                                &[u64_ty, hi_shifted, hi_64, shift32],
                            );
                            let combined = self.emitter.alloc_id();
                            self.emitter.emit_fn_inst(
                                spirv::Op::BitwiseOr,
                                &[u64_ty, combined, lo_64, hi_shifted],
                            );
                            self.coerce_to_ty(combined, u64_ty, target_ty)
                        } else {
                            self.coerce_to_ty(raw_u32, u32_ty, target_ty)
                        };
                        (val, target_ty)
                    }
                } else {
                    (self.emit_zero_const(target_ty), target_ty)
                }
            }
            InstData::Phi { incoming } => self.emit_phi_operands(id, &incoming),
            InstData::BdaOffset {
                base_ptr,
                index_i64,
                stride_bytes,
                ..
            } => {
                let u64_ty = self.u64_ty();
                let i64_ty = self.i64_ty();
                let raw_base = self.ensure_inst(base_ptr);
                let base_addr = if self.bda_ptr_insts.contains(&base_ptr) {
                    let addr_id = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::ConvertPtrToU, &[u64_ty, addr_id, raw_base]);
                    addr_id
                } else {
                    let base_src_ty = self.inst_types.get(&base_ptr).copied().unwrap_or(u64_ty);
                    self.coerce_to_ty(raw_base, base_src_ty, u64_ty)
                };
                let raw_idx = self.ensure_inst(index_i64);
                let idx_src_ty = self.inst_types.get(&index_i64).copied().unwrap_or(i64_ty);
                let idx_u64 = if self.is_signed_int_spv_ty(idx_src_ty) {
                    let idx_i64 = self.coerce_to_ty(raw_idx, idx_src_ty, i64_ty);
                    let casted = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::Bitcast, &[u64_ty, casted, idx_i64]);
                    casted
                } else {
                    self.coerce_to_ty(raw_idx, idx_src_ty, u64_ty)
                };
                let stride_const = self.emitter.emit_constant_64(u64_ty, stride_bytes);
                let byte_offset = self.emitter.alloc_id();
                self.emitter.emit_fn_inst(
                    spirv::Op::IMul,
                    &[u64_ty, byte_offset, idx_u64, stride_const],
                );
                let target_addr = self.emitter.alloc_id();
                self.emitter.emit_fn_inst(
                    spirv::Op::IAdd,
                    &[u64_ty, target_addr, base_addr, byte_offset],
                );
                let (elem_ty, elem_stride, mat_col_stride) =
                    if let IrType::Pointer { pointee, .. } = &self.func.insts[id].ir_ty
                        && **pointee != IrType::Void
                    {
                        let p_clone = pointee.as_ref().clone();
                        let spv_ty = self.emitter.emit_ir_type_with_layout(&p_clone, true);
                        (
                            spv_ty,
                            ir_type_stride(&p_clone),
                            matrix_col_stride(&p_clone),
                        )
                    } else {
                        (self.u32_ty(), 4, None)
                    };
                let ptr_ty = self.bda_ptr_ty_for(elem_ty, elem_stride, mat_col_stride);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::ConvertUToPtr, &[ptr_ty, res, target_addr]);
                is_bda_ptr = true;
                self.inst_ptr_storage.insert(id, self.bda_storage_class());
                (res, ptr_ty)
            }
            InstData::LoadBuiltin { kind, component } => {
                if let Some(&bi) = self.builtins.get(&(kind, false)) {
                    let loaded = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::Load, &[bi.value_ty, loaded, bi.var_id]);
                    if let Some(vec_len) = bi.vector_len {
                        if component.is_none()
                            && matches!(self.func.insts[id].ir_ty, IrType::Vector { .. })
                        {
                            (loaded, bi.value_ty)
                        } else {
                            let comp = component.unwrap_or(0).min(vec_len.saturating_sub(1));
                            let res = self.emitter.alloc_id();
                            self.emitter.emit_fn_inst(
                                spirv::Op::CompositeExtract,
                                &[bi.scalar_ty, res, loaded, comp],
                            );
                            (res, bi.scalar_ty)
                        }
                    } else {
                        (loaded, bi.scalar_ty)
                    }
                } else {
                    let ty = match kind {
                        BuiltinKind::FragCoord | BuiltinKind::Position | BuiltinKind::FragDepth => {
                            self.f32_ty()
                        }
                        BuiltinKind::FrontFacing => self.bool_ty(),
                        _ => self.u32_ty(),
                    };
                    (self.emit_zero_const(ty), ty)
                }
            }
            InstData::LoadStageInput { var_index } => {
                if let Some(io) = self.stage_io_vars.get(var_index as usize)
                    && !io.is_output
                {
                    let res = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::Load, &[io.val_ty, res, io.var_id]);
                    (res, io.val_ty)
                } else {
                    let u32_ty = self.u32_ty();
                    (self.emit_zero_const(u32_ty), u32_ty)
                }
            }
            InstData::StoreStageOutput { var_index, value } => {
                let raw_val = self.ensure_inst(value);
                if let Some(io) = self.stage_io_vars.get(var_index as usize)
                    && io.is_output
                {
                    let val_ty = self.inst_types.get(&value).copied().unwrap_or(io.val_ty);
                    let coerced = self.coerce_to_ty(raw_val, val_ty, io.val_ty);
                    self.emitter
                        .emit_fn_inst(spirv::Op::Store, &[io.var_id, coerced]);
                }
                (0, self.void_ty)
            }
            InstData::StoreBuiltin { kind, value } => {
                let raw_val = self.ensure_inst(value);
                if let Some(&bi) = self.builtins.get(&(kind, true)) {
                    let val_ty = self.inst_types.get(&value).copied().unwrap_or(bi.value_ty);
                    let coerced = self.coerce_to_ty(raw_val, val_ty, bi.value_ty);
                    self.emitter
                        .emit_fn_inst(spirv::Op::Store, &[bi.var_id, coerced]);
                }
                (0, self.void_ty)
            }
            InstData::DemoteToHelperInvocation => {
                self.emitter
                    .emit_capability(spirv::Capability::DemoteToHelperInvocation);
                self.emitter
                    .emit_extension("SPV_EXT_demote_to_helper_invocation");
                self.emitter
                    .emit_fn_inst(spirv::Op::DemoteToHelperInvocation, &[]);
                (0, self.void_ty)
            }
            InstData::LoadPushConstant { field_index } => {
                let u32_ty = self.u32_ty();
                if let Some(pc) = self.push_constant
                    && !pc.members.is_empty()
                {
                    let clamped_idx = (field_index as usize).min(pc.members.len() - 1);
                    let (member_ty, scalar_ty, is_composite, is_f16) = pc.members[clamped_idx];
                    let member_const = self.emitter.emit_constant_32(u32_ty, clamped_idx as u32);
                    let member_ptr_ty = self
                        .emitter
                        .emit_type_pointer(spirv::StorageClass::PushConstant, member_ty);
                    let member_ptr = self.emitter.alloc_id();
                    self.emitter.emit_fn_inst(
                        spirv::Op::AccessChain,
                        &[member_ptr_ty, member_ptr, pc.var_id, member_const],
                    );
                    let loaded = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::Load, &[member_ty, loaded, member_ptr]);
                    let desired_ir_ty = self.func.insts[id].ir_ty.clone();
                    if let IrType::Matrix { scalar, cols, rows } = desired_ir_ty {
                        let desired_spv_ty = self.emitter.emit_ir_type(&desired_ir_ty);
                        if member_ty == desired_spv_ty {
                            (loaded, desired_spv_ty)
                        } else {
                            let col_ty = self
                                .emitter
                                .emit_ir_type(&IrType::Vector { scalar, len: rows });
                            let target_scalar_ty =
                                self.emitter.emit_ir_type(&IrType::Scalar(scalar));
                            let cols_u32 = u32::from(cols.clamp(2, 4));
                            let rows_u32 = u32::from(rows.clamp(2, 4));
                            let pc_bytes = self
                                .func
                                .push_constants
                                .as_ref()
                                .and_then(|b| b.fields.get(clamped_idx))
                                .map_or(cols_u32 * rows_u32 * scalar.byte_size().max(4), |f| {
                                    f.byte_size
                                });
                            let total_elems =
                                (pc_bytes / scalar.byte_size().max(1)).max(cols_u32 * rows_u32);
                            let col_stride = (total_elems / cols_u32).max(rows_u32);
                            let mut col_ids = Vec::with_capacity(cols_u32 as usize);
                            for c in 0..cols_u32 {
                                let mut row_ids = Vec::with_capacity(rows_u32 as usize);
                                for r in 0..rows_u32 {
                                    let flat_idx = c * col_stride + r;
                                    let elem_id = self.emitter.alloc_id();
                                    self.emitter.emit_fn_inst(
                                        spirv::Op::CompositeExtract,
                                        &[scalar_ty, elem_id, loaded, flat_idx],
                                    );
                                    row_ids.push(self.coerce_to_ty(
                                        elem_id,
                                        scalar_ty,
                                        target_scalar_ty,
                                    ));
                                }
                                let col_id = self.emitter.alloc_id();
                                let mut ops = Vec::with_capacity(2 + row_ids.len());
                                ops.push(col_ty);
                                ops.push(col_id);
                                ops.extend_from_slice(&row_ids);
                                self.emitter
                                    .emit_fn_inst(spirv::Op::CompositeConstruct, &ops);
                                col_ids.push(col_id);
                            }
                            let mat_id = self.emitter.alloc_id();
                            let mut ops = Vec::with_capacity(2 + col_ids.len());
                            ops.push(desired_spv_ty);
                            ops.push(mat_id);
                            ops.extend_from_slice(&col_ids);
                            self.emitter
                                .emit_fn_inst(spirv::Op::CompositeConstruct, &ops);
                            (mat_id, desired_spv_ty)
                        }
                    } else if matches!(desired_ir_ty, IrType::Vector { .. } | IrType::Array { .. })
                    {
                        let desired_spv_ty = self.emitter.emit_ir_type(&desired_ir_ty);
                        let val = self.coerce_to_ty(loaded, member_ty, desired_spv_ty);
                        (val, desired_spv_ty)
                    } else {
                        let raw_scalar = if is_composite {
                            let scalar = self.emitter.alloc_id();
                            self.emitter.emit_fn_inst(
                                spirv::Op::CompositeExtract,
                                &[scalar_ty, scalar, loaded, 0],
                            );
                            scalar
                        } else {
                            loaded
                        };
                        if is_f16 && !desired_ir_ty.is_f16() {
                            let f32_ty = self.f32_ty();
                            let widened = self.emitter.alloc_id();
                            self.emitter
                                .emit_fn_inst(spirv::Op::FConvert, &[f32_ty, widened, raw_scalar]);
                            (widened, f32_ty)
                        } else {
                            (raw_scalar, scalar_ty)
                        }
                    }
                } else {
                    (self.emitter.emit_constant_32(u32_ty, 0), u32_ty)
                }
            }
            InstData::RayQueryInitialize {
                ray_query_id,
                tlas_descriptor_index,
                flags,
                cull_mask,
                origin,
                t_min,
                dir,
                t_max,
            } => {
                let raw_flags = self.ensure_inst(flags);
                let raw_mask = self.ensure_inst(cull_mask);
                let raw_origin = self.ensure_inst(origin);
                let raw_tmin = self.ensure_inst(t_min);
                let raw_dir = self.ensure_inst(dir);
                let raw_tmax = self.ensure_inst(t_max);
                if let Some(&rq_var) = self.ray_query_vars.get(&ray_query_id)
                    && let Some(desc) = self
                        .descriptors
                        .get(tlas_descriptor_index as usize)
                        .cloned()
                    && desc.kind == DescriptorKind::AccelerationStructure
                    && !desc.is_arrayed_descriptor()
                {
                    let u32_ty = self.u32_ty();
                    let f32_ty = self.f32_ty();
                    let flags_ty = self.inst_types.get(&flags).copied().unwrap_or(u32_ty);
                    let mask_ty = self.inst_types.get(&cull_mask).copied().unwrap_or(u32_ty);
                    let origin_ty = self.inst_types.get(&origin).copied().unwrap_or(f32_ty);
                    let tmin_ty = self.inst_types.get(&t_min).copied().unwrap_or(f32_ty);
                    let dir_ty = self.inst_types.get(&dir).copied().unwrap_or(f32_ty);
                    let tmax_ty = self.inst_types.get(&t_max).copied().unwrap_or(f32_ty);

                    let flags_u32 = self.coerce_to_ty(raw_flags, flags_ty, u32_ty);
                    let mask_u32 = self.coerce_to_ty(raw_mask, mask_ty, u32_ty);
                    let origin_v3 = self.emit_vec3_f32(raw_origin, origin_ty);
                    let tmin_f32 = self.coerce_to_ty(raw_tmin, tmin_ty, f32_ty);
                    let dir_v3 = self.emit_vec3_f32(raw_dir, dir_ty);
                    let tmax_f32 = self.coerce_to_ty(raw_tmax, tmax_ty, f32_ty);

                    let tlas_val = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::Load, &[desc.element_ty, tlas_val, desc.var_id]);

                    self.emitter.emit_fn_inst(
                        spirv::Op::RayQueryInitializeKHR,
                        &[
                            rq_var, tlas_val, flags_u32, mask_u32, origin_v3, tmin_f32, dir_v3,
                            tmax_f32,
                        ],
                    );
                }
                (0, self.void_ty)
            }
            InstData::RayQueryProceed(rq_id) => {
                let bool_ty = self.bool_ty();
                if let Some(&rq_var) = self.ray_query_vars.get(&rq_id) {
                    let res = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::RayQueryProceedKHR, &[bool_ty, res, rq_var]);
                    (res, bool_ty)
                } else {
                    (self.emitter.emit_constant_bool(bool_ty, false), bool_ty)
                }
            }
            InstData::RayQueryCommittedType(rq_id) => {
                self.emit_ray_query_u32(rq_id, spirv::Op::RayQueryGetIntersectionTypeKHR, 1)
            }
            InstData::RayQueryCandidateType(rq_id) => {
                self.emit_ray_query_u32(rq_id, spirv::Op::RayQueryGetIntersectionTypeKHR, 0)
            }
            InstData::RayQueryConfirmIntersection(rq_id) => {
                if let Some(&rq_var) = self.ray_query_vars.get(&rq_id) {
                    self.emitter
                        .emit_fn_inst(spirv::Op::RayQueryConfirmIntersectionKHR, &[rq_var]);
                }
                (0, self.void_ty)
            }
            InstData::RayQueryTerminate(rq_id) => {
                if let Some(&rq_var) = self.ray_query_vars.get(&rq_id) {
                    self.emitter
                        .emit_fn_inst(spirv::Op::RayQueryTerminateKHR, &[rq_var]);
                }
                (0, self.void_ty)
            }
            InstData::RayQueryGenerateIntersection {
                ray_query_id,
                t_hit,
            } => {
                let raw_t = self.ensure_inst(t_hit);
                if let Some(&rq_var) = self.ray_query_vars.get(&ray_query_id) {
                    let f32_ty = self.f32_ty();
                    let t_ty = self.inst_types.get(&t_hit).copied().unwrap_or(f32_ty);
                    let t_f32 = self.coerce_to_ty(raw_t, t_ty, f32_ty);
                    self.emitter
                        .emit_fn_inst(spirv::Op::RayQueryGenerateIntersectionKHR, &[rq_var, t_f32]);
                }
                (0, self.void_ty)
            }
            InstData::RayQueryCommittedT(rq_id) => self.emit_ray_query_t(rq_id, 1),
            InstData::RayQueryCandidateT(rq_id) => self.emit_ray_query_t(rq_id, 0),
            InstData::RayQueryCommittedInstanceId(rq_id) => {
                self.emit_ray_query_u32(rq_id, spirv::Op::RayQueryGetIntersectionInstanceIdKHR, 1)
            }
            InstData::RayQueryCommittedInstanceCustomIndex(rq_id) => self.emit_ray_query_u32(
                rq_id,
                spirv::Op::RayQueryGetIntersectionInstanceCustomIndexKHR,
                1,
            ),
            InstData::RayQueryCommittedPrimitiveIndex(rq_id) => self.emit_ray_query_u32(
                rq_id,
                spirv::Op::RayQueryGetIntersectionPrimitiveIndexKHR,
                1,
            ),
            InstData::RayQueryCommittedGeometryIndex(rq_id) => self.emit_ray_query_u32(
                rq_id,
                spirv::Op::RayQueryGetIntersectionGeometryIndexKHR,
                1,
            ),
            InstData::RayQueryCommittedObjectToWorld(rq_id) => self
                .emit_ray_query_matrix(rq_id, spirv::Op::RayQueryGetIntersectionObjectToWorldKHR),
            InstData::RayQueryCommittedWorldToObject(rq_id) => self
                .emit_ray_query_matrix(rq_id, spirv::Op::RayQueryGetIntersectionWorldToObjectKHR),
            InstData::RayQueryCommittedFrontFace(rq_id) => {
                let bool_ty = self.bool_ty();
                if let Some(&rq_var) = self.ray_query_vars.get(&rq_id) {
                    let u32_ty = self.u32_ty();
                    let committed = self.emitter.emit_constant_32(u32_ty, 1);
                    let res = self.emitter.alloc_id();
                    self.emitter.emit_fn_inst(
                        spirv::Op::RayQueryGetIntersectionFrontFaceKHR,
                        &[bool_ty, res, rq_var, committed],
                    );
                    (res, bool_ty)
                } else {
                    (self.emitter.emit_constant_bool(bool_ty, false), bool_ty)
                }
            }
            InstData::RayQueryCommittedBarycentrics(rq_id) => {
                let f32_ty = self.f32_ty();
                let vec2_f32_ty = self.emitter.emit_type_vector(f32_ty, 2);
                if let Some(&rq_var) = self.ray_query_vars.get(&rq_id) {
                    let u32_ty = self.u32_ty();
                    let committed = self.emitter.emit_constant_32(u32_ty, 1);
                    let v2_res = self.emitter.alloc_id();
                    self.emitter.emit_fn_inst(
                        spirv::Op::RayQueryGetIntersectionBarycentricsKHR,
                        &[vec2_f32_ty, v2_res, rq_var, committed],
                    );
                    if matches!(self.func.insts[id].ir_ty, IrType::Scalar(_)) {
                        let res = self.emitter.alloc_id();
                        self.emitter
                            .emit_fn_inst(spirv::Op::CompositeExtract, &[f32_ty, res, v2_res, 0]);
                        (res, f32_ty)
                    } else {
                        (v2_res, vec2_f32_ty)
                    }
                } else {
                    (self.emit_zero_const(vec2_f32_ty), vec2_f32_ty)
                }
            }
            InstData::MakeVec2(x, y) => {
                let raw_x = self.ensure_inst(x);
                let raw_y = self.ensure_inst(y);
                let i32_ty = self.i32_ty();
                let u32_ty = self.u32_ty();
                let f32_ty = self.f32_ty();
                let x_ty = self.inst_types.get(&x).copied().unwrap_or(i32_ty);
                let y_ty = self.inst_types.get(&y).copied().unwrap_or(x_ty);
                let elem_ty = if x_ty == f32_ty || y_ty == f32_ty {
                    f32_ty
                } else if x_ty == i32_ty || x_ty == u32_ty {
                    x_ty
                } else {
                    i32_ty
                };
                let cx = self.coerce_to_ty(raw_x, x_ty, elem_ty);
                let cy = self.coerce_to_ty(raw_y, y_ty, elem_ty);
                let vec_ty = self.emitter.emit_type_vector(elem_ty, 2);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::CompositeConstruct, &[vec_ty, res, cx, cy]);
                (res, vec_ty)
            }
            InstData::MakeVec3(x, y, z) => {
                let rx = self.ensure_inst(x);
                let ry = self.ensure_inst(y);
                let rz = self.ensure_inst(z);
                let f32_ty = self.f32_ty();
                let u32_ty = self.u32_ty();
                let i32_ty = self.i32_ty();
                let tx = self.inst_types.get(&x).copied().unwrap_or(f32_ty);
                let ty = self.inst_types.get(&y).copied().unwrap_or(f32_ty);
                let tz = self.inst_types.get(&z).copied().unwrap_or(f32_ty);
                let elem_ty = if tx == f32_ty || ty == f32_ty || tz == f32_ty {
                    f32_ty
                } else if tx == i32_ty || ty == i32_ty || tz == i32_ty {
                    i32_ty
                } else {
                    u32_ty
                };
                let cx = self.coerce_to_ty(rx, tx, elem_ty);
                let cy = self.coerce_to_ty(ry, ty, elem_ty);
                let cz = self.coerce_to_ty(rz, tz, elem_ty);
                let vec_ty = self.emitter.emit_type_vector(elem_ty, 3);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::CompositeConstruct, &[vec_ty, res, cx, cy, cz]);
                (res, vec_ty)
            }
            InstData::MakeVec4(x, y, z, w) => {
                let rx = self.ensure_inst(x);
                let ry = self.ensure_inst(y);
                let rz = self.ensure_inst(z);
                let rw = self.ensure_inst(w);
                let f32_ty = self.f32_ty();
                let u32_ty = self.u32_ty();
                let i32_ty = self.i32_ty();
                let tx = self.inst_types.get(&x).copied().unwrap_or(f32_ty);
                let ty = self.inst_types.get(&y).copied().unwrap_or(f32_ty);
                let tz = self.inst_types.get(&z).copied().unwrap_or(f32_ty);
                let tw = self.inst_types.get(&w).copied().unwrap_or(f32_ty);
                let elem_ty = if tx == f32_ty || ty == f32_ty || tz == f32_ty || tw == f32_ty {
                    f32_ty
                } else if tx == i32_ty || ty == i32_ty || tz == i32_ty || tw == i32_ty {
                    i32_ty
                } else {
                    u32_ty
                };
                let cx = self.coerce_to_ty(rx, tx, elem_ty);
                let cy = self.coerce_to_ty(ry, ty, elem_ty);
                let cz = self.coerce_to_ty(rz, tz, elem_ty);
                let cw = self.coerce_to_ty(rw, tw, elem_ty);
                let vec_ty = self.emitter.emit_type_vector(elem_ty, 4);
                let res = self.emitter.alloc_id();
                self.emitter.emit_fn_inst(
                    spirv::Op::CompositeConstruct,
                    &[vec_ty, res, cx, cy, cz, cw],
                );
                (res, vec_ty)
            }
            InstData::ExtractComponent { vec, index } => {
                let raw = self.ensure_inst(vec);
                let f32_ty = self.f32_ty();
                let vec_ty = self.inst_types.get(&vec).copied().unwrap_or(f32_ty);
                if let Some((elem_ty, lanes)) = self.vector_component_info(vec_ty) {
                    let clamped = index.min(lanes - 1);
                    let res = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::CompositeExtract, &[elem_ty, res, raw, clamped]);
                    (res, elem_ty)
                } else {
                    (raw, vec_ty)
                }
            }
            InstData::TraceRayKhr {
                tlas_descriptor_index,
                ray_flags,
                cull_mask,
                sbt_offset,
                sbt_stride,
                miss_index,
                origin,
                t_min,
                direction,
                t_max,
                payload_var_index,
            } => {
                let u32_ty = self.u32_ty();
                let f32_ty = self.f32_ty();
                let raw_flags = self.ensure_inst(ray_flags);
                let flags_ty = self.inst_types.get(&ray_flags).copied().unwrap_or(u32_ty);
                let flags_u32 = self.coerce_to_ty(raw_flags, flags_ty, u32_ty);
                let raw_mask = self.ensure_inst(cull_mask);
                let mask_ty = self.inst_types.get(&cull_mask).copied().unwrap_or(u32_ty);
                let mask_u32 = self.coerce_to_ty(raw_mask, mask_ty, u32_ty);
                let raw_sbt_off = self.ensure_inst(sbt_offset);
                let sbt_off_ty = self.inst_types.get(&sbt_offset).copied().unwrap_or(u32_ty);
                let sbt_off_u32 = self.coerce_to_ty(raw_sbt_off, sbt_off_ty, u32_ty);
                let raw_sbt_stride = self.ensure_inst(sbt_stride);
                let sbt_stride_ty = self.inst_types.get(&sbt_stride).copied().unwrap_or(u32_ty);
                let sbt_stride_u32 = self.coerce_to_ty(raw_sbt_stride, sbt_stride_ty, u32_ty);
                let raw_miss = self.ensure_inst(miss_index);
                let miss_ty = self.inst_types.get(&miss_index).copied().unwrap_or(u32_ty);
                let miss_u32 = self.coerce_to_ty(raw_miss, miss_ty, u32_ty);
                let raw_origin = self.ensure_inst(origin);
                let origin_ty = self.inst_types.get(&origin).copied().unwrap_or(f32_ty);
                let origin_v3 = self.emit_vec3_f32(raw_origin, origin_ty);
                let raw_tmin = self.ensure_inst(t_min);
                let tmin_ty = self.inst_types.get(&t_min).copied().unwrap_or(f32_ty);
                let tmin_f32 = self.coerce_to_ty(raw_tmin, tmin_ty, f32_ty);
                let raw_dir = self.ensure_inst(direction);
                let dir_ty = self.inst_types.get(&direction).copied().unwrap_or(f32_ty);
                let dir_v3 = self.emit_vec3_f32(raw_dir, dir_ty);
                let raw_tmax = self.ensure_inst(t_max);
                let tmax_ty = self.inst_types.get(&t_max).copied().unwrap_or(f32_ty);
                let tmax_f32 = self.coerce_to_ty(raw_tmax, tmax_ty, f32_ty);
                if let Some(desc) = self.descriptors.get(tlas_descriptor_index as usize)
                    && desc.kind == DescriptorKind::AccelerationStructure
                    && let Some(payload) = self.stage_vars.get(payload_var_index as usize)
                {
                    let tlas_val = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::Load, &[desc.element_ty, tlas_val, desc.var_id]);
                    self.emitter.emit_fn_inst_raw(
                        4445, // OpTraceRayKHR
                        &[
                            tlas_val,
                            flags_u32,
                            mask_u32,
                            sbt_off_u32,
                            sbt_stride_u32,
                            miss_u32,
                            origin_v3,
                            tmin_f32,
                            dir_v3,
                            tmax_f32,
                            payload.var_id,
                        ],
                    );
                }
                (0, self.void_ty)
            }
            InstData::IgnoreIntersectionKhr => {
                self.emitter.emit_fn_inst_raw(4448, &[]); // OpIgnoreIntersectionKHR
                (0, self.void_ty)
            }
            InstData::TerminateRayKhr => {
                self.emitter.emit_fn_inst_raw(4449, &[]); // OpTerminateRayKHR
                (0, self.void_ty)
            }
            InstData::ReportIntersectionKhr { hit_t, hit_kind } => {
                let bool_ty = self.bool_ty();
                let f32_ty = self.f32_ty();
                let u32_ty = self.u32_ty();
                let raw_t = self.ensure_inst(hit_t);
                let t_ty = self.inst_types.get(&hit_t).copied().unwrap_or(f32_ty);
                let t_f32 = self.coerce_to_ty(raw_t, t_ty, f32_ty);
                let raw_k = self.ensure_inst(hit_kind);
                let k_ty = self.inst_types.get(&hit_kind).copied().unwrap_or(u32_ty);
                let k_u32 = self.coerce_to_ty(raw_k, k_ty, u32_ty);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst_raw(5334, &[bool_ty, res, t_f32, k_u32]); // OpReportIntersectionKHR
                (res, bool_ty)
            }
            InstData::ExecuteCallableKhr {
                sbt_index,
                callable_var_index,
            } => {
                let u32_ty = self.u32_ty();
                let raw_idx = self.ensure_inst(sbt_index);
                let idx_ty = self.inst_types.get(&sbt_index).copied().unwrap_or(u32_ty);
                let idx_u32 = self.coerce_to_ty(raw_idx, idx_ty, u32_ty);
                if let Some(cvar) = self.stage_vars.get(callable_var_index as usize) {
                    self.emitter.emit_fn_inst_raw(4446, &[idx_u32, cvar.var_id]); // OpExecuteCallableKHR
                }
                (0, self.void_ty)
            }
            InstData::EmitMeshTasksExt {
                group_x,
                group_y,
                group_z,
                payload_var_index,
            } => {
                let u32_ty = self.u32_ty();
                let rx = self.ensure_inst(group_x);
                let tx = self.inst_types.get(&group_x).copied().unwrap_or(u32_ty);
                let gx = self.coerce_to_ty(rx, tx, u32_ty);
                let ry = self.ensure_inst(group_y);
                let ty = self.inst_types.get(&group_y).copied().unwrap_or(u32_ty);
                let gy = self.coerce_to_ty(ry, ty, u32_ty);
                let rz = self.ensure_inst(group_z);
                let tz = self.inst_types.get(&group_z).copied().unwrap_or(u32_ty);
                let gz = self.coerce_to_ty(rz, tz, u32_ty);
                let mut ops = vec![gx, gy, gz];
                if let Some(pidx) = payload_var_index
                    && let Some(pvar) = self.stage_vars.get(pidx as usize)
                {
                    ops.push(pvar.var_id);
                }
                self.emitter.emit_fn_inst_raw(5294, &ops); // OpEmitMeshTasksEXT
                (0, self.void_ty)
            }
            InstData::SetMeshOutputsExt {
                vertex_count,
                primitive_count,
            } => {
                let u32_ty = self.u32_ty();
                let rv = self.ensure_inst(vertex_count);
                let tv = self.inst_types.get(&vertex_count).copied().unwrap_or(u32_ty);
                let vc = self.coerce_to_ty(rv, tv, u32_ty);
                let rp = self.ensure_inst(primitive_count);
                let tp = self.inst_types.get(&primitive_count).copied().unwrap_or(u32_ty);
                let pc = self.coerce_to_ty(rp, tp, u32_ty);
                self.emitter.emit_fn_inst_raw(5295, &[vc, pc]); // OpSetMeshOutputsEXT
                (0, self.void_ty)
            }
            InstData::EmitVertex => {
                self.emitter.emit_fn_inst_raw(218, &[]); // OpEmitVertex
                (0, self.void_ty)
            }
            InstData::EndPrimitive => {
                self.emitter.emit_fn_inst_raw(219, &[]); // OpEndPrimitive
                (0, self.void_ty)
            }
            InstData::LoadStageVar { var_index } => {
                if let Some(sv) = self.stage_vars.get(var_index as usize) {
                    let res = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::Load, &[sv.val_ty, res, sv.var_id]);
                    (res, sv.val_ty)
                } else {
                    let u32_ty = self.u32_ty();
                    (self.emit_zero_const(u32_ty), u32_ty)
                }
            }
            InstData::StoreStageVar { var_index, value } => {
                let raw_val = self.ensure_inst(value);
                if let Some(sv) = self.stage_vars.get(var_index as usize) {
                    let val_ty = self.inst_types.get(&value).copied().unwrap_or(sv.val_ty);
                    let coerced = self.coerce_to_ty(raw_val, val_ty, sv.val_ty);
                    self.emitter
                        .emit_fn_inst(spirv::Op::Store, &[sv.var_id, coerced]);
                }
                (0, self.void_ty)
            }
            InstData::HitObjectTraceRay {
                hit_object_id,
                tlas_descriptor_index,
                ray_flags,
                cull_mask,
                sbt_offset,
                sbt_stride,
                miss_index,
                origin,
                t_min,
                direction,
                t_max,
                payload_var_index,
            } => {
                let u32_ty = self.u32_ty();
                let f32_ty = self.f32_ty();
                let flags_u32 = self.coerce_to_ty(
                    self.ensure_inst(ray_flags),
                    self.inst_types.get(&ray_flags).copied().unwrap_or(u32_ty),
                    u32_ty,
                );
                let mask_u32 = self.coerce_to_ty(
                    self.ensure_inst(cull_mask),
                    self.inst_types.get(&cull_mask).copied().unwrap_or(u32_ty),
                    u32_ty,
                );
                let sbt_off_u32 = self.coerce_to_ty(
                    self.ensure_inst(sbt_offset),
                    self.inst_types.get(&sbt_offset).copied().unwrap_or(u32_ty),
                    u32_ty,
                );
                let sbt_stride_u32 = self.coerce_to_ty(
                    self.ensure_inst(sbt_stride),
                    self.inst_types.get(&sbt_stride).copied().unwrap_or(u32_ty),
                    u32_ty,
                );
                let miss_u32 = self.coerce_to_ty(
                    self.ensure_inst(miss_index),
                    self.inst_types.get(&miss_index).copied().unwrap_or(u32_ty),
                    u32_ty,
                );
                let raw_origin = self.ensure_inst(origin);
                let origin_v3 = self.emit_vec3_f32(
                    raw_origin,
                    self.inst_types.get(&origin).copied().unwrap_or(f32_ty),
                );
                let tmin_f32 = self.coerce_to_ty(
                    self.ensure_inst(t_min),
                    self.inst_types.get(&t_min).copied().unwrap_or(f32_ty),
                    f32_ty,
                );
                let raw_dir = self.ensure_inst(direction);
                let dir_v3 = self.emit_vec3_f32(
                    raw_dir,
                    self.inst_types.get(&direction).copied().unwrap_or(f32_ty),
                );
                let tmax_f32 = self.coerce_to_ty(
                    self.ensure_inst(t_max),
                    self.inst_types.get(&t_max).copied().unwrap_or(f32_ty),
                    f32_ty,
                );
                if let Some(&ho_var) = self.hit_object_vars.get(&hit_object_id)
                    && let Some(desc) = self.descriptors.get(tlas_descriptor_index as usize)
                    && desc.kind == DescriptorKind::AccelerationStructure
                    && let Some(payload) = self.stage_vars.get(payload_var_index as usize)
                {
                    let tlas_val = self.emitter.alloc_id();
                    self.emitter
                        .emit_fn_inst(spirv::Op::Load, &[desc.element_ty, tlas_val, desc.var_id]);
                    let op: u16 = if self.use_ser_ext { 5316 } else { 5260 };
                    self.emitter.emit_fn_inst_raw(
                        op,
                        &[
                            ho_var,
                            tlas_val,
                            flags_u32,
                            mask_u32,
                            sbt_off_u32,
                            sbt_stride_u32,
                            miss_u32,
                            origin_v3,
                            tmin_f32,
                            dir_v3,
                            tmax_f32,
                            payload.var_id,
                        ],
                    );
                }
                (0, self.void_ty)
            }
            InstData::HitObjectRecordEmpty(ho_id) => {
                if let Some(&ho_var) = self.hit_object_vars.get(&ho_id) {
                    let op: u16 = if self.use_ser_ext { 5318 } else { 5259 };
                    self.emitter.emit_fn_inst_raw(op, &[ho_var]);
                }
                (0, self.void_ty)
            }
            InstData::HitObjectIsHit(ho_id) => {
                let bool_ty = self.bool_ty();
                if let Some(&ho_var) = self.hit_object_vars.get(&ho_id) {
                    let res = self.emitter.alloc_id();
                    let op: u16 = if self.use_ser_ext { 5351 } else { 5277 };
                    self.emitter.emit_fn_inst_raw(op, &[bool_ty, res, ho_var]);
                    (res, bool_ty)
                } else {
                    (self.emitter.emit_constant_bool(bool_ty, false), bool_ty)
                }
            }
            InstData::HitObjectGetRayTMax(ho_id) => {
                let f32_ty = self.f32_ty();
                if let Some(&ho_var) = self.hit_object_vars.get(&ho_id) {
                    let res = self.emitter.alloc_id();
                    let op: u16 = if self.use_ser_ext { 5333 } else { 5274 };
                    self.emitter.emit_fn_inst_raw(op, &[f32_ty, res, ho_var]);
                    (res, f32_ty)
                } else {
                    (self.emitter.emit_constant_32(f32_ty, 0), f32_ty)
                }
            }
            InstData::HitObjectGetSbtRecordIndex(ho_id) => {
                let u32_ty = self.u32_ty();
                if let Some(&ho_var) = self.hit_object_vars.get(&ho_id) {
                    let res = self.emitter.alloc_id();
                    let op: u16 = if self.use_ser_ext { 5348 } else { 5258 };
                    self.emitter.emit_fn_inst_raw(op, &[u32_ty, res, ho_var]);
                    (res, u32_ty)
                } else {
                    (self.emitter.emit_constant_32(u32_ty, 0), u32_ty)
                }
            }
            InstData::HitObjectExecuteShader {
                hit_object_id,
                payload_var_index,
            } => {
                if let Some(&ho_var) = self.hit_object_vars.get(&hit_object_id)
                    && let Some(payload) = self.stage_vars.get(payload_var_index as usize)
                {
                    let op: u16 = if self.use_ser_ext { 5319 } else { 5264 };
                    self.emitter.emit_fn_inst_raw(op, &[ho_var, payload.var_id]);
                }
                (0, self.void_ty)
            }
            InstData::ReorderThreadWithHint { hint, bits } => {
                let u32_ty = self.u32_ty();
                let h = self.coerce_to_ty(
                    self.ensure_inst(hint),
                    self.inst_types.get(&hint).copied().unwrap_or(u32_ty),
                    u32_ty,
                );
                let b = self.coerce_to_ty(
                    self.ensure_inst(bits),
                    self.inst_types.get(&bits).copied().unwrap_or(u32_ty),
                    u32_ty,
                );
                let op: u16 = if self.use_ser_ext { 5314 } else { 5280 };
                self.emitter.emit_fn_inst_raw(op, &[h, b]);
                (0, self.void_ty)
            }
            InstData::ReorderThreadWithHitObject {
                hit_object_id,
                hint,
                bits,
            } => {
                let u32_ty = self.u32_ty();
                let mut ops = Vec::new();
                if let Some(&ho_var) = self.hit_object_vars.get(&hit_object_id) {
                    ops.push(ho_var);
                    if let (Some(h_inst), Some(b_inst)) = (hint, bits) {
                        let h = self.coerce_to_ty(
                            self.ensure_inst(h_inst),
                            self.inst_types.get(&h_inst).copied().unwrap_or(u32_ty),
                            u32_ty,
                        );
                        let b = self.coerce_to_ty(
                            self.ensure_inst(b_inst),
                            self.inst_types.get(&b_inst).copied().unwrap_or(u32_ty),
                            u32_ty,
                        );
                        ops.push(h);
                        ops.push(b);
                    }
                    let op: u16 = if self.use_ser_ext { 5315 } else { 5279 };
                    self.emitter.emit_fn_inst_raw(op, &ops);
                }
                (0, self.void_ty)
            }
            InstData::RayQueryTriangleVertexPositions {
                ray_query_id,
                committed,
            } => {
                let f32_ty = self.f32_ty();
                let u32_ty = self.u32_ty();
                let v3_ty = self.emitter.emit_type_vector(f32_ty, 3);
                let len3 = self.emitter.emit_constant_32(u32_ty, 3);
                let arr_ty = self.emitter.emit_type_array(v3_ty, len3);
                if let Some(&rq_var) = self.ray_query_vars.get(&ray_query_id) {
                    let inter_const = self
                        .emitter
                        .emit_constant_32(u32_ty, u32::from(committed));
                    let res = self.emitter.alloc_id();
                    self.emitter.emit_fn_inst_raw(
                        5340, // OpRayQueryGetIntersectionTriangleVertexPositionsKHR
                        &[arr_ty, res, rq_var, inter_const],
                    );
                    (res, arr_ty)
                } else {
                    (self.emitter.emit_constant_null(arr_ty), arr_ty)
                }
            }
            InstData::CoopMatrixLoad {
                ptr,
                layout,
                stride,
            } => {
                let u32_ty = self.u32_ty();
                let ptr_id = self.ensure_inst(ptr);
                let mat_ir_ty = self.func.insts[id].ir_ty.clone();
                let mat_ty = self.emitter.emit_ir_type(&mat_ir_ty);
                let layout_id = self.emitter.emit_constant_32(u32_ty, layout as u32);
                let res = self.emitter.alloc_id();
                let mut ops = vec![mat_ty, res, ptr_id, layout_id];
                if let Some(s_inst) = stride {
                    let raw_s = self.ensure_inst(s_inst);
                    let s_ty = self.inst_types.get(&s_inst).copied().unwrap_or(u32_ty);
                    let s_u32 = self.coerce_to_ty(raw_s, s_ty, u32_ty);
                    ops.push(s_u32);
                }
                self.emitter.emit_fn_inst_raw(4457, &ops); // OpCooperativeMatrixLoadKHR
                (res, mat_ty)
            }
            InstData::CoopMatrixStore {
                ptr,
                value,
                layout,
                stride,
            } => {
                let u32_ty = self.u32_ty();
                let ptr_id = self.ensure_inst(ptr);
                let val_id = self.ensure_inst(value);
                let layout_id = self.emitter.emit_constant_32(u32_ty, layout as u32);
                let mut ops = vec![ptr_id, val_id, layout_id];
                if let Some(s_inst) = stride {
                    let raw_s = self.ensure_inst(s_inst);
                    let s_ty = self.inst_types.get(&s_inst).copied().unwrap_or(u32_ty);
                    let s_u32 = self.coerce_to_ty(raw_s, s_ty, u32_ty);
                    ops.push(s_u32);
                }
                self.emitter.emit_fn_inst_raw(4458, &ops); // OpCooperativeMatrixStoreKHR
                (0, self.void_ty)
            }
            InstData::CoopMatrixMulAdd {
                a,
                b,
                c,
                operands_mask,
            } => {
                let a_id = self.ensure_inst(a);
                let b_id = self.ensure_inst(b);
                let c_id = self.ensure_inst(c);
                let c_ty = self
                    .inst_types
                    .get(&c)
                    .copied()
                    .unwrap_or_else(|| self.emitter.emit_ir_type(&self.func.insts[id].ir_ty));
                let res = self.emitter.alloc_id();
                let mut ops = vec![c_ty, res, a_id, b_id, c_id];
                if operands_mask != 0 {
                    ops.push(operands_mask);
                }
                self.emitter.emit_fn_inst_raw(4459, &ops); // OpCooperativeMatrixMulAddKHR
                (res, c_ty)
            }
            InstData::CoopMatrixLength { mat_ty } => {
                let u32_ty = self.u32_ty();
                let spv_mat_ty = self.emitter.emit_ir_type(&mat_ty);
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst_raw(4460, &[u32_ty, res, spv_mat_ty]); // OpCooperativeMatrixLengthKHR
                (res, u32_ty)
            }
            InstData::SubpassLoad {
                descriptor_index,
                sample,
            } => {
                let f32_ty = self.f32_ty();
                let vec4_ty = self.emitter.emit_type_vector(f32_ty, 4);
                if let Some(desc) = self.descriptors.get(descriptor_index as usize).cloned() {
                    if let Some(img_id) = self.load_raw_image(&desc) {
                        let i32_ty = self.i32_ty();
                        let zero_i = self.emitter.emit_constant_32(i32_ty, 0);
                        let ivec2_ty = self.emitter.emit_type_vector(i32_ty, 2);
                        let zero_coord =
                            self.emitter.emit_constant_composite(ivec2_ty, &[zero_i, zero_i]);
                        let res_ty = self.emitter.emit_type_vector(desc.sampled_ty, 4);
                        let res = self.emitter.alloc_id();
                        if let Some(s) = sample {
                            let s_id = self.get_val(s);
                            self.emitter.emit_fn_inst(
                                spirv::Op::ImageRead,
                                &[res_ty, res, img_id, zero_coord, 0x40, s_id],
                            );
                        } else {
                            self.emitter.emit_fn_inst(
                                spirv::Op::ImageRead,
                                &[res_ty, res, img_id, zero_coord],
                            );
                        }
                        (res, res_ty)
                    } else {
                        let z = self.emitter.emit_constant_32(f32_ty, 0);
                        (
                            self.emitter.emit_constant_composite(vec4_ty, &[z, z, z, z]),
                            vec4_ty,
                        )
                    }
                } else {
                    let z = self.emitter.emit_constant_32(f32_ty, 0);
                    (
                        self.emitter.emit_constant_composite(vec4_ty, &[z, z, z, z]),
                        vec4_ty,
                    )
                }
            }
            InstData::ImageFetchMs {
                descriptor_index,
                coord,
                sample,
            } => {
                let f32_ty = self.f32_ty();
                let vec4_ty = self.emitter.emit_type_vector(f32_ty, 4);
                let coord_id = self.get_val(coord);
                let sample_id = self.get_val(sample);
                if let Some(desc) = self.descriptors.get(descriptor_index as usize).cloned() {
                    if let Some(img_id) = self.load_raw_image(&desc) {
                        let res_ty = self.emitter.emit_type_vector(desc.sampled_ty, 4);
                        let res = self.emitter.alloc_id();
                        self.emitter.emit_fn_inst(
                            spirv::Op::ImageFetch,
                            &[res_ty, res, img_id, coord_id, 0x40, sample_id],
                        );
                        (res, res_ty)
                    } else {
                        let z = self.emitter.emit_constant_32(f32_ty, 0);
                        (
                            self.emitter.emit_constant_composite(vec4_ty, &[z, z, z, z]),
                            vec4_ty,
                        )
                    }
                } else {
                    let z = self.emitter.emit_constant_32(f32_ty, 0);
                    (
                        self.emitter.emit_constant_composite(vec4_ty, &[z, z, z, z]),
                        vec4_ty,
                    )
                }
            }
            InstData::BeginInvocationInterlockExt => {
                self.emitter
                    .emit_extension("SPV_EXT_fragment_shader_interlock");
                self.emitter.emit_fn_inst_raw(5364, &[]); // OpBeginInvocationInterlockEXT
                (0, self.void_ty)
            }
            InstData::EndInvocationInterlockExt => {
                self.emitter
                    .emit_extension("SPV_EXT_fragment_shader_interlock");
                self.emitter.emit_fn_inst_raw(5365, &[]); // OpEndInvocationInterlockEXT
                (0, self.void_ty)
            }
            InstData::DebugPrintf { format, args } => {
                let ext_id = self.emitter.emit_debug_printf_import();
                let fmt_id = self.emitter.emit_debug_string(format.as_str());
                let arg_ids: Vec<u32> = args.iter().map(|&a| self.ensure_inst(a)).collect();
                let res = self.emitter.alloc_id();
                let mut ops = Vec::with_capacity(5 + arg_ids.len());
                ops.push(self.void_ty);
                ops.push(res);
                ops.push(ext_id);
                ops.push(1); // NonSemantic.DebugPrintf: DebugPrintf = 1
                ops.push(fmt_id);
                ops.extend(arg_ids);
                self.emitter.emit_fn_inst(spirv::Op::ExtInst, &ops);
                (0, self.void_ty)
            }
            InstData::DebugLine { line, col } => {
                if let Some((ext_id, debug_source)) = self.debug_ctx {
                    let u32_ty = self.u32_ty();
                    let line_id = self.emitter.emit_constant_32(u32_ty, line);
                    let col_id = self.emitter.emit_constant_32(u32_ty, col);
                    let res = self.emitter.alloc_id();
                    self.emitter.emit_fn_inst(
                        spirv::Op::ExtInst,
                        &[
                            self.void_ty,
                            res,
                            ext_id,
                            103, // DebugLine
                            debug_source,
                            line_id,
                            line_id,
                            col_id,
                            col_id,
                        ],
                    );
                }
                (0, self.void_ty)
            }
            InstData::Nop => (0, self.void_ty),
        };

        self.inst_ids.insert(id, spv_id);
        self.inst_types.insert(id, res_ty);
        if is_bda_ptr {
            self.bda_ptr_insts.insert(id);
        }
        if is_workgroup_ptr {
            self.workgroup_ptr_insts.insert(id);
        }

        if spv_id != 0
            && !is_global_var
            && !self.emitter.is_const_id(spv_id)
            && self.func.insts[id].non_uniform
            && self.decorated_non_uniform.insert(spv_id)
        {
            self.emitter
                .emit_decorate(spv_id, spirv::Decoration::NonUniform, &[]);
        }

        spv_id
    }

    fn emit_ray_query_u32(&mut self, rq_id: u32, op: spirv::Op, intersection: u32) -> (u32, u32) {
        let u32_ty = self.u32_ty();
        if let Some(&rq_var) = self.ray_query_vars.get(&rq_id) {
            let inter_const = self.emitter.emit_constant_32(u32_ty, intersection);
            let res = self.emitter.alloc_id();
            self.emitter
                .emit_fn_inst(op, &[u32_ty, res, rq_var, inter_const]);
            (res, u32_ty)
        } else {
            (self.emitter.emit_constant_32(u32_ty, 0), u32_ty)
        }
    }

    fn emit_ray_query_t(&mut self, rq_id: u32, intersection: u32) -> (u32, u32) {
        let f32_ty = self.f32_ty();
        if let Some(&rq_var) = self.ray_query_vars.get(&rq_id) {
            let u32_ty = self.u32_ty();
            let inter_const = self.emitter.emit_constant_32(u32_ty, intersection);
            let res = self.emitter.alloc_id();
            self.emitter.emit_fn_inst(
                spirv::Op::RayQueryGetIntersectionTKHR,
                &[f32_ty, res, rq_var, inter_const],
            );
            (res, f32_ty)
        } else {
            (
                self.emitter.emit_constant_32(f32_ty, 0.0_f32.to_bits()),
                f32_ty,
            )
        }
    }

    fn emit_ray_query_matrix(&mut self, rq_id: u32, op: spirv::Op) -> (u32, u32) {
        let f32_ty = self.f32_ty();
        let col_ty = self.emitter.emit_type_vector(f32_ty, 3);
        let mat4x3_ty = self.emitter.emit_type_matrix(col_ty, 4);
        if let Some(&rq_var) = self.ray_query_vars.get(&rq_id) {
            let u32_ty = self.u32_ty();
            let committed = self.emitter.emit_constant_32(u32_ty, 1);
            let res = self.emitter.alloc_id();
            self.emitter
                .emit_fn_inst(op, &[mat4x3_ty, res, rq_var, committed]);
            (res, mat4x3_ty)
        } else {
            (self.emit_zero_const(mat4x3_ty), mat4x3_ty)
        }
    }

    fn emit_subgroup_ballot_bit_count(
        &mut self,
        ballot: InstId,
        group_op: spirv::GroupOperation,
    ) -> (u32, u32) {
        self.emitter
            .emit_capability(spirv::Capability::GroupNonUniformBallot);
        let raw = self.ensure_inst(ballot);
        let u32_ty = self.u32_ty();
        let vec4_u32 = self.emitter.emit_type_vector(u32_ty, 4);
        let ballot_ty = self.inst_types.get(&ballot).copied().unwrap_or(u32_ty);
        let ballot_v4 = if ballot_ty == vec4_u32 {
            raw
        } else {
            let s = self.coerce_to_ty(raw, ballot_ty, u32_ty);
            let zero = self.emitter.emit_constant_32(u32_ty, 0);
            let v = self.emitter.alloc_id();
            self.emitter.emit_fn_inst(
                spirv::Op::CompositeConstruct,
                &[vec4_u32, v, s, zero, zero, zero],
            );
            v
        };
        let scope = self
            .emitter
            .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
        let res = self.emitter.alloc_id();
        self.emitter.emit_fn_inst(
            spirv::Op::GroupNonUniformBallotBitCount,
            &[u32_ty, res, scope, group_op as u32, ballot_v4],
        );
        (res, u32_ty)
    }

    fn emit_subgroup_arith(
        &mut self,
        val: InstId,
        kind: SubgroupArithKind,
        group_op: spirv::GroupOperation,
    ) -> (u32, u32) {
        self.emitter
            .emit_capability(spirv::Capability::GroupNonUniformArithmetic);
        let raw = self.ensure_inst(val);
        let u32_ty = self.u32_ty();
        let val_ty = self.inst_types.get(&val).copied().unwrap_or(u32_ty);
        let spv_op = match (
            kind,
            self.is_float_spv_ty(val_ty),
            self.is_signed_int_spv_ty(val_ty),
        ) {
            (SubgroupArithKind::Add, true, _) => spirv::Op::GroupNonUniformFAdd,
            (SubgroupArithKind::Add, false, _) => spirv::Op::GroupNonUniformIAdd,
            (SubgroupArithKind::Mul, true, _) => spirv::Op::GroupNonUniformFMul,
            (SubgroupArithKind::Mul, false, _) => spirv::Op::GroupNonUniformIMul,
            (SubgroupArithKind::Min, true, _) => spirv::Op::GroupNonUniformFMin,
            (SubgroupArithKind::Min, false, true) => spirv::Op::GroupNonUniformSMin,
            (SubgroupArithKind::Min, false, false) => spirv::Op::GroupNonUniformUMin,
            (SubgroupArithKind::Max, true, _) => spirv::Op::GroupNonUniformFMax,
            (SubgroupArithKind::Max, false, true) => spirv::Op::GroupNonUniformSMax,
            (SubgroupArithKind::Max, false, false) => spirv::Op::GroupNonUniformUMax,
        };
        let scope = self
            .emitter
            .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
        let res = self.emitter.alloc_id();
        self.emitter
            .emit_fn_inst(spv_op, &[val_ty, res, scope, group_op as u32, raw]);
        (res, val_ty)
    }

    fn emit_subgroup_arith_clustered(
        &mut self,
        val: InstId,
        kind: SubgroupArithKind,
        cluster_size_id: u32,
    ) -> (u32, u32) {
        self.emitter
            .emit_capability(spirv::Capability::GroupNonUniformArithmetic);
        let raw = self.ensure_inst(val);
        let u32_ty = self.u32_ty();
        let val_ty = self.inst_types.get(&val).copied().unwrap_or(u32_ty);
        let spv_op = match (
            kind,
            self.is_float_spv_ty(val_ty),
            self.is_signed_int_spv_ty(val_ty),
        ) {
            (SubgroupArithKind::Add, true, _) => spirv::Op::GroupNonUniformFAdd,
            (SubgroupArithKind::Add, false, _) => spirv::Op::GroupNonUniformIAdd,
            (SubgroupArithKind::Mul, true, _) => spirv::Op::GroupNonUniformFMul,
            (SubgroupArithKind::Mul, false, _) => spirv::Op::GroupNonUniformIMul,
            (SubgroupArithKind::Min, true, _) => spirv::Op::GroupNonUniformFMin,
            (SubgroupArithKind::Min, false, true) => spirv::Op::GroupNonUniformSMin,
            (SubgroupArithKind::Min, false, false) => spirv::Op::GroupNonUniformUMin,
            (SubgroupArithKind::Max, true, _) => spirv::Op::GroupNonUniformFMax,
            (SubgroupArithKind::Max, false, true) => spirv::Op::GroupNonUniformSMax,
            (SubgroupArithKind::Max, false, false) => spirv::Op::GroupNonUniformUMax,
        };
        let scope = self
            .emitter
            .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
        let res = self.emitter.alloc_id();
        self.emitter.emit_fn_inst(
            spv_op,
            &[
                val_ty,
                res,
                scope,
                spirv::GroupOperation::ClusteredReduce as u32,
                raw,
                cluster_size_id,
            ],
        );
        (res, val_ty)
    }

    fn emit_subgroup_bitwise(
        &mut self,
        val: InstId,
        opcode: u16,
        group_op: spirv::GroupOperation,
        cluster_size_id: Option<u32>,
    ) -> (u32, u32) {
        self.emitter
            .emit_capability(spirv::Capability::GroupNonUniformArithmetic);
        let raw = self.ensure_inst(val);
        let u32_ty = self.u32_ty();
        let val_ty = self.inst_types.get(&val).copied().unwrap_or(u32_ty);
        let scope = self
            .emitter
            .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
        let res = self.emitter.alloc_id();
        let mut ops = vec![val_ty, res, scope, group_op as u32, raw];
        if let Some(cs) = cluster_size_id {
            ops.push(cs);
        }
        self.emitter.emit_fn_inst_raw(opcode, &ops);
        (res, val_ty)
    }

    fn emit_subgroup_logical(&mut self, val: InstId, opcode: u16) -> (u32, u32) {
        self.emitter
            .emit_capability(spirv::Capability::GroupNonUniformArithmetic);
        let pred_id = self.ensure_bool(val);
        let u32_ty = self.u32_ty();
        let bool_ty = self.bool_ty();
        let scope = self
            .emitter
            .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
        let res = self.emitter.alloc_id();
        self.emitter.emit_fn_inst_raw(
            opcode,
            &[
                bool_ty,
                res,
                scope,
                spirv::GroupOperation::Reduce as u32,
                pred_id,
            ],
        );
        (res, bool_ty)
    }

    fn emit_quad_swap(&mut self, val: InstId, direction: u32) -> (u32, u32) {
        self.emitter
            .emit_capability(spirv::Capability::GroupNonUniformQuad);
        let raw = self.ensure_inst(val);
        let u32_ty = self.u32_ty();
        let val_ty = self.inst_types.get(&val).copied().unwrap_or(u32_ty);
        let scope = self
            .emitter
            .emit_constant_32(u32_ty, spirv::Scope::Subgroup as u32);
        let dir_const = self.emitter.emit_constant_32(u32_ty, direction);
        let res = self.emitter.alloc_id();
        self.emitter.emit_fn_inst(
            spirv::Op::GroupNonUniformQuadSwap,
            &[val_ty, res, scope, raw, dir_const],
        );
        (res, val_ty)
    }

    /// Emits a 32-bit float screen-space derivative (`OpDPdx`/`OpDPdy`/`OpFwidth`),
    /// coercing scalar or vector operands to 32-bit float as required by SPIR-V.
    fn emit_derivative_op(&mut self, op: spirv::Op, val: InstId) -> (u32, u32) {
        let raw = self.ensure_inst(val);
        let f32_ty = self.f32_ty();
        let src_ty = self.inst_types.get(&val).copied().unwrap_or(f32_ty);
        let target_ty = if let Some((_, len)) = self.vector_component_info(src_ty) {
            self.emitter.emit_type_vector(f32_ty, len)
        } else {
            f32_ty
        };
        let coerced = self.coerce_to_ty(raw, src_ty, target_ty);
        let res = self.emitter.alloc_id();
        self.emitter.emit_fn_inst(op, &[target_ty, res, coerced]);
        (res, target_ty)
    }

    /// Resolves `(ptr_id, elem_ty, storage_class)` for an atomic pointer operand across
    /// `Workgroup`, `StorageBuffer`, `PhysicalStorageBuffer`, and `Image`.
    fn resolve_atomic_ptr(
        &mut self,
        inst_id: InstId,
        ptr: InstId,
        raw_ptr_id: u32,
        val_ty: u32,
    ) -> Option<(u32, u32, spirv::StorageClass)> {
        let u32_ty = self.u32_ty();
        let i32_ty = self.i32_ty();
        let u64_cached = self.cached_u64_ty();
        let i64_cached = self.cached_i64_ty();
        let f32_ty = self.f32_ty();

        if let Some(&sc) = self.inst_ptr_storage.get(&ptr)
            && matches!(
                sc,
                spirv::StorageClass::Workgroup
                    | spirv::StorageClass::StorageBuffer
                    | spirv::StorageClass::Image
            )
            && raw_ptr_id != 0
        {
            let elem_ty = self.inst_pointee_types.get(&ptr).copied().unwrap_or(val_ty);
            return Some((raw_ptr_id, elem_ty, sc));
        }

        if self.workgroup_ptr_insts.contains(&ptr) && raw_ptr_id != 0 {
            let elem_ty = self.inst_pointee_types.get(&ptr).copied().unwrap_or(u32_ty);
            return Some((raw_ptr_id, elem_ty, spirv::StorageClass::Workgroup));
        }

        if self.uses_bda || self.bda_ptr_insts.contains(&ptr) {
            let elem_ty = if self.func.insts[inst_id].ir_ty != IrType::Void {
                let ir_ty = self.func.insts[inst_id].ir_ty.clone();
                self.emitter.emit_ir_type(&ir_ty)
            } else if u64_cached == Some(val_ty)
                || i64_cached == Some(val_ty)
                || val_ty == f32_ty
                || val_ty == i32_ty
            {
                val_ty
            } else {
                u32_ty
            };
            let stride =
                if self.cached_u64_ty() == Some(elem_ty) || self.cached_i64_ty() == Some(elem_ty) {
                    8
                } else {
                    4
                };
            let block_ptr = self.ensure_bda_block_ptr(ptr, elem_ty, stride, None);
            let elem_ptr = self.bda_elem_ptr_for(block_ptr, elem_ty);
            return Some((elem_ptr, elem_ty, self.bda_storage_class()));
        }

        None
    }

    fn emit_atomic_op(
        &mut self,
        inst_id: InstId,
        kind: AtomicOpKind,
        ptr: InstId,
        value: InstId,
        scope: AtomicScope,
        order: AtomicOrder,
    ) -> (u32, u32) {
        let raw_ptr = self.ensure_inst(ptr);
        let raw_val = self.ensure_inst(value);
        let u32_ty = self.u32_ty();
        let i32_ty = self.i32_ty();
        let f32_ty = self.f32_ty();
        let val_ty = self.inst_types.get(&value).copied().unwrap_or(u32_ty);

        let Some((ptr_id, elem_ty, sc)) = self.resolve_atomic_ptr(inst_id, ptr, raw_ptr, val_ty)
        else {
            return (self.emit_zero_const(u32_ty), u32_ty);
        };

        let u64_cached = self.cached_u64_ty();
        let i64_cached = self.cached_i64_ty();
        if u64_cached == Some(elem_ty) || i64_cached == Some(elem_ty) {
            self.emitter.emit_capability(spirv::Capability::Int64);
            self.emitter
                .emit_capability(spirv::Capability::Int64Atomics);
        } else if elem_ty == f32_ty {
            match kind {
                AtomicOpKind::Add => {
                    self.emitter
                        .emit_capability(spirv::Capability::AtomicFloat32AddEXT);
                    self.emitter
                        .emit_extension("SPV_EXT_shader_atomic_float_add");
                }
                AtomicOpKind::Min | AtomicOpKind::Max => {
                    self.emitter
                        .emit_capability(spirv::Capability::AtomicFloat32MinMaxEXT);
                    self.emitter
                        .emit_extension("SPV_EXT_shader_atomic_float_min_max");
                }
            }
        }

        let spv_op = match kind {
            AtomicOpKind::Add => {
                if elem_ty == f32_ty {
                    spirv::Op::AtomicFAddEXT
                } else {
                    spirv::Op::AtomicIAdd
                }
            }
            AtomicOpKind::Min => {
                if elem_ty == f32_ty {
                    spirv::Op::AtomicFMinEXT
                } else if elem_ty == i32_ty || i64_cached == Some(elem_ty) {
                    spirv::Op::AtomicSMin
                } else {
                    spirv::Op::AtomicUMin
                }
            }
            AtomicOpKind::Max => {
                if elem_ty == f32_ty {
                    spirv::Op::AtomicFMaxEXT
                } else if elem_ty == i32_ty || i64_cached == Some(elem_ty) {
                    spirv::Op::AtomicSMax
                } else {
                    spirv::Op::AtomicUMax
                }
            }
        };

        let val_id = self.coerce_to_ty(raw_val, val_ty, elem_ty);
        let scope_const = self
            .emitter
            .emit_constant_32(u32_ty, map_atomic_scope(scope));
        let sem_const = self
            .emitter
            .emit_constant_32(u32_ty, map_atomic_semantics_sc(order, sc));
        let res = self.emitter.alloc_id();
        self.emitter.emit_fn_inst(
            spv_op,
            &[elem_ty, res, ptr_id, scope_const, sem_const, val_id],
        );
        (res, elem_ty)
    }

    fn emit_atomic_cas(
        &mut self,
        inst_id: InstId,
        ptr: InstId,
        expected: InstId,
        desired: InstId,
        scope: AtomicScope,
        order: AtomicOrder,
    ) -> (u32, u32) {
        let raw_ptr = self.ensure_inst(ptr);
        let raw_exp = self.ensure_inst(expected);
        let raw_des = self.ensure_inst(desired);
        let u32_ty = self.u32_ty();
        let exp_ty = self.inst_types.get(&expected).copied().unwrap_or(u32_ty);
        let des_ty = self.inst_types.get(&desired).copied().unwrap_or(u32_ty);

        let Some((ptr_id, elem_ty, sc)) = self.resolve_atomic_ptr(inst_id, ptr, raw_ptr, exp_ty)
        else {
            return (self.emit_zero_const(u32_ty), u32_ty);
        };

        if self.cached_u64_ty() == Some(elem_ty) || self.cached_i64_ty() == Some(elem_ty) {
            self.emitter.emit_capability(spirv::Capability::Int64);
            self.emitter
                .emit_capability(spirv::Capability::Int64Atomics);
        }

        let exp_id = self.coerce_to_ty(raw_exp, exp_ty, elem_ty);
        let des_id = self.coerce_to_ty(raw_des, des_ty, elem_ty);
        let scope_const = self
            .emitter
            .emit_constant_32(u32_ty, map_atomic_scope(scope));
        let eq_sem = self
            .emitter
            .emit_constant_32(u32_ty, map_atomic_semantics_sc(order, sc));
        // SPIR-V forbids Release / AcquireRelease on the Unequal semantics operand.
        let neq_order = match order {
            AtomicOrder::Release => AtomicOrder::Relaxed,
            AtomicOrder::AcqRel => AtomicOrder::Acquire,
            other => other,
        };
        let neq_sem = self
            .emitter
            .emit_constant_32(u32_ty, map_atomic_semantics_sc(neq_order, sc));
        let res = self.emitter.alloc_id();
        self.emitter.emit_fn_inst(
            spirv::Op::AtomicCompareExchange,
            &[
                elem_ty,
                res,
                ptr_id,
                scope_const,
                eq_sem,
                neq_sem,
                des_id,
                exp_id,
            ],
        );
        (res, elem_ty)
    }

    /// Emits an `OpPhi` instruction whose incoming operands have already been materialized
    /// and coerced to `phi_ty` in their respective predecessor blocks (or are module-level constants).
    fn emit_phi_operands(&mut self, phi_id: InstId, incoming: &[(InstId, u32)]) -> (u32, u32) {
        let ty = self.infer_phi_ty(phi_id);
        let mut resolved_pairs: Vec<(u32, u32)> =
            Vec::with_capacity(self.current_block_preds.len().max(incoming.len()));
        if self.current_block_preds.is_empty() {
            if let Some(&(val_inst, _)) = incoming.first() {
                let val_id = self
                    .resolve_phi_incoming_no_emit(val_inst, ty)
                    .unwrap_or_else(|| self.emitter.emit_constant_null(ty));
                resolved_pairs.push((val_id, self.entry_label));
            }
        } else {
            let preds = self.current_block_preds.clone();
            let mut seen_labels: IndexSet<u32> = IndexSet::new();
            for (pred_tag, pred_label) in preds {
                if !seen_labels.insert(pred_label) {
                    continue;
                }
                let val_id =
                    if let Some(&coerced) = self.phi_coerced_inputs.get(&(phi_id, pred_label)) {
                        coerced
                    } else if let Some(&(val_inst, _)) = incoming
                        .iter()
                        .find(|&&(_, tag)| tag == pred_tag)
                        .or_else(|| incoming.first())
                    {
                        self.resolve_phi_incoming_no_emit(val_inst, ty)
                            .unwrap_or_else(|| self.emitter.emit_constant_null(ty))
                    } else {
                        self.emitter.emit_constant_null(ty)
                    };
                resolved_pairs.push((val_id, pred_label));
            }
        }
        let res = self.emitter.alloc_id();
        let mut operands = Vec::with_capacity(2 + resolved_pairs.len() * 2);
        operands.push(ty);
        operands.push(res);
        for (val_id, pred_label) in resolved_pairs {
            operands.push(val_id);
            operands.push(pred_label);
        }
        self.emitter.emit_fn_inst(spirv::Op::Phi, &operands);
        (res, ty)
    }

    /// Collects unemitted `InstData::Phi` instructions that belong to the merge block
    /// immediately following a structured control-flow construct.
    fn collect_merge_phis(&self, following_stmts: &[StructuredStmt]) -> Vec<InstId> {
        let mut phis = Vec::new();
        let mut seen = IndexSet::new();
        let mut visited = IndexSet::new();
        for stmt in following_stmts {
            match stmt {
                StructuredStmt::Eval(id) | StructuredStmt::Return(Some(id)) => {
                    self.collect_phis_in_inst(*id, &mut visited, &mut seen, &mut phis);
                }
                StructuredStmt::If { cond, .. } => {
                    self.collect_phis_in_inst(*cond, &mut visited, &mut seen, &mut phis);
                    break;
                }
                StructuredStmt::Switch { selector, .. } => {
                    self.collect_phis_in_inst(*selector, &mut visited, &mut seen, &mut phis);
                    break;
                }
                StructuredStmt::Loop { induction_init, .. } => {
                    if let Some(ind_id) = induction_init {
                        self.collect_phis_in_inst(*ind_id, &mut visited, &mut seen, &mut phis);
                    }
                    break;
                }
                StructuredStmt::Break | StructuredStmt::Continue | StructuredStmt::Return(None) => {
                    break;
                }
            }
        }
        phis
    }

    fn collect_phis_in_inst(
        &self,
        id: InstId,
        visited: &mut IndexSet<InstId>,
        seen_phis: &mut IndexSet<InstId>,
        out: &mut Vec<InstId>,
    ) {
        if self.inst_ids.contains_key(&id) || !visited.insert(id) {
            return;
        }
        if matches!(self.func.insts[id].data, InstData::Phi { .. }) {
            if seen_phis.insert(id) {
                out.push(id);
            }
            return;
        }
        for op in self.func.insts[id].data.operands() {
            self.collect_phis_in_inst(op, visited, seen_phis, out);
        }
    }

    /// Returns `true` if `id` transitively depends on an unemitted `Phi` or side-effecting
    /// instruction, making it unsafe to hoist above a control-flow boundary.
    fn depends_on_unemitted_phi_or_effect(
        &self,
        id: InstId,
        visited: &mut IndexSet<InstId>,
    ) -> bool {
        if self.inst_ids.contains_key(&id) || !visited.insert(id) {
            return false;
        }
        let inst = &self.func.insts[id];
        if matches!(inst.data, InstData::Phi { .. }) || inst.has_side_effects() {
            return true;
        }
        for op in inst.data.operands() {
            if self.depends_on_unemitted_phi_or_effect(op, visited) {
                return true;
            }
        }
        false
    }

    /// Pre-materializes pure, phi-independent SSA instructions referenced across two or more
    /// distinct regions (sibling branches, loop body/continuing, or a branch and following code)
    /// inside the dominating header/preheader block so nested branches never lazily emit an
    /// instruction that is later used outside or in a sibling branch.
    fn prematerialize_shared_branch_insts(
        &mut self,
        regions: &[&[StructuredStmt]],
        merge_phis: &[InstId],
    ) {
        if regions.len() < 2 && merge_phis.is_empty() {
            return;
        }
        let mut ref_counts: IndexMap<InstId, usize> = IndexMap::new();
        for region in regions {
            let mut region_refs: IndexSet<InstId> = IndexSet::new();
            let mut visited: IndexSet<InstId> = IndexSet::new();
            self.collect_branch_refs(region, &mut visited, &mut region_refs);
            for id in region_refs {
                *ref_counts.entry(id).or_insert(0) += 1;
            }
        }
        if !merge_phis.is_empty() {
            let mut phi_refs: IndexSet<InstId> = IndexSet::new();
            let mut visited: IndexSet<InstId> = IndexSet::new();
            for &phi_id in merge_phis {
                if let InstData::Phi { incoming } = &self.func.insts[phi_id].data {
                    for &(val_inst, _) in incoming {
                        self.collect_branch_inst_refs(val_inst, &mut visited, &mut phi_refs);
                    }
                }
            }
            for id in phi_refs {
                *ref_counts.entry(id).or_insert(0) += 1;
            }
        }
        let to_materialize: Vec<InstId> = ref_counts
            .into_iter()
            .filter_map(|(id, count)| {
                if count < 2
                    || self.inst_ids.contains_key(&id)
                    || is_const_inst_data(&self.func.insts[id].data)
                    || matches!(
                        self.func.insts[id].data,
                        InstData::Phi { .. } | InstData::Nop
                    )
                    || self.func.insts[id].has_side_effects()
                {
                    return None;
                }
                let mut check_vis = IndexSet::new();
                (!self.depends_on_unemitted_phi_or_effect(id, &mut check_vis)).then_some(id)
            })
            .collect();
        for id in to_materialize {
            if !self.inst_ids.contains_key(&id) {
                let _ = self.ensure_inst(id);
            }
        }
    }

    fn collect_branch_refs(
        &self,
        stmts: &[StructuredStmt],
        visited: &mut IndexSet<InstId>,
        out: &mut IndexSet<InstId>,
    ) {
        for stmt in stmts {
            match stmt {
                StructuredStmt::Eval(id) | StructuredStmt::Return(Some(id)) => {
                    self.collect_branch_inst_refs(*id, visited, out);
                }
                StructuredStmt::If {
                    cond,
                    then_body,
                    else_body,
                } => {
                    self.collect_branch_inst_refs(*cond, visited, out);
                    self.collect_branch_refs(then_body, visited, out);
                    self.collect_branch_refs(else_body, visited, out);
                }
                StructuredStmt::Switch {
                    selector,
                    cases,
                    default,
                } => {
                    self.collect_branch_inst_refs(*selector, visited, out);
                    for (_, case_body) in cases {
                        self.collect_branch_refs(case_body, visited, out);
                    }
                    self.collect_branch_refs(default, visited, out);
                }
                StructuredStmt::Loop {
                    body,
                    continuing,
                    induction_init,
                    ..
                } => {
                    if let Some(ind_id) = induction_init {
                        self.collect_branch_inst_refs(*ind_id, visited, out);
                    }
                    self.collect_branch_refs(body, visited, out);
                    self.collect_branch_refs(continuing, visited, out);
                }
                StructuredStmt::Break | StructuredStmt::Continue | StructuredStmt::Return(None) => {
                }
            }
        }
    }

    fn collect_branch_inst_refs(
        &self,
        id: InstId,
        visited: &mut IndexSet<InstId>,
        out: &mut IndexSet<InstId>,
    ) {
        if self.inst_ids.contains_key(&id) || !visited.insert(id) {
            return;
        }
        let inst = &self.func.insts[id];
        if matches!(inst.data, InstData::Phi { .. }) {
            return;
        }
        if !inst.has_side_effects() {
            out.insert(id);
        }
        for op in inst.data.operands() {
            self.collect_branch_inst_refs(op, visited, out);
        }
    }

    /// Pre-materializes and coerces incoming operands for `merge_phis` matching `pred_tag`
    /// in the currently active predecessor block before its terminating `OpBranch`.
    fn prematerialize_phi_inputs(&mut self, merge_phis: &[InstId], pred_tag: u32) {
        for &phi_id in merge_phis {
            let InstData::Phi { incoming } = &self.func.insts[phi_id].data else {
                continue;
            };
            let incoming = incoming.clone();
            let target_ty = self.infer_phi_ty(phi_id);
            for (val_inst, tag) in incoming {
                if tag == pred_tag {
                    let raw_id = self.ensure_inst(val_inst);
                    let src_ty = self.inst_types.get(&val_inst).copied().unwrap_or(target_ty);
                    let coerced = self.coerce_to_ty(raw_id, src_ty, target_ty);
                    self.phi_coerced_inputs
                        .insert((phi_id, self.current_label), coerced);
                }
            }
        }
    }

    /// Emits all pending merge-block `OpPhi` instructions immediately after `OpLabel`.
    fn emit_merge_phis(&mut self, merge_phis: &[InstId]) {
        if self.current_block_preds.is_empty() {
            return;
        }
        for &phi_id in merge_phis {
            if !self.inst_ids.contains_key(&phi_id) {
                let _ = self.ensure_inst(phi_id);
            }
        }
    }

    /// Collects all unemitted loop-header `Phi` instructions referenced anywhere in `body`
    /// (including inside nested control flow) or `continuing` whose `incoming` has `(init, 0)`
    /// and `(back, 1)` where `back` is defined/evaluated in `continuing`.
    fn collect_loop_header_phis(
        &self,
        body: &[StructuredStmt],
        continuing: &[StructuredStmt],
    ) -> Vec<InstId> {
        let mut continuing_refs: IndexSet<InstId> = IndexSet::new();
        self.collect_all_stmt_insts(continuing, &mut continuing_refs);
        if continuing_refs.is_empty() {
            return Vec::new();
        }
        let mut candidate_phis = Vec::new();
        let mut seen_phis = IndexSet::new();
        let mut visited = IndexSet::new();
        self.collect_all_phis_in_stmts(body, &mut visited, &mut seen_phis, &mut candidate_phis);
        self.collect_all_phis_in_stmts(
            continuing,
            &mut visited,
            &mut seen_phis,
            &mut candidate_phis,
        );

        candidate_phis
            .into_iter()
            .filter(|&phi_id| {
                let InstData::Phi { incoming } = &self.func.insts[phi_id].data else {
                    return false;
                };
                if incoming.len() != 2 {
                    return false;
                }
                let has_init = incoming.iter().any(|&(_, tag)| tag == 0);
                let back_in_continuing =
                    incoming
                        .iter()
                        .find(|&&(_, tag)| tag == 1)
                        .is_some_and(|&(back_inst, _)| {
                            !self.inst_ids.contains_key(&back_inst)
                                && continuing_refs.contains(&back_inst)
                        });
                has_init && back_in_continuing
            })
            .collect()
    }

    fn collect_all_stmt_insts(&self, stmts: &[StructuredStmt], out: &mut IndexSet<InstId>) {
        for stmt in stmts {
            match stmt {
                StructuredStmt::Eval(id) | StructuredStmt::Return(Some(id)) => {
                    self.collect_non_phi_inst_tree(*id, out);
                }
                StructuredStmt::If {
                    cond,
                    then_body,
                    else_body,
                } => {
                    self.collect_non_phi_inst_tree(*cond, out);
                    self.collect_all_stmt_insts(then_body, out);
                    self.collect_all_stmt_insts(else_body, out);
                }
                StructuredStmt::Switch {
                    selector,
                    cases,
                    default,
                } => {
                    self.collect_non_phi_inst_tree(*selector, out);
                    for (_, case_body) in cases {
                        self.collect_all_stmt_insts(case_body, out);
                    }
                    self.collect_all_stmt_insts(default, out);
                }
                StructuredStmt::Loop {
                    body,
                    continuing,
                    induction_init,
                    ..
                } => {
                    if let Some(ind_id) = induction_init {
                        self.collect_non_phi_inst_tree(*ind_id, out);
                    }
                    self.collect_all_stmt_insts(body, out);
                    self.collect_all_stmt_insts(continuing, out);
                }
                StructuredStmt::Break | StructuredStmt::Continue | StructuredStmt::Return(None) => {
                }
            }
        }
    }

    fn collect_non_phi_inst_tree(&self, id: InstId, out: &mut IndexSet<InstId>) {
        if self.inst_ids.contains_key(&id) || !out.insert(id) {
            return;
        }
        if matches!(self.func.insts[id].data, InstData::Phi { .. }) {
            return;
        }
        for op in self.func.insts[id].data.operands() {
            self.collect_non_phi_inst_tree(op, out);
        }
    }

    fn collect_all_phis_in_stmts(
        &self,
        stmts: &[StructuredStmt],
        visited: &mut IndexSet<InstId>,
        seen_phis: &mut IndexSet<InstId>,
        out: &mut Vec<InstId>,
    ) {
        for stmt in stmts {
            match stmt {
                StructuredStmt::Eval(id) | StructuredStmt::Return(Some(id)) => {
                    self.collect_phis_in_inst(*id, visited, seen_phis, out);
                }
                StructuredStmt::If {
                    cond,
                    then_body,
                    else_body,
                } => {
                    self.collect_phis_in_inst(*cond, visited, seen_phis, out);
                    self.collect_all_phis_in_stmts(then_body, visited, seen_phis, out);
                    self.collect_all_phis_in_stmts(else_body, visited, seen_phis, out);
                }
                StructuredStmt::Switch {
                    selector,
                    cases,
                    default,
                } => {
                    self.collect_phis_in_inst(*selector, visited, seen_phis, out);
                    for (_, case_body) in cases {
                        self.collect_all_phis_in_stmts(case_body, visited, seen_phis, out);
                    }
                    self.collect_all_phis_in_stmts(default, visited, seen_phis, out);
                }
                StructuredStmt::Loop {
                    body,
                    continuing,
                    induction_init,
                    ..
                } => {
                    if let Some(ind_id) = induction_init {
                        self.collect_phis_in_inst(*ind_id, visited, seen_phis, out);
                    }
                    self.collect_all_phis_in_stmts(body, visited, seen_phis, out);
                    self.collect_all_phis_in_stmts(continuing, visited, seen_phis, out);
                }
                StructuredStmt::Break | StructuredStmt::Continue | StructuredStmt::Return(None) => {
                }
            }
        }
    }

    /// Consumes a preallocated loop-header `OpPhi` back-edge ID only when the instruction's
    /// actual emitted SPIR-V type matches the `OpPhi` expected type; otherwise allocates a fresh
    /// ID so `continue_label` can safely coerce and bind the back-edge via `OpCopyObject`.
    fn take_preallocated_id(&mut self, inst_id: Option<InstId>, actual_ty: u32) -> u32 {
        if let Some(key) = inst_id
            && let Some(&(spv_id, expected_ty)) = self.preallocated_ids.get(&key)
            && expected_ty == actual_ty
        {
            self.preallocated_ids.shift_remove(&key);
            return spv_id;
        }
        self.emitter.alloc_id()
    }

    fn take_preallocated_exact(&mut self, inst_id: InstId, actual_ty: u32) -> Option<u32> {
        if let Some(&(spv_id, expected_ty)) = self.preallocated_ids.get(&inst_id)
            && expected_ty == actual_ty
        {
            self.preallocated_ids.shift_remove(&inst_id);
            return Some(spv_id);
        }
        None
    }

    fn emit_binop_32_with_id(
        &mut self,
        op: spirv::Op,
        a: InstId,
        b: InstId,
        inst_id: Option<InstId>,
    ) -> (u32, u32) {
        let raw_lhs = self.ensure_inst(a);
        let raw_rhs = self.ensure_inst(b);
        let i32_ty = self.i32_ty();
        let u32_ty = self.u32_ty();
        let i64_cached = self.cached_i64_ty();
        let u64_cached = self.cached_u64_ty();
        let lhs_ty = self.inst_types.get(&a).copied().unwrap_or(u32_ty);
        let rhs_ty = self.inst_types.get(&b).copied().unwrap_or(u32_ty);
        let ir_ty = inst_id.map(|iid| self.func.insts[iid].ir_ty.clone());
        let ty = if let Some(it) = ir_ty.as_ref()
            && it.is_64bit()
        {
            self.emitter.emit_ir_type(it)
        } else if let Some((comp_ty, len)) = self
            .vector_component_info(lhs_ty)
            .or_else(|| self.vector_component_info(rhs_ty))
        {
            let scalar = if i64_cached == Some(comp_ty) {
                self.i64_ty()
            } else if u64_cached == Some(comp_ty) {
                self.u64_ty()
            } else if comp_ty == i32_ty {
                i32_ty
            } else {
                u32_ty
            };
            self.emitter.emit_type_vector(scalar, len)
        } else if i64_cached == Some(lhs_ty)
            || u64_cached == Some(lhs_ty)
            || lhs_ty == i32_ty
            || lhs_ty == u32_ty
        {
            lhs_ty
        } else if i64_cached == Some(rhs_ty)
            || u64_cached == Some(rhs_ty)
            || rhs_ty == i32_ty
            || rhs_ty == u32_ty
        {
            rhs_ty
        } else {
            u32_ty
        };
        let lhs = self.coerce_to_ty(raw_lhs, lhs_ty, ty);
        let rhs = self.coerce_to_ty(raw_rhs, rhs_ty, ty);
        let res = self.take_preallocated_id(inst_id, ty);
        self.emitter.emit_fn_inst(op, &[ty, res, lhs, rhs]);
        (res, ty)
    }

    fn emit_signed_or_unsigned_binop_32_with_id(
        &mut self,
        signed_op: spirv::Op,
        unsigned_op: spirv::Op,
        a: InstId,
        b: InstId,
        inst_id: Option<InstId>,
    ) -> (u32, u32) {
        let raw_lhs = self.ensure_inst(a);
        let raw_rhs = self.ensure_inst(b);
        let i32_ty = self.i32_ty();
        let u32_ty = self.u32_ty();
        let i64_cached = self.cached_i64_ty();
        let u64_cached = self.cached_u64_ty();
        let lhs_ty = self.inst_types.get(&a).copied().unwrap_or(u32_ty);
        let rhs_ty = self.inst_types.get(&b).copied().unwrap_or(u32_ty);
        let ir_ty = inst_id.map(|iid| self.func.insts[iid].ir_ty.clone());
        let lhs_scalar = self
            .vector_component_info(lhs_ty)
            .map_or(lhs_ty, |(c, _)| c);
        let rhs_scalar = self
            .vector_component_info(rhs_ty)
            .map_or(rhs_ty, |(c, _)| c);
        let is_64bit = ir_ty.as_ref().is_some_and(IrType::is_64bit)
            || i64_cached == Some(lhs_scalar)
            || u64_cached == Some(lhs_scalar)
            || i64_cached == Some(rhs_scalar)
            || u64_cached == Some(rhs_scalar);
        let is_signed = if let Some(it) = ir_ty.as_ref()
            && (it.is_signed_int() || it.is_unsigned_int())
        {
            it.is_signed_int()
        } else {
            self.is_signed_int_spv_ty(lhs_ty)
        };
        let scalar = if is_64bit {
            if is_signed {
                self.i64_ty()
            } else {
                self.u64_ty()
            }
        } else if is_signed {
            i32_ty
        } else {
            u32_ty
        };
        let ty = if let Some((_, len)) = self
            .vector_component_info(lhs_ty)
            .or_else(|| self.vector_component_info(rhs_ty))
        {
            self.emitter.emit_type_vector(scalar, len)
        } else {
            scalar
        };
        let op = if is_signed { signed_op } else { unsigned_op };
        let lhs = self.coerce_to_ty(raw_lhs, lhs_ty, ty);
        let rhs = self.coerce_to_ty(raw_rhs, rhs_ty, ty);
        let res = self.take_preallocated_id(inst_id, ty);
        self.emitter.emit_fn_inst(op, &[ty, res, lhs, rhs]);
        (res, ty)
    }

    fn emit_binop_64_with_id(
        &mut self,
        op: spirv::Op,
        a: InstId,
        b: InstId,
        inst_id: Option<InstId>,
    ) -> (u32, u32) {
        let raw_lhs = self.ensure_inst(a);
        let raw_rhs = self.ensure_inst(b);
        let i64_ty = self.i64_ty();
        let u64_ty = self.u64_ty();
        let lhs_ty = self.inst_types.get(&a).copied().unwrap_or(u64_ty);
        let rhs_ty = self.inst_types.get(&b).copied().unwrap_or(u64_ty);
        let ty = if lhs_ty == i64_ty || lhs_ty == u64_ty {
            lhs_ty
        } else if rhs_ty == i64_ty || rhs_ty == u64_ty {
            rhs_ty
        } else {
            u64_ty
        };
        let lhs = self.coerce_to_ty(raw_lhs, lhs_ty, ty);
        let rhs = self.coerce_to_ty(raw_rhs, rhs_ty, ty);
        let res = self.take_preallocated_id(inst_id, ty);
        self.emitter.emit_fn_inst(op, &[ty, res, lhs, rhs]);
        (res, ty)
    }

    fn emit_float_binop_with_id(
        &mut self,
        op: spirv::Op,
        a: InstId,
        b: InstId,
        inst_id: Option<InstId>,
    ) -> (u32, u32) {
        let raw_lhs = self.ensure_inst(a);
        let raw_rhs = self.ensure_inst(b);
        let f32_ty = self.f32_ty();
        let f16_cached = self.emitter.type_cache.get(&TypeKey::Float(16)).copied();
        let lhs_ty = self.inst_types.get(&a).copied().unwrap_or(f32_ty);
        let rhs_ty = self.inst_types.get(&b).copied().unwrap_or(f32_ty);
        let scalar_ty = if f16_cached.is_some_and(|f16| {
            self.vector_component_info(lhs_ty)
                .map_or(lhs_ty, |(c, _)| c)
                == f16
                && self
                    .vector_component_info(rhs_ty)
                    .map_or(rhs_ty, |(c, _)| c)
                    == f16
        }) {
            f16_cached.unwrap_or(f32_ty)
        } else {
            f32_ty
        };
        let ty = if let Some((_, len)) = self
            .vector_component_info(lhs_ty)
            .or_else(|| self.vector_component_info(rhs_ty))
        {
            self.emitter.emit_type_vector(scalar_ty, len)
        } else {
            scalar_ty
        };
        let lhs = self.coerce_to_ty(raw_lhs, lhs_ty, ty);
        let rhs = self.coerce_to_ty(raw_rhs, rhs_ty, ty);
        let res = self.take_preallocated_id(inst_id, ty);
        self.emitter.emit_fn_inst(op, &[ty, res, lhs, rhs]);
        (res, ty)
    }

    /// Resolves a `Phi` incoming value without emitting any function-body instructions
    /// (SPIR-V requires all `OpPhi` instructions to appear at the very start of a block).
    /// Pure constants are safely materialized in the global `types_globals` section.
    fn resolve_phi_incoming_no_emit(&mut self, val_inst: InstId, target_ty: u32) -> Option<u32> {
        if let Some(&existing) = self.inst_ids.get(&val_inst) {
            let src_ty = self.inst_types.get(&val_inst).copied().unwrap_or(target_ty);
            return (src_ty == target_ty && existing != 0).then_some(existing);
        }
        let (const_id, const_ty) = match self.func.insts[val_inst].data {
            InstData::ConstBool(v) => {
                let ty = self.bool_ty();
                (self.emitter.emit_constant_bool(ty, v), ty)
            }
            InstData::ConstI32(v) => {
                let ty = self.i32_ty();
                (self.emitter.emit_constant_32(ty, v.cast_unsigned()), ty)
            }
            InstData::ConstU32(v) => {
                let ty = self.u32_ty();
                (self.emitter.emit_constant_32(ty, v), ty)
            }
            InstData::ConstF32(bits) => {
                let ty = self.f32_ty();
                (self.emitter.emit_constant_32(ty, bits), ty)
            }
            InstData::ConstF16(bits) => {
                let ty = self.f16_ty();
                (self.emitter.emit_constant_32(ty, u32::from(bits)), ty)
            }
            InstData::ConstI64(v) => {
                let ty = self.i64_ty();
                (self.emitter.emit_constant_64(ty, v.cast_unsigned()), ty)
            }
            InstData::ConstU64(v) => {
                let ty = self.u64_ty();
                (self.emitter.emit_constant_64(ty, v), ty)
            }
            _ => return None,
        };
        (const_ty == target_ty).then_some(const_id)
    }

    fn emit_glsl_std_450_op(&mut self, op: GlslStd450Op, args: &[InstId]) -> (u32, u32) {
        let f32_ty = self.f32_ty();
        let i32_ty = self.i32_ty();
        let u32_ty = self.u32_ty();

        // Special-case pack/unpack ops with fixed operand and return shapes.
        match op {
            GlslStd450Op::PackHalf2x16 => {
                let ext_id = self.emitter.emit_glsl_std_450_import();
                let v2_ty = self.emitter.emit_type_vector(f32_ty, 2);
                let arg0 = match args.first() {
                    Some(&a) => {
                        let raw = self.ensure_inst(a);
                        let src_ty = self.inst_types.get(&a).copied().unwrap_or(v2_ty);
                        self.coerce_to_ty(raw, src_ty, v2_ty)
                    }
                    None => self.emit_zero_const(v2_ty),
                };
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::ExtInst, &[u32_ty, res, ext_id, 58, arg0]);
                return (res, u32_ty);
            }
            GlslStd450Op::UnpackHalf2x16 => {
                let ext_id = self.emitter.emit_glsl_std_450_import();
                let v2_ty = self.emitter.emit_type_vector(f32_ty, 2);
                let arg0 = match args.first() {
                    Some(&a) => {
                        let raw = self.ensure_inst(a);
                        let src_ty = self.inst_types.get(&a).copied().unwrap_or(u32_ty);
                        self.coerce_to_ty(raw, src_ty, u32_ty)
                    }
                    None => self.emit_zero_const(u32_ty),
                };
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::ExtInst, &[v2_ty, res, ext_id, 62, arg0]);
                return (res, v2_ty);
            }
            GlslStd450Op::PackUnorm4x8 => {
                let ext_id = self.emitter.emit_glsl_std_450_import();
                let v4_ty = self.emitter.emit_type_vector(f32_ty, 4);
                let arg0 = match args.first() {
                    Some(&a) => {
                        let raw = self.ensure_inst(a);
                        let src_ty = self.inst_types.get(&a).copied().unwrap_or(v4_ty);
                        self.coerce_to_ty(raw, src_ty, v4_ty)
                    }
                    None => self.emit_zero_const(v4_ty),
                };
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::ExtInst, &[u32_ty, res, ext_id, 55, arg0]);
                return (res, u32_ty);
            }
            GlslStd450Op::UnpackUnorm4x8 => {
                let ext_id = self.emitter.emit_glsl_std_450_import();
                let v4_ty = self.emitter.emit_type_vector(f32_ty, 4);
                let arg0 = match args.first() {
                    Some(&a) => {
                        let raw = self.ensure_inst(a);
                        let src_ty = self.inst_types.get(&a).copied().unwrap_or(u32_ty);
                        self.coerce_to_ty(raw, src_ty, u32_ty)
                    }
                    None => self.emit_zero_const(u32_ty),
                };
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::ExtInst, &[v4_ty, res, ext_id, 64, arg0]);
                return (res, v4_ty);
            }
            GlslStd450Op::Mod => {
                let raw_a = args.first().map(|&a| (a, self.ensure_inst(a)));
                let raw_b = args.get(1).map(|&b| (b, self.ensure_inst(b)));
                let a_ty = raw_a
                    .and_then(|(a, _)| self.inst_types.get(&a).copied())
                    .unwrap_or(f32_ty);
                let b_ty = raw_b
                    .and_then(|(b, _)| self.inst_types.get(&b).copied())
                    .unwrap_or(a_ty);
                let res_ty = if let Some((_, len)) = self
                    .vector_component_info(a_ty)
                    .or_else(|| self.vector_component_info(b_ty))
                {
                    self.emitter.emit_type_vector(f32_ty, len)
                } else {
                    f32_ty
                };
                let lhs = match raw_a {
                    Some((_, id)) => self.coerce_to_ty(id, a_ty, res_ty),
                    None => self.emit_zero_const(res_ty),
                };
                let rhs = match raw_b {
                    Some((_, id)) => self.coerce_to_ty(id, b_ty, res_ty),
                    None => self.emit_zero_const(res_ty),
                };
                let res = self.emitter.alloc_id();
                self.emitter
                    .emit_fn_inst(spirv::Op::FMod, &[res_ty, res, lhs, rhs]);
                return (res, res_ty);
            }
            _ => {}
        }

        let scalar_target_ty = match op {
            GlslStd450Op::SAbs
            | GlslStd450Op::SMin
            | GlslStd450Op::SMax
            | GlslStd450Op::SClamp
            | GlslStd450Op::FindSMsb
            | GlslStd450Op::FindILsb => i32_ty,
            GlslStd450Op::UMin
            | GlslStd450Op::UMax
            | GlslStd450Op::UClamp
            | GlslStd450Op::FindUMsb => u32_ty,
            _ => f32_ty,
        };

        let mut raw_spv_args = Vec::with_capacity(args.len());
        let mut vec_ty_hint: Option<u32> = None;
        for &arg in args {
            let raw = self.ensure_inst(arg);
            let src_ty = self
                .inst_types
                .get(&arg)
                .copied()
                .unwrap_or(scalar_target_ty);
            if let Some((_, len)) = self.vector_component_info(src_ty) {
                vec_ty_hint = Some(self.emitter.emit_type_vector(scalar_target_ty, len));
            }
            raw_spv_args.push((raw, src_ty));
        }

        let arg_target_ty = vec_ty_hint.unwrap_or(scalar_target_ty);
        let mut spv_args = Vec::with_capacity(raw_spv_args.len());
        for (raw, src_ty) in raw_spv_args {
            if op == GlslStd450Op::Cross {
                let v3_ty = self.emitter.emit_type_vector(f32_ty, 3);
                spv_args.push(self.coerce_to_ty(raw, src_ty, v3_ty));
            } else {
                spv_args.push(self.coerce_to_ty(raw, src_ty, arg_target_ty));
            }
        }

        if op == GlslStd450Op::Dot {
            let a = spv_args
                .first()
                .copied()
                .unwrap_or_else(|| self.emit_zero_const(arg_target_ty));
            let b = spv_args
                .get(1)
                .copied()
                .unwrap_or_else(|| self.emit_zero_const(arg_target_ty));
            let res = self.emitter.alloc_id();
            if vec_ty_hint.is_some() {
                self.emitter
                    .emit_fn_inst(spirv::Op::Dot, &[f32_ty, res, a, b]);
            } else {
                self.emitter
                    .emit_fn_inst(spirv::Op::FMul, &[f32_ty, res, a, b]);
            }
            return (res, f32_ty);
        }

        let ext_id = self.emitter.emit_glsl_std_450_import();
        // Official Khronos GLSL.std.450 instruction numbers.
        let ext_op: u32 = match op {
            GlslStd450Op::Round => 1,
            GlslStd450Op::Trunc => 3,
            GlslStd450Op::FAbs => 4,
            GlslStd450Op::SAbs => 5,
            GlslStd450Op::Sign => 6,
            GlslStd450Op::Floor => 8,
            GlslStd450Op::Ceil => 9,
            GlslStd450Op::Fract => 10,
            GlslStd450Op::Sin => 13,
            GlslStd450Op::Cos => 14,
            GlslStd450Op::Tan => 15,
            GlslStd450Op::Asin => 16,
            GlslStd450Op::Acos => 17,
            GlslStd450Op::Atan => 18,
            GlslStd450Op::Atan2 => 25,
            GlslStd450Op::Pow => 26,
            GlslStd450Op::Exp => 27,
            GlslStd450Op::Log => 28,
            GlslStd450Op::Exp2 => 29,
            GlslStd450Op::Log2 => 30,
            GlslStd450Op::Sqrt => 31,
            GlslStd450Op::InverseSqrt => 32,
            GlslStd450Op::FMin => 37,
            GlslStd450Op::UMin => 38,
            GlslStd450Op::SMin => 39,
            GlslStd450Op::FMax => 40,
            GlslStd450Op::UMax => 41,
            GlslStd450Op::SMax => 42,
            GlslStd450Op::FClamp => 43,
            GlslStd450Op::UClamp => 44,
            GlslStd450Op::SClamp => 45,
            GlslStd450Op::FMix => 46,
            GlslStd450Op::Step => 48,
            GlslStd450Op::SmoothStep => 49,
            GlslStd450Op::Fma => 50,
            GlslStd450Op::Length => 66,
            GlslStd450Op::Cross => 68,
            GlslStd450Op::Normalize => 69,
            GlslStd450Op::Reflect => 71,
            GlslStd450Op::FindILsb => 73,
            GlslStd450Op::FindSMsb => 74,
            GlslStd450Op::FindUMsb => 75,
            GlslStd450Op::Dot
            | GlslStd450Op::Mod
            | GlslStd450Op::PackHalf2x16
            | GlslStd450Op::UnpackHalf2x16
            | GlslStd450Op::PackUnorm4x8
            | GlslStd450Op::UnpackUnorm4x8 => unreachable!(),
        };

        let res_ty = match op {
            GlslStd450Op::Cross => self.emitter.emit_type_vector(f32_ty, 3),
            GlslStd450Op::Length => f32_ty,
            _ => arg_target_ty,
        };
        let res = self.emitter.alloc_id();
        let mut ops = Vec::with_capacity(4 + spv_args.len());
        ops.push(res_ty);
        ops.push(res);
        ops.push(ext_id);
        ops.push(ext_op);
        ops.extend_from_slice(&spv_args);
        self.emitter.emit_fn_inst(spirv::Op::ExtInst, &ops);
        (res, res_ty)
    }

    fn emit_fcmp_op(&mut self, op: spirv::Op, a: InstId, b: InstId) -> (u32, u32) {
        let raw_lhs = self.ensure_inst(a);
        let raw_rhs = self.ensure_inst(b);
        let f32_ty = self.f32_ty();
        let bool_ty = self.bool_ty();
        let lhs_ty = self.inst_types.get(&a).copied().unwrap_or(f32_ty);
        let rhs_ty = self.inst_types.get(&b).copied().unwrap_or(f32_ty);
        let (cmp_ty, res_ty) = if let Some((_, len)) = self
            .vector_component_info(lhs_ty)
            .or_else(|| self.vector_component_info(rhs_ty))
        {
            (
                self.emitter.emit_type_vector(f32_ty, len),
                self.emitter.emit_type_vector(bool_ty, len),
            )
        } else {
            (f32_ty, bool_ty)
        };
        let lhs = self.coerce_to_ty(raw_lhs, lhs_ty, cmp_ty);
        let rhs = self.coerce_to_ty(raw_rhs, rhs_ty, cmp_ty);
        let res = self.emitter.alloc_id();
        self.emitter.emit_fn_inst(op, &[res_ty, res, lhs, rhs]);
        (res, res_ty)
    }

    fn emit_cmp_op(&mut self, op: spirv::Op, a: InstId, b: InstId) -> (u32, u32) {
        let raw_lhs = self.ensure_inst(a);
        let raw_rhs = self.ensure_inst(b);
        let i32_ty = self.i32_ty();
        let u32_ty = self.u32_ty();
        let i64_cached = self.cached_i64_ty();
        let u64_cached = self.cached_u64_ty();
        let bool_ty = self.bool_ty();
        let lhs_ty = self.inst_types.get(&a).copied().unwrap_or(u32_ty);
        let rhs_ty = self.inst_types.get(&b).copied().unwrap_or(u32_ty);
        let (cmp_ty, res_ty) = if let Some((c, len)) = self
            .vector_component_info(lhs_ty)
            .or_else(|| self.vector_component_info(rhs_ty))
        {
            let scalar = if c == i32_ty { i32_ty } else { u32_ty };
            (
                self.emitter.emit_type_vector(scalar, len),
                self.emitter.emit_type_vector(bool_ty, len),
            )
        } else if u64_cached == Some(lhs_ty) || u64_cached == Some(rhs_ty) {
            (self.u64_ty(), bool_ty)
        } else if i64_cached == Some(lhs_ty) || i64_cached == Some(rhs_ty) {
            (self.i64_ty(), bool_ty)
        } else if lhs_ty == i32_ty || rhs_ty == i32_ty {
            (i32_ty, bool_ty)
        } else {
            (u32_ty, bool_ty)
        };
        let lhs = self.coerce_to_ty(raw_lhs, lhs_ty, cmp_ty);
        let rhs = self.coerce_to_ty(raw_rhs, rhs_ty, cmp_ty);
        let res = self.emitter.alloc_id();
        self.emitter.emit_fn_inst(op, &[res_ty, res, lhs, rhs]);
        (res, res_ty)
    }

    fn emit_int_cmp_op(
        &mut self,
        signed_op: spirv::Op,
        unsigned_op: spirv::Op,
        a: InstId,
        b: InstId,
    ) -> (u32, u32) {
        let raw_lhs = self.ensure_inst(a);
        let raw_rhs = self.ensure_inst(b);
        let i32_ty = self.i32_ty();
        let u32_ty = self.u32_ty();
        let i64_cached = self.cached_i64_ty();
        let u64_cached = self.cached_u64_ty();
        let bool_ty = self.bool_ty();
        let lhs_ty = self.inst_types.get(&a).copied().unwrap_or(u32_ty);
        let rhs_ty = self.inst_types.get(&b).copied().unwrap_or(u32_ty);
        let (cmp_ty, res_ty, op) = if let Some((c, len)) = self
            .vector_component_info(lhs_ty)
            .or_else(|| self.vector_component_info(rhs_ty))
        {
            let is_unsigned = c == u32_ty;
            let scalar = if is_unsigned { u32_ty } else { i32_ty };
            (
                self.emitter.emit_type_vector(scalar, len),
                self.emitter.emit_type_vector(bool_ty, len),
                if is_unsigned { unsigned_op } else { signed_op },
            )
        } else if u64_cached == Some(lhs_ty) && u64_cached == Some(rhs_ty) {
            (self.u64_ty(), bool_ty, unsigned_op)
        } else if i64_cached == Some(lhs_ty)
            || i64_cached == Some(rhs_ty)
            || u64_cached == Some(lhs_ty)
            || u64_cached == Some(rhs_ty)
        {
            (self.i64_ty(), bool_ty, signed_op)
        } else if lhs_ty == u32_ty && rhs_ty == u32_ty {
            (u32_ty, bool_ty, unsigned_op)
        } else {
            (i32_ty, bool_ty, signed_op)
        };
        let lhs = self.coerce_to_ty(raw_lhs, lhs_ty, cmp_ty);
        let rhs = self.coerce_to_ty(raw_rhs, rhs_ty, cmp_ty);
        let res = self.emitter.alloc_id();
        self.emitter.emit_fn_inst(op, &[res_ty, res, lhs, rhs]);
        (res, res_ty)
    }

    /// Emits structured statements in strict block order, returning `true` if the active basic
    /// block was terminated.
    fn emit_stmts(&mut self, stmts: &[StructuredStmt]) -> bool {
        let mut terminated = false;
        for (stmt_idx, stmt) in stmts.iter().enumerate() {
            if terminated {
                break;
            }
            let following = &stmts[stmt_idx + 1..];
            match stmt {
                StructuredStmt::Eval(id) => {
                    let _ = self.ensure_inst(*id);
                    if matches!(
                        self.func.insts[*id].data,
                        InstData::IgnoreIntersectionKhr
                            | InstData::TerminateRayKhr
                            | InstData::EmitMeshTasksExt { .. }
                    ) {
                        return true;
                    }
                }
                StructuredStmt::If {
                    cond,
                    then_body,
                    else_body,
                } => {
                    let merge_phis = self.collect_merge_phis(following);
                    let cond_id = self.ensure_bool(*cond);
                    // Pre-materialize any pure operand shared across branches or used after the If
                    // in the dominating header block before OpSelectionMerge.
                    self.prematerialize_shared_branch_insts(
                        &[then_body.as_slice(), else_body.as_slice(), following],
                        &merge_phis,
                    );
                    if else_body.is_empty() {
                        self.prematerialize_phi_inputs(&merge_phis, 1);
                    }
                    let header_label = self.current_label;
                    let then_label = self.emitter.alloc_id();
                    let merge_label = self.emitter.alloc_id();
                    let else_label = if else_body.is_empty() {
                        merge_label
                    } else {
                        self.emitter.alloc_id()
                    };

                    self.emitter.emit_fn_inst(
                        spirv::Op::SelectionMerge,
                        &[merge_label, spirv::SelectionControl::NONE.bits()],
                    );
                    self.emitter.emit_fn_inst(
                        spirv::Op::BranchConditional,
                        &[cond_id, then_label, else_label],
                    );

                    let mut merge_preds: Vec<(u32, u32)> = Vec::with_capacity(2);

                    self.current_label = then_label;
                    self.current_block_preds = vec![(0, header_label)];
                    self.emitter.emit_fn_inst(spirv::Op::Label, &[then_label]);
                    if !self.emit_stmts(then_body) {
                        self.prematerialize_phi_inputs(&merge_phis, 0);
                        merge_preds.push((0, self.current_label));
                        self.emitter.emit_fn_inst(spirv::Op::Branch, &[merge_label]);
                    }

                    if else_body.is_empty() {
                        merge_preds.push((1, header_label));
                    } else {
                        self.current_label = else_label;
                        self.current_block_preds = vec![(0, header_label)];
                        self.emitter.emit_fn_inst(spirv::Op::Label, &[else_label]);
                        if !self.emit_stmts(else_body) {
                            self.prematerialize_phi_inputs(&merge_phis, 1);
                            merge_preds.push((1, self.current_label));
                            self.emitter.emit_fn_inst(spirv::Op::Branch, &[merge_label]);
                        }
                    }

                    self.current_label = merge_label;
                    self.current_block_preds = merge_preds;
                    self.emitter.emit_fn_inst(spirv::Op::Label, &[merge_label]);
                    if self.current_block_preds.is_empty() {
                        self.emitter.emit_fn_inst(spirv::Op::Unreachable, &[]);
                        return true;
                    }
                    self.emit_merge_phis(&merge_phis);
                }
                StructuredStmt::Switch {
                    selector,
                    cases,
                    default,
                } => {
                    let merge_phis = self.collect_merge_phis(following);
                    let raw_sel_id = self.ensure_inst(*selector);
                    let i32_ty = self.i32_ty();
                    let u32_ty = self.u32_ty();
                    let sel_src_ty = self.inst_types.get(selector).copied().unwrap_or(i32_ty);
                    let sel_id = if sel_src_ty == u32_ty {
                        raw_sel_id
                    } else {
                        self.coerce_to_ty(raw_sel_id, sel_src_ty, i32_ty)
                    };
                    let mut switch_regions: Vec<&[StructuredStmt]> =
                        Vec::with_capacity(cases.len() + 2);
                    for (_, case_body) in cases {
                        switch_regions.push(case_body.as_slice());
                    }
                    switch_regions.push(default.as_slice());
                    switch_regions.push(following);
                    self.prematerialize_shared_branch_insts(&switch_regions, &merge_phis);
                    let header_label = self.current_label;
                    let merge_label = self.emitter.alloc_id();
                    let default_label = self.emitter.alloc_id();
                    let case_labels: Vec<(i64, u32, &[StructuredStmt])> = cases
                        .iter()
                        .map(|(val, body)| (*val, self.emitter.alloc_id(), body.as_slice()))
                        .collect();

                    self.emitter.emit_fn_inst(
                        spirv::Op::SelectionMerge,
                        &[merge_label, spirv::SelectionControl::NONE.bits()],
                    );
                    let mut switch_ops = Vec::with_capacity(2 + case_labels.len() * 2);
                    switch_ops.push(sel_id);
                    switch_ops.push(default_label);
                    for (val, label, _) in &case_labels {
                        switch_ops.push((*val as i32).cast_unsigned());
                        switch_ops.push(*label);
                    }
                    self.emitter.emit_fn_inst(spirv::Op::Switch, &switch_ops);

                    let mut merge_preds: Vec<(u32, u32)> =
                        Vec::with_capacity(case_labels.len() + 1);
                    let default_tag = case_labels.len() as u32;

                    for (idx, (_, label, body)) in case_labels.into_iter().enumerate() {
                        let tag = idx as u32;
                        self.current_label = label;
                        self.current_block_preds = vec![(0, header_label)];
                        self.emitter.emit_fn_inst(spirv::Op::Label, &[label]);
                        if !self.emit_stmts(body) {
                            self.prematerialize_phi_inputs(&merge_phis, tag);
                            merge_preds.push((tag, self.current_label));
                            self.emitter.emit_fn_inst(spirv::Op::Branch, &[merge_label]);
                        }
                    }

                    self.current_label = default_label;
                    self.current_block_preds = vec![(0, header_label)];
                    self.emitter
                        .emit_fn_inst(spirv::Op::Label, &[default_label]);
                    if !self.emit_stmts(default) {
                        self.prematerialize_phi_inputs(&merge_phis, default_tag);
                        merge_preds.push((default_tag, self.current_label));
                        self.emitter.emit_fn_inst(spirv::Op::Branch, &[merge_label]);
                    }

                    self.current_label = merge_label;
                    self.current_block_preds = merge_preds;
                    self.emitter.emit_fn_inst(spirv::Op::Label, &[merge_label]);
                    if self.current_block_preds.is_empty() {
                        self.emitter.emit_fn_inst(spirv::Op::Unreachable, &[]);
                        return true;
                    }
                    self.emit_merge_phis(&merge_phis);
                }
                StructuredStmt::Loop {
                    body,
                    continuing,
                    unroll,
                    ..
                } => {
                    let header_phis = self.collect_loop_header_phis(body, continuing);
                    let mut merge_phis = self.collect_merge_phis(following);
                    merge_phis.retain(|id| !header_phis.contains(id));
                    // Pre-materialize pure loop-invariant operands shared across body/continuing/after
                    // in the dominating preheader before branching to the loop header.
                    self.prematerialize_shared_branch_insts(
                        &[body.as_slice(), continuing.as_slice(), following],
                        &merge_phis,
                    );
                    let preheader_label = self.current_label;
                    let header_label = self.emitter.alloc_id();
                    let body_label = self.emitter.alloc_id();
                    let continue_label = self.emitter.alloc_id();
                    let merge_label = self.emitter.alloc_id();
                    let loop_ctrl = if *unroll {
                        spirv::LoopControl::UNROLL.bits()
                    } else {
                        spirv::LoopControl::NONE.bits()
                    };

                    // Pre-materialize preheader inputs (tag 0) before branching to loop header.
                    let mut pending_header_phis: Vec<(InstId, u32, u32, u32)> = Vec::new();
                    let mut header_phi_backedges: Vec<(InstId, u32, u32)> = Vec::new();
                    for &phi_id in &header_phis {
                        if let InstData::Phi { incoming } = &self.func.insts[phi_id].data
                            && incoming.len() == 2
                            && let Some(&(init_inst, _)) =
                                incoming.iter().find(|&&(_, tag)| tag == 0)
                            && let Some(&(back_inst, _)) =
                                incoming.iter().find(|&&(_, tag)| tag == 1)
                            && !self.inst_ids.contains_key(&back_inst)
                        {
                            let inferred_ty = self.infer_phi_ty(phi_id);
                            let target_ty = match self.func.insts[back_inst].data {
                                InstData::IAdd32(_, _)
                                | InstData::ISub32(_, _)
                                | InstData::IMul32(_, _)
                                | InstData::IDiv32(_, _)
                                | InstData::IRem32(_, _)
                                | InstData::BitAnd32(_, _)
                                | InstData::BitOr32(_, _)
                                | InstData::BitXor32(_, _)
                                | InstData::BitNot32(_)
                                | InstData::Shl32(_, _)
                                | InstData::Shr32(_, _) => {
                                    let back_ir_ty = self.func.insts[back_inst].ir_ty.clone();
                                    if back_ir_ty.is_64bit() {
                                        self.emitter.emit_ir_type(&back_ir_ty)
                                    } else if self.cached_i64_ty() == Some(inferred_ty)
                                        || self.cached_u64_ty() == Some(inferred_ty)
                                    {
                                        inferred_ty
                                    } else if inferred_ty == self.i32_ty() {
                                        self.i32_ty()
                                    } else {
                                        self.u32_ty()
                                    }
                                }
                                InstData::IAdd64(_, _) | InstData::IMul64(_, _) => {
                                    if inferred_ty == self.i64_ty() {
                                        self.i64_ty()
                                    } else {
                                        self.u64_ty()
                                    }
                                }
                                InstData::FAdd32(_, _)
                                | InstData::FSub32(_, _)
                                | InstData::FMul32(_, _)
                                | InstData::FDiv32(_, _) => self.f32_ty(),
                                InstData::CastToU32(_) => self.u32_ty(),
                                InstData::CastI64ToU64(_) => self.u64_ty(),
                                _ => self.peek_value_ty(back_inst).unwrap_or(inferred_ty),
                            };
                            let raw_init = self.ensure_inst(init_inst);
                            let src_ty = self
                                .inst_types
                                .get(&init_inst)
                                .copied()
                                .unwrap_or(target_ty);
                            let init_spv = self.coerce_to_ty(raw_init, src_ty, target_ty);
                            let (back_spv, _) = *self
                                .preallocated_ids
                                .entry(back_inst)
                                .or_insert_with(|| (self.emitter.alloc_id(), target_ty));
                            pending_header_phis.push((phi_id, target_ty, init_spv, back_spv));
                            header_phi_backedges.push((back_inst, back_spv, target_ty));
                        }
                    }

                    self.emitter
                        .emit_fn_inst(spirv::Op::Branch, &[header_label]);

                    self.current_label = header_label;
                    self.current_block_preds = vec![(0, preheader_label), (1, continue_label)];
                    self.emitter.emit_fn_inst(spirv::Op::Label, &[header_label]);
                    let mut header_phi_offsets: Vec<usize> =
                        Vec::with_capacity(pending_header_phis.len());
                    for (phi_id, target_ty, init_spv, back_spv) in pending_header_phis {
                        let phi_word_offset = self.emitter.functions.len();
                        let res = self.emitter.alloc_id();
                        self.emitter.emit_fn_inst(
                            spirv::Op::Phi,
                            &[
                                target_ty,
                                res,
                                init_spv,
                                preheader_label,
                                back_spv,
                                continue_label,
                            ],
                        );
                        header_phi_offsets.push(phi_word_offset);
                        self.inst_ids.insert(phi_id, res);
                        self.inst_types.insert(phi_id, target_ty);
                    }
                    self.emitter.emit_fn_inst(
                        spirv::Op::LoopMerge,
                        &[merge_label, continue_label, loop_ctrl],
                    );
                    self.emitter.emit_fn_inst(spirv::Op::Branch, &[body_label]);

                    self.loop_stack.push(LoopFrame {
                        merge_label,
                        continue_label,
                        merge_phis: merge_phis.clone(),
                        next_break_tag: 0,
                        break_preds: Vec::new(),
                    });

                    self.current_label = body_label;
                    self.current_block_preds = vec![(0, header_label)];
                    self.emitter.emit_fn_inst(spirv::Op::Label, &[body_label]);
                    if !self.emit_stmts(body) {
                        self.emitter
                            .emit_fn_inst(spirv::Op::Branch, &[continue_label]);
                    }

                    self.current_label = continue_label;
                    self.current_block_preds = vec![(0, body_label)];
                    self.emitter
                        .emit_fn_inst(spirv::Op::Label, &[continue_label]);
                    if self.emit_stmts(continuing) {
                        for (back_inst, _, _) in header_phi_backedges {
                            self.preallocated_ids.shift_remove(&back_inst);
                        }
                        // No back-edge reaches header_label: shrink each 7-word header OpPhi
                        // (2 incoming pairs) to a 5-word OpPhi (preheader only) in reverse order.
                        for &phi_offset in header_phi_offsets.iter().rev() {
                            self.emitter.functions[phi_offset] =
                                (5 << 16) | (spirv::Op::Phi as u32);
                            self.emitter
                                .functions
                                .drain((phi_offset + 5)..(phi_offset + 7));
                        }
                    } else {
                        for (back_inst, back_spv, target_ty) in header_phi_backedges {
                            if self.preallocated_ids.shift_remove(&back_inst).is_some() {
                                let raw = self.ensure_inst(back_inst);
                                let src_ty = self
                                    .inst_types
                                    .get(&back_inst)
                                    .copied()
                                    .unwrap_or(target_ty);
                                let coerced = self.coerce_to_ty(raw, src_ty, target_ty);
                                self.emitter.emit_fn_inst(
                                    spirv::Op::CopyObject,
                                    &[target_ty, back_spv, coerced],
                                );
                            }
                        }
                        // Route header OpPhi back-edge predecessor to the actual latch block
                        // (which differs from continue_label when continuing has nested control flow).
                        let latch_label = self.current_label;
                        for &phi_offset in &header_phi_offsets {
                            self.emitter.functions[phi_offset + 6] = latch_label;
                        }
                        self.emitter
                            .emit_fn_inst(spirv::Op::Branch, &[header_label]);
                    }

                    let break_preds = self
                        .loop_stack
                        .pop()
                        .map_or_else(Vec::new, |frame| frame.break_preds);

                    self.current_label = merge_label;
                    self.current_block_preds = break_preds;
                    self.emitter.emit_fn_inst(spirv::Op::Label, &[merge_label]);
                    if self.current_block_preds.is_empty() {
                        self.emitter.emit_fn_inst(spirv::Op::Unreachable, &[]);
                        return true;
                    }
                    self.emit_merge_phis(&merge_phis);
                }
                StructuredStmt::Break => {
                    if let Some(frame) = self.loop_stack.last_mut() {
                        let merge_label = frame.merge_label;
                        let break_tag = frame.next_break_tag;
                        frame.next_break_tag += 1;
                        let loop_merge_phis = frame.merge_phis.clone();
                        self.prematerialize_phi_inputs(&loop_merge_phis, break_tag);
                        let break_label = self.current_label;
                        if let Some(frame) = self.loop_stack.last_mut() {
                            frame.break_preds.push((break_tag, break_label));
                        }
                        self.emitter.emit_fn_inst(spirv::Op::Branch, &[merge_label]);
                        terminated = true;
                    }
                }
                StructuredStmt::Continue => {
                    if let Some(frame) = self.loop_stack.last() {
                        let continue_label = frame.continue_label;
                        self.emitter
                            .emit_fn_inst(spirv::Op::Branch, &[continue_label]);
                        terminated = true;
                    }
                }
                StructuredStmt::Return(opt_val) => {
                    if self.fn_ret_ty == self.void_ty {
                        if let Some(val_id) = opt_val {
                            let _ = self.ensure_inst(*val_id);
                        }
                        self.emitter.emit_fn_inst(spirv::Op::Return, &[]);
                    } else {
                        let ret_val = if let Some(val_id) = opt_val {
                            let raw = self.ensure_inst(*val_id);
                            let src_ty = self
                                .inst_types
                                .get(val_id)
                                .copied()
                                .unwrap_or(self.fn_ret_ty);
                            self.coerce_to_ty(raw, src_ty, self.fn_ret_ty)
                        } else {
                            self.emit_zero_const(self.fn_ret_ty)
                        };
                        self.emitter
                            .emit_fn_inst(spirv::Op::ReturnValue, &[ret_val]);
                    }
                    terminated = true;
                }
            }
        }
        terminated
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AtomicOpKind {
    Add,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubgroupArithKind {
    Add,
    Mul,
    Min,
    Max,
}

const fn map_atomic_scope(scope: AtomicScope) -> u32 {
    match scope {
        AtomicScope::Device => spirv::Scope::Device as u32,
        AtomicScope::Workgroup => spirv::Scope::Workgroup as u32,
        AtomicScope::Subgroup => spirv::Scope::Subgroup as u32,
    }
}

const fn map_atomic_semantics_sc(order: AtomicOrder, sc: spirv::StorageClass) -> u32 {
    let storage = match sc {
        spirv::StorageClass::Workgroup => spirv::MemorySemantics::WORKGROUP_MEMORY.bits(),
        spirv::StorageClass::Image => spirv::MemorySemantics::IMAGE_MEMORY.bits(),
        _ => spirv::MemorySemantics::UNIFORM_MEMORY.bits(),
    };
    let ord = match order {
        AtomicOrder::Relaxed => spirv::MemorySemantics::empty().bits(),
        AtomicOrder::Acquire => spirv::MemorySemantics::ACQUIRE.bits(),
        AtomicOrder::Release => spirv::MemorySemantics::RELEASE.bits(),
        AtomicOrder::AcqRel => spirv::MemorySemantics::ACQUIRE_RELEASE.bits(),
    };
    if ord == 0 { 0 } else { ord | storage }
}

#[allow(clippy::too_many_arguments)]
fn emit_function_body(
    emitter: &mut SpirvEmitter,
    func: &IrFunction,
    fn_info: &EmittedFunctionInfo,
    emitted_functions: &IndexMap<String, EmittedFunctionInfo>,
    descriptors: &[EmittedDescriptor],
    workgroup_vars: &[EmittedWorkgroupVar],
    stage_io_vars: &[EmittedStageIoVar],
    stage_vars: &[EmittedStageVar],
    builtins: &IndexMap<(BuiltinKind, bool), EmittedBuiltinVar>,
    push_constant: Option<&EmittedPushConstant>,
    default_sampler_var: Option<u32>,
    void_ty: u32,
    uses_bda: bool,
    debug_ctx: Option<(u32, u32, u32)>,
) -> u32 {
    let fn_id = fn_info.fn_id;
    let entry_label = emitter.alloc_id();
    emitter.emit_fn_inst(
        spirv::Op::Function,
        &[
            fn_info.ret_ty,
            fn_id,
            spirv::FunctionControl::NONE.bits(),
            fn_info.fn_ty,
        ],
    );

    let mut param_ids = Vec::with_capacity(fn_info.param_tys.len());
    if func.stage.is_none() {
        for (idx, p) in func.params.iter().enumerate() {
            if let Some(&(spv_p_ty, is_ptr)) = fn_info.param_tys.get(idx) {
                let param_id = emitter.alloc_id();
                emitter.emit_fn_inst(spirv::Op::FunctionParameter, &[spv_p_ty, param_id]);
                if !p.name.is_empty() {
                    emitter.emit_name(param_id, p.name.as_str());
                }
                param_ids.push((param_id, spv_p_ty, is_ptr, p.ty.clone()));
            }
        }
    }

    emitter.emit_fn_inst(spirv::Op::Label, &[entry_label]);

    // SPIR-V spec 2.16.1: all StorageClass::Function OpVariable instructions must appear
    // in the function entry block before any non-variable instruction.
    let mut local_vars = Vec::with_capacity(func.local_vars.len());
    for lv in &func.local_vars {
        let val_ty = emitter.emit_ir_type(&lv.ty);
        let ptr_ty = emitter.emit_type_pointer(spirv::StorageClass::Function, val_ty);
        let var_id = emitter.alloc_id();
        emitter.emit_fn_inst(
            spirv::Op::Variable,
            &[ptr_ty, var_id, spirv::StorageClass::Function as u32],
        );
        if !lv.name.is_empty() {
            emitter.emit_name(var_id, lv.name.as_str());
        }
        local_vars.push(EmittedLocalVar {
            var_id,
            ptr_ty,
            val_ty,
            ir_ty: lv.ty.clone(),
        });
    }

    let mut rq_ids: IndexSet<u32> = IndexSet::new();
    for (_, inst) in func.insts.iter() {
        match inst.data {
            InstData::RayQueryInitialize { ray_query_id, .. }
            | InstData::RayQueryProceed(ray_query_id)
            | InstData::RayQueryCommittedType(ray_query_id)
            | InstData::RayQueryCandidateType(ray_query_id)
            | InstData::RayQueryConfirmIntersection(ray_query_id)
            | InstData::RayQueryTerminate(ray_query_id)
            | InstData::RayQueryGenerateIntersection { ray_query_id, .. }
            | InstData::RayQueryCommittedT(ray_query_id)
            | InstData::RayQueryCandidateT(ray_query_id)
            | InstData::RayQueryCommittedInstanceId(ray_query_id)
            | InstData::RayQueryCommittedInstanceCustomIndex(ray_query_id)
            | InstData::RayQueryCommittedPrimitiveIndex(ray_query_id)
            | InstData::RayQueryCommittedGeometryIndex(ray_query_id)
            | InstData::RayQueryCommittedObjectToWorld(ray_query_id)
            | InstData::RayQueryCommittedWorldToObject(ray_query_id)
            | InstData::RayQueryCommittedFrontFace(ray_query_id)
            | InstData::RayQueryCommittedBarycentrics(ray_query_id)
            | InstData::RayQueryTriangleVertexPositions { ray_query_id, .. } => {
                rq_ids.insert(ray_query_id);
            }
            _ => {}
        }
    }
    let use_ser_ext = func
        .required_caps
        .contains(HardwareCaps::SHADER_INVOCATION_REORDER_EXT);
    let mut hit_object_ids: IndexSet<u32> = IndexSet::new();
    for (_, inst) in func.insts.iter() {
        match inst.data {
            InstData::HitObjectTraceRay { hit_object_id, .. }
            | InstData::HitObjectRecordEmpty(hit_object_id)
            | InstData::HitObjectIsHit(hit_object_id)
            | InstData::HitObjectGetRayTMax(hit_object_id)
            | InstData::HitObjectGetSbtRecordIndex(hit_object_id)
            | InstData::HitObjectExecuteShader { hit_object_id, .. }
            | InstData::ReorderThreadWithHitObject { hit_object_id, .. } => {
                hit_object_ids.insert(hit_object_id);
            }
            _ => {}
        }
    }
    let mut hit_object_vars: IndexMap<u32, u32> = IndexMap::new();
    if !hit_object_ids.is_empty() {
        let ho_ty = emitter.emit_type_hit_object(use_ser_ext);
        let ho_ptr_ty = emitter.emit_type_pointer(spirv::StorageClass::Function, ho_ty);
        for ho_id in hit_object_ids {
            let var_id = emitter.alloc_id();
            emitter.emit_fn_inst(
                spirv::Op::Variable,
                &[ho_ptr_ty, var_id, spirv::StorageClass::Function as u32],
            );
            hit_object_vars.insert(ho_id, var_id);
        }
    }

    let mut ray_query_vars: IndexMap<u32, u32> = IndexMap::new();
    if !rq_ids.is_empty() {
        let rq_ty = emitter.emit_type_ray_query_khr();
        let rq_ptr_ty = emitter.emit_type_pointer(spirv::StorageClass::Function, rq_ty);
        for rq_id in rq_ids {
            let rq_var = emitter.alloc_id();
            emitter.emit_fn_inst(
                spirv::Op::Variable,
                &[rq_ptr_ty, rq_var, spirv::StorageClass::Function as u32],
            );
            ray_query_vars.insert(rq_id, rq_var);
        }
    }

    let mut ctx = FnLowerCtx {
        emitter,
        func,
        fn_ret_ty: fn_info.ret_ty,
        void_ty,
        emitted_functions,
        descriptors,
        workgroup_vars,
        stage_io_vars,
        stage_vars,
        local_vars,
        param_ids,
        builtins,
        push_constant,
        default_sampler_var,
        ray_query_vars,
        hit_object_vars,
        use_ser_ext,
        uses_bda,
        inst_ids: IndexMap::new(),
        preallocated_ids: IndexMap::new(),
        inst_types: IndexMap::new(),
        inst_pointee_types: IndexMap::new(),
        inst_ptr_storage: IndexMap::new(),
        inst_ptr_ir_pointee: IndexMap::new(),
        bda_ptr_insts: IndexSet::new(),
        workgroup_ptr_insts: IndexSet::new(),
        phi_target_types: IndexMap::new(),
        phi_coerced_inputs: IndexMap::new(),
        debug_ctx: debug_ctx.map(|(ext_id, debug_source, _)| (ext_id, debug_source)),
        decorated_non_uniform: IndexSet::new(),
        loop_stack: Vec::new(),
        current_label: entry_label,
        entry_label,
        current_block_preds: Vec::new(),
    };

    if let Some((ext_id, _, debug_fn)) = debug_ctx {
        let def_res = ctx.emitter.alloc_id();
        ctx.emitter.emit_fn_inst(
            spirv::Op::ExtInst,
            &[void_ty, def_res, ext_id, 101, debug_fn, fn_id], // DebugFunctionDefinition
        );
    }

    let terminated = ctx.emit_stmts(&func.body);
    if !terminated {
        if fn_info.ret_ty == void_ty {
            ctx.emitter.emit_fn_inst(spirv::Op::Return, &[]);
        } else {
            let zero = ctx.emit_zero_const(fn_info.ret_ty);
            ctx.emitter.emit_fn_inst(spirv::Op::ReturnValue, &[zero]);
        }
    }
    ctx.emitter.emit_fn_inst(spirv::Op::FunctionEnd, &[]);
    fn_id
}

const fn is_const_inst_data(data: &InstData) -> bool {
    matches!(
        data,
        InstData::ConstBool(_)
            | InstData::ConstI32(_)
            | InstData::ConstU32(_)
            | InstData::ConstF32(_)
            | InstData::ConstF16(_)
            | InstData::ConstI64(_)
            | InstData::ConstU64(_)
            | InstData::SpecConstant { .. }
            | InstData::SpecConstantOp { .. }
    )
}

fn push_inst(buf: &mut Vec<u32>, op: spirv::Op, operands: &[u32]) {
    push_raw_inst(buf, op as u16, operands);
}

fn push_raw_inst(buf: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
    let word_count = (1 + operands.len()) as u32;
    let opcode_word = (word_count << 16) | u32::from(opcode);
    buf.push(opcode_word);
    buf.extend_from_slice(operands);
}

fn map_ir_storage_class_raw(sc: IrStorageClass) -> u32 {
    match sc {
        IrStorageClass::Function => spirv::StorageClass::Function as u32,
        IrStorageClass::Workgroup => spirv::StorageClass::Workgroup as u32,
        IrStorageClass::Uniform => spirv::StorageClass::Uniform as u32,
        IrStorageClass::StorageBuffer => spirv::StorageClass::StorageBuffer as u32,
        IrStorageClass::PhysicalStorageBuffer => spirv::StorageClass::PhysicalStorageBuffer as u32,
        IrStorageClass::Image => spirv::StorageClass::Image as u32,
        IrStorageClass::RayPayloadKhr => 5338,
        IrStorageClass::IncomingRayPayloadKhr => 5342,
        IrStorageClass::HitAttributeKhr => 5339,
        IrStorageClass::CallableDataKhr => 5328,
        IrStorageClass::IncomingCallableDataKhr => 5329,
        IrStorageClass::ShaderRecordBufferKhr => 5343,
        IrStorageClass::TaskPayloadWorkgroupExt => 5402,
    }
}

fn encode_literal_string(s: &str) -> Vec<u32> {
    let bytes = s.as_bytes();
    let total_bytes = bytes.len() + 1; // NUL terminator
    let word_count = total_bytes.div_ceil(4);
    let mut words = vec![0u32; word_count];
    for (idx, &b) in bytes.iter().enumerate() {
        words[idx / 4] |= u32::from(b) << ((idx % 4) * 8);
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;
    use vsl_ir::{
        IrLocalVar, IrParam, IrPass, IrStageIoVar, LowerBdaOffsetsPass, MemoryAccess,
        PropagateNonUniformPass,
    };
    use vsl_mc::BINDLESS_IMAGE_COUNT;

    /// Decodes a SPIR-V word stream (skipping the 5-word header) into `(opcode, operands)` pairs.
    fn decode_instructions(words: &[u32]) -> Vec<(u32, &[u32])> {
        let mut out = Vec::new();
        let mut pc = 5;
        while pc < words.len() {
            let header = words[pc];
            let count = (header >> 16) as usize;
            let opcode = header & 0xFFFF;
            assert!(count >= 1, "invalid zero-length SPIR-V instruction at {pc}");
            assert!(
                pc + count <= words.len(),
                "truncated SPIR-V instruction at {pc}"
            );
            out.push((opcode, &words[pc + 1..pc + count]));
            pc += count;
        }
        out
    }

    fn validate_with_spirv_val_if_available(words: &[u32], tag: &str) {
        let spirv_val_available = std::process::Command::new("spirv-val")
            .arg("--version")
            .output()
            .is_ok_and(|out| out.status.success());
        let require_val = std::env::var("VSL_REQUIRE_SPIRV_VAL")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            || std::env::var("CI").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        if !spirv_val_available {
            assert!(
                !require_val,
                "spirv-val is required (VSL_REQUIRE_SPIRV_VAL or CI is set) but was not found on PATH while validating '{tag}'"
            );
            return;
        }
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let path = std::env::temp_dir().join(format!("vsl_spirv_{tag}_{}.spv", std::process::id()));
        std::fs::write(&path, &bytes).expect("failed to write temporary SPIR-V binary");
        let output = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&path)
            .output()
            .expect("failed to execute spirv-val");
        let _ = std::fs::remove_file(&path);
        assert!(
            output.status.success(),
            "spirv-val ({tag}) failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn emits_spirv_1_6_header_and_vulkan_memory_model() {
        let mut func = IrFunction::new("trace_glossy_reflections");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.uses_bda = true;
        func.required_caps = HardwareCaps::RAY_QUERY;

        let module = IrModule {
            name: "pack::reflections".into(),
            functions: vec![func],
        };

        let words = emit_module_spirv(&module);
        assert_eq!(words[0], spirv::MAGIC_NUMBER);
        assert_eq!(words[1], SPIRV_VERSION_1_6);
        assert!(words.len() > 10);
    }

    #[test]
    fn emits_descriptor_decorations_push_constant_offsets_and_debug_section_order() {
        let mut func = IrFunction::new("shade_gbuffer");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.descriptors.push(IrDescriptorVar {
            name: "instance_meta".into(),
            set: 1,
            binding: 1,
            kind: DescriptorKind::StorageBuffer,
            dim: None,
            sample_scalar: None,
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });
        func.descriptors.push(IrDescriptorVar {
            name: "bindless_textures".into(),
            set: 2,
            binding: 0,
            kind: DescriptorKind::SampledImage,
            dim: Some(TextureDim::Dim2DArray),
            sample_scalar: Some(SampleScalar::F32),
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: true,
            count: None,
            struct_layout: None,
        });
        func.descriptors.push(IrDescriptorVar {
            name: "out_moments".into(),
            set: 0,
            binding: 2,
            kind: DescriptorKind::StorageImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: Some(SampleScalar::F32),
            image_format: Some(ImageFormat::Rg16f),
            access: DescriptorAccess::WriteOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });
        func.workgroup_vars.push(IrWorkgroupVar {
            name: "shared_tiles".into(),
            element_is_u32: true,
            array_len: Some(64),
            ty: IrType::Void,
        });
        func.push_constants = Some(IrPushConstantBlock {
            name: "PassPush".into(),
            fields: vec![
                vsl_ir::IrPushConstantField {
                    name: "frame_index".into(),
                    byte_offset: 0,
                    byte_size: 4,
                    scalar: ScalarIrTy::U32,
                },
                vsl_ir::IrPushConstantField {
                    name: "exposure".into(),
                    byte_offset: 4,
                    byte_size: 4,
                    scalar: ScalarIrTy::F32,
                },
                vsl_ir::IrPushConstantField {
                    name: "camera_pos".into(),
                    byte_offset: 16,
                    byte_size: 16,
                    scalar: ScalarIrTy::F32,
                },
                vsl_ir::IrPushConstantField {
                    name: "view_proj".into(),
                    byte_offset: 32,
                    byte_size: 64,
                    scalar: ScalarIrTy::F32,
                },
            ],
            total_size: 96,
        });

        let module = IrModule {
            name: "pack::gbuffer".into(),
            functions: vec![func],
        };

        let words = emit_module_spirv(&module);
        assert_eq!(words[0], spirv::MAGIC_NUMBER);
        assert_eq!(words[1], SPIRV_VERSION_1_6);

        let insts = decode_instructions(&words);

        let has_ext_formats = insts.iter().any(|&(op, ops)| {
            op == spirv::Op::Capability as u32
                && ops == [spirv::Capability::StorageImageExtendedFormats as u32]
        });
        let has_desc_set_1 = insts.iter().any(|&(op, ops)| {
            op == spirv::Op::Decorate as u32
                && ops.len() == 3
                && ops[1] == spirv::Decoration::DescriptorSet as u32
                && ops[2] == 1
        });
        let has_binding_1 = insts.iter().any(|&(op, ops)| {
            op == spirv::Op::Decorate as u32
                && ops.len() == 3
                && ops[1] == spirv::Decoration::Binding as u32
                && ops[2] == 1
        });
        let has_non_writable = insts.iter().any(|&(op, ops)| {
            op == spirv::Op::Decorate as u32
                && ops.len() == 2
                && ops[1] == spirv::Decoration::NonWritable as u32
        });
        let has_non_readable = insts.iter().any(|&(op, ops)| {
            op == spirv::Op::Decorate as u32
                && ops.len() == 2
                && ops[1] == spirv::Decoration::NonReadable as u32
        });
        let has_workgroup_var = insts.iter().any(|&(op, ops)| {
            op == spirv::Op::Variable as u32
                && ops.len() == 3
                && ops[2] == spirv::StorageClass::Workgroup as u32
        });
        let has_arrayed_2d = insts.iter().any(|&(op, ops)| {
            op == spirv::Op::TypeImage as u32
                && ops.len() >= 8
                && ops[2] == spirv::Dim::Dim2D as u32
                && ops[4] == 1
        });
        let has_pc_offset_16 = insts.iter().any(|&(op, ops)| {
            op == spirv::Op::MemberDecorate as u32
                && ops.len() == 4
                && ops[1] == 2
                && ops[2] == spirv::Decoration::Offset as u32
                && ops[3] == 16
        });

        assert!(has_ext_formats, "missing StorageImageExtendedFormats");
        assert!(has_desc_set_1, "missing OpDecorate DescriptorSet 1");
        assert!(has_binding_1, "missing OpDecorate Binding 1");
        assert!(has_non_writable, "missing OpDecorate NonWritable");
        assert!(has_non_readable, "missing OpDecorate NonReadable");
        assert!(has_workgroup_var, "missing Workgroup OpVariable");
        assert!(has_arrayed_2d, "missing arrayed Dim2D OpTypeImage");
        assert!(has_pc_offset_16, "missing OpMemberDecorate Offset 16");

        // Verify SPIR-V logical layout: ExecutionModes < DebugNames < Decorations.
        let last_exec_mode = insts
            .iter()
            .rposition(|&(op, _)| op == spirv::Op::ExecutionMode as u32)
            .expect("expected OpExecutionMode");
        let first_name = insts
            .iter()
            .position(|&(op, _)| op == spirv::Op::Name as u32)
            .expect("expected OpName");
        let last_name = insts
            .iter()
            .rposition(|&(op, _)| {
                op == spirv::Op::Name as u32 || op == spirv::Op::MemberName as u32
            })
            .expect("expected OpName/OpMemberName");
        let first_decorate = insts
            .iter()
            .position(|&(op, _)| {
                op == spirv::Op::Decorate as u32 || op == spirv::Op::MemberDecorate as u32
            })
            .expect("expected OpDecorate");

        assert!(last_exec_mode < first_name);
        assert!(last_name < first_decorate);
    }

    #[test]
    fn emits_structured_control_flow_bda_phi_and_nonuniform_decorations() {
        let mut func = IrFunction::new("bda_and_cfg");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.descriptors.push(IrDescriptorVar {
            name: "bindless_tex".into(),
            set: 2,
            binding: 0,
            kind: DescriptorKind::SampledImage,
            dim: Some(TextureDim::DimCube),
            sample_scalar: Some(SampleScalar::F32),
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: true,
            count: None,
            struct_layout: None,
        });

        let cond = func.emit(None, InstData::ConstBool(true));
        let base_addr = func.emit(None, InstData::ConstU64(0x2000));
        let base_ptr = func.emit(
            None,
            InstData::ConvertUToPtr {
                addr_u64: base_addr,
                align: 16,
            },
        );
        let idx_i64 = func.emit(None, InstData::ConstI64(2));
        let bda_off = func.emit(
            None,
            InstData::BdaOffset {
                base_ptr,
                index_i64: idx_i64,
                stride_bytes: 16,
                align: 16,
            },
        );
        let bda_val = func.emit(
            None,
            InstData::BdaLoad {
                ptr: bda_off,
                access: MemoryAccess {
                    align: 16,
                    non_writable: false,
                    restrict: false,
                },
            },
        );
        let one = func.emit(None, InstData::ConstU32(1));
        let atomic = func.emit(
            None,
            InstData::AtomicAdd {
                ptr: bda_off,
                value: one,
                scope: AtomicScope::Device,
                order: AtomicOrder::AcqRel,
            },
        );
        let barrier = func.emit(None, InstData::WorkgroupBarrier);

        let tex_idx = func.emit(None, InstData::ConstU32(5));
        let nu_idx = func.emit(None, InstData::NonUniform(tex_idx));
        let chain = func.emit(
            None,
            InstData::ResourceAccessChain {
                descriptor_index: 0,
                index: nu_idx,
            },
        );
        let load = func.emit(None, InstData::ResourceLoad { ptr: chain });

        let then_val = func.emit(None, InstData::ConstU32(10));
        let else_val = func.emit(None, InstData::ConstU32(20));
        let merged_phi = func.emit(
            None,
            InstData::Phi {
                incoming: vec![(then_val, 0), (else_val, 1)],
            },
        );

        func.body.push(StructuredStmt::If {
            cond,
            then_body: vec![
                StructuredStmt::Eval(bda_val),
                StructuredStmt::Eval(atomic),
                StructuredStmt::Eval(barrier),
                StructuredStmt::Eval(load),
                StructuredStmt::Eval(then_val),
            ],
            else_body: vec![StructuredStmt::Eval(else_val)],
        });
        func.body.push(StructuredStmt::Eval(merged_phi));

        LowerBdaOffsetsPass.run(&mut func);
        PropagateNonUniformPass.run(&mut func);

        let module = IrModule {
            name: "pack::bda_cfg".into(),
            functions: vec![func],
        };

        let words = emit_module_spirv(&module);
        let insts = decode_instructions(&words);

        let has_device_scope_cap = insts.iter().any(|&(op, ops)| {
            op == spirv::Op::Capability as u32
                && ops == [spirv::Capability::VulkanMemoryModelDeviceScope as u32]
        });
        let has_selection_merge = insts
            .iter()
            .any(|&(op, _)| op == spirv::Op::SelectionMerge as u32);
        let has_convert_ptr_to_u = insts
            .iter()
            .any(|&(op, _)| op == spirv::Op::ConvertPtrToU as u32);
        let has_convert_u_to_ptr = insts
            .iter()
            .any(|&(op, _)| op == spirv::Op::ConvertUToPtr as u32);
        let has_control_barrier = insts
            .iter()
            .any(|&(op, _)| op == spirv::Op::ControlBarrier as u32);
        let has_non_uniform = insts.iter().any(|&(op, ops)| {
            op == spirv::Op::Decorate as u32
                && ops.len() == 2
                && ops[1] == spirv::Decoration::NonUniform as u32
        });
        let has_vulkan_load = insts.iter().any(|&(op, ops)| {
            let expected = (spirv::MemoryAccess::ALIGNED
                | spirv::MemoryAccess::MAKE_POINTER_VISIBLE
                | spirv::MemoryAccess::NON_PRIVATE_POINTER)
                .bits();
            op == spirv::Op::Load as u32 && ops.len() == 6 && ops[3] == expected
        });

        let phi_pos = insts
            .iter()
            .position(|&(op, _)| op == spirv::Op::Phi as u32)
            .expect("missing OpPhi");
        assert_eq!(
            insts[phi_pos - 1].0,
            spirv::Op::Label as u32,
            "OpPhi must immediately follow OpLabel"
        );
        let merge_label = insts[phi_pos - 1].1[0];
        let phi_ops = insts[phi_pos].1;
        assert_eq!(phi_ops.len(), 6);
        assert_ne!(phi_ops[3], merge_label);
        assert_ne!(phi_ops[5], merge_label);

        assert!(has_device_scope_cap, "missing VulkanMemoryModelDeviceScope");
        assert!(has_selection_merge, "missing OpSelectionMerge");
        assert!(has_convert_ptr_to_u, "missing OpConvertPtrToU");
        assert!(has_convert_u_to_ptr, "missing OpConvertUToPtr");
        assert!(has_control_barrier, "missing OpControlBarrier");
        assert!(has_non_uniform, "missing OpDecorate NonUniform");
        assert!(has_vulkan_load, "missing Vulkan memory-model BDA OpLoad");
    }

    #[test]
    fn deduplicates_descriptors_workgroup_vars_push_constants_and_emits_fragment_origin_upper_left()
    {
        let shared_desc = IrDescriptorVar {
            name: "scene_ubo".into(),
            set: 1,
            binding: 0,
            kind: DescriptorKind::UniformBuffer,
            dim: None,
            sample_scalar: None,
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        };
        let shared_wg = IrWorkgroupVar {
            name: "shared_cache".into(),
            element_is_u32: true,
            array_len: Some(16),
            ty: IrType::Void,
        };
        let shared_pc = IrPushConstantBlock {
            name: "FramePush".into(),
            fields: vec![vsl_ir::IrPushConstantField {
                name: "frame_index".into(),
                byte_offset: 0,
                byte_size: 4,
                scalar: ScalarIrTy::U32,
            }],
            total_size: 4,
        };

        let mut vert = IrFunction::new("vs_main");
        vert.stage = Some(ShaderStage::Vertex);
        vert.descriptors.push(shared_desc.clone());
        vert.workgroup_vars.push(shared_wg.clone());
        vert.push_constants = Some(shared_pc.clone());

        let mut frag = IrFunction::new("fs_main");
        frag.stage = Some(ShaderStage::Fragment);
        frag.descriptors.push(shared_desc.clone());
        frag.workgroup_vars.push(shared_wg.clone());
        frag.push_constants = Some(shared_pc.clone());

        let mut comp_a = IrFunction::new("cs_a");
        comp_a.stage = Some(ShaderStage::Compute);
        comp_a.workgroup_size = Some([8, 8, 1]);
        comp_a.descriptors.push(shared_desc.clone());
        comp_a.workgroup_vars.push(shared_wg.clone());
        comp_a.push_constants = Some(shared_pc.clone());

        let mut comp_b = IrFunction::new("cs_b");
        comp_b.stage = Some(ShaderStage::Compute);
        comp_b.workgroup_size = Some([8, 8, 1]);
        comp_b.descriptors.push(shared_desc);
        comp_b.workgroup_vars.push(shared_wg);
        comp_b.push_constants = Some(shared_pc);

        let module = IrModule {
            name: "pack::multi_entry".into(),
            functions: vec![vert, frag, comp_a, comp_b],
        };

        let words = emit_module_spirv(&module);
        let insts = decode_instructions(&words);

        let ubo_vars = insts
            .iter()
            .filter(|&&(op, ops)| {
                op == spirv::Op::Variable as u32
                    && ops.len() == 3
                    && ops[2] == spirv::StorageClass::Uniform as u32
            })
            .count();
        let wg_var_ids: Vec<u32> = insts
            .iter()
            .filter(|&&(op, ops)| {
                op == spirv::Op::Variable as u32
                    && ops.len() == 3
                    && ops[2] == spirv::StorageClass::Workgroup as u32
            })
            .map(|&(_, ops)| ops[1])
            .collect();
        let pc_vars = insts
            .iter()
            .filter(|&&(op, ops)| {
                op == spirv::Op::Variable as u32
                    && ops.len() == 3
                    && ops[2] == spirv::StorageClass::PushConstant as u32
            })
            .count();
        let has_origin_upper_left = insts.iter().any(|&(op, ops)| {
            op == spirv::Op::ExecutionMode as u32
                && ops.len() == 2
                && ops[1] == spirv::ExecutionMode::OriginUpperLeft as u32
        });

        assert_eq!(ubo_vars, 1, "shared (set, binding) must be deduplicated");
        assert_eq!(
            wg_var_ids.len(),
            1,
            "shared workgroup var must be deduplicated across compute entry points"
        );
        assert_eq!(pc_vars, 1, "shared push constant must be deduplicated");
        assert!(
            has_origin_upper_left,
            "Fragment stage must emit OriginUpperLeft"
        );

        let wg_var_id = wg_var_ids[0];
        for &(op, ops) in &insts {
            if op != spirv::Op::EntryPoint as u32 {
                continue;
            }
            let model = ops[0];
            let interface = &ops[4..];
            if model == spirv::ExecutionModel::GLCompute as u32 {
                assert!(
                    interface.contains(&wg_var_id),
                    "Compute entry point must include Workgroup variable in interface"
                );
            } else {
                assert!(
                    !interface.contains(&wg_var_id),
                    "Vertex/Fragment entry points must never attach Workgroup variables"
                );
            }
        }
    }

    #[test]
    fn emits_resource_access_chain_for_ssbo_unsized_and_single_image_and_ucmp_lt() {
        let mut func = IrFunction::new("resource_chain_and_cmp");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.descriptors.push(IrDescriptorVar {
            name: "ssbo_data".into(),
            set: 0,
            binding: 0,
            kind: DescriptorKind::StorageBuffer,
            dim: None,
            sample_scalar: None,
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });
        func.descriptors.push(IrDescriptorVar {
            name: "single_tex".into(),
            set: 0,
            binding: 1,
            kind: DescriptorKind::SampledImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: Some(SampleScalar::F32),
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });

        let idx = func.emit(None, InstData::ConstU32(2));
        let limit = func.emit(None, InstData::ConstU32(8));
        let ucmp = func.emit(None, InstData::UCmpLt(idx, limit));
        let ssbo_ptr = func.emit(
            None,
            InstData::ResourceAccessChain {
                descriptor_index: 0,
                index: idx,
            },
        );
        let ssbo_load = func.emit(None, InstData::ResourceLoad { ptr: ssbo_ptr });
        let single_ptr = func.emit(
            None,
            InstData::ResourceAccessChain {
                descriptor_index: 1,
                index: idx,
            },
        );
        let single_load = func.emit(None, InstData::ResourceLoad { ptr: single_ptr });

        func.body.push(StructuredStmt::Eval(ucmp));
        func.body.push(StructuredStmt::Eval(ssbo_load));
        func.body.push(StructuredStmt::Eval(single_load));

        let module = IrModule {
            name: "pack::resource_test".into(),
            functions: vec![func],
        };
        let words = emit_module_spirv(&module);
        let insts = decode_instructions(&words);

        let has_uless_than = insts
            .iter()
            .any(|&(op, _)| op == spirv::Op::ULessThan as u32);
        let ssbo_access_chain = insts
            .iter()
            .filter(|&&(op, ops)| op == spirv::Op::AccessChain as u32 && ops.len() == 5)
            .count();

        assert!(has_uless_than, "UCmpLt must emit OpULessThan");
        assert_eq!(
            ssbo_access_chain, 1,
            "StorageBuffer ResourceAccessChain must emit 2-index OpAccessChain [0, idx]"
        );
    }

    #[test]
    fn validates_emitted_spirv_with_spirv_val_when_available() {
        let mut func = IrFunction::new("validate_compute");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.descriptors.push(IrDescriptorVar {
            name: "bindless_2d".into(),
            set: 2,
            binding: 0,
            kind: DescriptorKind::SampledImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: Some(SampleScalar::F32),
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: true,
            count: None,
            struct_layout: None,
        });
        func.descriptors.push(IrDescriptorVar {
            name: "out_rg16f".into(),
            set: 0,
            binding: 0,
            kind: DescriptorKind::StorageImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: Some(SampleScalar::F32),
            image_format: Some(ImageFormat::Rg16f),
            access: DescriptorAccess::WriteOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });
        func.workgroup_vars.push(IrWorkgroupVar {
            name: "wg_scratch".into(),
            element_is_u32: true,
            array_len: Some(32),
            ty: IrType::Void,
        });

        let addr = func.emit(None, InstData::ConstU64(0x4000));
        let ptr = func.emit(
            None,
            InstData::ConvertUToPtr {
                addr_u64: addr,
                align: 16,
            },
        );
        let idx = func.emit(None, InstData::ConstI64(1));
        let off = func.emit(
            None,
            InstData::BdaOffset {
                base_ptr: ptr,
                index_i64: idx,
                stride_bytes: 16,
                align: 16,
            },
        );
        let loaded = func.emit(
            None,
            InstData::BdaLoad {
                ptr: off,
                access: MemoryAccess {
                    align: 16,
                    non_writable: false,
                    restrict: false,
                },
            },
        );
        let stored = func.emit(
            None,
            InstData::BdaStore {
                ptr: off,
                value: loaded,
                access: MemoryAccess {
                    align: 16,
                    non_writable: false,
                    restrict: false,
                },
            },
        );

        let one = func.emit(None, InstData::ConstU32(1));
        let atomic = func.emit(
            None,
            InstData::AtomicAdd {
                ptr: off,
                value: one,
                scope: AtomicScope::Device,
                order: AtomicOrder::AcqRel,
            },
        );

        let tex_idx = func.emit(None, InstData::ConstU32(3));
        let nu_idx = func.emit(None, InstData::NonUniform(tex_idx));
        let tex_chain = func.emit(
            None,
            InstData::ResourceAccessChain {
                descriptor_index: 0,
                index: nu_idx,
            },
        );
        let tex_load = func.emit(None, InstData::ResourceLoad { ptr: tex_chain });

        let coord = func.emit(None, InstData::ConstI32(0));
        let pixel_val = func.emit(None, InstData::const_f32(1.0));
        let img_store = func.emit(
            None,
            InstData::ImageStore {
                descriptor_index: 1,
                coord,
                value: pixel_val,
            },
        );

        let cond = func.emit(None, InstData::ConstBool(true));
        let ten = func.emit(None, InstData::ConstU32(10));
        let then_expr = func.emit(None, InstData::IAdd32(loaded, ten));
        let else_expr = func.emit(None, InstData::ISub32(loaded, one));
        let merged_phi = func.emit(
            None,
            InstData::Phi {
                incoming: vec![(then_expr, 0), (else_expr, 1)],
            },
        );
        let wg_idx = func.emit(None, InstData::ConstU32(0));
        let wg_store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: Some(wg_idx),
                value: merged_phi,
            },
        );

        func.body.push(StructuredStmt::Eval(stored));
        func.body.push(StructuredStmt::Eval(atomic));
        func.body.push(StructuredStmt::Eval(tex_load));
        func.body.push(StructuredStmt::Eval(img_store));
        func.body.push(StructuredStmt::If {
            cond,
            then_body: Vec::new(),
            else_body: vec![StructuredStmt::Eval(else_expr)],
        });
        func.body.push(StructuredStmt::Eval(wg_store));

        LowerBdaOffsetsPass.run(&mut func);
        PropagateNonUniformPass.run(&mut func);

        let module = IrModule {
            name: "pack::validate".into(),
            functions: vec![func],
        };
        let words = emit_module_spirv(&module);
        assert_eq!(words[0], spirv::MAGIC_NUMBER);
        assert_eq!(words[1], SPIRV_VERSION_1_6);

        let insts = decode_instructions(&words);
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::AtomicIAdd as u32),
            "missing OpAtomicIAdd"
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::ImageWrite as u32),
            "missing OpImageWrite"
        );
        let phi_pos = insts
            .iter()
            .position(|&(op, _)| op == spirv::Op::Phi as u32)
            .expect("missing OpPhi");
        assert_eq!(
            insts[phi_pos - 1].0,
            spirv::Op::Label as u32,
            "OpPhi must immediately follow OpLabel"
        );

        validate_with_spirv_val_if_available(&words, "validate_compute");
    }

    #[test]
    fn workgroup_addr_and_atomic_add_in_logical_mode_pass_validation() {
        let mut func = IrFunction::new("wg_atomic_logical");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.workgroup_vars.push(IrWorkgroupVar {
            name: "s_tile_count".into(),
            element_is_u32: true,
            array_len: None,
            ty: IrType::Void,
        });
        func.workgroup_vars.push(IrWorkgroupVar {
            name: "s_bins".into(),
            element_is_u32: true,
            array_len: Some(16),
            ty: IrType::Void,
        });

        let one = func.emit(None, InstData::ConstU32(1));
        let scalar_ptr = func.emit(
            None,
            InstData::WorkgroupAddr {
                var_index: 0,
                index: None,
            },
        );
        let atomic_scalar = func.emit(
            None,
            InstData::AtomicAdd {
                ptr: scalar_ptr,
                value: one,
                scope: AtomicScope::Workgroup,
                order: AtomicOrder::AcqRel,
            },
        );

        let hash_key = func.emit(None, InstData::ConstU64(0xDEAD_BEEF_0000_0007));
        let key_u32 = func.emit(None, InstData::CastToU32(hash_key));
        let mask = func.emit(None, InstData::ConstU32(0xF));
        let bin_idx = func.emit(None, InstData::BitAnd32(key_u32, mask));
        let array_ptr = func.emit(
            None,
            InstData::WorkgroupAddr {
                var_index: 1,
                index: Some(bin_idx),
            },
        );
        let atomic_array = func.emit(
            None,
            InstData::AtomicAdd {
                ptr: array_ptr,
                value: one,
                scope: AtomicScope::Workgroup,
                order: AtomicOrder::Relaxed,
            },
        );

        let dummy_ptr = func.emit(None, InstData::ConstU32(0));
        let fallback_atomic = func.emit(
            None,
            InstData::AtomicAdd {
                ptr: dummy_ptr,
                value: one,
                scope: AtomicScope::Workgroup,
                order: AtomicOrder::Relaxed,
            },
        );

        func.body.push(StructuredStmt::Eval(atomic_scalar));
        func.body.push(StructuredStmt::Eval(atomic_array));
        func.body.push(StructuredStmt::Eval(fallback_atomic));

        let module = IrModule {
            name: "pack::wg_atomic".into(),
            functions: vec![func],
        };
        let words = emit_module_spirv(&module);
        let insts = decode_instructions(&words);

        let is_logical = insts.iter().any(|&(op, ops)| {
            op == spirv::Op::MemoryModel as u32
                && ops
                    == [
                        spirv::AddressingModel::Logical as u32,
                        spirv::MemoryModel::GLSL450 as u32,
                    ]
        });
        let atomic_count = insts
            .iter()
            .filter(|&&(op, _)| op == spirv::Op::AtomicIAdd as u32)
            .count();
        let has_convert_u_to_ptr = insts
            .iter()
            .any(|&(op, _)| op == spirv::Op::ConvertUToPtr as u32);

        assert!(is_logical, "expected Logical addressing model");
        assert_eq!(atomic_count, 2, "expected 2 workgroup OpAtomicIAdd ops");
        assert!(
            !has_convert_u_to_ptr,
            "Logical mode must never emit OpConvertUToPtr"
        );
        validate_with_spirv_val_if_available(&words, "wg_atomic_logical");
    }

    #[test]
    fn image_sample_phi_and_image_store_pass_validation() {
        let mut func = IrFunction::new("sample_phi_store");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.descriptors.push(IrDescriptorVar {
            name: "scene_color".into(),
            set: 0,
            binding: 0,
            kind: DescriptorKind::SampledImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: Some(SampleScalar::F32),
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });
        func.descriptors.push(IrDescriptorVar {
            name: "prev_history".into(),
            set: 0,
            binding: 1,
            kind: DescriptorKind::StorageImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: Some(SampleScalar::F32),
            image_format: Some(ImageFormat::Rgba16f),
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });
        func.descriptors.push(IrDescriptorVar {
            name: "out_refl".into(),
            set: 0,
            binding: 2,
            kind: DescriptorKind::StorageImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: Some(SampleScalar::F32),
            image_format: Some(ImageFormat::Rg16f),
            access: DescriptorAccess::WriteOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });

        let cond_int = func.emit(None, InstData::ConstI32(1));
        let coord = func.emit(None, InstData::ConstI32(0));
        let lod = func.emit(None, InstData::ConstI32(0));
        let sampled_tex = func.emit(
            None,
            InstData::ImageSample {
                descriptor_index: 0,
                sampler_descriptor_index: None,
                image_inst: None,
                coord,
                lod,
                is_fetch: true,
            },
        );
        let sampled_storage = func.emit(
            None,
            InstData::ImageSample {
                descriptor_index: 1,
                sampler_descriptor_index: None,
                image_inst: None,
                coord,
                lod,
                is_fetch: true,
            },
        );
        let merged = func.emit(
            None,
            InstData::Phi {
                incoming: vec![(sampled_tex, 0), (sampled_storage, 1)],
            },
        );
        let img_store = func.emit(
            None,
            InstData::ImageStore {
                descriptor_index: 2,
                coord,
                value: merged,
            },
        );

        func.body.push(StructuredStmt::If {
            cond: cond_int,
            then_body: vec![StructuredStmt::Eval(sampled_tex)],
            else_body: vec![StructuredStmt::Eval(sampled_storage)],
        });
        func.body.push(StructuredStmt::Eval(img_store));

        let module = IrModule {
            name: "pack::image_sample_phi".into(),
            functions: vec![func],
        };
        let words = emit_module_spirv(&module);
        let insts = decode_instructions(&words);

        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::ImageFetch as u32),
            "expected OpImageFetch for Texture2D ImageSample"
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::ImageRead as u32),
            "expected OpImageRead for StorageImage ImageSample"
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::CompositeExtract as u32),
            "expected OpCompositeExtract to extract f32 scalar"
        );
        let phi_pos = insts
            .iter()
            .position(|&(op, _)| op == spirv::Op::Phi as u32)
            .expect("missing OpPhi");
        assert_eq!(
            insts[phi_pos - 1].0,
            spirv::Op::Label as u32,
            "OpPhi must immediately follow OpLabel"
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::ImageWrite as u32),
            "expected OpImageWrite"
        );
        validate_with_spirv_val_if_available(&words, "image_sample_phi_store");
    }

    #[test]
    fn loop_break_with_merge_phi_passes_validation() {
        let mut func = IrFunction::new("loop_break_phi");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.workgroup_vars.push(IrWorkgroupVar {
            name: "s_result".into(),
            element_is_u32: true,
            array_len: None,
            ty: IrType::Void,
        });

        let cond = func.emit(None, InstData::ConstBool(true));
        let base = func.emit(None, InstData::ConstU32(10));
        let delta_a = func.emit(None, InstData::ConstU32(3));
        let delta_b = func.emit(None, InstData::ConstU32(7));
        let val_a = func.emit(None, InstData::IAdd32(base, delta_a));
        let val_b = func.emit(None, InstData::IAdd32(base, delta_b));
        let merged = func.emit(
            None,
            InstData::Phi {
                incoming: vec![(val_a, 0), (val_b, 1)],
            },
        );
        let wg_store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: merged,
            },
        );

        func.body.push(StructuredStmt::Loop {
            body: vec![
                StructuredStmt::If {
                    cond,
                    then_body: vec![StructuredStmt::Eval(val_a), StructuredStmt::Break],
                    else_body: Vec::new(),
                },
                StructuredStmt::Eval(val_b),
                StructuredStmt::Break,
            ],
            continuing: Vec::new(),
            unroll: false,
            trip_count: None,
            induction_init: None,
            induction_step: 0,
        });
        func.body.push(StructuredStmt::Eval(wg_store));

        let module = IrModule {
            name: "pack::break_phi".into(),
            functions: vec![func],
        };
        let words = emit_module_spirv(&module);
        let insts = decode_instructions(&words);

        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::LoopMerge as u32),
            "expected OpLoopMerge"
        );
        let phi_pos = insts
            .iter()
            .position(|&(op, _)| op == spirv::Op::Phi as u32)
            .expect("missing OpPhi after loop break");
        assert_eq!(
            insts[phi_pos - 1].0,
            spirv::Op::Label as u32,
            "OpPhi must immediately follow merge OpLabel"
        );
        let merge_label = insts[phi_pos - 1].1[0];
        let phi_ops = insts[phi_pos].1;
        assert_eq!(phi_ops.len(), 6, "expected 2 incoming (val, label) pairs");
        assert_ne!(phi_ops[3], merge_label);
        assert_ne!(phi_ops[5], merge_label);
        assert_ne!(phi_ops[3], phi_ops[5]);
        validate_with_spirv_val_if_available(&words, "loop_break_phi");
    }

    #[test]
    fn emits_load_builtin_variables_entry_interface_and_component_extract() {
        let mut func = IrFunction::new("builtin_compute");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.workgroup_vars.push(IrWorkgroupVar {
            name: "s_out".into(),
            element_is_u32: true,
            array_len: None,
            ty: IrType::Void,
        });

        let gid_x = func.emit(
            None,
            InstData::LoadBuiltin {
                kind: BuiltinKind::GlobalInvocationId,
                component: Some(0),
            },
        );
        let gid_y = func.emit(
            None,
            InstData::LoadBuiltin {
                kind: BuiltinKind::GlobalInvocationId,
                component: Some(1),
            },
        );
        let local_idx = func.emit(
            None,
            InstData::LoadBuiltin {
                kind: BuiltinKind::LocalInvocationIndex,
                component: None,
            },
        );
        let sum_xy = func.emit(None, InstData::IAdd32(gid_x, gid_y));
        let total = func.emit(None, InstData::IAdd32(sum_xy, local_idx));
        let store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: total,
            },
        );
        func.body.push(StructuredStmt::Eval(store));

        let module = IrModule {
            name: "pack::builtins".into(),
            functions: vec![func],
        };
        let words = emit_module_spirv(&module);
        let insts = decode_instructions(&words);

        let builtin_gid_vars: Vec<u32> = insts
            .iter()
            .filter(|&&(op, ops)| {
                op == spirv::Op::Decorate as u32
                    && ops.len() == 3
                    && ops[1] == spirv::Decoration::BuiltIn as u32
                    && ops[2] == spirv::BuiltIn::GlobalInvocationId as u32
            })
            .map(|&(_, ops)| ops[0])
            .collect();
        let builtin_local_idx_vars: Vec<u32> = insts
            .iter()
            .filter(|&&(op, ops)| {
                op == spirv::Op::Decorate as u32
                    && ops.len() == 3
                    && ops[1] == spirv::Decoration::BuiltIn as u32
                    && ops[2] == spirv::BuiltIn::LocalInvocationIndex as u32
            })
            .map(|&(_, ops)| ops[0])
            .collect();

        assert_eq!(
            builtin_gid_vars.len(),
            1,
            "GlobalInvocationId Input variable must be deduplicated"
        );
        assert_eq!(builtin_local_idx_vars.len(), 1);

        let entry_ops = insts
            .iter()
            .find_map(|&(op, ops)| (op == spirv::Op::EntryPoint as u32).then_some(ops))
            .expect("missing OpEntryPoint");
        assert!(entry_ops.contains(&builtin_gid_vars[0]));
        assert!(entry_ops.contains(&builtin_local_idx_vars[0]));

        let extract_count = insts
            .iter()
            .filter(|&&(op, _)| op == spirv::Op::CompositeExtract as u32)
            .count();
        assert_eq!(
            extract_count, 2,
            "expected 2 OpCompositeExtract ops for GlobalInvocationId components 0 and 1"
        );
        validate_with_spirv_val_if_available(&words, "load_builtin");
    }

    #[test]
    fn emits_load_push_constant_scalar_composite_and_fallback() {
        let mut func = IrFunction::new("pc_load_test");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.workgroup_vars.push(IrWorkgroupVar {
            name: "s_pc_out".into(),
            element_is_u32: true,
            array_len: None,
            ty: IrType::Void,
        });
        func.push_constants = Some(IrPushConstantBlock {
            name: "FramePC".into(),
            fields: vec![
                vsl_ir::IrPushConstantField {
                    name: "frame_idx".into(),
                    byte_offset: 0,
                    byte_size: 4,
                    scalar: ScalarIrTy::U32,
                },
                vsl_ir::IrPushConstantField {
                    name: "cam_vec".into(),
                    byte_offset: 16,
                    byte_size: 16,
                    scalar: ScalarIrTy::U32,
                },
            ],
            total_size: 32,
        });

        let pc0 = func.emit(None, InstData::LoadPushConstant { field_index: 0 });
        let pc1 = func.emit(None, InstData::LoadPushConstant { field_index: 1 });
        let sum = func.emit(None, InstData::IAdd32(pc0, pc1));
        let store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: sum,
            },
        );
        func.body.push(StructuredStmt::Eval(store));

        let module = IrModule {
            name: "pack::push_const".into(),
            functions: vec![func],
        };
        let words = emit_module_spirv(&module);
        let insts = decode_instructions(&words);

        let access_chains = insts
            .iter()
            .filter(|&&(op, _)| op == spirv::Op::AccessChain as u32)
            .count();
        let composite_extracts = insts
            .iter()
            .filter(|&&(op, _)| op == spirv::Op::CompositeExtract as u32)
            .count();
        assert_eq!(
            access_chains, 2,
            "expected 2 PushConstant OpAccessChain ops"
        );
        assert_eq!(
            composite_extracts, 1,
            "expected OpCompositeExtract for 16-byte vector push constant member"
        );
        validate_with_spirv_val_if_available(&words, "load_push_constant");
    }

    #[test]
    fn emits_ray_query_khr_extension_function_var_and_traversal_ops() {
        let mut func = IrFunction::new("ray_query_compute");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.descriptors.push(IrDescriptorVar {
            name: "scene_tlas".into(),
            set: 0,
            binding: 0,
            kind: DescriptorKind::AccelerationStructure,
            dim: None,
            sample_scalar: None,
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });
        func.workgroup_vars.push(IrWorkgroupVar {
            name: "s_hit_id".into(),
            element_is_u32: true,
            array_len: None,
            ty: IrType::Void,
        });

        let flags = func.emit(None, InstData::ConstU32(0));
        let mask = func.emit(None, InstData::ConstU32(0xFF));
        let origin = func.emit(None, InstData::const_f32(0.0));
        let t_min = func.emit(None, InstData::const_f32(0.01));
        let dir = func.emit(None, InstData::const_f32(1.0));
        let t_max = func.emit(None, InstData::const_f32(100.0));

        let init = func.emit(
            None,
            InstData::RayQueryInitialize {
                ray_query_id: 0,
                tlas_descriptor_index: 0,
                flags,
                cull_mask: mask,
                origin,
                t_min,
                dir,
                t_max,
            },
        );
        let proceed = func.emit(None, InstData::RayQueryProceed(0));
        let hit_ty = func.emit(None, InstData::RayQueryCommittedType(0));
        let custom_idx = func.emit(None, InstData::RayQueryCommittedInstanceCustomIndex(0));
        let bary = func.emit(None, InstData::RayQueryCommittedBarycentrics(0));
        let bary_u32 = func.emit(None, InstData::CastToU32(bary));
        let combined = func.emit(None, InstData::IAdd32(hit_ty, custom_idx));
        let final_val = func.emit(None, InstData::IAdd32(combined, bary_u32));
        let store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: final_val,
            },
        );

        func.body.push(StructuredStmt::Eval(init));
        func.body.push(StructuredStmt::Eval(proceed));
        func.body.push(StructuredStmt::Eval(store));

        let module = IrModule {
            name: "pack::ray_query".into(),
            functions: vec![func],
        };
        let words = emit_module_spirv(&module);
        let insts = decode_instructions(&words);

        assert!(insts.iter().any(|&(op, ops)| {
            op == spirv::Op::Capability as u32 && ops == [spirv::Capability::RayQueryKHR as u32]
        }));
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::TypeRayQueryKHR as u32)
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::RayQueryInitializeKHR as u32)
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::RayQueryProceedKHR as u32)
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::RayQueryGetIntersectionTypeKHR as u32)
        );
        assert!(insts.iter().any(|&(op, _)| {
            op == spirv::Op::RayQueryGetIntersectionInstanceCustomIndexKHR as u32
        }));
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::RayQueryGetIntersectionBarycentricsKHR as u32)
        );

        let fn_pos = insts
            .iter()
            .position(|&(op, _)| op == spirv::Op::Function as u32)
            .expect("missing OpFunction");
        assert_eq!(insts[fn_pos + 1].0, spirv::Op::Label as u32);
        assert_eq!(insts[fn_pos + 2].0, spirv::Op::Variable as u32);
        assert_eq!(insts[fn_pos + 2].1[2], spirv::StorageClass::Function as u32);
        validate_with_spirv_val_if_available(&words, "ray_query_khr");
    }

    #[test]
    fn switch_coerces_u64_and_bool_selectors_and_both_branch_return_if_emits_unreachable() {
        let mut func = IrFunction::new("switch_and_unreachable_if");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);

        let sel_u64 = func.emit(None, InstData::ConstU64(1));
        let sel_bool = func.emit(None, InstData::ConstBool(true));
        let cond = func.emit(None, InstData::ConstBool(false));

        func.body.push(StructuredStmt::Switch {
            selector: sel_u64,
            cases: vec![(1, Vec::new())],
            default: Vec::new(),
        });
        func.body.push(StructuredStmt::Switch {
            selector: sel_bool,
            cases: vec![(1, Vec::new())],
            default: Vec::new(),
        });
        func.body.push(StructuredStmt::If {
            cond,
            then_body: vec![StructuredStmt::Return(None)],
            else_body: vec![StructuredStmt::Return(None)],
        });

        let module = IrModule {
            name: "pack::switch_unreachable".into(),
            functions: vec![func],
        };
        let words = emit_module_spirv(&module);
        let insts = decode_instructions(&words);

        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::SConvert as u32),
            "expected OpSConvert when coercing u64 switch selector to i32"
        );
        assert!(
            insts.iter().any(|&(op, _)| op == spirv::Op::Select as u32),
            "expected OpSelect when coercing bool switch selector to i32"
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::Unreachable as u32),
            "expected OpUnreachable in merge block when both If branches return"
        );
        validate_with_spirv_val_if_available(&words, "switch_unreachable_if");
    }

    #[test]
    fn bindless_image_sample_with_image_inst_preserves_nonuniform_decoration() {
        let mut func = IrFunction::new("bindless_nonuniform_sample");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.descriptors.push(IrDescriptorVar {
            name: "bindless_textures".into(),
            set: 2,
            binding: 0,
            kind: DescriptorKind::SampledImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: Some(SampleScalar::F32),
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: true,
            count: None,
            struct_layout: None,
        });
        func.descriptors.push(IrDescriptorVar {
            name: "out_color".into(),
            set: 0,
            binding: 0,
            kind: DescriptorKind::StorageImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: Some(SampleScalar::F32),
            image_format: Some(ImageFormat::Rgba16f),
            access: DescriptorAccess::WriteOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });

        let idx = func.emit(None, InstData::ConstU32(4));
        let nu_idx = func.emit(None, InstData::NonUniform(idx));
        let chain = func.emit(
            None,
            InstData::ResourceAccessChain {
                descriptor_index: 0,
                index: nu_idx,
            },
        );
        let resource_load = func.emit(None, InstData::ResourceLoad { ptr: chain });
        let coord = func.emit(None, InstData::ConstI32(0));
        let lod = func.emit(None, InstData::ConstI32(0));
        let sample = func.emit(
            None,
            InstData::ImageSample {
                descriptor_index: 0,
                sampler_descriptor_index: None,
                image_inst: Some(resource_load),
                coord,
                lod,
                is_fetch: true,
            },
        );
        let store = func.emit(
            None,
            InstData::ImageStore {
                descriptor_index: 1,
                coord,
                value: sample,
            },
        );
        func.body.push(StructuredStmt::Eval(store));

        PropagateNonUniformPass.run(&mut func);

        let module = IrModule {
            name: "pack::bindless_sample".into(),
            functions: vec![func],
        };
        let words = emit_module_spirv(&module);
        let insts = decode_instructions(&words);

        let fetch_ops = insts
            .iter()
            .find_map(|&(op, ops)| (op == spirv::Op::ImageFetch as u32).then_some(ops))
            .expect("expected OpImageFetch using preloaded bindless image");
        let fetched_img_id = fetch_ops[2];
        let img_decorated_nonuniform = insts.iter().any(|&(op, ops)| {
            op == spirv::Op::Decorate as u32
                && ops == [fetched_img_id, spirv::Decoration::NonUniform as u32]
        });
        assert!(
            img_decorated_nonuniform,
            "OpImageFetch image operand must carry OpDecorate NonUniform"
        );
        validate_with_spirv_val_if_available(&words, "bindless_nonuniform_sample");
    }

    #[test]
    fn bindless_combined_sampler_array_emits_4096_sampled_images() {
        let mut func = IrFunction::new("bindless_combined_sample");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.descriptors.push(IrDescriptorVar {
            name: "bindless".into(),
            set: 2,
            binding: 0,
            kind: DescriptorKind::CombinedImageSampler,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: Some(SampleScalar::F32),
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
            count: Some(BINDLESS_IMAGE_COUNT),
            struct_layout: None,
        });
        func.descriptors.push(IrDescriptorVar {
            name: "out_color".into(),
            set: 0,
            binding: 0,
            kind: DescriptorKind::StorageImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: Some(SampleScalar::F32),
            image_format: Some(ImageFormat::Rgba16f),
            access: DescriptorAccess::WriteOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });

        let idx = func.emit(None, InstData::ConstU32(4));
        let nu_idx = func.emit(None, InstData::NonUniform(idx));
        let chain = func.emit(
            None,
            InstData::ResourceAccessChain {
                descriptor_index: 0,
                index: nu_idx,
            },
        );
        let image_load = func.emit(None, InstData::ResourceLoad { ptr: chain });
        let coord = func.emit(None, InstData::ConstI32(0));
        let lod = func.emit(None, InstData::const_f32(0.0));
        let sample = func.emit(
            None,
            InstData::ImageSample {
                descriptor_index: 0,
                sampler_descriptor_index: None,
                image_inst: Some(image_load),
                coord,
                lod,
                is_fetch: false,
            },
        );
        let store = func.emit(
            None,
            InstData::ImageStore {
                descriptor_index: 1,
                coord,
                value: sample,
            },
        );
        func.body.push(StructuredStmt::Eval(store));

        PropagateNonUniformPass.run(&mut func);

        let words = emit_module_spirv(&IrModule {
            name: "pack::bindless_combined".into(),
            functions: vec![func],
        });
        let insts = decode_instructions(&words);

        let array_ops = insts
            .iter()
            .find_map(|&(op, ops)| (op == spirv::Op::TypeArray as u32).then_some(ops))
            .expect("expected OpTypeArray for the bindless descriptor");
        let len_id = array_ops[2];
        let array_len = insts.iter().find_map(|&(op, ops)| {
            (op == spirv::Op::Constant as u32 && ops.len() >= 3 && ops[1] == len_id).then(|| ops[2])
        });
        assert_eq!(array_len, Some(BINDLESS_IMAGE_COUNT));
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::TypeSampledImage as u32),
            "expected OpTypeSampledImage inside the bindless array"
        );
        assert!(
            !insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::TypeSampler as u32),
            "combined image samplers must not emit a standalone sampler type"
        );
        assert!(
            !insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::SampledImage as u32),
            "combined image samplers must be sampled directly without OpSampledImage"
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::ImageSampleExplicitLod as u32),
            "expected direct OpImageSampleExplicitLod on the combined sampler"
        );
        validate_with_spirv_val_if_available(&words, "bindless_combined_sampler_array");
    }

    #[test]
    fn loop_header_phi_found_through_nested_if_with_preallocated_select_backedge() {
        let mut func = IrFunction::new("loop_nested_if_phi");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.workgroup_vars.push(IrWorkgroupVar {
            name: "s_acc".into(),
            element_is_u32: true,
            array_len: None,
            ty: IrType::Void,
        });

        let init = func.emit(None, InstData::ConstU32(0));
        let limit = func.emit(None, InstData::ConstU32(8));
        let one = func.emit(None, InstData::ConstU32(1));
        let mask = func.emit(None, InstData::ConstU32(0xFF));
        let phi = func.emit(
            None,
            InstData::Phi {
                incoming: Vec::new(),
            },
        );
        let cond_break = func.emit(None, InstData::ICmpGe(phi, limit));
        let inc = func.emit(None, InstData::IAdd32(phi, one));
        let masked = func.emit(None, InstData::BitAnd32(inc, mask));
        let back_sel = func.emit(
            None,
            InstData::Select {
                cond: cond_break,
                on_true: init,
                on_false: masked,
            },
        );
        func.insts[phi].data = InstData::Phi {
            incoming: vec![(init, 0), (back_sel, 1)],
        };
        let store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: phi,
            },
        );

        func.body.push(StructuredStmt::Loop {
            body: vec![
                StructuredStmt::If {
                    cond: cond_break,
                    then_body: vec![StructuredStmt::Break],
                    else_body: Vec::new(),
                },
                StructuredStmt::Eval(store),
            ],
            continuing: vec![StructuredStmt::Eval(back_sel)],
            unroll: false,
            trip_count: None,
            induction_init: None,
            induction_step: 0,
        });

        let module = IrModule {
            name: "pack::loop_header_phi".into(),
            functions: vec![func],
        };
        let words = emit_module_spirv(&module);
        let insts = decode_instructions(&words);

        let phi_pos = insts
            .iter()
            .position(|&(op, _)| op == spirv::Op::Phi as u32)
            .expect("expected OpPhi in loop header");
        assert_eq!(
            insts[phi_pos - 1].0,
            spirv::Op::Label as u32,
            "OpPhi must immediately follow loop header OpLabel"
        );
        assert_eq!(
            insts[phi_pos + 1].0,
            spirv::Op::LoopMerge as u32,
            "OpLoopMerge must follow loop header OpPhi"
        );
        let back_val_id = insts[phi_pos].1[4];
        let back_select_defined = insts
            .iter()
            .any(|&(op, ops)| op == spirv::Op::Select as u32 && ops[1] == back_val_id);
        assert!(
            back_select_defined,
            "loop back-edge OpSelect must use the preallocated SSA result ID"
        );
        validate_with_spirv_val_if_available(&words, "loop_nested_if_phi");
    }

    #[test]
    fn shared_load_builtin_in_both_if_branches_prematerializes_in_header() {
        let mut func = IrFunction::new("shared_builtin_if");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.workgroup_vars.push(IrWorkgroupVar {
            name: "s_out".into(),
            element_is_u32: true,
            array_len: None,
            ty: IrType::Void,
        });
        func.push_constants = Some(IrPushConstantBlock {
            name: "PassPush".into(),
            fields: vec![vsl_ir::IrPushConstantField {
                name: "mode".into(),
                byte_offset: 0,
                byte_size: 4,
                scalar: ScalarIrTy::U32,
            }],
            total_size: 4,
        });

        let mode = func.emit(None, InstData::LoadPushConstant { field_index: 0 });
        let zero = func.emit(None, InstData::ConstU32(0));
        let cond = func.emit(None, InstData::ICmpEq(mode, zero));
        let gid_x = func.emit(
            None,
            InstData::LoadBuiltin {
                kind: BuiltinKind::GlobalInvocationId,
                component: Some(0),
            },
        );
        let one = func.emit(None, InstData::ConstU32(1));
        let two = func.emit(None, InstData::ConstU32(2));
        let then_val = func.emit(None, InstData::IAdd32(gid_x, one));
        let else_val = func.emit(None, InstData::IAdd32(gid_x, two));
        let then_store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: then_val,
            },
        );
        let else_store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: else_val,
            },
        );

        func.body.push(StructuredStmt::If {
            cond,
            then_body: vec![StructuredStmt::Eval(then_store)],
            else_body: vec![StructuredStmt::Eval(else_store)],
        });

        let module = IrModule {
            name: "pack::shared_builtin_if".into(),
            functions: vec![func],
        };
        let words = emit_module_spirv(&module);
        let insts = decode_instructions(&words);

        let extract_pos = insts
            .iter()
            .position(|&(op, _)| op == spirv::Op::CompositeExtract as u32)
            .expect("expected OpCompositeExtract for GlobalInvocationId.x");
        let sel_merge_pos = insts
            .iter()
            .position(|&(op, _)| op == spirv::Op::SelectionMerge as u32)
            .expect("expected OpSelectionMerge");
        assert!(
            extract_pos < sel_merge_pos,
            "shared LoadBuiltin must be materialized in dominating header block before OpSelectionMerge"
        );
        let gid_x_id = insts[extract_pos].1[1];
        let adds_using_gid: usize = insts
            .iter()
            .filter(|&&(op, ops)| op == spirv::Op::IAdd as u32 && ops[2] == gid_x_id)
            .count();
        assert_eq!(
            adds_using_gid, 2,
            "both branches should reuse the header-materialized GlobalInvocationId.x ID"
        );
        validate_with_spirv_val_if_available(&words, "shared_builtin_if");
    }

    #[test]
    fn infinite_loop_without_break_emits_unreachable_and_skips_trailing_stmt() {
        let mut func = IrFunction::new("infinite_loop_unreachable");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.workgroup_vars.push(IrWorkgroupVar {
            name: "s_out".into(),
            element_is_u32: true,
            array_len: None,
            ty: IrType::Void,
        });

        let one = func.emit(None, InstData::ConstU32(1));
        let two = func.emit(None, InstData::ConstU32(2));
        let loop_store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: one,
            },
        );
        let trailing_store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: two,
            },
        );

        func.body.push(StructuredStmt::Loop {
            body: vec![StructuredStmt::Eval(loop_store)],
            continuing: Vec::new(),
            unroll: false,
            trip_count: None,
            induction_init: None,
            induction_step: 0,
        });
        func.body.push(StructuredStmt::Eval(trailing_store));

        let module = IrModule {
            name: "pack::infinite_loop".into(),
            functions: vec![func],
        };
        let words = emit_module_spirv(&module);
        let insts = decode_instructions(&words);

        let unreachable_pos = insts
            .iter()
            .position(|&(op, _)| op == spirv::Op::Unreachable as u32)
            .expect("expected OpUnreachable in merge block of infinite loop");
        assert_eq!(
            insts[unreachable_pos - 1].0,
            spirv::Op::Label as u32,
            "OpUnreachable must immediately follow the loop merge OpLabel"
        );
        let store_count = insts
            .iter()
            .filter(|&&(op, _)| op == spirv::Op::Store as u32)
            .count();
        assert_eq!(
            store_count, 1,
            "trailing statement after infinite loop must not be emitted after OpUnreachable"
        );
        validate_with_spirv_val_if_available(&words, "infinite_loop_unreachable");
    }

    #[test]
    fn make_vec2_passed_to_image_store_emits_real_2d_coordinates() {
        let mut func = IrFunction::new("make_vec2_image_store");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.descriptors.push(IrDescriptorVar {
            name: "out_color".into(),
            set: 0,
            binding: 0,
            kind: DescriptorKind::StorageImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: Some(SampleScalar::F32),
            image_format: Some(ImageFormat::Rgba16f),
            access: DescriptorAccess::WriteOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });

        let gx = func.emit(
            None,
            InstData::LoadBuiltin {
                kind: BuiltinKind::GlobalInvocationId,
                component: Some(0),
            },
        );
        let gy = func.emit(
            None,
            InstData::LoadBuiltin {
                kind: BuiltinKind::GlobalInvocationId,
                component: Some(1),
            },
        );
        let coord = func.emit(None, InstData::MakeVec2(gx, gy));
        let val = func.emit(None, InstData::const_f32(1.0));
        let store = func.emit(
            None,
            InstData::ImageStore {
                descriptor_index: 0,
                coord,
                value: val,
            },
        );
        func.body.push(StructuredStmt::Eval(store));

        let module = IrModule {
            name: "pack::make_vec2_store".into(),
            functions: vec![func],
        };
        let words = emit_module_spirv(&module);
        let insts = decode_instructions(&words);

        let image_write_ops = insts
            .iter()
            .find_map(|&(op, ops)| (op == spirv::Op::ImageWrite as u32).then_some(ops))
            .expect("expected OpImageWrite");
        let coord_vec_id = image_write_ops[1];
        let coord_construct_ops = insts
            .iter()
            .find_map(|&(op, ops)| {
                (op == spirv::Op::CompositeConstruct as u32 && ops[1] == coord_vec_id)
                    .then_some(ops)
            })
            .expect("expected OpCompositeConstruct for ImageWrite coordinate");
        assert_eq!(
            coord_construct_ops.len(),
            4,
            "expected 2D coordinate vector"
        );
        assert_ne!(
            coord_construct_ops[2], coord_construct_ops[3],
            "ImageWrite 2D coordinate must use distinct (x, y) components instead of duplicating x"
        );
        validate_with_spirv_val_if_available(&words, "make_vec2_image_store");
    }

    #[test]
    fn loop_header_phi_non_binop_backedge_emits_copy_object_in_continue_block() {
        let mut func = IrFunction::new("loop_header_phi_copy_object");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.workgroup_vars.push(IrWorkgroupVar {
            name: "s_acc".into(),
            element_is_u32: true,
            array_len: None,
            ty: IrType::Void,
        });

        let init = func.emit(None, InstData::ConstU32(0));
        let limit = func.emit(None, InstData::ConstU32(4));
        let phi = func.emit(
            None,
            InstData::Phi {
                incoming: Vec::new(),
            },
        );
        let cond_break = func.emit(None, InstData::ICmpGe(phi, limit));
        let back_load = func.emit(
            None,
            InstData::WorkgroupLoad {
                var_index: 0,
                index: None,
            },
        );
        func.insts[phi].data = InstData::Phi {
            incoming: vec![(init, 0), (back_load, 1)],
        };
        let store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: phi,
            },
        );

        func.body.push(StructuredStmt::Loop {
            body: vec![
                StructuredStmt::If {
                    cond: cond_break,
                    then_body: vec![StructuredStmt::Break],
                    else_body: Vec::new(),
                },
                StructuredStmt::Eval(store),
            ],
            continuing: vec![StructuredStmt::Eval(back_load)],
            unroll: false,
            trip_count: None,
            induction_init: None,
            induction_step: 0,
        });

        let module = IrModule {
            name: "pack::loop_phi_copy".into(),
            functions: vec![func],
        };
        let words = emit_module_spirv(&module);
        let insts = decode_instructions(&words);

        let phi_ops = insts
            .iter()
            .find_map(|&(op, ops)| (op == spirv::Op::Phi as u32).then_some(ops))
            .expect("expected OpPhi in loop header");
        let back_val_id = phi_ops[4];
        let back_copy_defined = insts
            .iter()
            .any(|&(op, ops)| op == spirv::Op::CopyObject as u32 && ops[1] == back_val_id);
        assert!(
            back_copy_defined,
            "non-binop loop back-edge must be defined via OpCopyObject with the preallocated ID"
        );
        validate_with_spirv_val_if_available(&words, "loop_header_phi_copy_object");
    }

    #[test]
    fn emits_helper_function_with_inout_param_and_op_function_call() {
        let mut helper = IrFunction::new("add_and_bump");
        helper.return_ty = IrType::Scalar(ScalarIrTy::U32);
        helper.params.push(IrParam {
            name: "x".into(),
            ty: IrType::Scalar(ScalarIrTy::U32),
            is_inout: false,
            is_out: false,
        });
        helper.params.push(IrParam {
            name: "acc".into(),
            ty: IrType::Scalar(ScalarIrTy::U32),
            is_inout: true,
            is_out: false,
        });
        let p_x = helper.emit(None, InstData::Param { index: 0 });
        let p_acc = helper.emit_typed(
            IrType::Pointer {
                pointee: Box::new(IrType::Scalar(ScalarIrTy::U32)),
                align: 4,
                storage: IrStorageClass::Function,
            },
            InstData::Param { index: 1 },
        );
        let cur = helper.emit(None, InstData::LocalLoad { ptr: p_acc });
        let sum = helper.emit(None, InstData::IAdd32(p_x, cur));
        let st = helper.emit(
            None,
            InstData::LocalStore {
                ptr: p_acc,
                value: sum,
            },
        );
        helper.body.push(StructuredStmt::Eval(st));
        helper.body.push(StructuredStmt::Return(Some(sum)));

        let mut main_fn = IrFunction::new("call_main");
        main_fn.stage = Some(ShaderStage::Compute);
        main_fn.workgroup_size = Some([8, 8, 1]);
        main_fn.local_vars.push(IrLocalVar {
            name: "local_acc".into(),
            ty: IrType::Scalar(ScalarIrTy::U32),
        });
        main_fn.workgroup_vars.push(IrWorkgroupVar::typed(
            "s_out",
            IrType::Scalar(ScalarIrTy::U32),
        ));
        let init_val = main_fn.emit(None, InstData::ConstU32(5));
        let arg_x = main_fn.emit(None, InstData::ConstU32(7));
        let l_addr = main_fn.emit(None, InstData::LocalAddr { local_index: 0 });
        let l_init = main_fn.emit(
            None,
            InstData::LocalStore {
                ptr: l_addr,
                value: init_val,
            },
        );
        let call = main_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::U32),
            InstData::FunctionCall {
                callee: "add_and_bump".into(),
                args: vec![arg_x, l_addr],
            },
        );
        let wg_store = main_fn.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: call,
            },
        );
        main_fn.body.push(StructuredStmt::Eval(l_init));
        main_fn.body.push(StructuredStmt::Eval(wg_store));

        let words = emit_module_spirv(&IrModule {
            name: "pack::fn_call".into(),
            functions: vec![helper, main_fn],
        });
        let insts = decode_instructions(&words);

        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::FunctionParameter as u32),
            "expected OpFunctionParameter for helper function"
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::ReturnValue as u32),
            "expected OpReturnValue for non-void helper"
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::FunctionCall as u32),
            "expected OpFunctionCall in entry point"
        );
        validate_with_spirv_val_if_available(&words, "helper_fn_call");
    }

    #[test]
    fn emits_real_block_ubo_and_ssbo_structs_with_matrix_and_array_decorations() {
        let mut func = IrFunction::new("real_block_structs");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        // Set 0, binding 0 resolves to built-in WorldUBO (with Mat4 + Vec3 + U64 fields).
        func.descriptors.push(IrDescriptorVar {
            name: "world".into(),
            set: 0,
            binding: 0,
            kind: DescriptorKind::UniformBuffer,
            dim: None,
            sample_scalar: None,
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });
        // Set 1, binding 1 resolves to built-in InstanceMetaSSBO (RuntimeArray<InstanceMeta>).
        func.descriptors.push(IrDescriptorVar {
            name: "instances".into(),
            set: 1,
            binding: 1,
            kind: DescriptorKind::StorageBuffer,
            dim: None,
            sample_scalar: None,
            image_format: None,
            access: DescriptorAccess::ReadWrite,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });

        let zero = func.emit(None, InstData::ConstU32(0));
        let idx_mat_id = func.emit(None, InstData::ConstU32(2));
        let ptr = func.emit_typed(
            IrType::Pointer {
                pointee: Box::new(IrType::Scalar(ScalarIrTy::U32)),
                align: 4,
                storage: IrStorageClass::StorageBuffer,
            },
            InstData::DescriptorAccessChain {
                descriptor_index: 1,
                indices: vec![zero, zero, idx_mat_id],
            },
        );
        let val = func.emit(None, InstData::ConstU32(42));
        let store = func.emit(None, InstData::DescriptorStore { ptr, value: val });
        func.body.push(StructuredStmt::Eval(store));

        let words = emit_module_spirv(&IrModule {
            name: "pack::real_blocks".into(),
            functions: vec![func],
        });
        let insts = decode_instructions(&words);

        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::TypeMatrix as u32),
            "WorldUBO/InstanceMeta must emit OpTypeMatrix"
        );
        assert!(insts.iter().any(|&(op, ops)| {
            op == spirv::Op::MemberDecorate as u32
                && ops.len() == 3
                && ops[2] == spirv::Decoration::ColMajor as u32
        }));
        assert!(insts.iter().any(|&(op, ops)| {
            op == spirv::Op::MemberDecorate as u32
                && ops.len() == 4
                && ops[2] == spirv::Decoration::MatrixStride as u32
                && ops[3] == 16
        }));
        assert!(insts.iter().any(|&(op, ops)| {
            op == spirv::Op::Decorate as u32
                && ops.len() == 3
                && ops[1] == spirv::Decoration::ArrayStride as u32
                && ops[2] == 144
        }));
        validate_with_spirv_val_if_available(&words, "real_block_structs");
    }

    #[test]
    fn emits_vec4_image_sample_and_store_and_extended_image_ops() {
        let mut func = IrFunction::new("vec4_image_ops");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.descriptors.push(IrDescriptorVar {
            name: "tex_albedo".into(),
            set: 0,
            binding: 1,
            kind: DescriptorKind::CombinedImageSampler,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: Some(SampleScalar::F32),
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });
        func.descriptors.push(IrDescriptorVar {
            name: "out_rgba".into(),
            set: 0,
            binding: 2,
            kind: DescriptorKind::StorageImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: Some(SampleScalar::F32),
            image_format: Some(ImageFormat::Rgba16f),
            access: DescriptorAccess::WriteOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });

        let u0 = func.emit(None, InstData::const_f32(0.5));
        let uv = func.emit(None, InstData::MakeVec2(u0, u0));
        let lod = func.emit(None, InstData::const_f32(0.0));
        let vec4_f32 = IrType::Vector {
            scalar: ScalarIrTy::F32,
            len: 4,
        };
        let sampled = func.emit_typed(
            vec4_f32.clone(),
            InstData::ImageSample {
                descriptor_index: 0,
                sampler_descriptor_index: None,
                image_inst: None,
                coord: uv,
                lod,
                is_fetch: false,
            },
        );
        let grad = func.emit_typed(
            vec4_f32.clone(),
            InstData::ImageSampleGrad {
                descriptor_index: 0,
                sampler_descriptor_index: None,
                image_inst: None,
                coord: uv,
                ddx: uv,
                ddy: uv,
            },
        );
        let gathered = func.emit_typed(
            vec4_f32,
            InstData::ImageGather {
                descriptor_index: 0,
                sampler_descriptor_index: None,
                image_inst: None,
                coord: uv,
                component: 0,
            },
        );
        let lod0 = func.emit(None, InstData::ConstI32(0));
        let sz = func.emit_typed(
            IrType::Vector {
                scalar: ScalarIrTy::U32,
                len: 2,
            },
            InstData::ImageQuerySize {
                descriptor_index: 0,
                lod: Some(lod0),
            },
        );
        let lvls = func.emit(
            None,
            InstData::ImageQueryLevels {
                descriptor_index: 0,
            },
        );
        let sum = func.emit(None, InstData::FAdd32(sampled, grad));
        let sum2 = func.emit(None, InstData::FAdd32(sum, gathered));
        let c0 = func.emit(None, InstData::ConstI32(0));
        let icoord = func.emit(None, InstData::MakeVec2(c0, c0));
        let store = func.emit(
            None,
            InstData::ImageStore {
                descriptor_index: 1,
                coord: icoord,
                value: sum2,
            },
        );
        func.body.push(StructuredStmt::Eval(sz));
        func.body.push(StructuredStmt::Eval(lvls));
        func.body.push(StructuredStmt::Eval(store));

        let words = emit_module_spirv(&IrModule {
            name: "pack::vec4_images".into(),
            functions: vec![func],
        });
        let insts = decode_instructions(&words);

        assert!(
            !insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::CompositeExtract as u32),
            "vec4 ImageSample must not force OpCompositeExtract 0"
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::ImageGather as u32),
            "expected OpImageGather"
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::ImageQuerySizeLod as u32),
            "expected OpImageQuerySizeLod"
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::ImageQueryLevels as u32),
            "expected OpImageQueryLevels"
        );
        assert!(insts.iter().any(|&(op, ops)| {
            op == spirv::Op::Capability as u32 && ops == [spirv::Capability::ImageQuery as u32]
        }));
        validate_with_spirv_val_if_available(&words, "vec4_image_ops");
    }

    #[test]
    fn emits_full_ray_query_khr_ops_with_vec3_and_vec2_barycentrics() {
        let mut func = IrFunction::new("full_ray_query");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.descriptors.push(IrDescriptorVar {
            name: "tlas".into(),
            set: 1,
            binding: 0,
            kind: DescriptorKind::AccelerationStructure,
            dim: None,
            sample_scalar: None,
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });

        let flags = func.emit(None, InstData::ConstU32(0));
        let mask = func.emit(None, InstData::ConstU32(0xFF));
        let f0 = func.emit(None, InstData::const_f32(0.0));
        let f1 = func.emit(None, InstData::const_f32(1.0));
        let origin = func.emit(None, InstData::MakeVec3(f0, f0, f0));
        let dir = func.emit(None, InstData::MakeVec3(f0, f0, f1));
        let t_min = func.emit(None, InstData::const_f32(0.001));
        let t_max = func.emit(None, InstData::const_f32(100.0));

        let init = func.emit(
            None,
            InstData::RayQueryInitialize {
                ray_query_id: 0,
                tlas_descriptor_index: 0,
                flags,
                cull_mask: mask,
                origin,
                t_min,
                dir,
                t_max,
            },
        );
        let cand_ty = func.emit(None, InstData::RayQueryCandidateType(0));
        let confirm = func.emit(None, InstData::RayQueryConfirmIntersection(0));
        let gen_hit = func.emit(
            None,
            InstData::RayQueryGenerateIntersection {
                ray_query_id: 0,
                t_hit: t_min,
            },
        );
        let term = func.emit(None, InstData::RayQueryTerminate(0));
        let comm_t = func.emit(None, InstData::RayQueryCommittedT(0));
        let cand_t = func.emit(None, InstData::RayQueryCandidateT(0));
        let inst_id = func.emit(None, InstData::RayQueryCommittedInstanceId(0));
        let prim_idx = func.emit(None, InstData::RayQueryCommittedPrimitiveIndex(0));
        let geom_idx = func.emit(None, InstData::RayQueryCommittedGeometryIndex(0));
        let o2w = func.emit(None, InstData::RayQueryCommittedObjectToWorld(0));
        let w2o = func.emit(None, InstData::RayQueryCommittedWorldToObject(0));
        let ff = func.emit(None, InstData::RayQueryCommittedFrontFace(0));
        let bary = func.emit(None, InstData::RayQueryCommittedBarycentrics(0));

        for id in [
            init, cand_ty, confirm, gen_hit, term, comm_t, cand_t, inst_id, prim_idx, geom_idx,
            o2w, w2o, ff, bary,
        ] {
            func.body.push(StructuredStmt::Eval(id));
        }

        let words = emit_module_spirv(&IrModule {
            name: "pack::full_rq".into(),
            functions: vec![func],
        });
        let insts = decode_instructions(&words);

        for required_op in [
            spirv::Op::RayQueryConfirmIntersectionKHR,
            spirv::Op::RayQueryTerminateKHR,
            spirv::Op::RayQueryGenerateIntersectionKHR,
            spirv::Op::RayQueryGetIntersectionTKHR,
            spirv::Op::RayQueryGetIntersectionInstanceIdKHR,
            spirv::Op::RayQueryGetIntersectionPrimitiveIndexKHR,
            spirv::Op::RayQueryGetIntersectionGeometryIndexKHR,
            spirv::Op::RayQueryGetIntersectionObjectToWorldKHR,
            spirv::Op::RayQueryGetIntersectionWorldToObjectKHR,
            spirv::Op::RayQueryGetIntersectionFrontFaceKHR,
            spirv::Op::RayQueryGetIntersectionBarycentricsKHR,
        ] {
            assert!(
                insts.iter().any(|&(op, _)| op == required_op as u32),
                "missing expected RayQuery op {required_op:?}"
            );
        }
        assert!(
            !insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::CompositeExtract as u32),
            "vec2 RayQueryCommittedBarycentrics must not truncate to scalar"
        );
        validate_with_spirv_val_if_available(&words, "full_ray_query");
    }

    #[test]
    fn emits_64bit_and_f32_atomics_and_memory_barrier() {
        let mut func = IrFunction::new("atomics_64_f32");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.uses_bda = true;
        func.workgroup_vars.push(IrWorkgroupVar::typed(
            "wg_u64",
            IrType::Scalar(ScalarIrTy::U64),
        ));
        func.workgroup_vars.push(IrWorkgroupVar::typed(
            "wg_f32",
            IrType::Scalar(ScalarIrTy::F32),
        ));

        let wg_u64_ptr = func.emit(
            None,
            InstData::WorkgroupAddr {
                var_index: 0,
                index: None,
            },
        );
        let one_u64 = func.emit(None, InstData::ConstU64(1));
        let a64 = func.emit_typed(
            IrType::Scalar(ScalarIrTy::U64),
            InstData::AtomicAdd {
                ptr: wg_u64_ptr,
                value: one_u64,
                scope: AtomicScope::Workgroup,
                order: AtomicOrder::AcqRel,
            },
        );
        let wg_f32_ptr = func.emit(
            None,
            InstData::WorkgroupAddr {
                var_index: 1,
                index: None,
            },
        );
        let one_f32 = func.emit(None, InstData::const_f32(1.0));
        let af32 = func.emit_typed(
            IrType::Scalar(ScalarIrTy::F32),
            InstData::AtomicAdd {
                ptr: wg_f32_ptr,
                value: one_f32,
                scope: AtomicScope::Workgroup,
                order: AtomicOrder::Relaxed,
            },
        );
        let mem_bar = func.emit(
            None,
            InstData::MemoryBarrier {
                scope: AtomicScope::Device,
                order: AtomicOrder::AcqRel,
            },
        );
        func.body.push(StructuredStmt::Eval(a64));
        func.body.push(StructuredStmt::Eval(af32));
        func.body.push(StructuredStmt::Eval(mem_bar));

        let words = emit_module_spirv(&IrModule {
            name: "pack::atomics".into(),
            functions: vec![func],
        });
        let insts = decode_instructions(&words);

        assert!(insts.iter().any(|&(op, ops)| {
            op == spirv::Op::Capability as u32 && ops == [spirv::Capability::Int64Atomics as u32]
        }));
        assert!(insts.iter().any(|&(op, ops)| {
            op == spirv::Op::Capability as u32
                && ops == [spirv::Capability::AtomicFloat32AddEXT as u32]
        }));
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::AtomicIAdd as u32)
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::AtomicFAddEXT as u32)
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::MemoryBarrier as u32)
        );
        validate_with_spirv_val_if_available(&words, "atomics_64_f32");
    }

    #[test]
    fn emits_subgroup_exclusive_bitcount_arith_shuffle_vote_and_quad_ops() {
        let mut func = IrFunction::new("subgroup_quad_ops");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.uses_quad_control = true;

        let pred = func.emit(None, InstData::ConstBool(true));
        let ballot = func.emit(None, InstData::SubgroupBallot(pred));
        let excl = func.emit(None, InstData::SubgroupBallotExclusiveBitCount(ballot));
        let incl = func.emit(None, InstData::SubgroupBallotInclusiveBitCount(ballot));
        let v = func.emit(None, InstData::const_f32(2.0));
        let s_add = func.emit(None, InstData::SubgroupAdd(v));
        let lane = func.emit(
            None,
            InstData::LoadBuiltin {
                kind: BuiltinKind::SubgroupLocalInvocationId,
                component: None,
            },
        );
        let shuf = func.emit(None, InstData::SubgroupShuffle { value: v, lane });
        let any_p = func.emit(None, InstData::SubgroupAny(pred));
        let elect = func.emit(None, InstData::SubgroupElect);
        let qh = func.emit(None, InstData::QuadSwapHorizontal(v));

        for id in [excl, incl, s_add, shuf, any_p, elect, qh] {
            func.body.push(StructuredStmt::Eval(id));
        }

        let words = emit_module_spirv(&IrModule {
            name: "pack::subgroups".into(),
            functions: vec![func],
        });
        let insts = decode_instructions(&words);

        assert!(insts.iter().any(|&(op, ops)| {
            op == spirv::Op::GroupNonUniformBallotBitCount as u32
                && ops.len() == 5
                && ops[3] == spirv::GroupOperation::ExclusiveScan as u32
        }));
        assert!(insts.iter().any(|&(op, ops)| {
            op == spirv::Op::GroupNonUniformBallotBitCount as u32
                && ops.len() == 5
                && ops[3] == spirv::GroupOperation::InclusiveScan as u32
        }));
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::GroupNonUniformFAdd as u32)
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::GroupNonUniformShuffle as u32)
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::GroupNonUniformAny as u32)
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::GroupNonUniformElect as u32)
        );
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::GroupNonUniformQuadSwap as u32)
        );
        assert!(insts.iter().any(|&(op, ops)| {
            op == spirv::Op::ExecutionMode as u32
                && ops.len() == 2
                && ops[1] == spirv::ExecutionMode::QuadDerivativesKHR as u32
        }));
        validate_with_spirv_val_if_available(&words, "subgroup_quad_ops");
    }

    #[test]
    fn emits_vertex_and_fragment_stage_io_locations_and_demote() {
        let mut vert = IrFunction::new("vs_stage_io");
        vert.stage = Some(ShaderStage::Vertex);
        vert.stage_io.push(IrStageIoVar {
            name: "v_uv".into(),
            location: 0,
            is_output: true,
            ty: IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 2,
            },
            interpolation: IrInterpolation::Smooth,
            per_primitive: false,
        });
        let f0 = vert.emit(None, InstData::const_f32(0.0));
        let f1 = vert.emit(None, InstData::const_f32(1.0));
        let pos = vert.emit(None, InstData::MakeVec4(f0, f0, f0, f1));
        let uv = vert.emit(None, InstData::MakeVec2(f0, f1));
        let st_pos = vert.emit(
            None,
            InstData::StoreBuiltin {
                kind: BuiltinKind::Position,
                value: pos,
            },
        );
        let st_uv = vert.emit(
            None,
            InstData::StoreStageOutput {
                var_index: 0,
                value: uv,
            },
        );
        vert.body.push(StructuredStmt::Eval(st_pos));
        vert.body.push(StructuredStmt::Eval(st_uv));

        let mut frag = IrFunction::new("fs_stage_io");
        frag.stage = Some(ShaderStage::Fragment);
        frag.stage_io.push(IrStageIoVar {
            name: "v_flat_id".into(),
            location: 1,
            is_output: false,
            ty: IrType::Scalar(ScalarIrTy::U32),
            interpolation: IrInterpolation::Flat,
            per_primitive: false,
        });
        frag.stage_io.push(IrStageIoVar {
            name: "out_color".into(),
            location: 0,
            is_output: true,
            ty: IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 4,
            },
            interpolation: IrInterpolation::Smooth,
            per_primitive: false,
        });
        let flat_in = frag.emit(None, InstData::LoadStageInput { var_index: 0 });
        let ff = frag.emit(
            None,
            InstData::LoadBuiltin {
                kind: BuiltinKind::FrontFacing,
                component: None,
            },
        );
        let demote = frag.emit(None, InstData::DemoteToHelperInvocation);
        let depth_val = frag.emit(None, InstData::const_f32(0.5));
        let st_depth = frag.emit(
            None,
            InstData::StoreBuiltin {
                kind: BuiltinKind::FragDepth,
                value: depth_val,
            },
        );
        let c1 = frag.emit(None, InstData::const_f32(1.0));
        let col = frag.emit(None, InstData::MakeVec4(c1, c1, c1, c1));
        let st_col = frag.emit(
            None,
            InstData::StoreStageOutput {
                var_index: 1,
                value: col,
            },
        );
        frag.body.push(StructuredStmt::Eval(flat_in));
        frag.body.push(StructuredStmt::If {
            cond: ff,
            then_body: Vec::new(),
            else_body: vec![StructuredStmt::Eval(demote)],
        });
        frag.body.push(StructuredStmt::Eval(st_depth));
        frag.body.push(StructuredStmt::Eval(st_col));

        let words = emit_module_spirv(&IrModule {
            name: "pack::stage_io".into(),
            functions: vec![vert, frag],
        });
        let insts = decode_instructions(&words);

        assert!(insts.iter().any(|&(op, ops)| {
            op == spirv::Op::Decorate as u32
                && ops.len() == 3
                && ops[1] == spirv::Decoration::Location as u32
                && ops[2] == 1
        }));
        assert!(insts.iter().any(|&(op, ops)| {
            op == spirv::Op::Decorate as u32
                && ops.len() == 2
                && ops[1] == spirv::Decoration::Flat as u32
        }));
        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::DemoteToHelperInvocation as u32)
        );
        assert!(insts.iter().any(|&(op, ops)| {
            op == spirv::Op::ExecutionMode as u32
                && ops.len() == 2
                && ops[1] == spirv::ExecutionMode::DepthReplacing as u32
        }));
        validate_with_spirv_val_if_available(&words, "vertex_fragment_stage_io");
    }

    #[test]
    fn emits_spec_constant_op_and_local_size_id() {
        let mut func = IrFunction::new("spec_op_and_local_size_id");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([8, 8, 1]);
        func.local_size_spec_ids = Some([10, 11, 12]);
        func.workgroup_vars.push(IrWorkgroupVar::typed(
            "s_spec_out",
            IrType::Scalar(ScalarIrTy::U32),
        ));

        let sx = func.emit(
            None,
            InstData::SpecConstant {
                spec_id: 10,
                scalar: ScalarIrTy::U32,
                default_bits: 8,
            },
        );
        let sy = func.emit(
            None,
            InstData::SpecConstant {
                spec_id: 11,
                scalar: ScalarIrTy::U32,
                default_bits: 8,
            },
        );
        let threads = func.emit_typed(
            IrType::Scalar(ScalarIrTy::U32),
            InstData::SpecConstantOp {
                op: SpecConstantOpKind::IMul,
                operands: vec![sx, sy],
            },
        );
        let store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: threads,
            },
        );
        func.body.push(StructuredStmt::Eval(store));

        let words = emit_module_spirv(&IrModule {
            name: "pack::spec_local_size".into(),
            functions: vec![func],
        });
        let insts = decode_instructions(&words);

        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::SpecConstantOp as u32),
            "expected OpSpecConstantOp"
        );
        assert!(
            insts.iter().any(|&(op, ops)| {
                op == spirv::Op::ExecutionModeId as u32
                    && ops.len() == 5
                    && ops[1] == spirv::ExecutionMode::LocalSizeId as u32
            }),
            "expected OpExecutionModeId LocalSizeId with 3 spec constant IDs"
        );
        assert!(
            !insts.iter().any(|&(op, ops)| {
                op == spirv::Op::ExecutionMode as u32
                    && ops.len() >= 2
                    && ops[1] == spirv::ExecutionMode::LocalSize as u32
            }),
            "must not emit both LocalSize and LocalSizeId"
        );
        validate_with_spirv_val_if_available(&words, "spec_op_and_local_size_id");
    }

    #[test]
    fn unpack_half_2x16_emits_glsl_opcode_62() {
        let mut func = IrFunction::new("unpack_half_2x16_test");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([1, 1, 1]);
        func.workgroup_vars.push(IrWorkgroupVar::typed(
            "s_out",
            IrType::Scalar(ScalarIrTy::F32),
        ));

        let packed = func.emit(None, InstData::ConstU32(0x3C00_3C00));
        let unpacked = func.emit_typed(
            IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 2,
            },
            InstData::GlslStd450 {
                op: GlslStd450Op::UnpackHalf2x16,
                args: vec![packed],
            },
        );
        let x = func.emit_typed(
            IrType::Scalar(ScalarIrTy::F32),
            InstData::CompositeExtract {
                composite: unpacked,
                indices: vec![0],
            },
        );
        let store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: x,
            },
        );
        func.body.push(StructuredStmt::Eval(store));

        let words = emit_module_spirv(&IrModule {
            name: "pack::unpack_half_2x16".into(),
            functions: vec![func],
        });
        let insts = decode_instructions(&words);

        assert!(
            insts
                .iter()
                .any(|&(op, ops)| op == spirv::Op::ExtInst as u32
                    && ops.len() >= 4
                    && ops[3] == 62),
            "expected GLSL.std.450 UnpackHalf2x16 opcode 62"
        );
        assert!(
            !insts
                .iter()
                .any(|&(op, ops)| op == spirv::Op::ExtInst as u32
                    && ops.len() >= 4
                    && ops[3] == 60),
            "must not emit UnpackSnorm2x16 opcode 60 for UnpackHalf2x16"
        );
        validate_with_spirv_val_if_available(&words, "unpack_half_2x16_opcode_62");
    }

    #[test]
    fn spirv_emitter_default_starts_at_id_one() {
        let mut emitter = SpirvEmitter::default();
        assert_eq!(emitter.alloc_id(), 1);
        assert_eq!(emitter.alloc_id(), 2);
    }

    #[test]
    fn storage_image_query_size_always_emits_op_image_query_size() {
        let mut func = IrFunction::new("storage_img_query_size");
        func.stage = Some(ShaderStage::Compute);
        func.workgroup_size = Some([1, 1, 1]);
        func.descriptors.push(IrDescriptorVar {
            name: "u_img".into(),
            set: 0,
            binding: 1,
            kind: DescriptorKind::StorageImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: Some(SampleScalar::F32),
            image_format: Some(ImageFormat::Rgba16f),
            access: DescriptorAccess::ReadWrite,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });
        func.workgroup_vars.push(IrWorkgroupVar::typed(
            "s_width",
            IrType::Scalar(ScalarIrTy::U32),
        ));

        let lod_zero = func.emit(None, InstData::ConstI32(0));
        let size = func.emit_typed(
            IrType::Vector {
                scalar: ScalarIrTy::U32,
                len: 2,
            },
            InstData::ImageQuerySize {
                descriptor_index: 0,
                lod: Some(lod_zero),
            },
        );
        let width = func.emit_typed(
            IrType::Scalar(ScalarIrTy::U32),
            InstData::CompositeExtract {
                composite: size,
                indices: vec![0],
            },
        );
        let store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: width,
            },
        );
        func.body.push(StructuredStmt::Eval(store));

        let words = emit_module_spirv(&IrModule {
            name: "pack::storage_img_query_size".into(),
            functions: vec![func],
        });
        let insts = decode_instructions(&words);

        assert!(
            insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::ImageQuerySize as u32),
            "expected OpImageQuerySize for StorageImage"
        );
        assert!(
            !insts
                .iter()
                .any(|&(op, _)| op == spirv::Op::ImageQuerySizeLod as u32),
            "must not emit OpImageQuerySizeLod for StorageImage even when lod is Some"
        );
        validate_with_spirv_val_if_available(&words, "storage_img_query_size");
    }

    #[test]
    fn i64_arithmetic_and_capabilities_are_clean() {
        // Pure 32-bit shader must not emit spurious OpCapability Int64.
        let mut func32 = IrFunction::new("pure_32bit");
        func32.stage = Some(ShaderStage::Compute);
        func32.workgroup_size = Some([1, 1, 1]);
        func32.workgroup_vars.push(IrWorkgroupVar::typed(
            "s_out32",
            IrType::Scalar(ScalarIrTy::U32),
        ));
        let a32 = func32.emit(None, InstData::ConstU32(10));
        let b32 = func32.emit(None, InstData::ConstU32(3));
        let cmp32 = func32.emit(None, InstData::ICmpLt(b32, a32));
        let sum32 = func32.emit(None, InstData::IAdd32(a32, b32));
        let sel32 = func32.emit(
            None,
            InstData::Select {
                cond: cmp32,
                on_true: sum32,
                on_false: a32,
            },
        );
        let store32 = func32.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: sel32,
            },
        );
        func32.body.push(StructuredStmt::Eval(store32));

        let words32 = emit_module_spirv(&IrModule {
            name: "pack::pure_32bit".into(),
            functions: vec![func32],
        });
        let insts32 = decode_instructions(&words32);
        assert!(
            !insts32.iter().any(|&(op, ops)| {
                op == spirv::Op::Capability as u32 && ops == [spirv::Capability::Int64 as u32]
            }),
            "pure 32-bit shader must not emit OpCapability Int64"
        );
        validate_with_spirv_val_if_available(&words32, "pure_32bit_no_int64");

        // 64-bit integer arithmetic must emit OpCapability Int64 and 64-bit result types.
        let mut func64 = IrFunction::new("i64_arith");
        func64.stage = Some(ShaderStage::Compute);
        func64.workgroup_size = Some([1, 1, 1]);
        func64.workgroup_vars.push(IrWorkgroupVar::typed(
            "s_out64",
            IrType::Scalar(ScalarIrTy::I64),
        ));
        let lhs = func64.emit_typed(IrType::Scalar(ScalarIrTy::I64), InstData::ConstI64(100));
        let rhs = func64.emit_typed(IrType::Scalar(ScalarIrTy::I64), InstData::ConstI64(7));
        let shift = func64.emit_typed(IrType::Scalar(ScalarIrTy::I64), InstData::ConstI64(1));
        let sub = func64.emit_typed(IrType::Scalar(ScalarIrTy::I64), InstData::ISub32(lhs, rhs));
        let div = func64.emit_typed(IrType::Scalar(ScalarIrTy::I64), InstData::IDiv32(sub, rhs));
        let shr = func64.emit_typed(IrType::Scalar(ScalarIrTy::I64), InstData::Shr32(div, shift));
        let store64 = func64.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: shr,
            },
        );
        func64.body.push(StructuredStmt::Eval(store64));

        let words64 = emit_module_spirv(&IrModule {
            name: "pack::i64_arith".into(),
            functions: vec![func64],
        });
        let insts64 = decode_instructions(&words64);
        assert!(
            insts64.iter().any(|&(op, ops)| {
                op == spirv::Op::Capability as u32 && ops == [spirv::Capability::Int64 as u32]
            }),
            "expected OpCapability Int64 for i64 arithmetic"
        );
        let i64_ty_id = insts64
            .iter()
            .find_map(|&(op, ops)| {
                (op == spirv::Op::TypeInt as u32 && ops.len() == 3 && ops[1] == 64 && ops[2] == 1)
                    .then_some(ops[0])
            })
            .expect("expected OpTypeInt 64 1");
        assert!(
            insts64
                .iter()
                .any(|&(op, ops)| op == spirv::Op::ISub as u32 && ops[0] == i64_ty_id),
            "expected 64-bit OpISub"
        );
        assert!(
            insts64
                .iter()
                .any(|&(op, ops)| op == spirv::Op::SDiv as u32 && ops[0] == i64_ty_id),
            "expected 64-bit OpSDiv"
        );
        assert!(
            insts64.iter().any(|&(op, ops)| {
                op == spirv::Op::ShiftRightArithmetic as u32 && ops[0] == i64_ty_id
            }),
            "expected 64-bit OpShiftRightArithmetic"
        );
        validate_with_spirv_val_if_available(&words64, "i64_arith");
    }

    #[test]
    fn emits_rt_pipeline_and_mesh_and_legacy_stages() {
        // 1. RayGen with OpTraceRayKHR
        let mut rg = IrFunction::new("rgen_main");
        rg.stage = Some(ShaderStage::RayGen);
        rg.descriptors.push(IrDescriptorVar {
            name: "tlas".into(),
            set: 0,
            binding: 0,
            kind: DescriptorKind::AccelerationStructure,
            dim: None,
            sample_scalar: None,
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });
        rg.stage_vars.push(IrStageVar {
            name: "payload".into(),
            storage_class: IrStorageClass::RayPayloadKhr,
            ty: IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 4,
            },
            location: Some(0),
            struct_layout: None,
        });
        let u0 = rg.emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(0));
        let f0 = rg.emit_typed(IrType::Scalar(ScalarIrTy::F32), InstData::const_f32(0.0));
        let f1 = rg.emit_typed(IrType::Scalar(ScalarIrTy::F32), InstData::const_f32(1.0));
        let f100 = rg.emit_typed(IrType::Scalar(ScalarIrTy::F32), InstData::const_f32(100.0));
        let orig = rg.emit_typed(
            IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 3,
            },
            InstData::CompositeConstruct {
                components: vec![f0, f0, f0],
            },
        );
        let dir = rg.emit_typed(
            IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 3,
            },
            InstData::CompositeConstruct {
                components: vec![f0, f0, f1],
            },
        );
        let trace = rg.emit_typed(
            IrType::Void,
            InstData::TraceRayKhr {
                accel_descriptor_index: 0,
                ray_flags: u0,
                cull_mask: u0,
                sbt_offset: u0,
                sbt_stride: u0,
                miss_index: u0,
                origin: orig,
                tmin: f0,
                direction: dir,
                tmax: f100,
                payload_var_index: 0,
            },
        );
        rg.body.push(StructuredStmt::Eval(trace));
        let rg_words = emit_module_spirv(&IrModule {
            name: "pack::rt_pipeline".into(),
            functions: vec![rg],
        });
        let rg_insts = decode_instructions(&rg_words);
        assert!(rg_insts.iter().any(|&(op, ops)| op == spirv::Op::Capability as u32 && ops == [4479]));
        assert!(rg_insts.iter().any(|&(op, _)| op == 4445)); // OpTraceRayKHR

        // 2. Mesh stage with OpSetMeshOutputsEXT + OutputVertices/OutputPrimitivesEXT/OutputTrianglesEXT
        let mut ms = IrFunction::new("mesh_main");
        ms.stage = Some(ShaderStage::Mesh);
        ms.workgroup_size = Some([32, 1, 1]);
        ms.stage_exec_meta.max_vertices = Some(64);
        ms.stage_exec_meta.max_primitives = Some(126);
        ms.stage_exec_meta.output_topology = Some(IrTopology::Triangles);
        let v3 = ms.emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(3));
        let p1 = ms.emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(1));
        let set_out = ms.emit_typed(
            IrType::Void,
            InstData::SetMeshOutputsExt {
                vertex_count: v3,
                primitive_count: p1,
            },
        );
        ms.body.push(StructuredStmt::Eval(set_out));
        let ms_words = emit_module_spirv(&IrModule {
            name: "pack::mesh_stage".into(),
            functions: vec![ms],
        });
        let ms_insts = decode_instructions(&ms_words);
        assert!(ms_insts.iter().any(|&(op, ops)| op == spirv::Op::Capability as u32 && ops == [5283]));
        assert!(ms_insts.iter().any(|&(op, _)| op == 5295)); // OpSetMeshOutputsEXT

        // 3. Geometry stage with OpEmitVertex + OpEndPrimitive
        let mut gs = IrFunction::new("geom_main");
        gs.stage = Some(ShaderStage::Geometry);
        gs.stage_exec_meta.max_vertices = Some(3);
        gs.stage_exec_meta.input_topology = Some(IrTopology::Triangles);
        gs.stage_exec_meta.output_topology = Some(IrTopology::TriangleStrip);
        let ev = gs.emit_typed(IrType::Void, InstData::EmitVertex);
        let ep = gs.emit_typed(IrType::Void, InstData::EndPrimitive);
        gs.body.push(StructuredStmt::Eval(ev));
        gs.body.push(StructuredStmt::Eval(ep));
        let gs_words = emit_module_spirv(&IrModule {
            name: "pack::geom_stage".into(),
            functions: vec![gs],
        });
        let gs_insts = decode_instructions(&gs_words);
        assert!(gs_insts.iter().any(|&(op, _)| op == spirv::Op::EmitVertex as u32));
        assert!(gs_insts.iter().any(|&(op, _)| op == spirv::Op::EndPrimitive as u32));
    }

    #[test]
    fn emits_ser_coop_matrix_raster_subgroups_and_debug_info() {
        // 1. Cooperative Matrix
        let mut cm = IrFunction::new("coop_main");
        cm.stage = Some(ShaderStage::Compute);
        cm.workgroup_size = Some([32, 1, 1]);
        let mat_ty = IrType::CoopMatrix {
            scalar: CoopMatrixScalar::F16,
            scope: AtomicScope::Subgroup,
            rows: 16,
            cols: 16,
            use_kind: CoopMatrixUse::MatrixA,
        };
        let len_id = cm.emit_typed(
            IrType::Scalar(ScalarIrTy::U32),
            InstData::CoopMatrixLength { mat_ty },
        );
        cm.body.push(StructuredStmt::Eval(len_id));
        let cm_words = emit_module_spirv(&IrModule {
            name: "pack::coop_matrix".into(),
            functions: vec![cm],
        });
        let cm_insts = decode_instructions(&cm_words);
        assert!(cm_insts.iter().any(|&(op, ops)| op == spirv::Op::Capability as u32 && ops == [6022]));
        assert!(cm_insts.iter().any(|&(op, _)| op == 4456)); // OpTypeCooperativeMatrixKHR
        assert!(cm_insts.iter().any(|&(op, _)| op == 4460)); // OpCooperativeMatrixLengthKHR

        // 2. Raster: EarlyFragmentTests + Fragment Interlock + SubpassData + Dim2DMS
        let mut frag = IrFunction::new("frag_raster");
        frag.stage = Some(ShaderStage::Fragment);
        frag.stage_exec_meta.early_fragment_tests = true;
        frag.stage_exec_meta.interlock_mode = Some(IrInterlockMode::PixelOrdered);
        frag.stage_exec_meta.input_attachment_indices.push((0, 0));
        frag.descriptors.push(IrDescriptorVar {
            name: "gbuf".into(),
            set: 0,
            binding: 0,
            kind: DescriptorKind::InputAttachment,
            dim: Some(TextureDim::SubpassData),
            sample_scalar: Some(SampleScalar::F32),
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
            count: None,
            struct_layout: None,
        });
        let b_il = frag.emit_typed(IrType::Void, InstData::BeginInvocationInterlockExt);
        let sub_ld = frag.emit_typed(
            IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 4,
            },
            InstData::SubpassLoad {
                descriptor_index: 0,
                sample: None,
            },
        );
        let e_il = frag.emit_typed(IrType::Void, InstData::EndInvocationInterlockExt);
        frag.body.push(StructuredStmt::Eval(b_il));
        frag.body.push(StructuredStmt::Eval(sub_ld));
        frag.body.push(StructuredStmt::Eval(e_il));
        let frag_words = emit_module_spirv(&IrModule {
            name: "pack::raster_ext".into(),
            functions: vec![frag],
        });
        let frag_insts = decode_instructions(&frag_words);
        assert!(frag_insts.iter().any(|&(op, _)| op == 5364)); // OpBeginInvocationInterlockEXT
        assert!(frag_insts.iter().any(|&(op, _)| op == 5365)); // OpEndInvocationInterlockEXT
        assert!(frag_insts.iter().any(|&(op, ops)| {
            op == spirv::Op::ExecutionMode as u32
                && ops.len() >= 2
                && ops[1] == spirv::ExecutionMode::EarlyFragmentTests as u32
        }));

        // 3. Clustered Subgroup + QuadBroadcast + DebugInfo.100 + DebugPrintf
        let mut dbg_fn = IrFunction::new("dbg_sg_main");
        dbg_fn.stage = Some(ShaderStage::Compute);
        dbg_fn.workgroup_size = Some([64, 1, 1]);
        dbg_fn.debug_source = Some(vsl_ir::IrDebugSource {
            file_name: "shader.vsl".into(),
            source_text: Some("@compute fn dbg_sg_main() {}".into()),
        });
        let v = dbg_fn.emit_typed(IrType::Scalar(ScalarIrTy::F32), InstData::const_f32(1.5));
        let cs = dbg_fn.emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(4));
        let cl = dbg_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::F32),
            InstData::SubgroupClustered {
                op: SubgroupClusteredOp::Add,
                value: v,
                cluster_size: cs,
            },
        );
        let pf = dbg_fn.emit_typed(
            IrType::Void,
            InstData::DebugPrintf {
                format: "cl=%f".into(),
                args: vec![cl],
            },
        );
        dbg_fn.body.push(StructuredStmt::Eval(cl));
        dbg_fn.body.push(StructuredStmt::Eval(pf));
        let dbg_words = emit_module_spirv(&IrModule {
            name: "pack::dbg_sg".into(),
            functions: vec![dbg_fn],
        });
        let dbg_insts = decode_instructions(&dbg_words);
        // Verify OpString appears BEFORE OpName per SPIR-V section 2.4
        let first_string = dbg_insts
            .iter()
            .position(|&(op, _)| op == spirv::Op::String as u32)
            .expect("expected OpString");
        let first_name = dbg_insts
            .iter()
            .position(|&(op, _)| op == spirv::Op::Name as u32)
            .expect("expected OpName");
        assert!(first_string < first_name, "OpString must precede OpName");
        assert!(dbg_insts.iter().any(|&(op, ops)| op == spirv::Op::Capability as u32 && ops == [67]));
    }
}
