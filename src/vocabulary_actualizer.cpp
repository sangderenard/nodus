#include "vocabulary_actualizer.h"

#include "canonical_ops.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// Emission helpers
// ---------------------------------------------------------------------------

bool atomic_write_file(const fs::path& target, const std::string& contents, std::string& err) {
    fs::path tmp;
    try {
        auto parent = target.parent_path();
        if (!parent.empty()) fs::create_directories(parent);
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::ostringstream ss;
        ss << std::hex << gen();
        tmp = parent / (target.filename().string() + ".tmp-" + ss.str());
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs.good()) {
            err = "failed to open temp file: " + tmp.string();
            return false;
        }
        ofs << contents;
        ofs.flush();
        ofs.close();
        // Overwrite-on-regenerate: the catalog is the authority, so stale
        // sources must refresh (unlike the module actualizer's
        // write_placeholder_if_missing policy, which protects hand edits).
        fs::rename(tmp, target);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        try { if (!tmp.empty() && fs::exists(tmp)) fs::remove(tmp); } catch (...) {}
        return false;
    }
}

std::string sanitize_identifier(const std::string& id) {
    std::string out;
    out.reserve(id.size());
    for (char ch : id) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) {
            out.push_back(ch);
        } else {
            out.push_back('_');
        }
    }
    if (out.empty() || (out[0] >= '0' && out[0] <= '9')) out.insert(out.begin(), '_');
    return out;
}

// ---------------------------------------------------------------------------
// Eigen expression catalog
//
// One expression per canonical operation, phrased over:
//   a, b : Eigen::Map<const Eigen::ArrayXf>   (b only for binary ops)
// assigned to r (Eigen::Map<Eigen::ArrayXf>). Anything Eigen's Array API does
// not provide natively goes through unaryExpr/binaryExpr with the <cmath>
// function, so semantics stay legible. The vocab checker diffs this group
// against the inmemory group per op, so a semantic divergence here is caught
// by test, not by review alone.
// ---------------------------------------------------------------------------

const std::unordered_map<std::string, std::string>& eigen_unary_exprs() {
    static const std::unordered_map<std::string, std::string> table = {
        {"sqrt",        "a.sqrt()"},
        {"exp",         "a.exp()"},
        {"log",         "a.log()"},
        {"neg",         "-a"},
        {"abs",         "a.abs()"},
        {"round",       "a.round()"},
        {"trunc",       "a.unaryExpr([](float v) { return std::trunc(v); })"},
        {"floor",       "a.floor()"},
        {"ceil",        "a.ceil()"},
        {"isfinite",    "a.isFinite().cast<float>()"},
        {"isnan",       "a.isNaN().cast<float>()"},
        {"isinf",       "a.isInf().cast<float>()"},
        {"logical_not", "(a == 0.0f).cast<float>()"},
        {"sin",         "a.sin()"},
        {"cos",         "a.cos()"},
        {"tan",         "a.tan()"},
        {"asin",        "a.asin()"},
        {"acos",        "a.acos()"},
        {"atan",        "a.atan()"},
        {"sinh",        "a.sinh()"},
        {"cosh",        "a.cosh()"},
        {"tanh",        "a.tanh()"},
        {"asinh",       "a.unaryExpr([](float v) { return std::asinh(v); })"},
        {"acosh",       "a.unaryExpr([](float v) { return std::acosh(v); })"},
        {"atanh",       "a.unaryExpr([](float v) { return std::atanh(v); })"},
    };
    return table;
}

const std::unordered_map<std::string, std::string>& eigen_binary_exprs() {
    static const std::unordered_map<std::string, std::string> table = {
        {"add",           "a + b"},
        {"sub",           "a - b"},
        {"mul",           "a * b"},
        {"truediv",       "a / b"},
        {"pow",           "a.binaryExpr(b, [](float x, float y) { return std::pow(x, y); })"},
        {"mod",           "a.binaryExpr(b, [](float x, float y) { return std::fmod(x, y); })"},
        {"floordiv",      "a.binaryExpr(b, [](float x, float y) { return std::floor(x / y); })"},
        {"less",          "(a < b).cast<float>()"},
        {"less_equal",    "(a <= b).cast<float>()"},
        {"greater",       "(a > b).cast<float>()"},
        {"greater_equal", "(a >= b).cast<float>()"},
        {"equal",         "(a == b).cast<float>()"},
        {"not_equal",     "(a != b).cast<float>()"},
        {"maximum",       "a.max(b)"},
        {"minimum",       "a.min(b)"},
    };
    return table;
}

// ---------------------------------------------------------------------------
// Per-op source generation
//
// Generated tools speak the cross-image-safe currency end to end:
//   - stack values are AbstractTensorHandle (VT_ABSTRACT_TENSOR) -- an 8-byte
//     POD id, never a C++ object pointer, so a tool loaded from a standalone
//     DLL agrees with in-process tools about what is on the stack;
//   - all allocation, description, payload access, and (for the inmemory
//     group) computation go through the nodus_tensor_* C ABI (tensor_abi.h),
//     which is the transport layer this repo built for exactly this crossing.
// The eigen group maps payloads through the same ABI and runs Eigen
// expressions locally -- Eigen is header-only, so the compute lives inside
// the generated tool while data access stays on the sanctioned boundary.
//
// Operand contract mirrors register_abstract_tensor_tool_ir: operands popped
// right-then-left, everything restored untouched on failure, one output
// pushed on success. Input handles are consumed from the stack but not
// destroyed -- lifetime stays with the producer, matching the registrar's
// caller-owns policy. The output handle is owned by whoever pops it.
// ---------------------------------------------------------------------------

struct GroupInfo {
    const char* dir_name;   // subdirectory under source/tools/vocab/
    const char* group_tag;  // value returned by nodus_tool_backend_group()
    bool id_qualified;      // prefix the tool id with "<group>." ?
};

void emit_stack_prologue(std::ostringstream& ss, int arity) {
    ss << "        if (!ctx.stack.raw) return;\n";
    ss << "        RawStackFrame& frame = *ctx.stack.raw;\n";
    ss << "        nodus::tensors::AbstractTensorHandle right{};\n";
    ss << "        nodus::tensors::AbstractTensorHandle left{};\n";
    if (arity == 2) {
        ss << "        if (!raw_stack_pop_abstract_tensor(frame, right)) return;\n";
    }
    ss << "        if (!raw_stack_pop_abstract_tensor(frame, left)) {\n";
    if (arity == 2) {
        ss << "            raw_stack_push_abstract_tensor(frame, right);\n";
    }
    ss << "            return;\n";
    ss << "        }\n";
    ss << "        auto restore = [&]() {\n";
    ss << "            raw_stack_push_abstract_tensor(frame, left);\n";
    if (arity == 2) {
        ss << "            raw_stack_push_abstract_tensor(frame, right);\n";
    }
    ss << "        };\n";
    ss << "        NodusTensorDesc left_desc{};\n";
    ss << "        if (nodus_tensor_describe(left.id, &left_desc) != NODUS_OK) {\n";
    ss << "            restore(); return;\n";
    ss << "        }\n";
    if (arity == 2) {
        ss << "        NodusTensorDesc right_desc{};\n";
        ss << "        if (nodus_tensor_describe(right.id, &right_desc) != NODUS_OK) {\n";
        ss << "            restore(); return;\n";
        ss << "        }\n";
    }
    ss << "        nodus::tensors::AbstractTensorHandle output{};\n";
    ss << "        output.id = nodus_tensor_create(left_desc.dtype, left_desc.rank,\n";
    ss << "                                        left_desc.shape);\n";
    ss << "        if (output.id == 0) { restore(); return; }\n";
    ss << "        auto fail = [&]() {\n";
    ss << "            nodus_tensor_destroy(output.id);\n";
    ss << "            restore();\n";
    ss << "        };\n";
}

void emit_stack_epilogue(std::ostringstream& ss) {
    ss << "        if (!raw_stack_push_abstract_tensor(frame, output)) { fail(); return; }\n";
}

std::string generate_inmemory_body(const nodus::ops::OpDesc& op) {
    std::ostringstream ss;
    emit_stack_prologue(ss, op.arity);
    if (op.arity == 1) {
        ss << "        if (nodus_tensor_unary(" << op.canonical_id
           << " /* " << op.name << " */, left.id, output.id) != NODUS_OK) {\n";
    } else {
        ss << "        if (nodus_tensor_binary(" << op.canonical_id
           << " /* " << op.name << " */, left.id, right.id, output.id) != NODUS_OK) {\n";
    }
    ss << "            fail(); return;\n";
    ss << "        }\n";
    emit_stack_epilogue(ss);
    return ss.str();
}

std::string generate_eigen_body(const nodus::ops::OpDesc& op, const std::string& expr) {
    std::ostringstream ss;
    emit_stack_prologue(ss, op.arity);
    // Eigen engine boundary: float32 payloads only for now. Anything else is
    // restored untouched -- the same honest-boundary policy as tensor_math.h's
    // dense helpers (prove semantics on the native path first, widen later).
    ss << "        const bool engine_ok =\n";
    ss << "            left_desc.dtype == NODUS_DTYPE_F32";
    if (op.arity == 2) {
        ss << " &&\n            right_desc.dtype == NODUS_DTYPE_F32 &&\n";
        ss << "            right_desc.element_count == left_desc.element_count";
    }
    ss << ";\n";
    ss << "        if (!engine_ok || left_desc.element_count == 0) { fail(); return; }\n";
    ss << "        const auto n = static_cast<Eigen::Index>(left_desc.element_count);\n";
    ss << "        void* left_data = nullptr; uint64_t left_bytes = 0;\n";
    ss << "        void* out_data = nullptr; uint64_t out_bytes = 0;\n";
    if (op.arity == 2) {
        ss << "        void* right_data = nullptr; uint64_t right_bytes = 0;\n";
    }
    ss << "        const bool mapped =\n";
    ss << "            nodus_tensor_map(left.id, &left_data, &left_bytes) == NODUS_OK &&\n";
    if (op.arity == 2) {
        ss << "            nodus_tensor_map(right.id, &right_data, &right_bytes) == NODUS_OK &&\n";
    }
    ss << "            nodus_tensor_map(output.id, &out_data, &out_bytes) == NODUS_OK;\n";
    ss << "        if (mapped) {\n";
    ss << "            Eigen::Map<const Eigen::ArrayXf> a(static_cast<const float*>(left_data), n);\n";
    if (op.arity == 2) {
        ss << "            Eigen::Map<const Eigen::ArrayXf> b(static_cast<const float*>(right_data), n);\n";
    }
    ss << "            Eigen::Map<Eigen::ArrayXf> r(static_cast<float*>(out_data), n);\n";
    ss << "            r = " << expr << ";\n";
    ss << "        }\n";
    ss << "        if (left_data) nodus_tensor_unmap(left.id);\n";
    if (op.arity == 2) {
        ss << "        if (right_data) nodus_tensor_unmap(right.id);\n";
    }
    ss << "        if (out_data) nodus_tensor_unmap(output.id);\n";
    ss << "        if (!mapped) { fail(); return; }\n";
    emit_stack_epilogue(ss);
    return ss.str();
}

std::string generate_op_source(const nodus::ops::OpDesc& op,
                               const GroupInfo& group,
                               const std::string& tool_id,
                               const std::string& body) {
    const std::string sym = sanitize_identifier(tool_id);
    const bool is_eigen = std::string(group.group_tag) == "eigen";
    std::ostringstream ss;
    ss << "// Generated by vocabulary_actualizer -- DO NOT EDIT.\n";
    ss << "// Catalog authority: ops/canonical_ops.json (via include/canonical_ops.h).\n";
    ss << "// op: " << op.name << "  canonical_id: " << op.canonical_id
       << "  arity: " << static_cast<int>(op.arity)
       << "  backend_group: " << group.group_tag << "\n";
    ss << "#include \"tool_api.h\"\n";
    ss << "#include \"value_types.h\"\n";
    ss << "#include \"common/tensors/abstraction/abstract_tensor_handle.h\"\n";
    ss << "#include \"common/tensors/abstraction/tensor_abi.h\"\n";
    if (is_eigen) {
        ss << "#include <Eigen/Dense>\n";
    }
    ss << "#include <cmath>\n";
    ss << "\n";
    ss << "namespace {\n";
    ss << "class VocabTool_" << sym << " final : public ITool {\n";
    ss << "public:\n";
    ss << "    const char* id_cstr() const noexcept override { return \"" << tool_id << "\"; }\n";
    ss << "    const char* name_cstr() const noexcept override { return \""
       << group.group_tag << " " << op.name << "\"; }\n";
    ss << "    ToolCaps caps() const override { return static_cast<ToolCaps>(0); }\n";
    ss << "    void initialize(const ToolInitContext&) override {}\n";
    ss << "    void shutdown() override {}\n";
    ss << "    void tick(double, HostAPI&) override {}\n";
    ss << "    void render(RenderContext&) override {}\n";
    ss << "    int32_t port_count() const override { return 2; }\n";
    ss << "    ToolPortSpec port_spec(int32_t idx) const override {\n";
    ss << "        if (idx == 0) return ToolPortSpec{ToolPortKind::Argument, "
       << static_cast<int>(op.arity) << "};\n";
    ss << "        if (idx == 1) return ToolPortSpec{ToolPortKind::Return, 1};\n";
    ss << "        return ToolPortSpec{};\n";
    ss << "    }\n";
    ss << "    void execute_stack(ToolStackContext& ctx) override {\n";
    ss << body;
    ss << "    }\n";
    ss << "};\n";
    ss << "} // namespace\n";
    ss << "\n";
    ss << "extern \"C\" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() {\n";
    ss << "    return new VocabTool_" << sym << "();\n";
    ss << "}\n";
    ss << "extern \"C\" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* t) {\n";
    ss << "    delete t;\n";
    ss << "}\n";
    ss << "extern \"C\" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() {\n";
    ss << "    return __FILE__;\n";
    ss << "}\n";
    ss << "#if !defined(NODUS_PLUGIN_COMPOSITE)\n";
    ss << "extern \"C\" NODUS_PLUGIN_EXPORT const char* nodus_tool_backend_group() {\n";
    ss << "    return \"" << group.group_tag << "\";\n";
    ss << "}\n";
    ss << "#endif\n";
    return ss.str();
}

} // namespace

extern "C" int gp_vocabulary_actualize_tools(const char* output_root,
                                             uint32_t group_mask,
                                             GP_VocabActualizeReport* report) {
    if (!output_root || !*output_root || group_mask == 0u) return 0;

    GP_VocabActualizeReport local{};
    const fs::path root = output_root;
    const fs::path vocab_dir = root / "source" / "tools" / "vocab";

    std::vector<GroupInfo> groups;
    if (group_mask & GP_VOCAB_GROUP_INMEMORY) groups.push_back({"inmemory", "inmemory", false});
    if (group_mask & GP_VOCAB_GROUP_EIGEN) groups.push_back({"eigen", "eigen", true});

    std::ostringstream manifest;
    manifest << "VOCABLIB V1\n";

    bool io_ok = true;
    for (const auto& op : nodus::ops::kOps) {
        ++local.ops_considered;
        // The membrane gate: exactly the subset the CPU evaluator implements.
        // Same filter as register_abstract_tensor_tool_ir and the canvas
        // bridge -- widening it ahead of the evaluator mints tools that
        // compute wrong, so it must stay in lockstep with them.
        if (op.ct_value < 0 || (op.arity != 1 && op.arity != 2)) continue;
        ++local.ops_eligible;

        for (const auto& group : groups) {
            const std::string name(op.name);
            std::string body;
            if (std::string(group.group_tag) == "inmemory") {
                body = generate_inmemory_body(op);
            } else { // eigen
                const auto& table = (op.arity == 1) ? eigen_unary_exprs() : eigen_binary_exprs();
                auto it = table.find(name);
                if (it == table.end()) {
                    std::cerr << "vocabulary_actualizer: shortfall: op '" << name
                              << "' has no eigen expression registered; source withheld\n";
                    ++local.shortfalls;
                    continue;
                }
                body = generate_eigen_body(op, it->second);
            }

            const std::string tool_id = group.id_qualified
                ? (std::string(group.group_tag) + ".abstract_tensor." + name)
                : ("abstract_tensor." + name);
            const fs::path out_path =
                vocab_dir / group.dir_name / ("abstract_tensor_" + name + ".cpp");
            const std::string contents = generate_op_source(op, group, tool_id, body);
            std::string err;
            if (!atomic_write_file(out_path, contents, err)) {
                std::cerr << "vocabulary_actualizer: failed to write "
                          << out_path.generic_string() << ": " << err << "\n";
                io_ok = false;
                continue;
            }
            ++local.sources_written;
            manifest << "VOCABTOOL " << tool_id << " " << group.group_tag << " \""
                     << out_path.generic_string() << "\"\n";
            ++local.manifest_entries;
        }
    }

    std::string err;
    if (!atomic_write_file(root / "vocab_manifest.txt", manifest.str(), err)) {
        std::cerr << "vocabulary_actualizer: failed to write manifest: " << err << "\n";
        io_ok = false;
    }

    if (report) *report = local;
    std::cerr << "vocabulary_actualizer: considered=" << local.ops_considered
              << " eligible=" << local.ops_eligible
              << " written=" << local.sources_written
              << " shortfalls=" << local.shortfalls << "\n";
    return io_ok ? 1 : 0;
}
