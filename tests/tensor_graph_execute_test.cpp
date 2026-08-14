// tensor_graph_execute_test.cpp
//
// Executes a GraphIR script emitted by a frontend (turing's
// process_graph_to_nodus_graph_ir) through the REGISTERED abstract_tensor
// tool vocabulary, and compares the numeric result against an expected
// vector computed by the source ecosystem.
//
// This is the destination half of shallow interpretation: a composite
// operation arrives as a graph of canonical operator names, every name
// resolves to a tool this ecosystem already owns, and the values prove the
// composition computes the same thing here that it computes at the source.
// Nothing in this file knows which composite is coming.
//
// Usage: tensor_graph_execute <script.graphir> <values.txt>
//   values.txt lines:
//     input <v0> <v1> ...     one line per graph input, in emission order
//     expect <v0> <v1> ...    the expected output vector
//     tolerance <t>           optional, default 1e-9
#include "common/tensors/abstraction/graph_ir.h"
#include "common/tensors/abstraction/abstract_op_graph.h"
#include "common/tensors/abstraction/abstract_tensor_graph_ir.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/graph_sparse.h"
#include "tool_registry.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace nodus::tensors;

namespace {

struct NodeRecord {
    std::string kind;                    // tool id: abstract_tensor.<op> / .structural
    std::string op;                      // tensor.op attribute (canonical or label)
    bool has_constant = false;
    bool has_axis = false;               // process.dim/keepdim attr present
    double constant = 0.0;
    std::vector<uint32_t> input_ports;   // creation order == operand order
    std::vector<uint32_t> output_ports;
};

struct Loaded {
    std::map<uint32_t, NodeRecord> nodes;          // emission-ordered (map keeps id order)
    std::map<uint32_t, uint32_t> port_owner;       // port id -> node id
    std::vector<std::pair<uint32_t, uint32_t>> connections; // src port -> dst port
};

bool fail(const std::string& what) {
    std::cerr << "[EXEC] FAILED: " << what << "\n";
    return false;
}

double as_double(const GraphIrValue& value, bool* ok) {
    *ok = true;
    if (const auto* d = std::get_if<double>(&value)) return *d;
    if (const auto* u = std::get_if<uint32_t>(&value)) return static_cast<double>(*u);
    if (const auto* s = std::get_if<std::string>(&value)) {
        try { return std::stod(*s); } catch (...) {}
    }
    *ok = false;
    return 0.0;
}

// Replay the edit log into an executable picture of the graph.
Loaded load(const GraphEditBuilder& edits) {
    Loaded out;
    for (const GraphEdit& edit : edits.edits()) {
        switch (edit.kind) {
        case GraphEditKind::AddNode:
            out.nodes[edit.a].kind = edit.key;
            break;
        case GraphEditKind::AddPort: {
            const uint32_t node = edit.a, port = edit.b;
            out.port_owner[port] = node;
            bool ok = false;
            const uint32_t flags =
                static_cast<uint32_t>(as_double(edit.value, &ok));
            if (flags & kPortFlagInput) out.nodes[node].input_ports.push_back(port);
            if (flags & kPortFlagOutput) out.nodes[node].output_ports.push_back(port);
            break;
        }
        case GraphEditKind::Connect:
            out.connections.emplace_back(edit.a, edit.b);
            break;
        case GraphEditKind::SetAttr: {
            auto found = out.nodes.find(edit.a);
            if (found == out.nodes.end()) break;   // port attr; not needed here
            if (edit.key == "tensor.op") {
                if (const auto* s = std::get_if<std::string>(&edit.value))
                    found->second.op = *s;
            } else if (edit.key == "process.constant") {
                bool ok = false;
                const double v = as_double(edit.value, &ok);
                if (ok) { found->second.has_constant = true; found->second.constant = v; }
            } else if (edit.key == "process.dim" || edit.key == "process.keepdim") {
                found->second.has_axis = true;
            }
            break;
        }
        default:
            break;
        }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: tensor_graph_execute <script.graphir> <values.txt>\n";
        return 2;
    }

    std::ifstream script_stream(argv[1], std::ios::binary);
    if (!script_stream.good()) { fail(std::string("open ") + argv[1]); return 1; }
    std::ostringstream script;
    script << script_stream.rdbuf();

    std::vector<std::vector<double>> inputs;
    std::vector<double> expected;
    double tolerance = 1e-9;
    {
        std::ifstream values(argv[2]);
        if (!values.good()) { fail(std::string("open ") + argv[2]); return 1; }
        std::string line;
        while (std::getline(values, line)) {
            std::istringstream row(line);
            std::string record;
            if (!(row >> record) || record.empty() || record[0] == '#') continue;
            if (record == "tolerance") { row >> tolerance; continue; }
            std::vector<double> data;
            double v;
            while (row >> v) data.push_back(v);
            if (record == "input") inputs.push_back(std::move(data));
            else if (record == "expect") expected = std::move(data);
            else { fail("unknown values record: " + record); return 1; }
        }
    }
    if (inputs.empty() || expected.empty()) {
        fail("values file needs at least one input and one expect line");
        return 1;
    }
    const size_t length = inputs.front().size();

    AbstractOpGraphRunResult result;
    std::string error;
    if (!run_graph_ir_on_abstract_graph(
            script.str(), make_abstract_tensor_graph_ir_ops(), result, &error)) {
        fail("evaluate GraphIR script: " + error);
        return 1;
    }
    Loaded graph = load(result.edits);
    if (graph.nodes.empty()) { fail("script produced no nodes"); return 1; }

    // Node-level dataflow from port-level connections.
    std::map<uint32_t, std::vector<std::pair<size_t, uint32_t>>> operands; // dst node -> (slot, src node)
    std::map<uint32_t, size_t> outgoing;
    for (const auto& [src_port, dst_port] : graph.connections) {
        const uint32_t src_node = graph.port_owner.at(src_port);
        const uint32_t dst_node = graph.port_owner.at(dst_port);
        const auto& ports = graph.nodes.at(dst_node).input_ports;
        size_t slot = 0;
        for (; slot < ports.size(); ++slot) if (ports[slot] == dst_port) break;
        operands[dst_node].emplace_back(slot, src_node);
        ++outgoing[src_node];
    }

    ToolRegistry registry;
    register_abstract_tensor_tool_ir(registry);
    auto& backend = in_memory_backend_singleton();
    const TensorDesc desc{TensorDType::F64,
                          {{static_cast<uint32_t>(length)}}};
    const ValueTypeId pointer_type =
        ValueTypeRegistry::global().builtin(VT_VOID_PTR);

    auto materialize = [&](const std::vector<double>& data) {
        auto* tensor = new AbstractTensor(desc, &backend);
        void* raw = nullptr; size_t bytes = 0;
        backend.map(tensor->handle(), &raw, &bytes);
        auto* values = static_cast<double*>(raw);
        for (size_t i = 0; i < length; ++i)
            values[i] = i < data.size() ? data[i] : data.back();
        backend.unmap(tensor->handle());
        return tensor;
    };

    std::map<uint32_t, AbstractTensor*> computed;
    size_t next_input = 0;
    uint32_t last_sink = 0;

    // Emission order IS topological order (the serializer walks a topological
    // sort), so a single ordered pass executes the whole graph.
    for (auto& [node_id, node] : graph.nodes) {
        auto its_operands = operands.find(node_id);
        const size_t arity =
            its_operands == operands.end() ? 0 : its_operands->second.size();

        if (arity == 0) {
            if (node.has_constant) {
                computed[node_id] = materialize({node.constant});
            } else {
                if (next_input >= inputs.size()) {
                    fail("graph wants more inputs than the values file provides "
                         "(node op '" + node.op + "')");
                    return 1;
                }
                computed[node_id] = materialize(inputs[next_input++]);
            }
            continue;
        }

        std::vector<AbstractTensor*> args(arity, nullptr);
        for (const auto& [slot, src] : its_operands->second) {
            if (!computed.count(src)) { fail("operand not computed for op '" + node.op + "'"); return 1; }
            args[slot < arity ? slot : 0] = computed[src];
        }

        if (node.kind == "abstract_tensor.structural") {
            if (arity == 1) { computed[node_id] = args[0]; continue; }   // pass-through
            fail("structural node '" + node.op + "' with arity " +
                 std::to_string(arity) + " is not executable");
            return 1;
        }

        // The reduction tools reduce over ALL elements; a graph asking for a
        // dimensional reduction must be refused, not silently full-reduced.
        if (node.has_axis &&
            (node.kind == "abstract_tensor.sum" ||
             node.kind == "abstract_tensor.mean")) {
            fail("dimensional reduction ('" + node.op +
                 "' with a dim attribute) has no tool yet");
            return 1;
        }

        auto tool = registry.create(node.kind);
        if (!tool) { fail("no registered tool for '" + node.kind + "'"); return 1; }

        RawStackFrame raw{};
        if (!raw_stack_init_frame(raw, 1024)) { fail("stack init"); return 1; }
        for (AbstractTensor* argument : args) {
            void* pointer = argument;
            raw_stack_push_typed(raw, &pointer, pointer_type);
        }
        ToolStackContext context{};
        context.stack.raw = &raw;
        tool->execute_stack(context);
        void* out_pointer = nullptr;
        const bool popped = raw_stack_pop_typed(raw, &out_pointer, pointer_type) != 0;
        raw_stack_free_mask(raw);
        auto* output = static_cast<AbstractTensor*>(out_pointer);
        if (!popped || !output || !output->valid()) {
            fail("tool '" + node.kind + "' did not produce a result");
            return 1;
        }
        computed[node_id] = output;
    }

    // The result is the last computed node nothing consumes -- typically the
    // 'return' pass-through the frontend emitted.
    for (const auto& [node_id, node] : graph.nodes) {
        if (computed.count(node_id) && outgoing.find(node_id) == outgoing.end())
            last_sink = node_id;
    }
    if (last_sink == 0) { fail("no executed sink node"); return 1; }
    void* raw = nullptr; size_t bytes = 0;
    backend.map(computed[last_sink]->handle(), &raw, &bytes);
    const auto* values = static_cast<double*>(raw);
    bool matches = true;
    for (size_t i = 0; i < length && i < expected.size(); ++i) {
        if (std::abs(values[i] - expected[i]) > tolerance) {
            std::cerr << "[EXEC] mismatch at [" << i << "]: got " << values[i]
                      << " expected " << expected[i] << "\n";
            matches = false;
        }
    }
    std::cout << "[EXEC] result:";
    for (size_t i = 0; i < length; ++i) std::cout << " " << values[i];
    std::cout << "\n";
    backend.unmap(computed[last_sink]->handle());

    if (!matches) { fail("output does not match expectation"); return 1; }
    std::cout << "[EXEC] OK: " << graph.nodes.size() << " node(s) executed through "
              << "registered abstract_tensor tools; result matches the source "
              << "ecosystem within " << tolerance << "\n";
    return 0;
}
