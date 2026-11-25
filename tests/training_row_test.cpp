// End-to-end test of a stateful training row (NODUS BUFFER ABI V2 STATE).
//
// The product is made by tools/compile_tensor_network_train_row.py: the loss in examples/tensor_network_train_row.py,
// compiled by Turing's compile_native_training_schedule (backward graph and native SGD update). Its reference.txt holds
// a batch, initial weights, and the weights after N steps of train_aapl_qualified_row.PolicyNetwork.step (AbstractTensor
// autograd in Python).
//
// This test adopts the product through gp_artifact_import_buffer_abi, builds and loads the generated wrapper, loads the
// initial weights through the row's state ports, runs N calls with the batch on the stack, and compares the weights the
// row saves through serialize() with the reference. A second instance restored by deserialize() must save the same
// bytes.
//
//     test_training_row <product dir> <reference.txt>
//
// Without arguments it reads NODUS_TRAINING_ROW_PRODUCT and NODUS_TRAINING_ROW_REFERENCE, and is skipped when they
// are unset: the product needs a Turing checkout to compile.
#include "artifact_importer.h"
#include "common/tensors/abstraction/abstract_tensor_handle.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "tool_api.h"
#include "value_types.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

const char* const kStates[] = {"input_weight", "hidden_bias", "output_weight", "output_bias"};
const std::vector<uint64_t> kStateShapes[] = {{6, 4}, {1, 4}, {4, 3}, {1, 3}};

bool write_text(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
    return out.good();
}

bool run(const std::string& command) { return std::system(command.c_str()) == 0; }

fs::path find_core_lib(const fs::path& build) {
    for (const auto& path : {build / "Release" / "nodus_tensor_core.lib", build / "Debug" / "nodus_tensor_core.lib",
                             build / "nodus_tensor_core.lib"}) {
        if (fs::exists(path)) return path;
    }
    return {};
}

std::map<std::string, std::vector<double>> read_reference(const fs::path& path) {
    std::map<std::string, std::vector<double>> records;
    std::ifstream in(path);
    std::string name;
    size_t count = 0;
    while (in >> name >> count) {
        std::vector<double>& values = records[name];
        values.resize(count);
        for (double& value : values) {
            std::string text;
            in >> text;
            value = std::stod(text);
        }
    }
    return records;
}

// The receipt's ABI and BINARY entries, relative to the product root.
bool read_receipt(const fs::path& root, fs::path* abi, fs::path* binary) {
    std::ifstream in(root / "nodus-row-tool-product.receipt");
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag, value;
        ss >> tag >> std::quoted(value);
        if (tag == "ABI") *abi = root / value;
        if (tag == "BINARY") *binary = root / value;
    }
    return !abi->empty() && !binary->empty();
}

uint64_t tensor_f64(const std::vector<uint64_t>& shape, const std::vector<double>& values) {
    const uint64_t handle = nodus_tensor_create(NODUS_DTYPE_F64, static_cast<uint32_t>(shape.size()), shape.data());
    if (!handle) return 0;
    const int64_t bytes = static_cast<int64_t>(values.size() * sizeof(double));
    if (nodus_tensor_write(handle, 0, values.data(), static_cast<size_t>(bytes)) != bytes) {
        nodus_tensor_destroy(handle);
        return 0;
    }
    return handle;
}

double tensor_scalar(uint64_t handle) {
    double value = std::nan("");
    nodus_tensor_read(handle, 0, &value, sizeof(value));
    return value;
}

// The row's ports as a host sees them: one queue of tensor handles per port.
struct PortHost {
    std::vector<std::deque<uint64_t>> waiting;
    static int32_t consume(void* host, int32_t port, void* out, size_t capacity, size_t* written) {
        auto* self = static_cast<PortHost*>(host);
        if (port < 0 || static_cast<size_t>(port) >= self->waiting.size() || capacity < sizeof(uint64_t) ||
            self->waiting[static_cast<size_t>(port)].empty()) return 0;
        const uint64_t handle = self->waiting[static_cast<size_t>(port)].front();
        self->waiting[static_cast<size_t>(port)].pop_front();
        std::memcpy(out, &handle, sizeof(handle));
        *written = sizeof(handle);
        return 1;
    }
};

struct BytesOut final : OutputArchive {
    std::vector<uint8_t> bytes;
    bool write(const void* data, size_t len) override {
        const auto* p = static_cast<const uint8_t*>(data);
        bytes.insert(bytes.end(), p, p + len);
        return true;
    }
};

struct BytesIn final : InputArchive {
    const std::vector<uint8_t>& bytes;
    size_t at = 0;
    explicit BytesIn(const std::vector<uint8_t>& b) : bytes(b) {}
    bool read(void* data, size_t len) override {
        if (at + len > bytes.size()) return false;
        std::memcpy(data, bytes.data() + at, len);
        at += len;
        return true;
    }
};

// One call: the batch on the stack, the port host bound; returns the loss, or NaN when the call failed.
double call(ITool& tool, PortHost& ports, uint64_t features, uint64_t targets) {
    RawStackFrame raw{};
    if (!raw_stack_init_frame(raw, 4096)) return std::nan("");
    nodus::tensors::AbstractTensorHandle handle{};
    handle.id = features;
    bool ok = raw_stack_push_abstract_tensor(raw, handle);
    handle.id = targets;
    ok = ok && raw_stack_push_abstract_tensor(raw, handle);
    ToolRowPortAccess access{};
    access.host = &ports;
    access.consume = &PortHost::consume;
    ToolStackContext ctx{};
    ctx.stack.raw = &raw;
    ctx.row_ports = &access;
    if (ok) tool.execute_stack(ctx);
    double loss = std::nan("");
    nodus::tensors::AbstractTensorHandle prediction{}, loss_handle{};
    if (ok && ctx.succeeded && raw_stack_pop_abstract_tensor(raw, prediction) &&
        raw_stack_pop_abstract_tensor(raw, loss_handle)) {
        loss = tensor_scalar(loss_handle.id);
    }
    if (prediction.id) nodus_tensor_destroy(prediction.id);
    if (loss_handle.id) nodus_tensor_destroy(loss_handle.id);
    raw_stack_free_mask(raw);
    if (raw.backend) gp_mem_backend_release(raw.backend);
    return loss;
}

// The saved states: header, then per state a byte count and its bytes, in the descriptor's state order.
bool saved_states(const std::vector<uint8_t>& bytes, std::vector<std::vector<uint8_t>>* states) {
    BytesIn in(bytes);
    uint32_t header[2] = {};
    if (!in.read(header, sizeof(header)) || header[0] != 0x31545353u) return false;
    states->assign(header[1], {});
    for (auto& state : *states) {
        uint64_t size = 0;
        if (!in.read(&size, sizeof(size))) return false;
        state.resize(static_cast<size_t>(size));
        if (size && !in.read(state.data(), state.size())) return false;
    }
    return in.at == bytes.size();
}

std::vector<double> as_doubles(const std::vector<uint8_t>& bytes) {
    std::vector<double> values(bytes.size() / sizeof(double));
    std::memcpy(values.data(), bytes.data(), values.size() * sizeof(double));
    return values;
}

} // namespace

int main(int argc, char** argv) {
#ifndef _WIN32
    std::cout << "training_row_test skipped: the wrapper build here is Windows-only\n";
    return 0;
#else
    const char* product_arg = argc > 2 ? argv[1] : std::getenv("NODUS_TRAINING_ROW_PRODUCT");
    const char* reference_arg = argc > 2 ? argv[2] : std::getenv("NODUS_TRAINING_ROW_REFERENCE");
    if (!product_arg || !reference_arg) {
        std::cout << "training_row_test skipped: give <product dir> <reference.txt>\n";
        return 0;
    }
    const fs::path product = fs::path(product_arg);
    auto reference = read_reference(reference_arg);
    const size_t steps = reference["steps"].empty() ? 0 : static_cast<size_t>(reference["steps"][0]);
    const size_t batch = reference["features"].size() / 6;
    if (!steps || !batch) {
        std::cerr << "reference.txt is incomplete\n";
        return 1;
    }
    fs::path abi, binary;
    if (!read_receipt(product, &abi, &binary)) {
        std::cerr << "product receipt names no ABI and BINARY\n";
        return 1;
    }

    const fs::path repo = fs::path(NODUS_REPO_ROOT);
    const fs::path build = fs::path(NODUS_BUILD_DIR);
    const fs::path root = repo / "module_library" / "test_build" / "training_row_test";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    GP_ArtifactImportReport report{};
    char wrapper_buf[1024] = {};
    if (!gp_artifact_import_buffer_abi(abi.string().c_str(), binary.string().c_str(), root.string().c_str(), &report,
                                       wrapper_buf, sizeof(wrapper_buf), nullptr, 0) ||
        report.wrappers_written != 1) {
        std::cerr << "the training product's ABI was refused\n";
        return 1;
    }
    const fs::path core_lib = find_core_lib(build);
    const fs::path project = root / "wrapper_project";
    std::ostringstream cmake;
    cmake << "cmake_minimum_required(VERSION 3.15)\nproject(training_row LANGUAGES CXX)\n"
             "add_library(training_row SHARED \"" << fs::path(wrapper_buf).generic_string() << "\")\n"
          << "target_include_directories(training_row PRIVATE \"" << repo.generic_string() << "\" \""
          << (repo / "include").generic_string() << "\")\n"
          << "target_link_libraries(training_row PRIVATE \"" << core_lib.generic_string() << "\")\n"
             "set_target_properties(training_row PROPERTIES CXX_STANDARD 20 "
             "RUNTIME_OUTPUT_DIRECTORY \"${CMAKE_BINARY_DIR}/out\" RUNTIME_OUTPUT_DIRECTORY_RELEASE \"${CMAKE_BINARY_DIR}/out\")\n";
    if (core_lib.empty() || !write_text(project / "CMakeLists.txt", cmake.str()) ||
        !run("cmake -S \"" + project.string() + "\" -B \"" + (project / "build").string() + "\"") ||
        !run("cmake --build \"" + (project / "build").string() + "\" --config Release")) {
        std::cerr << "failed to build the generated training wrapper\n";
        return 1;
    }
    HMODULE module = LoadLibraryA((project / "build" / "out" / "training_row.dll").string().c_str());
    using CreateFn = ITool* (*)();
    using DestroyFn = void (*)(ITool*);
    auto create = module ? reinterpret_cast<CreateFn>(GetProcAddress(module, "create_tool")) : nullptr;
    auto destroy = module ? reinterpret_cast<DestroyFn>(GetProcAddress(module, "destroy_tool")) : nullptr;
    ITool* tool = create ? create() : nullptr;
    if (!tool || !destroy) {
        std::cerr << "the generated training wrapper did not load\n";
        return 1;
    }
    tool->initialize(ToolInitContext{});

    // Ports: two output blanks, then the six states in descriptor order.
    bool ok = tool->row_port_count() == 8;
    for (int k = 0; ok && k < 4; ++k) ok = std::string(tool->row_port(2 + k).name) == kStates[k];
    ok = ok && std::string(tool->row_port(6).name) == "steps" && std::string(tool->row_port(7).name) == "learning_rate";
    if (!ok) {
        std::cerr << "the training row's ports are wrong\n";
        return 1;
    }

    PortHost ports;
    ports.waiting.resize(8);
    const uint64_t features = tensor_f64({batch, 6}, reference["features"]);
    const uint64_t targets = tensor_f64({batch, 3}, reference["targets"]);
    std::vector<double> losses;
    // Call 0: the states start at zero; a tensor that does not fit input_weight's shape is destroyed and leaves it so.
    ports.waiting[2].push_back(tensor_f64({2, 2}, {1, 2, 3, 4}));
    for (size_t step = 0; step < steps + 1; ++step) {
        if (step == 1) {
            // Call 1 loads the reference's initial weights through the four state ports, replacing what call 0 trained.
            for (int k = 0; k < 4; ++k) {
                ports.waiting[static_cast<size_t>(2 + k)].push_back(
                    tensor_f64(kStateShapes[k], reference[std::string("initial.") + kStates[k]]));
            }
        }
        losses.push_back(call(*tool, ports, features, targets));
        if (std::isnan(losses.back())) {
            std::cerr << "call " << step << " failed\n";
            return 1;
        }
    }
    // Calls 1..steps are the reference's steps.
    BytesOut saved;
    std::vector<std::vector<uint8_t>> states;
    if (!tool->serialize(saved) || !saved_states(saved.bytes, &states) || states.size() != 6) {
        std::cerr << "the training row did not save its states\n";
        return 1;
    }
    double worst = 0.0;
    for (int k = 0; k < 4; ++k) {
        const auto got = as_doubles(states[static_cast<size_t>(k)]);
        const auto& want = reference[std::string("final.") + kStates[k]];
        if (got.size() != want.size()) {
            std::cerr << kStates[k] << " has " << got.size() << " values, want " << want.size() << "\n";
            return 1;
        }
        for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(got[i] - want[i]));
    }
    int32_t saved_steps = 0;
    double saved_rate = 0.0;
    std::memcpy(&saved_steps, states[4].data(), std::min(states[4].size(), sizeof(saved_steps)));
    std::memcpy(&saved_rate, states[5].data(), std::min(states[5].size(), sizeof(saved_rate)));

    // A second instance restored from the saved bytes saves the same bytes.
    ITool* restored = create();
    BytesIn in(saved.bytes);
    BytesOut again;
    const bool round_trip = restored && restored->deserialize(in) && restored->serialize(again) &&
                            again.bytes == saved.bytes;

    std::cout << std::setprecision(17)
              << "first loss row=" << losses[1] << " python=" << reference["first_loss"][0] << "\n"
              << "last loss  row=" << losses[steps] << " python=" << reference["last_loss"][0] << "\n"
              << "max |weight row - python| after " << steps << " steps = " << worst << "\n"
              << "steps=" << saved_steps << " learning_rate=" << saved_rate << " restore=" << round_trip << "\n";

    nodus_tensor_destroy(features);
    nodus_tensor_destroy(targets);
    tool->shutdown();
    destroy(tool);
    if (restored) destroy(restored);
    FreeLibrary(module);
    // The bound is the measurement of 2026-11-24 (6.9e-18 over 50 steps) with room for one ULP at 0.5.
    if (!round_trip || saved_steps != 1 || saved_rate != 0.01 || losses[1] != reference["first_loss"][0] ||
        worst > 1.2e-16) {
        std::cerr << "[TRAINING ROW] FAIL\n";
        return 1;
    }
    std::cout << "[TRAINING ROW] PASS\n";
    return 0;
#endif
}
