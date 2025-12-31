#pragma once

#include <functional>
#include <memory>
#include <string>
#include "tool_api.h"

struct ToolIR {
    std::string id;
    std::string name;
    ToolCaps caps = ToolCaps::None;

    // Optional lifecycle callbacks. If not set, no-op behavior is used.
    std::function<void(const ToolInitContext&)> initialize;
    std::function<void()> shutdown;
    std::function<void(double, HostAPI&)> tick;
    std::function<void(RenderContext&)> render;
    std::function<void(TableRenderArgs&)> render_table;
    std::function<void(LEDRenderArgs&)> render_led_cell;
    std::function<void(TextRenderArgs&)> render_text;

    std::function<int32_t()> port_count;
    std::function<ToolPortSpec(int32_t)> port_spec;

    std::function<void(ToolStackContext&)> execute_stack;

    std::function<bool(OutputArchive&)> serialize;
    std::function<bool(InputArchive&)> deserialize;
};

// A lightweight ITool implementation that delegates to a ToolIR instance.
class BuiltinTool : public ITool {
public:
    explicit BuiltinTool(std::shared_ptr<ToolIR> ir) : ir_(std::move(ir)) {}
    ~BuiltinTool() noexcept override = default;

    const char* id_cstr() const noexcept override { return ir_ ? ir_->id.c_str() : ""; }
    const char* name_cstr() const noexcept override { return ir_ ? ir_->name.c_str() : ""; }
    ToolCaps caps() const override { return ir_ ? ir_->caps : ToolCaps::None; }

    void initialize(const ToolInitContext& ctx) override { if (ir_ && ir_->initialize) ir_->initialize(ctx); }
    void shutdown() override { if (ir_ && ir_->shutdown) ir_->shutdown(); }
    void tick(double dt, HostAPI& host) override { if (ir_ && ir_->tick) ir_->tick(dt, host); }
    void render(RenderContext& ctx) override { if (ir_ && ir_->render) ir_->render(ctx); }

    void render_table(TableRenderArgs& args) override { if (ir_ && ir_->render_table) ir_->render_table(args); }
    void render_led_cell(LEDRenderArgs& args) override { if (ir_ && ir_->render_led_cell) ir_->render_led_cell(args); }
    void render_text(TextRenderArgs& args) override { if (ir_ && ir_->render_text) ir_->render_text(args); }

    int32_t port_count() const override { return (ir_ && ir_->port_count) ? ir_->port_count() : 0; }
    ToolPortSpec port_spec(int32_t idx) const override { return (ir_ && ir_->port_spec) ? ir_->port_spec(idx) : ToolPortSpec{}; }

    void execute_stack(ToolStackContext& ctx) override { if (ir_ && ir_->execute_stack) ir_->execute_stack(ctx); }

    bool serialize(OutputArchive& out) const override { return (ir_ && ir_->serialize) ? ir_->serialize(out) : false; }
    bool deserialize(InputArchive& in) override { return (ir_ && ir_->deserialize) ? ir_->deserialize(in) : false; }

private:
    std::shared_ptr<ToolIR> ir_;
};

// Helper utilities (implementation in tool_ir.cpp)
namespace tool_ir {
    // make_registry_entry is declared in tool_registry.h where ToolRegistry is visible.

    // Expose ToolIR-to-SPIR-V translation integration
    void tool_ir_translate_kernel_to_spirv(const nodus::spirv::KernelIR& ir);
}
