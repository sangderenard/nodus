#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tool_api.h"
#include "tool_ir.h"

class ToolRegistry {
public:
    using Factory = std::function<std::unique_ptr<ITool, std::function<void(ITool*)>>()>;

    struct Entry {
        std::string id;
        std::string name;
        ToolCaps caps = ToolCaps::None;
        std::string source_path;
        Factory factory;
        int auto_mouse_ports = 0;
        int auto_keyboard_ports = 0;
    };

    bool register_tool(Entry entry);
    const Entry* find(const std::string& id) const;
    std::unique_ptr<ITool, std::function<void(ITool*)>> create(const std::string& id) const;
    std::vector<Entry> entries() const;
    bool set_source_path(const std::string& id, const std::string& path);

    bool unregister_tool(const std::string& id);

    // Alias layer (boundary error register, E13): canvas ROWS and definition
    // modules name bare catalog ids ("abstract_tensor.tanh"), while
    // PluginLoader registers versioned unique ids so multiple loads coexist.
    // An alias makes a loaded tool answer to its bare id. Resolution policy,
    // stated once and enforced here:
    //   - find()/create() resolve exact real entries FIRST; an alias never
    //     shadows a real entry (an in-image reference implementation keeps
    //     priority over any loaded witness of the same id);
    //   - registering an alias whose name collides with a real entry is
    //     refused;
    //   - re-registering an existing alias re-points it (latest load wins;
    //     earlier versions stay reachable by their unique ids);
    //   - unregister_tool() drops every alias that points at the removed id.
    // Resolution is exactly one hop: aliases to aliases are refused.
    bool register_alias(const std::string& alias, const std::string& target_id);
    // The id find()/create() would act on for `id` (identity for real
    // entries, target for aliases, empty if unknown).
    std::string resolve(const std::string& id) const;

private:
    std::unordered_map<std::string, Entry> entries_;
    std::unordered_map<std::string, std::string> aliases_;
};

ToolRegistry& tool_registry_global();

namespace tool_ir { ToolRegistry::Entry make_registry_entry(std::shared_ptr<ToolIR> ir); }

// Helper to register a ToolIR directly.
inline bool register_tool_ir(std::shared_ptr<ToolIR> ir) {
    if (!ir) return false;
    return tool_registry_global().register_tool(tool_ir::make_registry_entry(ir));
}

class ToolRegistrar {
public:
    ToolRegistrar(const char* id, const char* name, ToolCaps caps, ToolRegistry::Factory factory);
};

#define REGISTER_TOOL(CLASS, ID, NAME, CAPS) \
    static ToolRegistrar g_tool_registrar_##CLASS(ID, NAME, CAPS, [] { \
        return std::unique_ptr<ITool, std::function<void(ITool*)>>(new CLASS(), [](ITool* p){ delete p; }); \
    })

