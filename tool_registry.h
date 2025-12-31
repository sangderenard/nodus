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

private:
    std::unordered_map<std::string, Entry> entries_;
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

