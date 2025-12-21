#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tool_api.h"

class ToolRegistry {
public:
    using Factory = std::function<std::unique_ptr<ITool>()>;

    struct Entry {
        std::string id;
        std::string name;
        ToolCaps caps = ToolCaps::None;
        Factory factory;
    };

    bool register_tool(Entry entry);
    const Entry* find(const std::string& id) const;
    std::unique_ptr<ITool> create(const std::string& id) const;
    std::vector<Entry> entries() const;

private:
    std::unordered_map<std::string, Entry> entries_;
};

ToolRegistry& tool_registry_global();

class ToolRegistrar {
public:
    ToolRegistrar(const char* id, const char* name, ToolCaps caps, ToolRegistry::Factory factory);
};

#define REGISTER_TOOL(CLASS, ID, NAME, CAPS) \
    static ToolRegistrar g_tool_registrar_##CLASS(ID, NAME, CAPS, [] { \
        return std::make_unique<CLASS>(); \
    })

