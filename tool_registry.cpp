#include "tool_registry.h"
#include <iostream>

ToolRegistry& tool_registry_global() {
    static ToolRegistry registry;
    return registry;
}

bool ToolRegistry::register_tool(Entry entry) {
    if (entry.id.empty() || !entry.factory) return false;
    bool inserted = entries_.emplace(entry.id, std::move(entry)).second;
    if (inserted) {
        std::cerr << "DEBUG: ToolRegistry registered tool id='" << entries_.find(entry.id)->second.id << "'" << " name='" << entries_.find(entry.id)->second.name << "'\n";
    } else {
        std::cerr << "DEBUG: ToolRegistry failed to register tool id='" << entry.id << "' (already exists?)\n";
    }
    return inserted;
}

bool ToolRegistry::unregister_tool(const std::string& id) {
    auto it = entries_.find(id);
    if (it == entries_.end()) return false;
    std::cerr << "DEBUG: ToolRegistry unregistering tool id='" << id << "'\n";
    entries_.erase(it);
    return true;
}

const ToolRegistry::Entry* ToolRegistry::find(const std::string& id) const {
    auto it = entries_.find(id);
    if (it == entries_.end()) return nullptr;
    return &it->second;
}

std::unique_ptr<ITool, std::function<void(ITool*)>> ToolRegistry::create(const std::string& id) const {
    auto* entry = find(id);
    if (!entry) return nullptr;
    return entry->factory();
}

std::vector<ToolRegistry::Entry> ToolRegistry::entries() const {
    std::vector<Entry> list;
    list.reserve(entries_.size());
    for (const auto& kv : entries_) {
        list.push_back(kv.second);
    }
    return list;
}

ToolRegistrar::ToolRegistrar(const char* id, const char* name, ToolCaps caps, ToolRegistry::Factory factory) {
    ToolRegistry::Entry entry;
    entry.id = id ? id : "";
    entry.name = name ? name : "";
    entry.caps = caps;
    entry.factory = std::move(factory);
    tool_registry_global().register_tool(std::move(entry));
}

