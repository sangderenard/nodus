#include "tool_registry.h"
#include <iostream>

ToolRegistry& tool_registry_global() {
    static ToolRegistry registry;
    return registry;
}

bool ToolRegistry::register_tool(Entry entry) {
    if (entry.id.empty() || !entry.factory) return false;
    auto result = entries_.emplace(entry.id, std::move(entry));
    bool inserted = result.second;
    if (inserted) {
        const auto& e = result.first->second;
        std::cerr << "DEBUG: ToolRegistry registered tool id='" << e.id << "'" << " name='" << e.name << "'"
                  << " auto_mouse_ports=" << e.auto_mouse_ports << " auto_keyboard_ports=" << e.auto_keyboard_ports << "\n";
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
    // Drop every alias that pointed at the removed entry -- a dangling alias
    // would resolve to nothing and read as a silent hole later (register E2's
    // lesson: never leave a name that quietly stops meaning anything).
    for (auto ait = aliases_.begin(); ait != aliases_.end();) {
        if (ait->second == id) {
            std::cerr << "DEBUG: ToolRegistry dropping alias '" << ait->first
                      << "' (target unregistered)\n";
            ait = aliases_.erase(ait);
        } else {
            ++ait;
        }
    }
    return true;
}

bool ToolRegistry::register_alias(const std::string& alias, const std::string& target_id) {
    if (alias.empty() || target_id.empty() || alias == target_id) return false;
    if (entries_.count(alias)) {
        // A real entry owns this name; the in-image implementation keeps
        // priority over any loaded witness (policy stated in the header).
        std::cerr << "DEBUG: ToolRegistry alias '" << alias
                  << "' refused: a real entry owns that id\n";
        return false;
    }
    if (!entries_.count(target_id)) {
        // One-hop rule: targets must be real entries, never other aliases.
        std::cerr << "DEBUG: ToolRegistry alias '" << alias << "' -> '" << target_id
                  << "' refused: target is not a registered entry\n";
        return false;
    }
    aliases_[alias] = target_id; // re-pointing an existing alias: latest wins
    std::cerr << "DEBUG: ToolRegistry alias '" << alias << "' -> '" << target_id << "'\n";
    return true;
}

std::string ToolRegistry::resolve(const std::string& id) const {
    if (entries_.count(id)) return id;
    auto it = aliases_.find(id);
    if (it != aliases_.end() && entries_.count(it->second)) return it->second;
    return {};
}

const ToolRegistry::Entry* ToolRegistry::find(const std::string& id) const {
    auto it = entries_.find(id);
    if (it != entries_.end()) return &it->second;
    auto ait = aliases_.find(id);
    if (ait != aliases_.end()) {
        auto tit = entries_.find(ait->second);
        if (tit != entries_.end()) return &tit->second;
    }
    return nullptr;
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

bool ToolRegistry::set_source_path(const std::string& id, const std::string& path) {
    auto it = entries_.find(id);
    if (it == entries_.end()) return false;
    it->second.source_path = path;
    return true;
}

ToolRegistrar::ToolRegistrar(const char* id, const char* name, ToolCaps caps, ToolRegistry::Factory factory) {
    ToolRegistry::Entry entry;
    entry.id = id ? id : "";
    entry.name = name ? name : "";
    entry.caps = caps;
    entry.factory = std::move(factory);
    tool_registry_global().register_tool(std::move(entry));
}

