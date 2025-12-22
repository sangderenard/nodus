#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>

struct LoadedModuleInfo;

class PluginLoader {
public:
    PluginLoader();
    ~PluginLoader();

    // Load a DLL at `path`. Optionally provide a HostAPI pointer for plugin initialization.
    // Returns the registered unique tool id on success, or empty string on failure.
    std::string load_module(const std::string& path, void* host = nullptr);

    // Unload previously loaded module by tool id. Returns true on success.
    bool unload_module(const std::string& tool_id);

    // List loaded module ids.
    std::vector<std::string> loaded_modules() const;

private:
    std::unordered_map<std::string, std::unique_ptr<LoadedModuleInfo>> modules_;
};
