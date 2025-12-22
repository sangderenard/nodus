#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>

#include "../plugin_loader.h"

namespace fs = std::filesystem;

static std::string timestamp_string() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    char buf[64];
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return std::string(buf);
}

int main(int argc, char** argv) {
    PluginLoader loader;
    if (argc < 2) {
        std::cout << "usage: modloader <command> [args]\n";
        std::cout << " commands:\n";
        std::cout << "  load <dll_path>                    - load and register tool from DLL\n";
        std::cout << "  unload <tool_id>                   - unload previously loaded tool\n";
        std::cout << "  list                               - list loaded tools\n";
        std::cout << "  buildload <builddir> <target> <built_dll_relpath> [dest_dir]\n";
        std::cout << "    - build the target via cmake, copy built DLL to dest_dir with timestamp, then load\n";
        return 1;
    }

    std::string cmd = argv[1];
    if (cmd == "load" && argc >= 3) {
        std::string path = argv[2];
        std::string id = loader.load_module(path);
        if (!id.empty()) std::cout << "loaded: " << id << "\n";
        else std::cout << "failed to load " << path << "\n";
        return id.empty() ? 1 : 0;
    }

    if (cmd == "unload" && argc >= 3) {
        std::string id = argv[2];
        if (loader.unload_module(id)) {
            std::cout << "unloaded: " << id << "\n";
            return 0;
        } else {
            std::cout << "failed to unload: " << id << "\n";
            return 1;
        }
    }

    if (cmd == "list") {
        auto list = loader.loaded_modules();
        for (auto& id : list) std::cout << id << "\n";
        return 0;
    }

    if (cmd == "buildload" && argc >= 5) {
        std::string builddir = argv[2];
        std::string target = argv[3];
        std::string built_rel = argv[4];
        std::string dest_dir = (argc >= 6) ? argv[5] : ".";

        // run cmake --build
        std::string cmdline = "cmake --build \"" + builddir + "\" --config Release --target \"" + target + "\"";
        std::cout << "Running: " << cmdline << "\n";
        int rc = std::system(cmdline.c_str());
        if (rc != 0) {
            std::cerr << "build failed: rc=" << rc << "\n";
            return 1;
        }

        fs::path built = fs::path(builddir) / fs::path(built_rel);
        if (!fs::exists(built)) {
            std::cerr << "built file not found: " << built.string() << "\n";
            return 1;
        }

        fs::create_directories(dest_dir);
        std::string ts = timestamp_string();
        fs::path dest = fs::path(dest_dir) / (built.stem().string() + "_" + ts + built.extension().string());
        try {
            fs::copy_file(built, dest, fs::copy_options::skip_existing);
        } catch (const std::exception& e) {
            std::cerr << "copy failed: " << e.what() << "\n";
            return 1;
        }

        std::cout << "Copied to: " << dest.string() << "\n";
        std::string id = loader.load_module(dest.string());
        if (id.empty()) {
            std::cerr << "failed to load " << dest.string() << "\n";
            return 1;
        }
        std::cout << "loaded: " << id << "\n";
        return 0;
    }

    std::cout << "unknown command\n";
    return 1;
}
