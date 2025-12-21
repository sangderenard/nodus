import glob
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


def _compiler():
    return os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")


def _shared_lib_ext():
    if sys.platform.startswith("win"):
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


class ToolDllTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.cmake = shutil.which("cmake")
        if not cls.cmake:
            raise unittest.SkipTest("No CMake available")
        # We will use CMake to build the test loader and tools, so no direct compiler is required here.
        cls.serialized = sorted(
            glob.glob(os.path.join(cls.repo_root, "module_library", "serialized", "*.gpmod"))
        )
        if not cls.serialized:
            raise unittest.SkipTest("No serialized modules to test")

    def test_serializations_to_tool_dlls(self):
        with tempfile.TemporaryDirectory(prefix="tool_dll_test_") as temp_root:
            modlib_path = os.path.join(temp_root, "module_library.txt")
            source_root = os.path.join(temp_root, "module_library")
            os.makedirs(source_root, exist_ok=True)

            module_lines = ["MODULELIB V1", f"ROOT \"{source_root}\""]
            for idx, ser_path in enumerate(self.serialized):
                module_id = os.path.splitext(os.path.basename(ser_path))[0]
                tool_id = f"tool_{module_id}"
                module_lines.append(
                    "MODULE {id} {idx} \"{label}\" \"{serialized}\" \"\" 1 \"{tool_id}\" 0".format(
                        id=module_id,
                        idx=idx,
                        label=module_id,
                        serialized=os.path.abspath(ser_path),
                        tool_id=tool_id,
                    )
                )

            with open(modlib_path, "w", encoding="utf-8") as f:
                f.write("\n".join(module_lines))
                f.write("\n")

            # Build the project's `modlib_actualize` target via CMake
            repo_build = os.path.join(self.repo_root, "build")
            subprocess.run([self.cmake, "--build", repo_build, "--target", "modlib_actualize", "--config", "Release"], check=True)

            # Locate built executable
            if sys.platform.startswith("win"):
                generator = os.path.join(repo_build, "Release", "modlib_actualize.exe")
            else:
                generator = os.path.join(repo_build, "modlib_actualize")
            if not os.path.exists(generator):
                # try common places
                for root, _, files in os.walk(repo_build):
                    for fn in files:
                        if fn.startswith("modlib_actualize"):
                            generator = os.path.join(root, fn)
                            break
                    if os.path.exists(generator):
                        break

            subprocess.run([generator, modlib_path, source_root], check=True)

            loader = os.path.join(temp_root, "tool_loader")
            loader_cpp = os.path.join(temp_root, "tool_loader.cpp")
            with open(loader_cpp, "w", encoding="utf-8") as f:
                f.write(
                    "#include \"tool_api.h\"\n"
                    "#include <iostream>\n"
                    "#include <memory>\n"
                    "#include <string>\n"
                    "#if defined(_WIN32)\n"
                    "#include <windows.h>\n"
                    "#else\n"
                    "#include <dlfcn.h>\n"
                    "#endif\n\n"
                    "class StubHost : public HostAPI {\n"
                    "public:\n"
                    "    void log(const char* message) override {\n"
                    "        if (message) { std::cerr << message << \"\\n\"; }\n"
                    "    }\n"
                    "    void draw_text(const TextRenderArgs& args) override { (void)args; }\n"
                    "};\n\n"
                    "int main(int argc, char** argv) {\n"
                    "    if (argc < 2) {\n"
                    "        std::cerr << \"usage: tool_loader <dll> [serialized]\\n\";\n"
                    "        return 2;\n"
                    "    }\n"
                    "    const char* path = argv[1];\n"
                    "    const char* ser = (argc > 2) ? argv[2] : nullptr;\n"
                    "#if defined(_WIN32)\n"
                    "    HMODULE handle = LoadLibraryA(path);\n"
                    "    if (!handle) {\n"
                    "        std::cerr << \"LoadLibrary failed\\n\";\n"
                    "        return 3;\n"
                    "    }\n"
                    "    auto create = reinterpret_cast<CreateToolFn>(GetProcAddress(handle, \"create_tool\"));\n"
                    "    if (!create) {\n"
                    "        std::cerr << \"create_tool not found\\n\";\n"
                    "        FreeLibrary(handle);\n"
                    "        return 4;\n"
                    "    }\n"
                    "#else\n"
                    "    void* handle = dlopen(path, RTLD_NOW);\n"
                    "    if (!handle) {\n"
                    "        std::cerr << dlerror() << \"\\n\";\n"
                    "        return 3;\n"
                    "    }\n"
                    "    auto create = reinterpret_cast<CreateToolFn>(dlsym(handle, \"create_tool\"));\n"
                    "    if (!create) {\n"
                    "        std::cerr << \"create_tool not found\\n\";\n"
                    "        dlclose(handle);\n"
                    "        return 4;\n"
                    "    }\n"
                    "#endif\n"
                    "    std::unique_ptr<ITool> tool(create());\n"
                    "    if (!tool) {\n"
                    "        std::cerr << \"create_tool returned null\\n\";\n"
                    "        return 5;\n"
                    "    }\n"
                    "    ToolInitContext ctx{};\n"
                    "    ctx.serialized_path = ser;\n"
                    "    StubHost host;\n"
                    "    RenderContext rctx{};\n"
                    "    tool->initialize(ctx);\n"
                    "    tool->tick(0.016, host);\n"
                    "    tool->render(rctx);\n"
                    "    tool->shutdown();\n"
                    "#if defined(_WIN32)\n"
                    "    FreeLibrary(handle);\n"
                    "#else\n"
                    "    dlclose(handle);\n"
                    "#endif\n"
                    "    return 0;\n"
                    "}\n"
                )

            # Build the loader with CMake to avoid relying on a specific compiler command.
            proj_dir = os.path.join(temp_root, "loader_cmake")
            build_dir = os.path.join(proj_dir, "build")
            os.makedirs(proj_dir, exist_ok=True)
            cmakelists = """
cmake_minimum_required(VERSION 3.15)
project(tool_loader LANGUAGES CXX)
add_executable(tool_loader "%s")
target_include_directories(tool_loader PRIVATE "%s")
set_target_properties(tool_loader PROPERTIES CXX_STANDARD 17)
""" % (loader_cpp.replace("\\", "/"), self.repo_root.replace("\\", "/"))
            if not sys.platform.startswith("win"):
                cmakelists += "\nfind_package(Threads REQUIRED)\nset(CMAKE_THREAD_LIBS_INIT ${CMAKE_THREAD_LIBS_INIT})\nadd_definitions(-D_POSIX_C_SOURCE=200112L)\n" \
                           + "target_link_libraries(tool_loader PRIVATE dl)\n"

            cmake_file = os.path.join(proj_dir, "CMakeLists.txt")
            with open(cmake_file, "w", encoding="utf-8") as f:
                f.write(cmakelists)

            subprocess.run([self.cmake, "-S", proj_dir, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release"], check=True)
            subprocess.run([self.cmake, "--build", build_dir, "--config", "Release"], check=True)

            if sys.platform.startswith("win"):
                loader_exec = os.path.join(build_dir, "Release", "tool_loader.exe")
            else:
                loader_exec = os.path.join(build_dir, "tool_loader")

            tool_sources = glob.glob(os.path.join(source_root, "source", "tools", "*.cpp"))
            self.assertTrue(tool_sources, "No tool sources generated")

            for tool_source in tool_sources:
                tool_name = os.path.splitext(os.path.basename(tool_source))[0]

                # Create a tiny CMake project to build this tool as a shared library.
                proj_dir = os.path.join(temp_root, "cmake_" + tool_name)
                build_dir = os.path.join(proj_dir, "build")
                os.makedirs(proj_dir, exist_ok=True)

                cmakelists = """
cmake_minimum_required(VERSION 3.15)
project(%s LANGUAGES CXX)
add_library(%s SHARED "%s")
target_include_directories(%s PRIVATE "%s")
set_target_properties(%s PROPERTIES CXX_STANDARD 17)
""" % (tool_name, tool_name, tool_source.replace('\\', '/'), tool_name, self.repo_root.replace('\\', '/'), tool_name)

                cmake_file = os.path.join(proj_dir, "CMakeLists.txt")
                with open(cmake_file, "w", encoding="utf-8") as f:
                    f.write(cmakelists)

                # Prefer using Ninja + g++/clang++ when available (avoids MSVC-specific compile issues).
                preferred_cxx = shutil.which("g++") or shutil.which("clang++")
                have_ninja = shutil.which("ninja") is not None
                if preferred_cxx and have_ninja:
                    subprocess.run([
                        "cmake",
                        "-S",
                        proj_dir,
                        "-B",
                        build_dir,
                        "-G",
                        "Ninja",
                        "-DCMAKE_BUILD_TYPE=Release",
                        "-DCMAKE_CXX_COMPILER=%s" % preferred_cxx,
                    ], check=True)
                    subprocess.run(["cmake", "--build", build_dir], check=True)
                else:
                    subprocess.run(["cmake", "-S", proj_dir, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release"], check=True)
                    # Use --config Release for multi-config generators (e.g., MSVC)
                    subprocess.run(["cmake", "--build", build_dir, "--config", "Release"], check=True)

                # Find the built shared library.
                tool_lib = None
                # Common locations: build/Release/<name>.dll (Windows) or build/lib<name>.so (Unix)
                if sys.platform.startswith("win"):
                    candidate = os.path.join(build_dir, "Release", tool_name + _shared_lib_ext())
                    if os.path.exists(candidate):
                        tool_lib = candidate
                if not tool_lib:
                    # search build dir for lib matching tool_name
                    for root, _, files in os.walk(build_dir):
                        for fn in files:
                            if fn.endswith(_shared_lib_ext()) and tool_name in fn:
                                tool_lib = os.path.join(root, fn)
                                break
                        if tool_lib:
                            break
                if not tool_lib:
                    raise RuntimeError("Failed to locate built tool library for %s" % tool_name)

                module_id = tool_name.replace("tool_", "", 1)
                serialized = next(
                    (p for p in self.serialized if os.path.splitext(os.path.basename(p))[0] == module_id),
                    None,
                )
                subprocess.run([loader_exec, tool_lib, serialized or ""], check=True)


if __name__ == "__main__":
    unittest.main()
