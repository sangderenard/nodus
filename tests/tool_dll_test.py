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
        cls.compiler = _compiler()
        if not cls.compiler:
            raise unittest.SkipTest("No C++ compiler available")
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

            generator = os.path.join(temp_root, "modlib_actualize")
            generator_cpp = os.path.join(temp_root, "modlib_actualize.cpp")
            with open(generator_cpp, "w", encoding="utf-8") as f:
                f.write(
                    "#include \"module_library_actualizer.h\"\n"
                    "#include <iostream>\n\n"
                    "int main(int argc, char** argv) {\n"
                    "    if (argc < 3) {\n"
                    "        std::cerr << \"usage: modlib_actualize <lib> <root>\\n\";\n"
                    "        return 2;\n"
                    "    }\n"
                    "    return gp_module_library_actualize_from_file(argv[1], argv[2]) ? 0 : 1;\n"
                    "}\n"
                )

            compile_cmd = [
                self.compiler,
                "-std=c++17",
                "-I",
                self.repo_root,
                generator_cpp,
                os.path.join(self.repo_root, "module_library_actualizer.cpp"),
                os.path.join(self.repo_root, "module_library.cpp"),
                "-o",
                generator,
            ]
            subprocess.run(compile_cmd, check=True)

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

            compile_cmd = [
                self.compiler,
                "-std=c++17",
                "-I",
                self.repo_root,
                loader_cpp,
                "-o",
                loader,
            ]
            if not sys.platform.startswith("win"):
                compile_cmd.append("-ldl")
            subprocess.run(compile_cmd, check=True)

            tool_sources = glob.glob(os.path.join(source_root, "source", "tools", "*.cpp"))
            self.assertTrue(tool_sources, "No tool sources generated")

            for tool_source in tool_sources:
                tool_name = os.path.splitext(os.path.basename(tool_source))[0]
                tool_lib = os.path.join(temp_root, tool_name + _shared_lib_ext())
                build_cmd = [
                    self.compiler,
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    "-I",
                    self.repo_root,
                    tool_source,
                    "-o",
                    tool_lib,
                ]
                if sys.platform.startswith("win"):
                    build_cmd = [
                        self.compiler,
                        "-std=c++17",
                        "-shared",
                        "-I",
                        self.repo_root,
                        tool_source,
                        "-o",
                        tool_lib,
                    ]
                subprocess.run(build_cmd, check=True)

                module_id = tool_name.replace("tool_", "", 1)
                serialized = next(
                    (p for p in self.serialized if os.path.splitext(os.path.basename(p))[0] == module_id),
                    None,
                )
                subprocess.run([loader, tool_lib, serialized or ""], check=True)


if __name__ == "__main__":
    unittest.main()
