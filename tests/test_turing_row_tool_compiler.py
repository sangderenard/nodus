from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.turing_row_tool_compiler import (
    ADOPTION_RECEIPT_NAME,
    BUFFER_ABI_SCHEMA,
    PRODUCT_SCHEMA,
    ProductError,
    compile_row_tool,
)


@dataclass
class FakeValue:
    id: int
    dtype: str = "float64"
    shape: tuple[int, ...] = ()


@dataclass
class FakeInstruction:
    op: str
    args: list[FakeValue]
    res: FakeValue | None


@dataclass
class FakeBlock:
    instrs: list[FakeInstruction] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)


@dataclass
class FakeFunction:
    name: str
    args: list[FakeValue]
    blocks: dict[str, FakeBlock]


@dataclass
class FakeModule:
    functions: dict[str, FakeFunction]


class FakeBufferArtifact:
    def __init__(self, backend: str, *, complete: bool = True):
        self.name = f"row_{backend}"
        self.complete = complete
        self.shortfalls = () if complete else ("unsupported",)
        self.source = "void row_c(void **buffers, long long *extents) {}\n"
        self.llvm_ir = "define void @row_llvm(ptr %buffers, ptr %extents) { ret void }\n"
        self.buffer_order = (1, 2, 99, 3)
        self.buffer_dtypes = ("double", "float64", "i64", "f64")
        self.buffer_shapes = ((), (), (4,), ())
        self.extent_order = ()
        self.library_path = None


class FakeFortranArtifact:
    name = "row_fortran"
    source = "module row_fortran\nend module\n"
    complete = True
    shortfalls = ()

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        source = directory / "row_fortran.f90"
        source.write_text(self.source, encoding="utf-8")
        source.with_suffix(".api.yaml").write_text(
            "schema: turing-compiled-program-api-v1\n"
            "module: row_fortran\nlanguage: fortran\nentry: row_fortran\n"
            "metadata: {}\nentry_points: []\n",
            encoding="utf-8",
        )
        return source


class FakeCompiler:
    head = "a" * 40
    toolchain = {"schema": "fake-toolchain-v1", "files": []}

    def __init__(self, *, build: bool = True, complete: bool = True):
        self.build = build
        self.complete = complete
        self.lower_calls = []

    def lower(self, source, entrypoint, **kwargs):
        self.lower_calls.append((source, entrypoint, kwargs))
        Path("generated_ssa_helpers.py").write_text("# retained helper\n", encoding="utf-8")
        identity_logs = Path("artifacts") / "identity_logs"
        identity_logs.mkdir(parents=True, exist_ok=True)
        (identity_logs / "fake.jsonl").write_text("{}\n", encoding="utf-8")
        left, right, result = FakeValue(1), FakeValue(2), FakeValue(3)
        workspace = FakeValue(99, "i64", (4,))
        block = FakeBlock([FakeInstruction("Add", [left, right], result)])
        module = FakeModule({"lowered_row": FakeFunction(
            "lowered_row", [left, right], {"entry": block}
        )})
        # Keep workspace reachable only through the backend's complete buffer
        # order; this proves the product does not silently omit it.
        assert workspace.id == 99
        return module, {"lowered_row": (result,)}, ("lowered_row",)

    def emit_c(self, module, root, *, entry_name):
        return FakeBufferArtifact("c", complete=self.complete)

    def emit_llvm(self, module, root, *, entry_name):
        return FakeBufferArtifact("llvm", complete=self.complete)

    def compile_c(self, artifact, directory):
        return self._build(artifact, directory)

    def compile_llvm(self, artifact, directory):
        return self._build(artifact, directory)

    def _build(self, artifact, directory):
        if not self.build:
            raise FileNotFoundError("test toolchain unavailable")
        artifact.library_path = directory / f"{artifact.name}.dll"
        artifact.library_path.write_bytes(b"fake-dll")
        return artifact

    def emit_fortran(self, module, *, name, outputs, exports):
        return FakeFortranArtifact()

    def compile_fortran(self, artifact, directory, *, entrypoint, name):
        if not self.build:
            raise FileNotFoundError("test Fortran compiler unavailable")
        library = directory / f"{name}.dll"
        library.write_bytes(b"fake-fortran-dll")
        return SimpleNamespace(
            executable_path=library,
            api_path=directory / "row_fortran.api.yaml",
        )


def request(tmp_path: Path, backend: str = "c") -> dict:
    contract = tmp_path / "extraction-contract.yaml"
    contract.write_text("test: fake-boundary-contract\n", encoding="utf-8")
    return {
        "source_text": "total = left + right\nreturn total\n",
        "global_scope": True,
        "row_entry": "combine",
        "columns": [
            {"name": "left", "role": "input", "dtype": "f64", "shape": []},
            {"name": "right", "role": "input", "dtype": "f64", "shape": []},
            {"name": "total", "role": "output", "dtype": "f64", "shape": []},
        ],
        "backend": backend,
        "output_dir": str(tmp_path / "product"),
        "extraction_contract": str(contract),
        "ssa_composition": {
            "directory": str(tmp_path / "composition"),
            "options": {"jobs": 2},
        },
    }


def test_c_product_uses_public_composer_and_retains_complete_buffer_abi(tmp_path):
    compiler = FakeCompiler()
    manifest = compile_row_tool(request(tmp_path), compiler=compiler)

    submitted, entrypoint, options = compiler.lower_calls[0]
    assert entrypoint == "combine"
    assert submitted == (
        "def combine(left, right):\n"
        "    total = left + right\n"
        "    return total\n"
    )
    assert options["partitioned_directory"] == (tmp_path / "composition").resolve()
    assert options["partitioned_options"] == {"jobs": 2, "ssa_composition": True}
    assert options["extraction_contract"] == (
        tmp_path / "product" / "extraction-contract.yaml"
    ).resolve()
    assert manifest["schema"] == PRODUCT_SCHEMA
    assert manifest["loadable"] is True
    assert manifest["compiler"]["public_entry"].endswith("lower_ast_source_to_ssa")
    assert manifest["buffer_abi"]["schema"] == BUFFER_ABI_SCHEMA
    assert manifest["buffer_abi"]["call"].endswith("int64_t *extents)")
    assert [item["role"] for item in manifest["buffer_abi"]["buffers"]] == [
        "input", "input", "workspace", "output"
    ]

    descriptor = (tmp_path / "product" / "backend" / "row_c.nodus_buffer_abi.txt").read_text()
    assert descriptor.startswith("NODUS BUFFER ABI V2\n")
    assert 'TOOL "imported.combine.c" "compiled combine (c)"' in descriptor
    assert 'EXTENT_DTYPE "i64"' in descriptor
    assert 'VALUE WORKSPACE "%t99" "i64" 1 4' in descriptor
    assert [line for line in descriptor.splitlines() if line.startswith("BUFFER ")] == [
        'BUFFER "%t1"', 'BUFFER "%t2"', 'BUFFER "%t99"', 'BUFFER "%t3"'
    ]
    assert (tmp_path / "product" / "repository-ssa.pkl").is_file()
    assert (tmp_path / "product" / "extraction-contract.yaml").is_file()
    assert (tmp_path / "product" / "repository-ssa.txt").is_file()
    assert (tmp_path / "product" / "compiler-work" / "generated_ssa_helpers.py").is_file()
    assert (
        tmp_path / "product" / "compiler-work" / "artifacts" / "identity_logs" / "fake.jsonl"
    ).is_file()
    assert sum(
        item["kind"] == "compiler-side-product" for item in manifest["artifacts"]
    ) == 2
    assert all(len(item["sha256"]) == 64 for item in manifest["artifacts"])
    assert manifest["adoption_receipt"] == ADOPTION_RECEIPT_NAME
    assert (tmp_path / "product" / ADOPTION_RECEIPT_NAME).read_text() == (
        "NODUS ROW TOOL PRODUCT V1\n"
        'BACKEND "c"\n'
        'MANIFEST "nodus-row-tool-product.json"\n'
        'ABI "backend/row_c.nodus_buffer_abi.txt"\n'
        'BINARY "backend/row_c.dll"\n'
    )


def test_llvm_unavailable_toolchain_retains_source_and_marks_non_loadable(tmp_path):
    compiler = FakeCompiler(build=False)
    manifest = compile_row_tool(request(tmp_path, "llvm"), compiler=compiler)
    assert manifest["loadable"] is False
    assert manifest["build_error"]["type"] == "FileNotFoundError"
    assert manifest["adoption_receipt"] is None
    assert not (tmp_path / "product" / ADOPTION_RECEIPT_NAME).exists()
    descriptor = (
        tmp_path / "product" / "backend" / "row_llvm.nodus_buffer_abi.txt"
    ).read_text()
    assert 'EXTENT_DTYPE "i32"' in descriptor
    assert manifest["buffer_abi"]["call"].endswith("int32_t *extents)")
    assert (tmp_path / "product" / "backend" / "row_llvm.ll").is_file()


def test_non_loadable_rerun_removes_stale_adoption_receipt(tmp_path):
    first = request(tmp_path, "llvm")
    assert compile_row_tool(first, compiler=FakeCompiler())["loadable"] is True
    receipt = tmp_path / "product" / ADOPTION_RECEIPT_NAME
    assert receipt.is_file()
    second = compile_row_tool(first, compiler=FakeCompiler(build=False))
    assert second["loadable"] is False
    assert not receipt.exists()


def test_incomplete_backend_artifact_is_refused(tmp_path):
    with pytest.raises(ProductError, match="C emission incomplete"):
        compile_row_tool(request(tmp_path), compiler=FakeCompiler(complete=False))
    assert not (tmp_path / "product" / "nodus-row-tool-product.json").exists()


def test_request_requires_explicit_extraction_contract(tmp_path):
    malformed = request(tmp_path)
    del malformed["extraction_contract"]
    with pytest.raises(ValueError, match="explicit extraction_contract"):
        compile_row_tool(malformed, compiler=FakeCompiler())


def test_fortran_product_retains_turing_api_descriptor(tmp_path):
    manifest = compile_row_tool(request(tmp_path, "fortran"), compiler=FakeCompiler())
    assert manifest["loadable"] is True
    assert manifest["buffer_abi"] is None
    kinds = {item["kind"] for item in manifest["artifacts"]}
    assert "abi-descriptor" in kinds
    assert (tmp_path / "product" / "backend" / "row_fortran.api.yaml").is_file()
    assert (tmp_path / "product" / ADOPTION_RECEIPT_NAME).read_text() == (
        "NODUS ROW TOOL PRODUCT V1\n"
        'BACKEND "fortran"\n'
        'MANIFEST "nodus-row-tool-product.json"\n'
        'ABI "backend/row_fortran.api.yaml"\n'
        'BINARY "backend/combine.dll"\n'
    )


def test_non_shape_extent_is_refused_instead_of_mistranslated(tmp_path):
    compiler = FakeCompiler()
    original = compiler.emit_c

    def emit_c(*args, **kwargs):
        artifact = original(*args, **kwargs)
        artifact.extent_order = ((1, "numel", None),)
        return artifact

    compiler.emit_c = emit_c
    with pytest.raises(ProductError, match="axis-addressed shape extents"):
        compile_row_tool(request(tmp_path), compiler=compiler)


def test_dynamic_rank_and_shape_sources_survive_into_buffer_abi(tmp_path):
    compiler = FakeCompiler()
    original = compiler.emit_c

    def emit_c(*args, **kwargs):
        artifact = original(*args, **kwargs)
        artifact.extent_order = ((1, "dim", 0),)
        return artifact

    compiler.emit_c = emit_c
    dynamic = request(tmp_path)
    dynamic["columns"] = [
        {"name": "left", "role": "input", "dtype": "f64",
         "rank": 1, "shape": None},
        {"name": "right", "role": "input", "dtype": "f64",
         "rank": 1, "shape": None, "shape_from": "left"},
        {"name": "total", "role": "output", "dtype": "f64",
         "rank": 1, "shape": None, "shape_from": "left"},
    ]

    manifest = compile_row_tool(dynamic, compiler=compiler)
    descriptor = (
        tmp_path / "product" / "backend" / "row_c.nodus_buffer_abi.txt"
    ).read_text()
    assert manifest["buffer_abi"]["columns"] == dynamic["columns"]
    assert 'VALUE INPUT "%t1" "f64" 1 ?' in descriptor
    assert 'VALUE INPUT "%t2" "f64" 1 ?' in descriptor
    assert 'VALUE OUTPUT "%t3" "f64" 1 ?' in descriptor
    assert 'EXTENT "%t1" 0' in descriptor
    assert 'SHAPE "%t2" 0 FROM "%t1" 0' in descriptor
    assert 'SHAPE "%t3" 0 FROM "%t1" 0' in descriptor


def test_dynamic_output_requires_explicit_shape_source(tmp_path):
    malformed = request(tmp_path)
    malformed["columns"][-1] = {
        "name": "total", "role": "output", "dtype": "f64",
        "rank": 1, "shape": None,
    }
    with pytest.raises(ValueError, match="requires shape_from"):
        compile_row_tool(malformed, compiler=FakeCompiler())


def test_dynamic_fortran_refuses_before_legacy_importer_can_narrow_it(tmp_path):
    dynamic = request(tmp_path, "fortran")
    dynamic["columns"] = [
        {"name": "left", "role": "input", "dtype": "f64",
         "rank": 1, "shape": None},
        {"name": "right", "role": "input", "dtype": "f64",
         "rank": 1, "shape": None, "shape_from": "left"},
        {"name": "total", "role": "output", "dtype": "f64",
         "rank": 1, "shape": None, "shape_from": "left"},
    ]
    with pytest.raises(ProductError, match="C/LLVM buffer ABI"):
        compile_row_tool(dynamic, compiler=FakeCompiler())


def test_declared_input_count_must_equal_root_argument_count(tmp_path):
    malformed = request(tmp_path)
    malformed["columns"] = malformed["columns"][1:]
    with pytest.raises(ProductError, match="input/inout column count"):
        compile_row_tool(malformed, compiler=FakeCompiler())


def test_declared_output_count_must_equal_published_output_count(tmp_path):
    malformed = request(tmp_path)
    malformed["columns"].append(
        {"name": "extra", "role": "output", "dtype": "f64", "shape": []}
    )
    with pytest.raises(ProductError, match="output/inout column count"):
        compile_row_tool(malformed, compiler=FakeCompiler())


@pytest.mark.skipif(
    os.environ.get("NODUS_RUN_TURING_ROW_REAL") != "1",
    reason="set NODUS_RUN_TURING_ROW_REAL=1 for the real compiler backend tests",
)
@pytest.mark.parametrize("backend", ("c", "llvm", "fortran"))
def test_real_turing_public_boundary_opt_in(tmp_path, backend):
    turing_root = Path(__file__).resolve().parents[2] / "turing"
    if not turing_root.is_dir():
        pytest.skip("sibling Turing checkout is absent")
    real_request = {
        "source_text": "def add(left, right):\n    return left + right\n",
        "row_entry": "add",
        "columns": [
            {"name": "left", "role": "input", "dtype": "f64"},
            {"name": "right", "role": "input", "dtype": "f64"},
            {"name": "total", "role": "output", "dtype": "f64"},
        ],
        "backend": backend,
        "output_dir": str(tmp_path / f"real-product-{backend}"),
        "turing_root": str(turing_root),
        "extraction_contract": str(
            turing_root / "extraction_contracts" / "program_extraction.yaml"
        ),
        "allow_uncompiled": False,
    }
    nodus_root = Path(__file__).resolve().parents[1]
    side_paths = (
        nodus_root / "generated_ssa_helpers.py",
        nodus_root / "ssa_helper_metadata.json",
    )
    before = {path: path.stat().st_mtime_ns if path.exists() else None for path in side_paths}
    identity_root = nodus_root / "artifacts" / "identity_logs"
    before_logs = set(identity_root.glob("*")) if identity_root.exists() else set()
    manifest = compile_row_tool(real_request)
    assert manifest["schema"] == PRODUCT_SCHEMA
    assert manifest["compiler"]["git_head"] != "unknown"
    assert manifest["loadable"] is True
    product = tmp_path / f"real-product-{backend}"
    assert (product / "repository-ssa.pkl").is_file()
    assert (product / ADOPTION_RECEIPT_NAME).is_file()
    assert before == {
        path: path.stat().st_mtime_ns if path.exists() else None for path in side_paths
    }
    after_logs = set(identity_root.glob("*")) if identity_root.exists() else set()
    assert after_logs == before_logs
