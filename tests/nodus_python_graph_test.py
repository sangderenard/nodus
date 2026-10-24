"""Native acceptance for the Pythonic Nodus graph surface.

Run through CTest so NODUS_HEADLESS_TEST_DLL names the just-built library.
"""

from __future__ import annotations

import math
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from nodus_graph import (
    BuiltinTool,
    ExecutionContract,
    FifoConfig,
    Graph,
    GraphDefinition,
    Input,
    ModuleDefinition,
    Output,
    PluginTool,
    PublishResult,
    SchedulerMode,
    Tool,
    UnsupportedNativeContract,
    graph_definition_from_row_product,
)


def assert_close(actual: float | None, expected: float) -> None:
    assert actual is not None
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6), (actual, expected)


def build_scalar_graph(graph: Graph):
    module = graph.add_module("(a-b)*c", stable_id=0xA11CE)
    module.set_rows(
        Input(2),
        Tool(BuiltinTool.SUBTRACT),
        Input(1),
        Tool(BuiltinTool.MULTIPLY),
        Output(1),
    )
    graph.bind_input("a", module.input(0))
    graph.bind_input("b", module.input(1))
    graph.bind_input("c", module.input(2))
    graph.bind_output("result", module.output(0))
    return module


def push_case(graph: Graph, a: float, b: float, c: float, expected: float) -> None:
    graph.push_f32("a", a)
    graph.push_f32("b", b)
    graph.push_f32("c", c)
    graph.step()
    assert_close(graph.pull_f32("result"), expected)


def test_scalar_stack_and_native_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "scalar.canvas"
        with Graph() as graph:
            module = build_scalar_graph(graph)
            serialized = module.export_to_root(pathlib.Path(tmp) / "materialized")
            blob = serialized.read_bytes()
            assert b"INPUT" in blob and b"Subtract" in blob
            assert b"Multiply" in blob and b"OUTPUT" in blob
            # First-token visibility, stack operand order, and a later tick.
            push_case(graph, 9.0, 4.0, 3.0, 15.0)
            push_case(graph, 10.0, 2.0, 4.0, 32.0)
            graph.save(path)
            identity = module.stable_id

        with Graph() as restored:
            restored.load(path)
            module = restored.module_by_id(identity, input_count=3, output_count=1)
            assert module.stable_id == identity
            restored.bind_input("a", module.input(0))
            restored.bind_input("b", module.input(1))
            restored.bind_input("c", module.input(2))
            restored.bind_output("result", module.output(0))
            push_case(restored, 7.0, 2.0, 5.0, 25.0)


def test_two_module_fifo() -> None:
    with Graph() as graph:
        producer = graph.add_module("producer").set_rows(
            Input(2), Tool(BuiltinTool.ADD), Output(1),
        )
        consumer = graph.add_module("consumer").set_rows(
            Input(2), Tool(BuiltinTool.MULTIPLY), Output(1),
        )
        graph.connect(producer.output(0), consumer.input(0))
        state = graph.edge_fifo_state(graph.edges[0])
        assert state.config == FifoConfig.scalar_f32()
        assert state.unread == 0
        graph.bind_input("x", producer.input(0))
        graph.bind_input("y", producer.input(1))
        graph.bind_input("scale", consumer.input(1))
        graph.bind_output("result", consumer.output(0))
        graph.push_f32("x", 2.0)
        graph.push_f32("y", 3.0)
        graph.push_f32("scale", 4.0)
        graph.step()
        assert_close(graph.pull_f32("result"), 20.0)
        assert graph.edge_fifo_state(graph.edges[0]).unread == 0


def test_vulkan_fifo_storage_is_communicated_and_read_back() -> None:
    fifo = FifoConfig(token_bytes=256, slots=2, top_k=0, type_id=91,
                      layout=0, dtype=14, storage="vulkan")
    try:
        with Graph() as graph:
            producer = graph.add_module("shader A").set_rows(Input(), Output())
            consumer = graph.add_module("shader B").set_rows(Input(), Output())
            edge = graph.connect(producer.output(0), consumer.input(0), fifo=fifo)
            assert graph.edge_fifo_state(edge).config == fifo
    except RuntimeError as error:
        if "FIFO configuration failed" in str(error):
            pytest.skip("no Vulkan compute backend available")
        raise


def test_fifo_reject_drop_inspection_and_invalid_calls() -> None:
    lossless = FifoConfig.scalar_f32(slots=2, top_k=0)
    with Graph() as graph:
        module = graph.add_module("lossless").set_rows(Input(), Output())
        graph.bind_input("in", module.input(0), fifo=lossless)
        assert graph.port_fifo_state("in").config == lossless
        assert graph.port_fifo_state("in").unread == 0
        assert graph.try_push_f32("in", 1.0) == PublishResult(True, False)
        assert graph.try_push_f32("in", 2.0) == PublishResult(True, False)
        assert graph.port_fifo_state("in").unread == 2
        assert graph.try_push_f32("in", 3.0) == PublishResult(False, False)
        assert graph.port_fifo_state("in").unread == 2
        try:
            graph.try_push_bytes("in", b"bad")
        except RuntimeError as exc:
            assert "invalid" in str(exc)
        else:
            raise AssertionError("wrong-sized FIFO token was accepted")
        try:
            graph.port_fifo_state("missing")
        except KeyError:
            pass
        else:
            raise AssertionError("missing port inspection succeeded")

    lossy = FifoConfig.scalar_f32(slots=2, top_k=1)
    with Graph() as graph:
        module = graph.add_module("lossy").set_rows(Input(), Output())
        graph.bind_input("in", module.input(0), fifo=lossy)
        assert graph.try_push_f32("in", 1.0) == PublishResult(True, False)
        assert graph.try_push_f32("in", 2.0) == PublishResult(True, False)
        assert graph.try_push_f32("in", 3.0) == PublishResult(True, True)
        state = graph.port_fifo_state("in")
        assert state.config == lossy
        assert state.unread == 1


def test_native_stack_display_snapshots() -> None:
    with Graph() as graph:
        module = graph.add_module("stack-inspection").set_rows(
            Input(2),
            Tool(BuiltinTool.STACK_DISPLAY),
            Tool(BuiltinTool.SUBTRACT),
            Tool(BuiltinTool.STACK_DISPLAY),
            Input(1),
            Tool(BuiltinTool.STACK_DISPLAY),
            Tool(BuiltinTool.MULTIPLY),
            Tool(BuiltinTool.STACK_DISPLAY),
            Output(1),
        )
        graph.bind_input("a", module.input(0))
        graph.bind_input("b", module.input(1))
        graph.bind_input("c", module.input(2))
        graph.bind_output("result", module.output(0))
        push_case(graph, 9.0, 4.0, 3.0, 15.0)

        expected = {
            1: (9.0, 4.0),
            3: (5.0,),
            5: (5.0, 3.0),
            7: (15.0,),
        }
        for row_index, values in expected.items():
            snapshot = module.stack_snapshot(row_index)
            assert snapshot.values == values
            assert len(snapshot.items) == len(values)
            assert all(item.element_bytes > 0 for item in snapshot.items)
            assert len({item.type_id for item in snapshot.items}) == 1
        assert module.stack_tail() == ()
        assert module.stack_snapshot(0).values == ()
        try:
            module.stack_snapshot(99)
        except IndexError:
            pass
        else:
            raise AssertionError("invalid semantic row stack inspection succeeded")


def test_two_module_fifo_reload_requires_explicit_reattachment() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "pipeline.canvas"
        with Graph() as graph:
            producer = graph.add_module("producer", stable_id=0xA001).set_rows(
                Input(2), Tool(BuiltinTool.ADD), Output(1),
            )
            consumer = graph.add_module("consumer", stable_id=0xA002).set_rows(
                Input(2), Tool(BuiltinTool.MULTIPLY), Output(1),
            )
            graph.connect(
                producer.output(0), consumer.input(0),
                fifo=FifoConfig.scalar_f32(slots=7, top_k=0),
            )
            graph.save(path)

        with Graph() as restored:
            restored.load(path)
            producer = restored.module_by_id(0xA001, input_count=2, output_count=1)
            consumer = restored.module_by_id(0xA002, input_count=2, output_count=1)
            try:
                restored.step()
            except UnsupportedNativeContract as exc:
                assert "does not retain FIFO specs" in str(exc)
            else:
                raise AssertionError("loaded edge silently invented FIFO persistence")

            restored.attach_edge(producer.output(0), consumer.input(0), fifo=None)
            try:
                restored.step()
            except UnsupportedNativeContract as exc:
                assert "explicit FIFO config" in str(exc)
            else:
                raise AssertionError("explicitly absent restored FIFO was executed")

            restored.attach_edge(
                producer.output(0), consumer.input(0),
                fifo=FifoConfig.scalar_f32(slots=7, top_k=0),
            )
            restored.bind_input("x", producer.input(0))
            restored.bind_input("y", producer.input(1))
            restored.bind_input("scale", consumer.input(1))
            restored.bind_output("result", consumer.output(0))
            restored.push_f32("x", 6.0)
            restored.push_f32("y", 1.0)
            restored.push_f32("scale", 3.0)
            restored.step()
            assert_close(restored.pull_f32("result"), 21.0)


def test_failed_load_preserves_bindings_and_successful_load_invalidates_handles() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        saved = pathlib.Path(tmp) / "valid.canvas"
        missing = pathlib.Path(tmp) / "missing.canvas"
        with Graph() as graph:
            stale = build_scalar_graph(graph)
            graph.save(saved)
            try:
                graph.load(missing)
            except RuntimeError:
                pass
            else:
                raise AssertionError("missing native canvas unexpectedly loaded")
            push_case(graph, 8.0, 3.0, 2.0, 10.0)
            graph.load(saved)
            try:
                stale.input(0)
            except UnsupportedNativeContract as exc:
                assert "stale" in str(exc)
            else:
                raise AssertionError("pre-load module handle remained usable")


def test_pointer_contract_reaches_native_scheduler() -> None:
    with Graph() as graph:
        module = graph.add_module("tensor")
        module.set_rows(
            Input(2),
            PluginTool(
                "abstract_tensor.add", 2,
                contract=ExecutionContract.ABSTRACT_TENSOR_POINTER,
            ),
            Output(1),
        )
        assert module.native_contract is ExecutionContract.ABSTRACT_TENSOR_POINTER
        # Typed connector transport now owns this contract. A missing plugin
        # produces no value, but the graph is no longer rejected merely for
        # carrying AbstractTensorHandle values.
        graph.step()


def test_typed_fifo_does_not_select_stack_transfer() -> None:
    """FIFO schema and module stack currency remain independent choices."""
    typed_f32 = FifoConfig(token_bytes=4, slots=4, top_k=1, type_id=0)
    with Graph() as graph:
        module = graph.add_module("typed-fifo-scalar-stack").set_rows(
            Input(2), Tool(BuiltinTool.ADD), Output(1),
        )
        assert module.native_contract is ExecutionContract.SCALAR_F32
        graph.bind_input("left", module.input(0), fifo=typed_f32)
        graph.bind_input("right", module.input(1), fifo=typed_f32)
        graph.bind_output("sum", module.output(0), fifo=typed_f32)
        graph.push_f32("left", 6.25)
        graph.push_f32("right", 1.75)
        graph.step()
        assert_close(graph.pull_f32("sum"), 8.0)


def test_simultaneous_graphs_are_refused() -> None:
    with Graph():
        try:
            Graph()
        except UnsupportedNativeContract as exc:
            assert "one live native graph" in str(exc)
        else:
            raise AssertionError("second process-global canvas was allowed")


def test_fifo_absence_and_invalid_capacity_are_explicit() -> None:
    try:
        FifoConfig(slots=0)
    except ValueError as exc:
        assert "use None" in str(exc)
    else:
        raise AssertionError("zero-capacity FIFO was accepted as absence")

    with Graph() as graph:
        producer = graph.add_module("producer").set_rows(Input(), Output())
        consumer = graph.add_module("consumer").set_rows(Input(), Output())
        graph.connect(producer.output(0), consumer.input(0), fifo=None)
        assert graph.edge_fifo_state(graph.edges[0]).config is None
        try:
            graph.step()
        except UnsupportedNativeContract as exc:
            assert "explicit FIFO config" in str(exc)
        else:
            raise AssertionError("unconfigured edge was silently executed")


def test_duplicate_stable_identity_is_refused() -> None:
    with Graph() as graph:
        graph.add_module("first", stable_id=0xD00D)
        try:
            graph.add_module("second", stable_id=0xD00D)
        except ValueError as exc:
            assert "already in use" in str(exc)
        else:
            raise AssertionError("duplicate stable identity was accepted")


def test_complete_definition_roundtrip_and_execution() -> None:
    edge_fifo = FifoConfig.scalar_f32(slots=7, top_k=0)
    port_fifo = FifoConfig.scalar_f32(slots=5, top_k=1)
    with Graph() as source:
        producer = source.add_module(
            "producer", x=10, y=20, width=300, height=180, stable_id=0xB001,
        ).set_rows(Input(2), Tool(BuiltinTool.ADD), Output(1))
        consumer = source.add_module(
            "consumer", x=400, y=20, width=320, height=200, stable_id=0xB002,
        ).set_rows(Input(2), Tool(BuiltinTool.MULTIPLY), Output(1))
        source.connect(producer.output(0), consumer.input(0), fifo=edge_fifo)
        source.bind_input("x", producer.input(0), fifo=port_fifo)
        source.bind_input("y", producer.input(1), fifo=port_fifo)
        source.bind_input("scale", consumer.input(1), fifo=port_fifo)
        source.bind_output("result", consumer.output(0), fifo=port_fifo, start_at_head=False)
        definition = source.definition()

    wire = definition.to_mapping()
    received = GraphDefinition.from_mapping(wire)
    assert received == definition
    assert received.edges[0].source.module_id == 0xB001
    assert received.edges[0].target.module_id == 0xB002
    assert received.edges[0].fifo == edge_fifo

    with Graph(scheduler=received.scheduler) as restored:
        restored.apply_definition(received)
        assert restored.definition() == received
        # Negative control: corrupt every executable Python-side cache that
        # the former implementation used to echo.  Definition readback must
        # still come from Canvas/Table and the native named-port bindings.
        restored.scheduler = SchedulerMode.FREE_SPINNING
        restored.modules[0].label = "poisoned Python cache"
        restored.modules[0].x = -999
        restored.modules[0].rows = ()
        restored.modules[0].contract = ExecutionContract.OPAQUE
        restored.edges.clear()
        restored._input_ports.clear()
        restored._output_ports.clear()
        assert restored.definition() == received
        restored.push_f32("x", 2.0)
        restored.push_f32("y", 3.0)
        restored.push_f32("scale", 4.0)
        restored.step()
        assert_close(restored.pull_f32("result"), 20.0)


def test_definition_refuses_contract_disagreement_before_native_adoption() -> None:
    try:
        GraphDefinition(modules=(ModuleDefinition(
            stable_id=1,
            label="wrong-contract",
            rows=(Input(2), Tool(BuiltinTool.ADD), Output(1)),
            contract=ExecutionContract.OPAQUE,
        ),))
    except ValueError as exc:
        assert "does not match declared" in str(exc)
    else:
        raise AssertionError("graph definition accepted a row-contract disagreement")


def test_runtime_shape_sources_are_communicable() -> None:
    product = {
        "buffer_abi": {
            "schema": "nodus-turing-buffer-abi-v2",
            "tool_id": "imported.dynamic.c",
            "tool_name": "dynamic tensor row",
            "arguments": [10],
            "outputs": [20],
            "buffers": [
                {
                    "ssa_value_id": 10, "column": "source", "dtype": "f64",
                    "shape": [None, None],
                    "shape_sources": [
                        {"ssa_value_id": 10, "axis": 0},
                        {"ssa_value_id": 10, "axis": 1},
                    ],
                },
                {
                    "ssa_value_id": 20, "column": "result", "dtype": "f64",
                    "shape": [None, None],
                    "shape_sources": [
                        {"ssa_value_id": 10, "axis": 0},
                        {"ssa_value_id": 10, "axis": 1},
                    ],
                },
            ],
        },
    }
    definition = graph_definition_from_row_product(
        product, stable_id=0xD1A6, abstract_tensor_type_id=17,
    )
    source, result = definition.connectors
    assert source.tensor.shape == (None, None)
    assert tuple(item.connector for item in source.tensor.shape_sources) == (
        source.reference, source.reference,
    )
    assert tuple(item.axis for item in result.tensor.shape_sources) == (0, 1)
    assert all(item.connector == source.reference for item in result.tensor.shape_sources)
    assert GraphDefinition.from_mapping(definition.to_mapping()) == definition


def main() -> int:
    cases = [
        test_scalar_stack_and_native_roundtrip,
        test_two_module_fifo,
        test_fifo_reject_drop_inspection_and_invalid_calls,
        test_native_stack_display_snapshots,
        test_two_module_fifo_reload_requires_explicit_reattachment,
        test_failed_load_preserves_bindings_and_successful_load_invalidates_handles,
        test_pointer_contract_reaches_native_scheduler,
        test_typed_fifo_does_not_select_stack_transfer,
        test_simultaneous_graphs_are_refused,
        test_fifo_absence_and_invalid_capacity_are_explicit,
        test_duplicate_stable_identity_is_refused,
        test_complete_definition_roundtrip_and_execution,
        test_definition_refuses_contract_disagreement_before_native_adoption,
        test_runtime_shape_sources_are_communicable,
    ]
    for case in cases:
        case()
        print(f"{case.__name__}: OK")
    print(f"nodus_python_graph_test: {len(cases)} cases OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
