"""Native acceptance for the Pythonic Nodus graph surface.

Run through CTest so NODUS_HEADLESS_TEST_DLL names the just-built library.
"""

from __future__ import annotations

import math
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from nodus_graph import (
    BuiltinTool,
    ExecutionContract,
    FifoConfig,
    Graph,
    Input,
    Output,
    PluginTool,
    Tool,
    UnsupportedNativeContract,
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
        graph.bind_input("x", producer.input(0))
        graph.bind_input("y", producer.input(1))
        graph.bind_input("scale", consumer.input(1))
        graph.bind_output("result", consumer.output(0))
        graph.push_f32("x", 2.0)
        graph.push_f32("y", 3.0)
        graph.push_f32("scale", 4.0)
        graph.step()
        assert_close(graph.pull_f32("result"), 20.0)


def test_pointer_contract_is_explicitly_refused() -> None:
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
        try:
            graph.step()
        except UnsupportedNativeContract as exc:
            assert "AbstractTensor pointers" in str(exc)
        else:
            raise AssertionError("pointer/scalar contract mismatch was not refused")


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
        try:
            graph.step()
        except UnsupportedNativeContract as exc:
            assert "explicit FIFO config" in str(exc)
        else:
            raise AssertionError("unconfigured edge was silently executed")


def main() -> int:
    test_scalar_stack_and_native_roundtrip()
    test_two_module_fifo()
    test_pointer_contract_is_explicitly_refused()
    test_simultaneous_graphs_are_refused()
    test_fifo_absence_and_invalid_capacity_are_explicit()
    print("nodus_python_graph_test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
