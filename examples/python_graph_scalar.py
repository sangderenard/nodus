"""Build and execute a real Nodus canvas row program from Python."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from nodus_graph import BuiltinTool, FifoConfig, Graph, Input, Output, Tool


with Graph() as graph:
    arithmetic = graph.add_module("(a-b)*c", stable_id=0xA11CE)
    arithmetic.set_rows(
        Input(2),
        Tool(BuiltinTool.SUBTRACT),
        Input(1),
        Tool(BuiltinTool.MULTIPLY),
        Output(1),
    )

    graph.bind_input("a", arithmetic.input(0))
    graph.bind_input("b", arithmetic.input(1))
    graph.bind_input("c", arithmetic.input(2))
    graph.bind_output(
        "result", arithmetic.output(0), fifo=FifoConfig.scalar_f32(slots=8, top_k=1),
    )

    graph.push_f32("a", 9.0)
    graph.push_f32("b", 4.0)
    graph.push_f32("c", 3.0)
    graph.step()
    print(graph.pull_f32("result"))  # 15.0
