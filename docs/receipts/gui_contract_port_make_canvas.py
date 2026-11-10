import ctypes, pathlib, sys
ROOT = pathlib.Path(r"C:/dev/Powershell/nodus")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"tools")); sys.path.insert(0, str(ROOT/"tests"))
from nodus_graph import FifoConfig, Graph, PluginTool
from nodus_graph.native import _fifo
from nodus_headless_bridge import CanvasEdgeDesc
PREFIX, CAP, PORT = "gui.proof", 4, 47611
out = pathlib.Path(__file__).parent / "messenger.canvas"
with Graph() as graph:
    graph.acquire_contract_pair(PREFIX, CAP)
    market = graph.add_module("market"); worker = graph.add_module("worker")
    market.set_rows(PluginTool(PREFIX + ".market", 0, 0))
    worker.set_rows(PluginTool(PREFIX + ".worker", 0, 0))
    desc = CanvasEdgeDesc(a_module=market.index, a_contact_idx=0, b_module=worker.index, b_contact_idx=0)
    e = graph.lib.nodus_headless_add_edge(graph._handle, ctypes.byref(desc), 0)
    assert e >= 0
    fifo = FifoConfig(token_bytes=64, slots=4, top_k=0, type_id=0x534A5231, layout=2, dtype=8)
    assert graph.lib.nodus_headless_configure_edge_fifo(graph._handle, e, ctypes.byref(_fifo(fifo)))
    p = graph.declare_contract_port(PREFIX, CAP, e, PORT)
    print("bound port", p)
    graph.save(out)
for ln in out.read_text(errors="replace").splitlines():
    if ln.startswith("CONTRACT_PORT"): print(ln)
