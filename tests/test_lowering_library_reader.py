import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nodus_graph.lowering_library import LoweringLibrary, row_tool_candidates  # noqa: E402

DDL = """
CREATE TABLE lowering_cores (core_id TEXT PRIMARY KEY, family TEXT NOT NULL, name TEXT NOT NULL, source_function TEXT, logical_abi_digest TEXT, source_digest TEXT, extraction_contract_sha256 TEXT, first_seen REAL NOT NULL);
CREATE TABLE lowerings (core_id TEXT NOT NULL REFERENCES lowering_cores(core_id), backend TEXT NOT NULL, contract TEXT, entry TEXT NOT NULL, api_path TEXT NOT NULL, abi_path TEXT, product_path TEXT, product_sha256 TEXT, abi_schema TEXT, arch TEXT, verdict TEXT NOT NULL, scanned REAL NOT NULL, PRIMARY KEY (core_id, backend, api_path));
"""


def _lowering(core, backend, verdict, tag):
    return (core, backend, None, f"{tag}_entry", f"{tag}/api.yaml", f"{tag}/abi.json",
            f"{tag}/product.bin", "sha", "abi-v1", None, verdict, 2.0)


class LibraryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "accounts.db"
        conn = sqlite3.connect(self.db)
        conn.executescript(DDL)
        conn.executemany("INSERT INTO lowering_cores (core_id, family, name, first_seen) VALUES (?,?,?,?)",
                         [("fountain:def", "fountain", "fountain", 1.0),
                          ("entity:abc", "entity", "entity", 1.0)])
        conn.executemany("INSERT INTO lowerings VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
            _lowering("entity:abc", "wasm", "built", "ew"),
            _lowering("entity:abc", "c", "built", "ec"),
            _lowering("fountain:def", "wgsl", "built", "fg"),
            _lowering("fountain:def", "wasm", "described", "fw"),
        ])
        conn.commit()
        conn.close()
        self.lib = LoweringLibrary(self.db, self.root)

    def test_cores(self):
        self.assertEqual([c.core_id for c in self.lib.cores()], ["entity:abc", "fountain:def"])

    def test_filter_and_group(self):
        self.assertEqual({l.core_id for l in self.lib.lowerings(backend="wasm")},
                         {"entity:abc", "fountain:def"})
        self.assertEqual(len(self.lib.lowerings(core_id="entity:abc", verdict="built")), 2)
        groups = self.lib.by_core()
        self.assertEqual({k: len(v) for k, v in groups.items()},
                         {"entity:abc": 2, "fountain:def": 2})

    def test_menu_order(self):
        menu = self.lib.menu()
        self.assertEqual([m.core.core_id for m in menu], ["entity:abc", "fountain:def"])
        self.assertEqual([l.backend for l in menu[0].lowerings], ["c", "wasm"])
        self.assertEqual([l.backend for l in menu[1].lowerings], ["wasm", "wgsl"])

    def test_row_tool_candidates(self):
        menu = self.lib.menu()
        got = row_tool_candidates(menu[0])
        self.assertEqual([(l.core_id, l.backend) for l in got], [("entity:abc", "c")])
        self.assertEqual(row_tool_candidates(menu[1]), [])

    def test_resolve(self):
        low = self.lib.lowerings(backend="c")[0]
        paths = low.resolve(self.root)
        self.assertEqual(paths["api"], (self.root / "ec/api.yaml").resolve())
        self.assertEqual(paths["abi"], (self.root / "ec/abi.json").resolve())
        self.assertEqual(paths["product"], (self.root / "ec/product.bin").resolve())
        self.assertTrue(all(p.is_absolute() for p in paths.values()))

    def test_empty_database_refused(self):
        empty = self.root / "empty.db"
        sqlite3.connect(empty).close()
        with self.assertRaises(LookupError) as ctx:
            LoweringLibrary(empty, self.root).cores()
        self.assertIn("lowering_cores", str(ctx.exception))
        conn = sqlite3.connect(empty)
        self.assertEqual(conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0], 0)
        conn.close()


if __name__ == "__main__":
    unittest.main()
