"""Read-only view of the Turing-side lowering library (accounts.db).

One core (a logical function) owns several lowerings (backends). Nodus
organizes them as siblings and offers them; it never writes the database.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

BACKEND_ORDER = ("c", "llvm", "fortran", "wasm", "wgsl", "spirv", "cpython")
ROW_TOOL_BACKENDS = ("c", "llvm", "fortran")
_TABLES = ("lowering_cores", "lowerings")


@dataclass(frozen=True)
class Core:
    core_id: str
    family: str
    name: str
    first_seen: float
    source_function: str | None = None
    logical_abi_digest: str | None = None
    source_digest: str | None = None
    extraction_contract_sha256: str | None = None


@dataclass(frozen=True)
class Lowering:
    core_id: str
    backend: str
    entry: str
    api_path: str
    verdict: str
    scanned: float
    contract: str | None = None
    abi_path: str | None = None
    product_path: str | None = None
    product_sha256: str | None = None
    abi_schema: str | None = None
    arch: str | None = None

    def resolve(self, root: Path) -> dict[str, Path]:
        """Absolute api/abi/product paths (only those the row records)."""
        base = Path(root).resolve()
        out: dict[str, Path] = {}
        for key, value in (
            ("api", self.api_path),
            ("abi", self.abi_path),
            ("product", self.product_path),
        ):
            if value:
                out[key] = (base / value).resolve()
        return out


@dataclass(frozen=True)
class MenuEntry:
    core: Core
    lowerings: tuple[Lowering, ...]


def _backend_rank(backend: str) -> int:
    try:
        return BACKEND_ORDER.index(backend)
    except ValueError:
        return len(BACKEND_ORDER)


class LoweringLibrary:
    def __init__(self, db_path: str | Path, root: str | Path):
        self.db_path = Path(db_path)
        self.root = Path(root)

    def _connect(self) -> sqlite3.Connection:
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        present = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table in _TABLES:
            if table not in present:
                conn.close()
                raise LookupError(f"lowering library table missing: {table}")
        return conn

    def _query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        conn = self._connect()
        try:
            return conn.execute(sql, args).fetchall()
        finally:
            conn.close()

    def cores(self) -> list[Core]:
        rows = self._query(
            "SELECT core_id, family, name, source_function, logical_abi_digest, "
            "source_digest, extraction_contract_sha256, first_seen "
            "FROM lowering_cores ORDER BY family, name, core_id"
        )
        return [Core(**dict(r)) for r in rows]

    def lowerings(
        self,
        core_id: str | None = None,
        backend: str | None = None,
        verdict: str | None = None,
    ) -> list[Lowering]:
        where, args = [], []
        for column, value in (("core_id", core_id), ("backend", backend), ("verdict", verdict)):
            if value is not None:
                where.append(f"{column} = ?")
                args.append(value)
        sql = (
            "SELECT core_id, backend, contract, entry, api_path, abi_path, "
            "product_path, product_sha256, abi_schema, arch, verdict, scanned "
            "FROM lowerings"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY core_id, backend, api_path"
        return [Lowering(**dict(r)) for r in self._query(sql, tuple(args))]

    def by_core(self) -> dict[str, list[Lowering]]:
        grouped: dict[str, list[Lowering]] = {}
        for item in self.lowerings():
            grouped.setdefault(item.core_id, []).append(item)
        return grouped

    def menu(self) -> list[MenuEntry]:
        grouped = self.by_core()
        entries = []
        for core in sorted(self.cores(), key=lambda c: (c.family, c.name, c.core_id)):
            items = sorted(
                grouped.get(core.core_id, ()),
                key=lambda l: (_backend_rank(l.backend), l.api_path),
            )
            entries.append(MenuEntry(core, tuple(items)))
        return entries


def row_tool_candidates(entry: MenuEntry) -> list[Lowering]:
    """Lowerings the row adopter (src/turing_product_adopter.cpp) can adopt."""
    return [
        l
        for l in entry.lowerings
        if l.backend in ROW_TOOL_BACKENDS and l.verdict == "built"
    ]
