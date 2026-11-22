"""mesh_crypto: the mesh's signing and sealing, as interpreted Python rows (ABI v2 bytes ports, python_row_tool.h).

docs/MESH_CANVAS_UP_PORTS_2026-11-21.md, stages 3-4; the site's KEY_CHANGE_PROTOCOL.md (workspace root) for how keys
and suites change. Owner, 2026-11-21: "we need to be able to make the correct keys for a quantum future safely on both
linux and windows"; "establish protocol for forcing key changes across the site safely, any time we need to change our
crypto". The crypto runs in pyca/cryptography (prebuilt wheels on Windows and Linux, OpenSSL 4 built in), called out
from Nodus rows until a native build replaces it behind the same rows.

Suite 1 (every format names its suite; a holder accepts only the suites in its allowlist):
  signing  Ed25519 + ML-DSA-65 (FIPS 204): a signature holds only if both do
  sealing  X25519 + ML-KEM-768 (FIPS 203): a reader's wrapping key is HKDF-SHA256 of both shared secrets
  content  ChaCha20-Poly1305, one fresh key per snapshot

Identity. A node's key has an epoch; its *lineage* is the kid of its epoch-0 key and never changes. Writers name their
readers by lineage and seal to each one's current key; a replica names its writer by lineage, so a rotation never
breaks the one-writer rule. kid = SHA-256 of a public block.

A node's folder (NODUS_MESH_KEYS names it; the node's identity: process state, never in a canvas):
  sign.ed25519 sign.mldsa65 kem.x25519 kem.mlkem768   the current private keys (0600); node.pub, their public block
  retired/<epoch>/   earlier private keys, kept until retired (sealed data addressed to them can still be opened)
  peers/<kid>.pub    public blocks of known keys, any epoch      revoked/<kid>   refused at once (a force record)
  cutover/<kid>      the time (ns) after which that older epoch is refused (a rotation record)
  root.pub           the site root's public block (force records)  suites   allowed suite ids, one per line (default 1)
  records/           the newest record time seen per kind (no replays)

Public block "NMP2": suite u16 | epoch u32 | lineage 32 (zeros at epoch 0) | Ed25519 32 | ML-DSA-65 1952 |
                     X25519 32 | ML-KEM-768 1184

Rows, in the order a snapshot travels (each one duty):
  seal    "NRS1" | n u32 | n reader lineages | snapshot -> "NMC1" sealed to their current keys, "NMC0" when n = 0
  sign    sealed -> "NMS2" | suite | kid | generation u64 | length u64 | body | Ed25519 sig | ML-DSA-65 sig
  verify  envelope -> "NMV1" | writer lineage | generation | body: a known, unrevoked, not cut-over key of an allowed
          suite, both signatures; a rotation ("NMK1") or force ("NMF1") record arriving here is applied, and yields b""
  open    verified -> "NMO1" | writer lineage | generation | readable u8 | snapshot (empty for a non-reader)
A refusal returns b"" (nothing goes on).

Records:
  rotation "NMK1": suite | old kid | new public block | effective ns | grace ns | old Ed25519 | old ML-DSA | new Ed25519 |
           new ML-DSA (over everything before the signatures). Accepted only if the old kid is trusted now, the new
           block is the old lineage's next epoch, and the record is newer than the last one seen.
  force    "NMF1": suite | time ns | n u32 | n revoked kids | has replacement u8 | [public block] | root Ed25519 |
           root ML-DSA. Accepted only under root.pub and newer than the last force record seen. No grace: at once.

Operators:  python mesh_crypto.py keygen <folder>        a node's first keys (never replaces any)
            python mesh_crypto.py rotate <folder> [grace s]  next epoch; prints the rotation record's path
            python mesh_crypto.py root-keygen <folder>   the site root (keep the folder offline)
            python mesh_crypto.py force <root folder> <out file> <revoked kid hex>... [--replace <node.pub>]
            python mesh_crypto.py accept <folder> <record file>   apply a rotation or force record
"""
from __future__ import annotations

import hashlib
import os
import struct
import sys
import threading
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization as _ser
from cryptography.hazmat.primitives.asymmetric import mldsa, mlkem
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

SUITE = 1
ED_PUB, ML_PUB, X_PUB, KEM_PUB, ED_SIG, ML_SIG, KEM_CT, KID = 32, 1952, 32, 1184, 64, 3309, 1088, 32
PUB_HEAD = 4 + 2 + 4 + KID
PUB_BYTES = PUB_HEAD + ED_PUB + ML_PUB + X_PUB + KEM_PUB
SIGS = ED_SIG + ML_SIG
_RAW = (_ser.Encoding.Raw, _ser.PublicFormat.Raw)
_PRIV = (_ser.Encoding.Raw, _ser.PrivateFormat.Raw, _ser.NoEncryption())
_lock = threading.Lock()
_last_generation = [0]


class MeshKeyError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------- key blocks
class Public:
    def __init__(self, block: bytes):
        if len(block) != PUB_BYTES or block[:4] != b"NMP2":
            raise MeshKeyError("not a mesh public block")
        self.block = block
        self.suite, self.epoch = struct.unpack_from("<HI", block, 4)
        lineage = block[10:10 + KID]
        self.kid = hashlib.sha256(block).digest()
        self.lineage = self.kid if self.epoch == 0 else lineage
        at = PUB_HEAD
        self.ed = Ed25519PublicKey.from_public_bytes(block[at:at + ED_PUB]); at += ED_PUB
        self.ml = mldsa.MLDSA65PublicKey.from_public_bytes(block[at:at + ML_PUB]); at += ML_PUB
        self.x = X25519PublicKey.from_public_bytes(block[at:at + X_PUB]); at += X_PUB
        self.kem = mlkem.MLKEM768PublicKey.from_public_bytes(block[at:at + KEM_PUB])

    def verify(self, signatures: bytes, message: bytes) -> None:
        self.ed.verify(signatures[:ED_SIG], message)
        self.ml.verify(signatures[ED_SIG:], message)


class Private:
    def __init__(self, folder: Path):
        self.ed = Ed25519PrivateKey.from_private_bytes((folder / "sign.ed25519").read_bytes())
        self.ml = mldsa.MLDSA65PrivateKey.from_seed_bytes((folder / "sign.mldsa65").read_bytes())
        self.x = X25519PrivateKey.from_private_bytes((folder / "kem.x25519").read_bytes())
        self.kem = mlkem.MLKEM768PrivateKey.from_seed_bytes((folder / "kem.mlkem768").read_bytes())
        self.public = Public((folder / "node.pub").read_bytes())

    def sign(self, message: bytes) -> bytes:
        return self.ed.sign(message) + self.ml.sign(message)


def _block(epoch: int, lineage: bytes, ed, ml, x, kem) -> bytes:
    return (b"NMP2" + struct.pack("<HI", SUITE, epoch) + (lineage if epoch else b"\0" * KID)
            + ed.public_key().public_bytes(*_RAW) + ml.public_key().public_bytes(*_RAW)
            + x.public_key().public_bytes(*_RAW) + kem.public_key().public_bytes_raw())


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def _make_keys(folder: Path, epoch: int, lineage: bytes) -> bytes:
    ed, x = Ed25519PrivateKey.generate(), X25519PrivateKey.generate()
    ml_seed, kem_seed = os.urandom(32), os.urandom(64)
    ml, kem = mldsa.MLDSA65PrivateKey.from_seed_bytes(ml_seed), mlkem.MLKEM768PrivateKey.from_seed_bytes(kem_seed)
    for name, data in (("sign.ed25519", ed.private_bytes(*_PRIV)), ("sign.mldsa65", ml_seed),
                       ("kem.x25519", x.private_bytes(*_PRIV)), ("kem.mlkem768", kem_seed)):
        _write_private(folder / name, data)
    block = _block(epoch, lineage, ed, ml, x, kem)
    (folder / "node.pub").write_bytes(block)
    return block


# ------------------------------------------------------------------------------------------------------- the folder
def _folder() -> Path:
    name = os.environ.get("NODUS_MESH_KEYS", "")
    if not name:
        raise MeshKeyError("NODUS_MESH_KEYS names no key folder")
    return Path(name)


def _own(folder: Path | None = None) -> Private:
    return Private(folder or _folder())


def _retired(folder: Path) -> list[Private]:
    out = []
    for d in sorted((folder / "retired").glob("*"), reverse=True):
        try:
            out.append(Private(d))
        except (OSError, MeshKeyError, ValueError):
            continue
    return out


def _suites(folder: Path) -> set[int]:
    try:
        return {int(line) for line in (folder / "suites").read_text().split() if line.strip()}
    except OSError:
        return {SUITE}


def _known(folder: Path, kid: bytes) -> Public | None:
    """A key this node trusts now: itself (any epoch it holds) or a peer, not revoked, not past its cutover."""
    if (folder / "revoked" / kid.hex()).exists():
        return None
    try:
        cut = int((folder / "cutover" / kid.hex()).read_text())
        if time.time_ns() > cut:
            return None
    except (OSError, ValueError):
        pass
    for path in (folder / "node.pub", folder / "peers" / (kid.hex() + ".pub")):
        try:
            p = Public(path.read_bytes())
        except (OSError, MeshKeyError):
            continue
        if p.kid == kid:
            return p
    return None


def _current_of(folder: Path, lineage: bytes) -> Public | None:
    """A lineage's newest trusted key: the one a writer seals to."""
    best = None
    candidates = [folder / "node.pub"] + sorted((folder / "peers").glob("*.pub"))
    for path in candidates:
        try:
            p = Public(path.read_bytes())
        except (OSError, MeshKeyError):
            continue
        if p.lineage == lineage and _known(folder, p.kid) and (best is None or p.epoch > best.epoch):
            best = p
    return best


def _newer(folder: Path, kind: str, stamp: int) -> bool:
    path = folder / "records" / kind
    try:
        if stamp <= int(path.read_text()):
            return False
    except (OSError, ValueError):
        pass
    path.parent.mkdir(exist_ok=True)
    path.write_text(str(stamp))
    return True


# ------------------------------------------------------------------------------------------------------- operators
def keygen(folder) -> bytes:
    """A node's first keys (epoch 0) in `folder`; refuses to replace any. Returns its public block (node.pub)."""
    folder = Path(folder)
    if (folder / "node.pub").exists() or (folder / "sign.ed25519").exists():
        raise MeshKeyError(f"{folder} already holds mesh keys; keys are never replaced in place")
    folder.mkdir(parents=True, exist_ok=True)
    for sub in ("peers", "revoked", "cutover", "retired", "records"):
        (folder / sub).mkdir(exist_ok=True)
    return _make_keys(folder, 0, b"\0" * KID)


def rotate(folder, grace_s: float = 3600.0) -> bytes:
    """The node's next epoch: the current keys move to retired/<epoch>/, new ones take their place, and the rotation
    record (signed by both) is written to rotation-<epoch>.rec and returned. Peers accept it with `accept`."""
    folder = Path(folder)
    old = _own(folder)
    retired = folder / "retired" / str(old.public.epoch)
    retired.mkdir(parents=True)
    for name in ("sign.ed25519", "sign.mldsa65", "kem.x25519", "kem.mlkem768", "node.pub"):
        os.replace(folder / name, retired / name)
    block = _make_keys(folder, old.public.epoch + 1, old.public.lineage)
    new = _own(folder)
    effective, grace = time.time_ns(), int(grace_s * 1e9)
    message = b"NMK1" + struct.pack("<H", SUITE) + old.public.kid + block + struct.pack("<QQ", effective, grace)
    record = message + old.sign(message) + new.sign(message)
    (folder / f"rotation-{new.public.epoch}.rec").write_bytes(record)
    return record


def root_keygen(folder) -> bytes:
    """The site root (sign keys only; keep `folder` offline). Returns root.pub for every node's folder."""
    folder = Path(folder)
    if (folder / "node.pub").exists():
        raise MeshKeyError(f"{folder} already holds a root")
    folder.mkdir(parents=True, exist_ok=True)
    return _make_keys(folder, 0, b"\0" * KID)


def force(root_folder, revoked: list[bytes], replacement: bytes | None = None) -> bytes:
    """A force record under the site root: `revoked` kids refused at once, and `replacement` (a public block)
    installed as its lineage's next key."""
    root = _own(Path(root_folder))
    body = struct.pack("<HQI", SUITE, time.time_ns(), len(revoked)) + b"".join(revoked)
    body += (b"\1" + replacement) if replacement else b"\0"
    message = b"NMF1" + body
    return message + root.sign(message)


def accept(folder, record: bytes) -> str:
    """Apply a rotation or force record to `folder`. Returns what it did; raises MeshKeyError when refused."""
    folder = Path(folder)
    if record[:4] == b"NMK1":
        head = 4 + 2 + KID + PUB_BYTES + 16
        if len(record) != head + 2 * SIGS:
            raise MeshKeyError("malformed rotation record")
        (suite,) = struct.unpack_from("<H", record, 4)
        old_kid = record[6:6 + KID]
        new = Public(record[6 + KID:6 + KID + PUB_BYTES])
        effective, grace = struct.unpack_from("<QQ", record, 6 + KID + PUB_BYTES)
        message = record[:head]
        old = _known(folder, old_kid)
        if suite not in _suites(folder) or old is None:
            raise MeshKeyError("rotation from a key this node does not trust now")
        if new.lineage != old.lineage or new.epoch != old.epoch + 1:
            raise MeshKeyError("the new key is not that lineage's next epoch")
        old.verify(record[head:head + SIGS], message)
        new.verify(record[head + SIGS:], message)
        if not _newer(folder, "rotation-" + old.lineage.hex(), effective):
            raise MeshKeyError("an old or replayed rotation record")
        if new.kid != _own(folder).public.kid:
            (folder / "peers" / (new.kid.hex() + ".pub")).write_bytes(new.block)
        (folder / "cutover").mkdir(exist_ok=True)
        (folder / "cutover" / old_kid.hex()).write_text(str(effective + grace))
        return f"rotation: lineage {old.lineage.hex()[:8]} epoch {old.epoch} -> {new.epoch}"
    if record[:4] == b"NMF1":
        root = Public((folder / "root.pub").read_bytes())
        message, signatures = record[:-SIGS], record[-SIGS:]
        root.verify(signatures, message)
        suite, stamp, n = struct.unpack_from("<HQI", record, 4)
        at = 4 + 14
        revoked = [record[at + i * KID: at + (i + 1) * KID] for i in range(n)]
        at += n * KID
        replacement = Public(record[at + 1:at + 1 + PUB_BYTES]) if record[at:at + 1] == b"\1" else None
        if not _newer(folder, "force", stamp):
            raise MeshKeyError("an old or replayed force record")
        (folder / "revoked").mkdir(exist_ok=True)
        for kid in revoked:
            (folder / "revoked" / kid.hex()).write_text(str(stamp))
        if replacement:
            (folder / "peers" / (replacement.kid.hex() + ".pub")).write_bytes(replacement.block)
        return f"force: {n} revoked" + (f", replacement epoch {replacement.epoch}" if replacement else "")
    raise MeshKeyError("not a rotation or force record")


# ------------------------------------------------------------------------------------------------------------- rows
def _wrap_key(x_shared: bytes, kem_shared: bytes, kid: bytes, eph: bytes, ct: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=kid,
                info=b"nodus mesh seal v1" + eph + ct).derive(x_shared + kem_shared)


def seal(request: bytes) -> bytes:
    """ "NRS1" | n | n reader lineages | snapshot -> "NMC1" sealed to their current keys, "NMC0" (public) when n = 0."""
    try:
        if len(request) < 8 or request[:4] != b"NRS1":
            return b""
        (n,) = struct.unpack_from("<I", request, 4)
        at = 8 + n * KID
        if n > 4096 or len(request) < at:
            return b""
        snapshot = request[at:]
        if n == 0:
            return b"NMC0" + snapshot
        folder = _folder()
        content_key, nonce = ChaCha20Poly1305.generate_key(), os.urandom(12)
        entries = []
        for i in range(n):
            reader = _current_of(folder, request[8 + i * KID: 8 + (i + 1) * KID])
            if reader is None:
                return b""                       # a named reader with no trusted key: refuse, never drop a reader
            eph = X25519PrivateKey.generate()
            eph_pub = eph.public_key().public_bytes(*_RAW)
            kem_shared, ct = reader.kem.encapsulate()
            wrap = _wrap_key(eph.exchange(reader.x), kem_shared, reader.kid, eph_pub, ct)
            entries.append(reader.kid + eph_pub + ct + ChaCha20Poly1305(wrap).encrypt(b"\0" * 12, content_key, reader.kid))
        header = b"NMC1" + struct.pack("<HI", SUITE, n) + b"".join(entries) + nonce
        body = ChaCha20Poly1305(content_key).encrypt(nonce, snapshot, header)
        return header + struct.pack("<Q", len(body)) + body
    except MeshKeyError:
        return b""


def sign(body: bytes) -> bytes:
    """body -> "NMS2" | suite | writer kid | generation | length | body | Ed25519 + ML-DSA-65 signatures."""
    own = _own()
    with _lock:
        generation = max(time.time_ns(), _last_generation[0] + 1)
        _last_generation[0] = generation
    message = b"NMS2" + struct.pack("<H", SUITE) + own.public.kid + struct.pack("<QQ", generation, len(body)) + body
    return message + own.sign(message)


def verify(envelope: bytes) -> bytes:
    """A signed envelope from a trusted key -> "NMV1" | writer lineage | generation | body; records are applied."""
    try:
        folder = _folder()
        if envelope[:4] in (b"NMK1", b"NMF1"):
            accept(folder, envelope)
            return b""
        head = 4 + 2 + KID + 16
        if len(envelope) < head + SIGS or envelope[:4] != b"NMS2":
            return b""
        (suite,) = struct.unpack_from("<H", envelope, 4)
        kid = envelope[6:6 + KID]
        generation, length = struct.unpack_from("<QQ", envelope, 6 + KID)
        if suite not in _suites(folder) or len(envelope) != head + length + SIGS:
            return b""
        writer = _known(folder, kid)
        if writer is None or writer.suite != suite:
            return b""
        writer.verify(envelope[head + length:], envelope[:head + length])
        return b"NMV1" + writer.lineage + struct.pack("<Q", generation) + envelope[head:head + length]
    except Exception:
        return b""


def open(verified: bytes) -> bytes:  # noqa: A001 - the row's name
    """ "NMV1" | writer | generation | sealed -> "NMO1" | writer | generation | readable | snapshot (or empty)."""
    if len(verified) < 4 + KID + 8 + 4 or verified[:4] != b"NMV1":
        return b""
    writer, generation = verified[4:4 + KID], verified[4 + KID:4 + KID + 8]
    sealed = verified[4 + KID + 8:]
    out = b"NMO1" + writer + generation
    if sealed[:4] == b"NMC0":
        return out + b"\1" + sealed[4:]
    if sealed[:4] != b"NMC1" or len(sealed) < 10:
        return b""
    try:
        folder = _folder()
        (_suite, n) = struct.unpack_from("<HI", sealed, 4)
        entry = KID + X_PUB + KEM_CT + 48
        nonce_at = 10 + n * entry
        if n > 4096 or len(sealed) < nonce_at + 20:
            return b""
        header, nonce = sealed[:nonce_at + 12], sealed[nonce_at:nonce_at + 12]
        (length,) = struct.unpack_from("<Q", sealed, nonce_at + 12)
        body = sealed[nonce_at + 20:]
        if len(body) != length:
            return b""
        mine = {p.public.kid: p for p in [_own(folder)] + _retired(folder)}   # current and not yet retired epochs
        for i in range(n):
            e = sealed[10 + i * entry: 10 + (i + 1) * entry]
            kid, eph_pub, ct, wrapped = e[:KID], e[KID:KID + X_PUB], e[KID + X_PUB:KID + X_PUB + KEM_CT], e[-48:]
            key = mine.get(kid)
            if key is None:
                continue
            shared = key.x.exchange(X25519PublicKey.from_public_bytes(eph_pub))
            wrap = _wrap_key(shared, key.kem.decapsulate(ct), kid, eph_pub, ct)
            content_key = ChaCha20Poly1305(wrap).decrypt(b"\0" * 12, wrapped, kid)
            return out + b"\1" + ChaCha20Poly1305(content_key).decrypt(nonce, body, header)
        return out + b"\0"                       # not a reader: only that the writer exists, and its generation
    except Exception:
        return b""


def _main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "keygen":
        p = Public(keygen(argv[1]))
        print(f"node lineage {p.lineage.hex()} kid {p.kid.hex()} epoch 0")
    elif len(argv) in (2, 3) and argv[0] == "rotate":
        record = rotate(argv[1], float(argv[2]) if len(argv) == 3 else 3600.0)
        p = Public(record[6 + KID:6 + KID + PUB_BYTES])
        print(f"rotated: lineage {p.lineage.hex()[:16]} epoch {p.epoch} kid {p.kid.hex()}; record "
              f"{Path(argv[1]) / f'rotation-{p.epoch}.rec'}")
    elif len(argv) == 2 and argv[0] == "root-keygen":
        root_keygen(argv[1])
        print(f"site root made in {argv[1]}: copy its node.pub to every node folder as root.pub; keep the folder offline")
    elif len(argv) >= 4 and argv[0] == "force":
        replacement = None
        rest = argv[3:]
        if "--replace" in rest:
            i = rest.index("--replace")
            replacement = Path(rest[i + 1]).read_bytes()
            rest = rest[:i] + rest[i + 2:]
        Path(argv[2]).write_bytes(force(argv[1], [bytes.fromhex(k) for k in rest], replacement))
        print(f"force record written to {argv[2]}")
    elif len(argv) == 3 and argv[0] == "accept":
        print(accept(argv[1], Path(argv[2]).read_bytes()))
    else:
        print(__doc__.split("Operators:")[1].split('"""')[0] if "Operators:" in __doc__ else "usage")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
