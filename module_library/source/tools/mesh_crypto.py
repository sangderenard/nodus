"""mesh_crypto: the mesh's signing and sealing, as interpreted Python rows (ABI v2 bytes ports, python_row_tool.h).

docs/MESH_CANVAS_UP_PORTS_2026-11-21.md, stages 3-4. Owner, 2026-11-21: "we need to be able to make the correct keys for
a quantum future safely on both linux and windows"; the crypto runs in pyca/cryptography (prebuilt wheels on Windows
and Linux, OpenSSL 4 built in), called out from Nodus rows until a native build replaces it behind the same rows.

Every key is hybrid, a classical half and a post-quantum half, so a break of either alone breaks nothing:
  signing  Ed25519 + ML-DSA-65 (FIPS 204): an envelope verifies only if both signatures do
  sealing  X25519 + ML-KEM-768 (FIPS 203): a reader's wrapping key is HKDF-SHA256 of both shared secrets
  content  ChaCha20-Poly1305, one fresh key per snapshot

A node's keys live in the folder NODUS_MESH_KEYS names (the node's identity: process state, never in a canvas):
  sign.ed25519 (32 bytes) sign.mldsa65 (32-byte seed) kem.x25519 (32 bytes) kem.mlkem768 (64-byte seed)  private, 0600
  node.pub   its public block: "NMP1" | Ed25519 32 | ML-DSA-65 1952 | X25519 32 | ML-KEM-768 1184
  peers/<kid hex>.pub   the public blocks of the nodes it knows (a node's kid = SHA-256 of its public block)

The rows, in the order a snapshot travels (each one duty):
  seal    "NRS1" | n u32 | n reader kids | snapshot  ->  "NMC1" sealed to those readers, or "NMC0" (n = 0: public)
  sign    sealed  ->  "NMS1" | writer kid | generation u64 | body length u64 | body | Ed25519 sig | ML-DSA-65 sig
  verify  envelope  ->  "NMV1" | writer kid | generation | body, if the writer is a known node and both signatures hold
  open    verified  ->  "NMO1" | writer kid | generation | readable u8 | snapshot (empty when this node is no reader)
A refusal returns b"" (nothing goes on). Everyone can verify a sealed snapshot and so learn the writer and generation;
only a named reader can open it.

Operators make a node's keys with:  python mesh_crypto.py keygen <folder>   (never replaces existing keys)
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

PUB_MAGIC, ED_PUB, ML_PUB, X_PUB, KEM_PUB = b"NMP1", 32, 1952, 32, 1184
PUB_BYTES = 4 + ED_PUB + ML_PUB + X_PUB + KEM_PUB
ED_SIG, ML_SIG, KEM_CT, KID = 64, 3309, 1088, 32
_RAW = (_ser.Encoding.Raw, _ser.PublicFormat.Raw)
_lock = threading.Lock()
_cache: dict = {}
_last_generation = [0]


class MeshKeyError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------------- keys
def _folder() -> Path:
    name = os.environ.get("NODUS_MESH_KEYS", "")
    if not name:
        raise MeshKeyError("NODUS_MESH_KEYS names no key folder")
    return Path(name)


def public_block(ed: Ed25519PublicKey, ml, x: X25519PublicKey, kem) -> bytes:
    return (PUB_MAGIC + ed.public_bytes(*_RAW) + ml.public_bytes(*_RAW) + x.public_bytes(*_RAW)
            + kem.public_bytes_raw())


def kid_of(block: bytes) -> bytes:
    return hashlib.sha256(block).digest()


def parse_public(block: bytes):
    if len(block) != PUB_BYTES or not block.startswith(PUB_MAGIC):
        raise MeshKeyError("not a mesh public block")
    at = 4
    ed = Ed25519PublicKey.from_public_bytes(block[at:at + ED_PUB]); at += ED_PUB
    ml = mldsa.MLDSA65PublicKey.from_public_bytes(block[at:at + ML_PUB]); at += ML_PUB
    x = X25519PublicKey.from_public_bytes(block[at:at + X_PUB]); at += X_PUB
    kem = mlkem.MLKEM768PublicKey.from_public_bytes(block[at:at + KEM_PUB])
    return ed, ml, x, kem


def _own():
    """This node's private keys and kid, read once (a single load event)."""
    with _lock:
        folder = _folder()
        cached = _cache.get(("own", str(folder)))
        if cached:
            return cached
        ed = Ed25519PrivateKey.from_private_bytes((folder / "sign.ed25519").read_bytes())
        ml = mldsa.MLDSA65PrivateKey.from_seed_bytes((folder / "sign.mldsa65").read_bytes())
        x = X25519PrivateKey.from_private_bytes((folder / "kem.x25519").read_bytes())
        kem = mlkem.MLKEM768PrivateKey.from_seed_bytes((folder / "kem.mlkem768").read_bytes())
        block = public_block(ed.public_key(), ml.public_key(), x.public_key(), kem.public_key())
        own = {"ed": ed, "ml": ml, "x": x, "kem": kem, "block": block, "kid": kid_of(block)}
        _cache[("own", str(folder))] = own
        return own


def _peer(kid: bytes):
    """A known node's public keys by kid: this node itself, or peers/<kid hex>.pub (whose hash must be the kid)."""
    own = _own()
    if kid == own["kid"]:
        return parse_public(own["block"])
    path = _folder() / "peers" / (kid.hex() + ".pub")
    try:
        block = path.read_bytes()
    except OSError:
        return None
    if kid_of(block) != kid:
        return None
    return parse_public(block)


def keygen(folder) -> bytes:
    """Make a node's keys in `folder`; refuses to replace any. Returns the node's public block (node.pub)."""
    folder = Path(folder)
    names = ("sign.ed25519", "sign.mldsa65", "kem.x25519", "kem.mlkem768", "node.pub")
    if any((folder / n).exists() for n in names):
        raise MeshKeyError(f"{folder} already holds mesh keys; keys are never replaced in place")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "peers").mkdir(exist_ok=True)
    ed, x = Ed25519PrivateKey.generate(), X25519PrivateKey.generate()
    ml_seed, kem_seed = os.urandom(32), os.urandom(64)
    ml, kem = mldsa.MLDSA65PrivateKey.from_seed_bytes(ml_seed), mlkem.MLKEM768PrivateKey.from_seed_bytes(kem_seed)
    secrets = {
        "sign.ed25519": ed.private_bytes(_ser.Encoding.Raw, _ser.PrivateFormat.Raw, _ser.NoEncryption()),
        "sign.mldsa65": ml_seed,
        "kem.x25519": x.private_bytes(_ser.Encoding.Raw, _ser.PrivateFormat.Raw, _ser.NoEncryption()),
        "kem.mlkem768": kem_seed,
    }
    for name, data in secrets.items():
        fd = os.open(folder / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    block = public_block(ed.public_key(), ml.public_key(), x.public_key(), kem.public_key())
    (folder / "node.pub").write_bytes(block)
    return block


# ------------------------------------------------------------------------------------------------------------- rows
def _wrap_key(x_shared: bytes, kem_shared: bytes, kid: bytes, eph: bytes, ct: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=kid,
                info=b"nodus mesh seal v1" + eph + ct).derive(x_shared + kem_shared)


def seal(request: bytes) -> bytes:
    """ "NRS1" | n | n reader kids | snapshot  ->  sealed to those readers ("NMC1"), or public ("NMC0") when n = 0."""
    try:
        if len(request) < 8 or request[:4] != b"NRS1":
            return b""
        (n,) = struct.unpack_from("<I", request, 4)
        at = 8 + n * KID
        if n > 4096 or len(request) < at:
            return b""
        readers = [request[8 + i * KID: 8 + (i + 1) * KID] for i in range(n)]
        snapshot = request[at:]
        if n == 0:
            return b"NMC0" + snapshot
        content_key, nonce = ChaCha20Poly1305.generate_key(), os.urandom(12)
        entries = []
        for kid in readers:
            peer = _peer(kid)
            if peer is None:
                return b""                       # a named reader this node does not know: refuse, never drop one
            _ed, _ml, x_pub, kem_pub = peer
            eph = X25519PrivateKey.generate()
            eph_pub = eph.public_key().public_bytes(*_RAW)
            kem_shared, ct = kem_pub.encapsulate()
            wrap = _wrap_key(eph.exchange(x_pub), kem_shared, kid, eph_pub, ct)
            wrapped = ChaCha20Poly1305(wrap).encrypt(b"\0" * 12, content_key, kid)
            entries.append(kid + eph_pub + ct + wrapped)
        header = b"NMC1" + struct.pack("<I", n) + b"".join(entries) + nonce
        body = ChaCha20Poly1305(content_key).encrypt(nonce, snapshot, header)
        return header + struct.pack("<Q", len(body)) + body
    except MeshKeyError:
        return b""


def sign(body: bytes) -> bytes:
    """body -> "NMS1" | writer kid | generation | length | body | Ed25519 sig | ML-DSA-65 sig (over all before them)."""
    own = _own()
    with _lock:
        generation = max(time.time_ns(), _last_generation[0] + 1)
        _last_generation[0] = generation
    message = b"NMS1" + own["kid"] + struct.pack("<QQ", generation, len(body)) + body
    return message + own["ed"].sign(message) + own["ml"].sign(message)


def verify(envelope: bytes) -> bytes:
    """An envelope from a known node with both signatures valid -> "NMV1" | writer kid | generation | body; else b""."""
    head = 4 + KID + 16
    if len(envelope) < head + ED_SIG + ML_SIG or envelope[:4] != b"NMS1":
        return b""
    kid = envelope[4:4 + KID]
    generation, length = struct.unpack_from("<QQ", envelope, 4 + KID)
    if len(envelope) != head + length + ED_SIG + ML_SIG:
        return b""
    message = envelope[:head + length]
    ed_sig = envelope[head + length: head + length + ED_SIG]
    ml_sig = envelope[head + length + ED_SIG:]
    try:
        peer = _peer(kid)
        if peer is None:
            return b""
        ed, ml, _x, _kem = peer
        ed.verify(ed_sig, message)
        ml.verify(ml_sig, message)
    except Exception:
        return b""
    return b"NMV1" + kid + struct.pack("<Q", generation) + envelope[head:head + length]


def open(verified: bytes) -> bytes:  # noqa: A001 - the row's name
    """ "NMV1" | writer | generation | sealed  ->  "NMO1" | writer | generation | readable | snapshot (or empty)."""
    if len(verified) < 4 + KID + 8 + 4 or verified[:4] != b"NMV1":
        return b""
    writer, generation = verified[4:4 + KID], verified[4 + KID:4 + KID + 8]
    sealed = verified[4 + KID + 8:]
    out = b"NMO1" + writer + generation
    if sealed[:4] == b"NMC0":
        return out + b"\1" + sealed[4:]
    if sealed[:4] != b"NMC1" or len(sealed) < 8:
        return b""
    try:
        own = _own()
        (n,) = struct.unpack_from("<I", sealed, 4)
        entry = KID + X_PUB + KEM_CT + 48
        nonce_at = 8 + n * entry
        if n > 4096 or len(sealed) < nonce_at + 12 + 8:
            return b""
        header = sealed[:nonce_at + 12]
        nonce = sealed[nonce_at:nonce_at + 12]
        (length,) = struct.unpack_from("<Q", sealed, nonce_at + 12)
        body = sealed[nonce_at + 20:]
        if len(body) != length:
            return b""
        for i in range(n):
            e = sealed[8 + i * entry: 8 + (i + 1) * entry]
            kid, eph_pub, ct, wrapped = e[:KID], e[KID:KID + X_PUB], e[KID + X_PUB:KID + X_PUB + KEM_CT], e[-48:]
            if kid != own["kid"]:
                continue
            x_shared = own["x"].exchange(X25519PublicKey.from_public_bytes(eph_pub))
            wrap = _wrap_key(x_shared, own["kem"].decapsulate(ct), kid, eph_pub, ct)
            content_key = ChaCha20Poly1305(wrap).decrypt(b"\0" * 12, wrapped, kid)
            return out + b"\1" + ChaCha20Poly1305(content_key).decrypt(nonce, body, header)
        return out + b"\0"                       # not a reader: only that the writer exists, and its generation
    except Exception:
        return b""


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "keygen":
        block = keygen(sys.argv[2])
        print(f"node kid {kid_of(block).hex()}  public block {Path(sys.argv[2]) / 'node.pub'}")
    else:
        print("usage: python mesh_crypto.py keygen <folder>")
        sys.exit(2)
