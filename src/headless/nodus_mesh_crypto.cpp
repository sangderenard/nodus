// The mesh's signing and sealing, native, on the pinned OpenSSL (scripts/build_openssl.cmake): the four mesh rows
// (module_library/source/tools/mesh_{seal,sign,verify,open}.cpp) and the key operators (tools/mesh_keys) call it
// through nodus_headless_abi.h. docs/MESH_CANVAS_UP_PORTS_2026-11-21.md, stages 3-4; KEY_CHANGE_PROTOCOL.md (workspace
// root) for how keys and suites change. Owner, 2026-11-21: "If we can vendor openSSL 4 ... we use the thing that's
// supposed to be used"; "all of them to use pre and post quantum pairs"; no Python for Nodus.
//
// Suite 1 (every format names its suite; a holder accepts only the suites in its allowlist):
//   signing  Ed25519 + ML-DSA-65 (FIPS 204): a signature holds only if both do
//   sealing  X25519 + ML-KEM-768 (FIPS 203): a reader's wrapping key is HKDF-SHA256 of both shared secrets
//   content  ChaCha20-Poly1305, one fresh key per snapshot
//
// Identity. A node's key has an epoch; its lineage is the kid of its epoch-0 key and never changes. Writers name readers
// by lineage and seal to each one's current key; a replica names its writer by lineage. kid = SHA-256 of a public block.
//
// A node's folder (NODUS_MESH_KEYS names it; the node's identity: process state, never in a canvas):
//   epochs/<n>/   one complete key set: sign.ed25519 sign.mldsa65 (seed) kem.x25519 kem.mlkem768 (seed) node.pub
//   current       the epoch in use. A key set is written whole under a hidden staging name and renamed into
//                 epochs/<n> before current names it, and current is replaced in one rename: a reader sees one whole
//                 key set, the old or the new (review eba4505a). Earlier epochs stay to open data sealed to them.
//   peers/<kid>.pub   known public blocks   revoked/<kid>   refused at once   cutover/<kid>   refused after (ns)
//   root.pub      the site root's public block   suites   allowed suite ids, one per line (default 1)
//   records/<kind>    the newest record time applied per kind; written last, so an interrupted accept is retried
//   rotation-<n>.rec  the record that made epoch n
//
// Public block "NMP2": suite u16 | epoch u32 | lineage 32 (zeros at epoch 0) | Ed25519 32 | ML-DSA-65 1952 |
//                      X25519 32 | ML-KEM-768 1184
// Rows (each one duty; a refusal yields nothing):
//   seal    "NRS1" | n u32 | n reader lineages | snapshot -> "NMC1" sealed to their current keys, "NMC0" when n = 0
//   sign    sealed -> "NMS2" | suite | kid | generation u64 | length u64 | body | Ed25519 sig | ML-DSA-65 sig
//   verify  envelope -> "NMV1" | writer lineage | generation | body; a rotation or force record is applied instead
//   open    verified -> "NMO1" | writer lineage | generation | readable u8 | snapshot (empty for a non-reader)
// Records:
//   rotation "NMK1": suite | old kid | new block | effective ns | grace ns | old sigs | new sigs (over all before them)
//   force    "NMF2": suite | time ns | n u32 | n revoked kids | has allowlist u8 | [count u16 | suite ids u16] |
//            has replacement u8 | [block] | root sigs. Under root.pub only, newest only, at once. An allowlist replaces
//            `suites`: a suite left out is refused everywhere, every key of it with it (a forced crypto change).
#include "nodus_headless_abi.h"

#ifndef NODUS_MESH_CRYPTO
#define NODUS_MESH_CRYPTO 0      // CMake defines it 1 when the pinned OpenSSL (ML-KEM, ML-DSA) is linked
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#if NODUS_MESH_CRYPTO
#include <openssl/core_names.h>
#include <openssl/evp.h>
#include <openssl/kdf.h>
#include <openssl/params.h>
#include <openssl/rand.h>
#endif

// ------------------------------------------------------------------------------------------------ platform region
// File creation that a crash cannot leave half-written: create exclusively, write, flush to the disk, close.
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
namespace nodus::mesh::platform {
inline bool write_new_durable(const std::filesystem::path& p, const uint8_t* d, size_t n) {
    int fd = -1;
    if (_wsopen_s(&fd, p.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _SH_DENYRW, _S_IREAD | _S_IWRITE) != 0) return false;
    size_t at = 0;
    while (at < n) {
        const int w = _write(fd, d + at, static_cast<unsigned>(std::min<size_t>(n - at, 1u << 30)));
        if (w <= 0) { _close(fd); return false; }
        at += static_cast<size_t>(w);
    }
    const bool ok = _commit(fd) == 0;
    return _close(fd) == 0 && ok;
}
inline void sync_dir(const std::filesystem::path&) {}     // NTFS: a rename is journalled with the file it moves
}  // namespace nodus::mesh::platform
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
namespace nodus::mesh::platform {
inline bool write_new_durable(const std::filesystem::path& p, const uint8_t* d, size_t n) {
    const int fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return false;
    size_t at = 0;
    while (at < n) {
        const ssize_t w = ::write(fd, d + at, n - at);
        if (w <= 0) { ::close(fd); return false; }
        at += static_cast<size_t>(w);
    }
    const bool ok = ::fsync(fd) == 0;
    return ::close(fd) == 0 && ok;
}
inline void sync_dir(const std::filesystem::path& dir) {
    const int fd = ::open(dir.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd >= 0) { ::fsync(fd); ::close(fd); }
}
}  // namespace nodus::mesh::platform
#endif
// ------------------------------------------------------------------------------------------- end platform region

namespace nodus::mesh {
namespace {

namespace fs = std::filesystem;
using Bytes = std::vector<uint8_t>;

constexpr uint16_t kSuite = 1;
constexpr size_t kEdPub = 32, kMlPub = 1952, kXPub = 32, kKemPub = 1184, kEdSig = 64, kMlSig = 3309, kKemCt = 1088,
                 kKid = 32;
constexpr size_t kPubHead = 4 + 2 + 4 + kKid;
constexpr size_t kPubBytes = kPubHead + kEdPub + kMlPub + kXPub + kKemPub;
constexpr size_t kSigs = kEdSig + kMlSig;
constexpr size_t kEntry = kKid + kXPub + kKemCt + 48;      // reader kid | ephemeral X25519 | ML-KEM ct | wrapped key
constexpr uint32_t kMaxReaders = 4096;
const char* const kKeyFiles[5] = {"sign.ed25519", "sign.mldsa65", "kem.x25519", "kem.mlkem768", "node.pub"};

template <typename T> void put(Bytes& out, const T& v) {
    const auto* b = reinterpret_cast<const uint8_t*>(&v);
    out.insert(out.end(), b, b + sizeof(T));
}
template <typename T> bool get(const Bytes& in, size_t at, T& v) {
    if (at > in.size() || sizeof(T) > in.size() - at) return false;
    std::memcpy(&v, in.data() + at, sizeof(T));
    return true;
}
void append(Bytes& out, const uint8_t* p, size_t n) { out.insert(out.end(), p, p + n); }
void append(Bytes& out, const Bytes& b) { out.insert(out.end(), b.begin(), b.end()); }
Bytes slice(const Bytes& b, size_t at, size_t n) {
    return at <= b.size() && n <= b.size() - at ? Bytes(b.begin() + static_cast<std::ptrdiff_t>(at),
                                                         b.begin() + static_cast<std::ptrdiff_t>(at + n))
                                                 : Bytes{};
}
std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; ++i) { s += d[p[i] >> 4]; s += d[p[i] & 15]; }
    return s;
}
std::string hex(const Bytes& b) { return hex(b.data(), b.size()); }
bool unhex(const std::string& s, Bytes& out) {
    auto v = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
    if (s.size() % 2) return false;
    out.clear();
    for (size_t i = 0; i < s.size(); i += 2) {
        const int a = v(s[i]), b = v(s[i + 1]);
        if (a < 0 || b < 0) return false;
        out.push_back(static_cast<uint8_t>(a * 16 + b));
    }
    return true;
}
uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

bool read_all(const fs::path& p, Bytes& out) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return false;
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}
bool read_text(const fs::path& p, std::string& out) {
    Bytes b;
    if (!read_all(p, b)) return false;
    out.assign(b.begin(), b.end());
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) out.pop_back();
    return true;
}
std::string random_tag() {
    uint8_t r[8] = {};
#if NODUS_MESH_CRYPTO
    RAND_bytes(r, sizeof(r));
#endif
    return hex(r, sizeof(r));
}
// Replace `p` whole: the new bytes go to a hidden name beside it, durably, then one rename puts them in place.
bool write_replace(const fs::path& p, const Bytes& data) {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    const fs::path tmp = p.parent_path() / ("." + p.filename().string() + ".tmp-" + random_tag());
    if (!platform::write_new_durable(tmp, data.data(), data.size())) { fs::remove(tmp, ec); return false; }
    fs::rename(tmp, p, ec);
    if (ec) { fs::remove(tmp, ec); return false; }
    platform::sync_dir(p.parent_path());
    return true;
}
bool write_replace(const fs::path& p, const std::string& text) { return write_replace(p, Bytes(text.begin(), text.end())); }

#if NODUS_MESH_CRYPTO
// ---------------------------------------------------------------------------------------------------- primitives
struct PkeyFree { void operator()(EVP_PKEY* k) const { EVP_PKEY_free(k); } };
struct CtxFree { void operator()(EVP_PKEY_CTX* c) const { EVP_PKEY_CTX_free(c); } };
struct MdFree { void operator()(EVP_MD_CTX* c) const { EVP_MD_CTX_free(c); } };
struct CipherFree { void operator()(EVP_CIPHER_CTX* c) const { EVP_CIPHER_CTX_free(c); } };
using Pkey = std::unique_ptr<EVP_PKEY, PkeyFree>;
using Ctx = std::unique_ptr<EVP_PKEY_CTX, CtxFree>;

Bytes random_bytes(size_t n) {
    Bytes b(n);
    if (RAND_bytes(b.data(), static_cast<int>(n)) != 1) b.clear();
    return b;
}
Bytes sha256(const uint8_t* p, size_t n) {
    Bytes d(32);
    unsigned len = 0;
    if (EVP_Digest(p, n, d.data(), &len, EVP_sha256(), nullptr) != 1 || len != 32) d.clear();
    return d;
}
Bytes sha256(const Bytes& b) { return sha256(b.data(), b.size()); }

// Ed25519 and X25519 from raw bytes; ML-DSA-65 and ML-KEM-768 from their seeds or raw public keys.
Pkey raw_private(const char* alg, const Bytes& b) {
    return Pkey(EVP_PKEY_new_raw_private_key_ex(nullptr, alg, nullptr, b.data(), b.size()));
}
Pkey raw_public(const char* alg, const uint8_t* p, size_t n) {
    return Pkey(EVP_PKEY_new_raw_public_key_ex(nullptr, alg, nullptr, p, n));
}
Pkey from_params(const char* alg, int selection, const char* param, const uint8_t* p, size_t n) {
    Ctx c(EVP_PKEY_CTX_new_from_name(nullptr, alg, nullptr));
    EVP_PKEY* k = nullptr;
    OSSL_PARAM params[2] = {OSSL_PARAM_construct_octet_string(param, const_cast<uint8_t*>(p), n), OSSL_PARAM_construct_end()};
    if (!c || EVP_PKEY_fromdata_init(c.get()) != 1 || EVP_PKEY_fromdata(c.get(), &k, selection, params) != 1) return nullptr;
    return Pkey(k);
}
Pkey from_seed(const char* alg, const Bytes& seed) {
    return from_params(alg, EVP_PKEY_KEYPAIR, OSSL_PKEY_PARAM_ML_DSA_SEED, seed.data(), seed.size());   // "seed": both
}
Pkey pq_public(const char* alg, const uint8_t* p, size_t n) {
    return from_params(alg, EVP_PKEY_PUBLIC_KEY, OSSL_PKEY_PARAM_PUB_KEY, p, n);
}
Bytes public_of(EVP_PKEY* k, bool pq) {
    Bytes b(4096);
    size_t n = b.size();
    const int ok = pq ? EVP_PKEY_get_octet_string_param(k, OSSL_PKEY_PARAM_PUB_KEY, b.data(), b.size(), &n)
                      : EVP_PKEY_get_raw_public_key(k, b.data(), &n);
    if (ok != 1) return {};
    b.resize(n);
    return b;
}
Bytes raw_private_of(EVP_PKEY* k) {
    Bytes b(64);
    size_t n = b.size();
    if (EVP_PKEY_get_raw_private_key(k, b.data(), &n) != 1) return {};
    b.resize(n);
    return b;
}
// One-shot signing with no digest: pure Ed25519 and pure ML-DSA (empty context).
Bytes sign_with(EVP_PKEY* k, const Bytes& m) {
    std::unique_ptr<EVP_MD_CTX, MdFree> c(EVP_MD_CTX_new());
    size_t n = 0;
    if (!c || EVP_DigestSignInit_ex(c.get(), nullptr, nullptr, nullptr, nullptr, k, nullptr) != 1 ||
        EVP_DigestSign(c.get(), nullptr, &n, m.data(), m.size()) != 1) return {};
    Bytes s(n);
    if (EVP_DigestSign(c.get(), s.data(), &n, m.data(), m.size()) != 1) return {};
    s.resize(n);
    return s;
}
bool verify_with(EVP_PKEY* k, const uint8_t* sig, size_t sn, const Bytes& m) {
    std::unique_ptr<EVP_MD_CTX, MdFree> c(EVP_MD_CTX_new());
    return c && EVP_DigestVerifyInit_ex(c.get(), nullptr, nullptr, nullptr, nullptr, k, nullptr) == 1 &&
           EVP_DigestVerify(c.get(), sig, sn, m.data(), m.size()) == 1;
}
Bytes x25519(EVP_PKEY* mine, EVP_PKEY* theirs) {
    Ctx c(EVP_PKEY_CTX_new_from_pkey(nullptr, mine, nullptr));
    size_t n = 0;
    if (!c || EVP_PKEY_derive_init(c.get()) != 1 || EVP_PKEY_derive_set_peer(c.get(), theirs) != 1 ||
        EVP_PKEY_derive(c.get(), nullptr, &n) != 1) return {};
    Bytes s(n);
    if (EVP_PKEY_derive(c.get(), s.data(), &n) != 1) return {};
    s.resize(n);
    return s;
}
bool kem_encapsulate(EVP_PKEY* pub, Bytes& ct, Bytes& shared) {
    Ctx c(EVP_PKEY_CTX_new_from_pkey(nullptr, pub, nullptr));
    size_t cn = 0, sn = 0;
    if (!c || EVP_PKEY_encapsulate_init(c.get(), nullptr) != 1 || EVP_PKEY_encapsulate(c.get(), nullptr, &cn, nullptr, &sn) != 1)
        return false;
    ct.resize(cn);
    shared.resize(sn);
    if (EVP_PKEY_encapsulate(c.get(), ct.data(), &cn, shared.data(), &sn) != 1) return false;
    ct.resize(cn);
    shared.resize(sn);
    return ct.size() == kKemCt;
}
Bytes kem_decapsulate(EVP_PKEY* priv, const uint8_t* ct, size_t cn) {
    Ctx c(EVP_PKEY_CTX_new_from_pkey(nullptr, priv, nullptr));
    size_t n = 0;
    if (!c || EVP_PKEY_decapsulate_init(c.get(), nullptr) != 1 || EVP_PKEY_decapsulate(c.get(), nullptr, &n, ct, cn) != 1)
        return {};
    Bytes s(n);
    if (EVP_PKEY_decapsulate(c.get(), s.data(), &n, ct, cn) != 1) return {};
    s.resize(n);
    return s;
}
Bytes hkdf_sha256(const Bytes& key, const Bytes& salt, const Bytes& info) {
    EVP_KDF* kdf = EVP_KDF_fetch(nullptr, "HKDF", nullptr);
    EVP_KDF_CTX* c = kdf ? EVP_KDF_CTX_new(kdf) : nullptr;
    EVP_KDF_free(kdf);
    if (!c) return {};
    char digest[] = "SHA256";
    OSSL_PARAM p[5] = {OSSL_PARAM_construct_utf8_string(OSSL_KDF_PARAM_DIGEST, digest, 0),
                       OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_KEY, const_cast<uint8_t*>(key.data()), key.size()),
                       OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_SALT, const_cast<uint8_t*>(salt.data()), salt.size()),
                       OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_INFO, const_cast<uint8_t*>(info.data()), info.size()),
                       OSSL_PARAM_construct_end()};
    Bytes out(32);
    const bool ok = EVP_KDF_derive(c, out.data(), out.size(), p) == 1;
    EVP_KDF_CTX_free(c);
    return ok ? out : Bytes{};
}
// ChaCha20-Poly1305: ciphertext then its 16-byte tag.
bool aead(bool encrypt, const Bytes& key, const uint8_t* nonce, const Bytes& aad, const uint8_t* in, size_t n, Bytes& out) {
    std::unique_ptr<EVP_CIPHER_CTX, CipherFree> c(EVP_CIPHER_CTX_new());
    if (!c || key.size() != 32) return false;
    size_t body = n;
    if (!encrypt) { if (n < 16) return false; body = n - 16; }
    if ((encrypt ? EVP_EncryptInit_ex2(c.get(), EVP_chacha20_poly1305(), key.data(), nonce, nullptr)
                 : EVP_DecryptInit_ex2(c.get(), EVP_chacha20_poly1305(), key.data(), nonce, nullptr)) != 1) return false;
    int len = 0;
    if (!aad.empty() && (encrypt ? EVP_EncryptUpdate(c.get(), nullptr, &len, aad.data(), static_cast<int>(aad.size()))
                                 : EVP_DecryptUpdate(c.get(), nullptr, &len, aad.data(), static_cast<int>(aad.size()))) != 1)
        return false;
    out.resize(body + (encrypt ? 16 : 0));
    for (size_t at = 0; at < body;) {                       // EVP updates take an int length
        const int step = static_cast<int>(std::min<size_t>(body - at, size_t(1) << 30));
        if ((encrypt ? EVP_EncryptUpdate(c.get(), out.data() + at, &len, in + at, step)
                     : EVP_DecryptUpdate(c.get(), out.data() + at, &len, in + at, step)) != 1) return false;
        at += static_cast<size_t>(step);
    }
    uint8_t tail[16] = {};
    if (encrypt) {
        if (EVP_EncryptFinal_ex(c.get(), tail, &len) != 1 ||
            EVP_CIPHER_CTX_ctrl(c.get(), EVP_CTRL_AEAD_GET_TAG, 16, out.data() + body) != 1) return false;
        return true;
    }
    std::memcpy(tail, in + body, 16);
    if (EVP_CIPHER_CTX_ctrl(c.get(), EVP_CTRL_AEAD_SET_TAG, 16, tail) != 1) return false;
    uint8_t fin[16] = {};
    return EVP_DecryptFinal_ex(c.get(), fin, &len) == 1;
}

// ---------------------------------------------------------------------------------------------------- key blocks
struct Public {
    Bytes block, kid, lineage;
    uint16_t suite = 0;
    uint32_t epoch = 0;
    Pkey ed, ml, x, kem;
    bool parse(const Bytes& b) {
        if (b.size() != kPubBytes || std::memcmp(b.data(), "NMP2", 4) != 0) return false;
        block = b;
        std::memcpy(&suite, b.data() + 4, 2);
        std::memcpy(&epoch, b.data() + 6, 4);
        kid = sha256(b);
        lineage = epoch == 0 ? kid : slice(b, 10, kKid);
        size_t at = kPubHead;
        ed = raw_public("ED25519", b.data() + at, kEdPub); at += kEdPub;
        ml = pq_public("ML-DSA-65", b.data() + at, kMlPub); at += kMlPub;
        x = raw_public("X25519", b.data() + at, kXPub); at += kXPub;
        kem = pq_public("ML-KEM-768", b.data() + at, kKemPub);
        return kid.size() == kKid && ed && ml && x && kem;
    }
    bool verify(const uint8_t* sigs, const Bytes& m) const {
        return verify_with(ed.get(), sigs, kEdSig, m) && verify_with(ml.get(), sigs + kEdSig, kMlSig, m);
    }
};
bool load_public(const fs::path& p, Public& out) {
    Bytes b;
    return read_all(p, b) && out.parse(b);
}

struct Private {
    Pkey ed, ml, x, kem;
    Public pub;
    bool load(const fs::path& dir) {
        Bytes e, m, xx, k;
        if (!read_all(dir / "sign.ed25519", e) || !read_all(dir / "sign.mldsa65", m) || !read_all(dir / "kem.x25519", xx) ||
            !read_all(dir / "kem.mlkem768", k) || !load_public(dir / "node.pub", pub)) return false;
        ed = raw_private("ED25519", e);
        ml = from_seed("ML-DSA-65", m);
        x = raw_private("X25519", xx);
        kem = from_params("ML-KEM-768", EVP_PKEY_KEYPAIR, OSSL_PKEY_PARAM_ML_KEM_SEED, k.data(), k.size());
        return ed && ml && x && kem;
    }
    Bytes sign(const Bytes& m) const {
        Bytes a = sign_with(ed.get(), m), b = sign_with(ml.get(), m);
        if (a.size() != kEdSig || b.size() != kMlSig) return {};
        append(a, b);
        return a;
    }
};

Bytes block_of(uint32_t epoch, const Bytes& lineage, EVP_PKEY* ed, EVP_PKEY* ml, EVP_PKEY* x, EVP_PKEY* kem) {
    Bytes b;
    append(b, reinterpret_cast<const uint8_t*>("NMP2"), 4);
    put(b, kSuite);
    put(b, epoch);
    if (epoch) append(b, lineage); else b.resize(b.size() + kKid, 0);
    EVP_PKEY* const keys[4] = {ed, ml, x, kem};
    const bool pq[4] = {false, true, false, true};
    for (int i = 0; i < 4; ++i) append(b, public_of(keys[i], pq[i]));
    return b.size() == kPubBytes ? b : Bytes{};
}

// A complete key set, written whole under a hidden staging name and renamed to `dir` (which must not exist).
bool make_key_set(const fs::path& dir, uint32_t epoch, const Bytes& lineage, Bytes& block) {
    Pkey ed(EVP_PKEY_Q_keygen(nullptr, nullptr, "ED25519")), x(EVP_PKEY_Q_keygen(nullptr, nullptr, "X25519"));
    const Bytes ml_seed = random_bytes(32), kem_seed = random_bytes(64);
    Pkey ml = from_seed("ML-DSA-65", ml_seed);
    Pkey kem = from_params("ML-KEM-768", EVP_PKEY_KEYPAIR, OSSL_PKEY_PARAM_ML_KEM_SEED, kem_seed.data(), kem_seed.size());
    if (!ed || !x || !ml || !kem || ml_seed.empty() || kem_seed.empty()) return false;
    block = block_of(epoch, lineage, ed.get(), ml.get(), x.get(), kem.get());
    const Bytes ed_raw = raw_private_of(ed.get()), x_raw = raw_private_of(x.get());
    if (block.empty() || ed_raw.size() != 32 || x_raw.size() != 32) return false;
    std::error_code ec;
    fs::create_directories(dir.parent_path(), ec);
    const fs::path staging = dir.parent_path() / (".staging-" + random_tag());
    fs::create_directory(staging, ec);
    const Bytes* data[5] = {&ed_raw, &ml_seed, &x_raw, &kem_seed, &block};
    bool ok = !ec;
    for (int i = 0; ok && i < 5; ++i) ok = platform::write_new_durable(staging / kKeyFiles[i], data[i]->data(), data[i]->size());
    if (ok) { platform::sync_dir(staging); fs::rename(staging, dir, ec); ok = !ec; }
    if (!ok) { fs::remove_all(staging, ec); return false; }
    platform::sync_dir(dir.parent_path());
    return true;
}

// ---------------------------------------------------------------------------------------------------- the folder
struct Folder {
    fs::path root;
    explicit Folder(fs::path r) : root(std::move(r)) {}
    fs::path epochs() const { return root / "epochs"; }
    bool current(uint32_t& n) const {
        std::string t;
        if (!read_text(root / "current", t) || t.empty() || t.size() > 10) return false;
        for (char c : t) if (c < '0' || c > '9') return false;
        n = static_cast<uint32_t>(std::stoul(t));
        return true;
    }
    bool own(Private& out) const {
        uint32_t n = 0;
        return current(n) && out.load(epochs() / std::to_string(n));
    }
    // Every epoch this node holds, newest first: the current one and the earlier ones kept to open old data.
    std::vector<uint32_t> held() const {
        std::vector<uint32_t> out;
        std::error_code ec;
        for (fs::directory_iterator it(epochs(), ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = it->path().filename().string();
            if (name.empty() || name.find_first_not_of("0123456789") != std::string::npos || name.size() > 10) continue;
            out.push_back(static_cast<uint32_t>(std::stoul(name)));
        }
        std::sort(out.rbegin(), out.rend());
        return out;
    }
    std::set<uint16_t> suites() const {
        std::string t;
        if (!read_text(root / "suites", t)) return {kSuite};
        std::set<uint16_t> out;
        size_t at = 0;
        while (at < t.size()) {
            const size_t end = t.find_first_of(" \r\n\t", at);
            const std::string w = t.substr(at, end == std::string::npos ? std::string::npos : end - at);
            if (!w.empty() && w.find_first_not_of("0123456789") == std::string::npos && w.size() <= 5) {
                const unsigned long v = std::stoul(w);
                if (v <= 0xFFFF) out.insert(static_cast<uint16_t>(v));
            }
            if (end == std::string::npos) break;
            at = end + 1;
        }
        return out;
    }
    // A key this node trusts now: one of its own epochs or a peer's, of an allowed suite, not revoked, not past cutover.
    bool known(const Bytes& kid, Public& out) const {
        std::error_code ec;
        const std::string k = hex(kid);
        if (fs::exists(root / "revoked" / k, ec)) return false;
        std::string cut;
        if (read_text(root / "cutover" / k, cut) && !cut.empty() && cut.find_first_not_of("0123456789") == std::string::npos &&
            now_ns() > std::stoull(cut)) return false;
        const std::set<uint16_t> allowed = suites();
        std::vector<fs::path> candidates;
        for (uint32_t e : held()) candidates.push_back(epochs() / std::to_string(e) / "node.pub");
        candidates.push_back(root / "peers" / (k + ".pub"));
        for (const auto& p : candidates) {
            Public pub;
            if (load_public(p, pub) && pub.kid == kid) {
                if (!allowed.count(pub.suite)) return false;
                out = std::move(pub);
                return true;
            }
        }
        return false;
    }
    // A lineage's newest trusted key: the one a writer seals to.
    bool current_of(const Bytes& lineage, Public& best) const {
        bool found = false;
        std::vector<fs::path> candidates;
        for (uint32_t e : held()) candidates.push_back(epochs() / std::to_string(e) / "node.pub");
        std::error_code ec;
        for (fs::directory_iterator it(root / "peers", ec), end; !ec && it != end; it.increment(ec))
            if (it->path().extension() == ".pub") candidates.push_back(it->path());
        for (const auto& p : candidates) {
            Public pub, trusted;
            if (!load_public(p, pub) || pub.lineage != lineage || !known(pub.kid, trusted)) continue;
            if (!found || pub.epoch > best.epoch) { best = std::move(pub); found = true; }
        }
        return found;
    }
    // The newest time applied for a record kind. Read here; written only after the record's changes are in place.
    bool newer(const std::string& kind, uint64_t stamp) const {
        std::string t;
        if (!read_text(root / "records" / kind, t) || t.empty() || t.find_first_not_of("0123456789") != std::string::npos) return true;
        return stamp > std::stoull(t);
    }
    bool applied(const std::string& kind, uint64_t stamp) const { return write_replace(root / "records" / kind, std::to_string(stamp)); }
};

bool keys_folder(fs::path& out) {
    const char* name = std::getenv("NODUS_MESH_KEYS");
    if (!name || !*name) return false;
    out = fs::path(name);
    return true;
}

// ------------------------------------------------------------------------------------------------------ records
std::string accept(const Folder& f, const Bytes& r, bool& ok) {
    ok = false;
    if (r.size() >= 4 && std::memcmp(r.data(), "NMK1", 4) == 0) {
        const size_t head = 4 + 2 + kKid + kPubBytes + 16;
        if (r.size() != head + 2 * kSigs) return "malformed rotation record";
        uint16_t suite = 0;
        get(r, 4, suite);
        const Bytes old_kid = slice(r, 6, kKid), message = slice(r, 0, head);
        Public next, old;
        if (!next.parse(slice(r, 6 + kKid, kPubBytes))) return "malformed rotation record";
        uint64_t effective = 0, grace = 0;
        get(r, 6 + kKid + kPubBytes, effective);
        get(r, 6 + kKid + kPubBytes + 8, grace);
        if (!f.suites().count(suite) || !f.known(old_kid, old)) return "rotation from a key this node does not trust now";
        if (next.lineage != old.lineage || next.epoch != old.epoch + 1 || next.suite != suite)
            return "the new key is not that lineage's next epoch";
        if (!old.verify(r.data() + head, message) || !next.verify(r.data() + head + kSigs, message))
            return "rotation signatures do not hold";
        const std::string kind = "rotation-" + hex(old.lineage);
        if (!f.newer(kind, effective)) return "an old or replayed rotation record";
        // Each change in one rename, the marker last: an interrupted accept is applied again whole.
        Public mine;
        bool own = false;
        for (uint32_t e : f.held()) own = own || (load_public(f.epochs() / std::to_string(e) / "node.pub", mine) && mine.kid == next.kid);
        if (!own && !write_replace(f.root / "peers" / (hex(next.kid) + ".pub"), next.block)) return "could not write the new key";
        if (!write_replace(f.root / "cutover" / hex(old_kid), std::to_string(effective + grace))) return "could not write the cutover";
        if (!f.applied(kind, effective)) return "could not record the rotation";
        ok = true;
        return "rotation: lineage " + hex(old.lineage).substr(0, 8) + " epoch " + std::to_string(old.epoch) + " -> " +
               std::to_string(next.epoch);
    }
    if (r.size() >= 4 && std::memcmp(r.data(), "NMF2", 4) == 0) {
        Public root;
        if (!load_public(f.root / "root.pub", root)) return "no site root (root.pub) here";
        if (r.size() < 4 + 14 + 2 + kSigs) return "malformed force record";
        const Bytes message = slice(r, 0, r.size() - kSigs);
        if (!root.verify(r.data() + r.size() - kSigs, message)) return "force record not signed by the site root";
        uint16_t suite = 0;
        uint64_t stamp = 0;
        uint32_t n = 0;
        get(r, 4, suite);
        get(r, 6, stamp);
        get(r, 14, n);
        size_t at = 18;
        if (n > kMaxReaders || message.size() < at + static_cast<size_t>(n) * kKid + 2) return "malformed force record";
        std::vector<Bytes> revoked;
        for (uint32_t i = 0; i < n; ++i, at += kKid) revoked.push_back(slice(message, at, kKid));
        std::vector<uint16_t> allow;
        const bool has_allow = message[at++] == 1;
        if (has_allow) {
            uint16_t count = 0;
            if (!get(message, at, count) || message.size() < at + 2 + count * 2u) return "malformed force record";
            at += 2;
            for (uint16_t i = 0; i < count; ++i, at += 2) { uint16_t s = 0; get(message, at, s); allow.push_back(s); }
        }
        if (at >= message.size()) return "malformed force record";
        const bool has_replacement = message[at++] == 1;
        Public replacement;
        if (has_replacement && (message.size() != at + kPubBytes || !replacement.parse(slice(message, at, kPubBytes))))
            return "malformed force record";
        if (!has_replacement && message.size() != at) return "malformed force record";
        if (!f.newer("force", stamp)) return "an old or replayed force record";
        for (const Bytes& kid : revoked)
            if (!write_replace(f.root / "revoked" / hex(kid), std::to_string(stamp))) return "could not write a revocation";
        if (has_allow) {
            std::string text;
            for (uint16_t s : allow) text += std::to_string(s) + "\n";
            if (!write_replace(f.root / "suites", text)) return "could not write the suites";
        }
        if (has_replacement && !write_replace(f.root / "peers" / (hex(replacement.kid) + ".pub"), replacement.block))
            return "could not write the replacement";
        if (!f.applied("force", stamp)) return "could not record the force";
        ok = true;
        std::string what = "force: " + std::to_string(n) + " revoked";
        if (has_allow) { what += ", suites now"; for (uint16_t s : allow) what += " " + std::to_string(s); }
        if (has_replacement) what += ", replacement epoch " + std::to_string(replacement.epoch);
        return what;
    }
    return "not a rotation or force record";
}

// --------------------------------------------------------------------------------------------------------- rows
Bytes wrap_key(const Bytes& x_shared, const Bytes& kem_shared, const Bytes& kid, const Bytes& eph, const Bytes& ct) {
    Bytes ikm = x_shared, info;
    append(ikm, kem_shared);
    const char label[] = "nodus mesh seal v1";
    append(info, reinterpret_cast<const uint8_t*>(label), sizeof(label) - 1);
    append(info, eph);
    append(info, ct);
    return hkdf_sha256(ikm, kid, info);
}

Bytes seal(const Bytes& in) {
    if (in.size() < 8 || std::memcmp(in.data(), "NRS1", 4) != 0) return {};
    uint32_t n = 0;
    get(in, 4, n);
    const size_t at = 8 + static_cast<size_t>(n) * kKid;
    if (n > kMaxReaders || in.size() < at) return {};
    Bytes out;
    if (n == 0) {
        append(out, reinterpret_cast<const uint8_t*>("NMC0"), 4);
        out.insert(out.end(), in.begin() + static_cast<std::ptrdiff_t>(at), in.end());
        return out;
    }
    fs::path keys;
    if (!keys_folder(keys)) return {};
    const Folder f(keys);
    const Bytes content_key = random_bytes(32), nonce = random_bytes(12), zero(12, 0);
    if (content_key.empty() || nonce.empty()) return {};
    Bytes header;
    append(header, reinterpret_cast<const uint8_t*>("NMC1"), 4);
    put(header, kSuite);
    put(header, n);
    for (uint32_t i = 0; i < n; ++i) {
        Public reader;
        if (!f.current_of(slice(in, 8 + i * kKid, kKid), reader)) return {};   // a named reader with no trusted key
        Pkey eph(EVP_PKEY_Q_keygen(nullptr, nullptr, "X25519"));
        Bytes ct, kem_shared;
        if (!eph || !kem_encapsulate(reader.kem.get(), ct, kem_shared)) return {};
        const Bytes eph_pub = public_of(eph.get(), false), shared = x25519(eph.get(), reader.x.get());
        const Bytes wrap = wrap_key(shared, kem_shared, reader.kid, eph_pub, ct);
        Bytes wrapped;
        if (wrap.empty() || shared.empty() || !aead(true, wrap, zero.data(), reader.kid, content_key.data(), 32, wrapped)) return {};
        append(header, reader.kid);
        append(header, eph_pub);
        append(header, ct);
        append(header, wrapped);
    }
    append(header, nonce);
    Bytes body;
    if (!aead(true, content_key, nonce.data(), header, in.data() + at, in.size() - at, body)) return {};
    out = std::move(header);
    put(out, static_cast<uint64_t>(body.size()));
    append(out, body);
    return out;
}

std::atomic<uint64_t> g_last_generation{0};

Bytes sign(const Bytes& body) {
    fs::path keys;
    Private own;
    if (!keys_folder(keys) || !Folder(keys).own(own)) return {};
    uint64_t generation = now_ns(), last = g_last_generation.load();
    do { generation = std::max(generation, last + 1); } while (!g_last_generation.compare_exchange_weak(last, generation));
    Bytes m;
    append(m, reinterpret_cast<const uint8_t*>("NMS2"), 4);
    put(m, kSuite);
    append(m, own.pub.kid);
    put(m, generation);
    put(m, static_cast<uint64_t>(body.size()));
    append(m, body);
    const Bytes s = own.sign(m);
    if (s.empty()) return {};
    append(m, s);
    return m;
}

Bytes verify(const Bytes& env) {
    fs::path keys;
    if (!keys_folder(keys)) return {};
    const Folder f(keys);
    if (env.size() >= 4 && (std::memcmp(env.data(), "NMK1", 4) == 0 || std::memcmp(env.data(), "NMF2", 4) == 0)) {
        bool ok = false;
        const std::string what = accept(f, env, ok);
        std::fprintf(stderr, "mesh: %s%s\n", ok ? "" : "record refused: ", what.c_str());
        return {};
    }
    const size_t head = 4 + 2 + kKid + 16;
    if (env.size() < head + kSigs || std::memcmp(env.data(), "NMS2", 4) != 0) return {};
    uint16_t suite = 0;
    uint64_t generation = 0, length = 0;
    get(env, 4, suite);
    get(env, 6 + kKid, generation);
    get(env, 6 + kKid + 8, length);
    if (!f.suites().count(suite) || length > env.size() || env.size() != head + length + kSigs) return {};
    Public writer;
    if (!f.known(slice(env, 6, kKid), writer) || writer.suite != suite) return {};
    if (!writer.verify(env.data() + head + length, slice(env, 0, head + static_cast<size_t>(length)))) return {};
    Bytes out;
    append(out, reinterpret_cast<const uint8_t*>("NMV1"), 4);
    append(out, writer.lineage);
    put(out, generation);
    out.insert(out.end(), env.begin() + static_cast<std::ptrdiff_t>(head), env.begin() + static_cast<std::ptrdiff_t>(head + length));
    return out;
}

Bytes open(const Bytes& v) {
    if (v.size() < 4 + kKid + 8 + 4 || std::memcmp(v.data(), "NMV1", 4) != 0) return {};
    Bytes out = slice(v, 0, 4 + kKid + 8);
    std::memcpy(out.data(), "NMO1", 4);
    const size_t s = 4 + kKid + 8;
    if (std::memcmp(v.data() + s, "NMC0", 4) == 0) {
        out.push_back(1);
        out.insert(out.end(), v.begin() + static_cast<std::ptrdiff_t>(s + 4), v.end());
        return out;
    }
    if (std::memcmp(v.data() + s, "NMC1", 4) != 0 || v.size() < s + 10) return {};
    uint32_t n = 0;
    get(v, s + 6, n);
    const size_t nonce_at = s + 10 + static_cast<size_t>(n) * kEntry;
    if (n > kMaxReaders || v.size() < nonce_at + 20) return {};
    uint64_t length = 0;
    get(v, nonce_at + 12, length);
    const size_t body_at = nonce_at + 20;
    if (v.size() - body_at != length) return {};
    const Bytes header = slice(v, s, nonce_at + 12 - s);
    fs::path keys;
    if (!keys_folder(keys)) return {};
    const Folder f(keys);
    std::map<Bytes, std::unique_ptr<Private>> mine;            // the current epoch and every earlier one still held
    for (uint32_t e : f.held()) {
        auto p = std::make_unique<Private>();
        if (p->load(f.epochs() / std::to_string(e))) { const Bytes kid = p->pub.kid; mine.emplace(kid, std::move(p)); }
    }
    const Bytes zero(12, 0);
    for (uint32_t i = 0; i < n; ++i) {
        const size_t e = s + 10 + static_cast<size_t>(i) * kEntry;
        const Bytes kid = slice(v, e, kKid), eph_pub = slice(v, e + kKid, kXPub), ct = slice(v, e + kKid + kXPub, kKemCt);
        const auto it = mine.find(kid);
        if (it == mine.end()) continue;
        Pkey eph = raw_public("X25519", eph_pub.data(), eph_pub.size());
        if (!eph) return {};
        const Bytes shared = x25519(it->second->x.get(), eph.get());
        const Bytes kem_shared = kem_decapsulate(it->second->kem.get(), ct.data(), ct.size());
        const Bytes wrap = wrap_key(shared, kem_shared, kid, eph_pub, ct);
        Bytes content_key, snapshot;
        if (wrap.empty() || !aead(false, wrap, zero.data(), kid, v.data() + e + kKid + kXPub + kKemCt, 48, content_key) ||
            !aead(false, content_key, v.data() + nonce_at, header, v.data() + body_at, static_cast<size_t>(length), snapshot))
            return {};
        out.push_back(1);
        append(out, snapshot);
        return out;
    }
    out.push_back(0);                                          // not a reader: only that the writer exists, and its generation
    return out;
}

// ---------------------------------------------------------------------------------------------------- operators
// Rotation, resumable at every stage (review eba4505a): the next key set is made whole and renamed into epochs/<n+1>,
// its record written, and only then does `current` move to it, in one rename. Run again after an interruption, it
// finishes what is there. `stop_after` (drills): 1 after the key set, 2 after the record; 0 runs to the end.
std::string rotate(const Folder& f, double grace_s, int stop_after, bool& ok) {
    ok = false;
    Private old;
    if (!f.own(old)) return f.root.string() + " holds no current key set";
    const uint32_t next_epoch = old.pub.epoch + 1;
    const fs::path next_dir = f.epochs() / std::to_string(next_epoch);
    std::error_code ec;
    for (fs::directory_iterator it(f.epochs(), ec), end; !ec && it != end; it.increment(ec))      // an earlier crash's
        if (it->path().filename().string().rfind(".staging-", 0) == 0) fs::remove_all(it->path(), ec);   // half-made set
    Private next;
    if (!fs::exists(next_dir / "node.pub", ec)) {
        Bytes block;
        if (!make_key_set(next_dir, next_epoch, old.pub.lineage, block)) return "could not make the next key set";
    }
    if (!next.load(next_dir) || next.pub.lineage != old.pub.lineage || next.pub.epoch != next_epoch)
        return "epochs/" + std::to_string(next_epoch) + " is not this lineage's next key set";
    if (stop_after == 1) { ok = true; return "stopped after the key set (drill)"; }
    const fs::path record_path = f.root / ("rotation-" + std::to_string(next_epoch) + ".rec");
    Bytes record;
    if (!read_all(record_path, record) || record.size() != 4 + 2 + kKid + kPubBytes + 16 + 2 * kSigs) {
        Bytes m;
        append(m, reinterpret_cast<const uint8_t*>("NMK1"), 4);
        put(m, kSuite);
        append(m, old.pub.kid);
        append(m, next.pub.block);
        put(m, now_ns());
        put(m, static_cast<uint64_t>(grace_s * 1e9));
        const Bytes a = old.sign(m), b = next.sign(m);
        if (a.empty() || b.empty()) return "could not sign the rotation record";
        record = m;
        append(record, a);
        append(record, b);
        if (!write_replace(record_path, record)) return "could not write the rotation record";
    }
    if (stop_after == 2) { ok = true; return "stopped after the record (drill)"; }
    if (!write_replace(f.root / "current", std::to_string(next_epoch))) return "could not switch to the next key set";
    ok = true;
    return "rotated: lineage " + hex(next.pub.lineage).substr(0, 16) + " epoch " + std::to_string(next_epoch) + " kid " +
           hex(next.pub.kid) + "; record " + record_path.string();
}

std::string keygen(const Folder& f, bool& ok) {
    ok = false;
    std::error_code ec;
    if (fs::exists(f.root / "current", ec) || fs::exists(f.epochs(), ec))
        return f.root.string() + " already holds mesh keys; keys are never replaced in place";
    for (const char* sub : {"peers", "revoked", "cutover", "records", "epochs"}) fs::create_directories(f.root / sub, ec);
    Bytes block;
    if (!make_key_set(f.epochs() / "0", 0, Bytes(kKid, 0), block) || !write_replace(f.root / "current", std::string("0")))
        return "could not make the key set";
    Public p;
    p.parse(block);
    ok = true;
    return "node lineage " + hex(p.lineage) + " kid " + hex(p.kid) + " epoch 0";
}

std::string force(const Folder& rootf, const fs::path& out, const std::vector<Bytes>& revoked, const std::vector<uint16_t>* allow,
                  const Bytes* replacement, bool& ok) {
    ok = false;
    Private root;
    if (!rootf.own(root)) return rootf.root.string() + " holds no site root";
    Bytes m;
    append(m, reinterpret_cast<const uint8_t*>("NMF2"), 4);
    put(m, kSuite);
    put(m, now_ns());
    put(m, static_cast<uint32_t>(revoked.size()));
    for (const Bytes& k : revoked) append(m, k);
    m.push_back(allow ? 1 : 0);
    if (allow) { put(m, static_cast<uint16_t>(allow->size())); for (uint16_t s : *allow) put(m, s); }
    m.push_back(replacement ? 1 : 0);
    if (replacement) append(m, *replacement);
    const Bytes s = root.sign(m);
    if (s.empty()) return "could not sign";
    append(m, s);
    if (!write_replace(out, m)) return "could not write " + out.string();
    ok = true;
    return "force record written to " + out.string();
}
#endif  // NODUS_MESH_CRYPTO

void report(char* out, size_t cap, const std::string& text) {
    if (!out || !cap) return;
    const size_t n = std::min(cap - 1, text.size());
    std::memcpy(out, text.data(), n);
    out[n] = '\0';
}

}  // namespace
}  // namespace nodus::mesh

extern "C" NODUS_HEADLESS_API int32_t nodus_mesh_available(void) { return NODUS_MESH_CRYPTO ? 1 : 0; }

extern "C" NODUS_HEADLESS_API uint64_t nodus_mesh_bound(int32_t op, const uint8_t* in, size_t n) {
    using namespace nodus::mesh;
    if (!in && n) return 0;
    switch (op) {
    case NODUS_MESH_SEAL: {
        uint32_t readers = 0;
        if (n >= 8) std::memcpy(&readers, in + 4, 4);
        if (readers > kMaxReaders) return 0;
        return n + 64 + static_cast<uint64_t>(readers) * kEntry;
    }
    case NODUS_MESH_SIGN: return n + 4 + 2 + kKid + 16 + kSigs;
    case NODUS_MESH_VERIFY: case NODUS_MESH_OPEN: return n + 1;
    default: return 0;
    }
}

extern "C" NODUS_HEADLESS_API int32_t nodus_mesh_row(int32_t op, const uint8_t* in, size_t n, uint8_t* out, size_t cap,
                                                     size_t* written) {
    if (written) *written = 0;
#if NODUS_MESH_CRYPTO
    using namespace nodus::mesh;
    if (!in && n) return 0;
    const Bytes input(in, in + n);
    Bytes result;
    try {
        switch (op) {
        case NODUS_MESH_SEAL: result = seal(input); break;
        case NODUS_MESH_SIGN: result = sign(input); break;
        case NODUS_MESH_VERIFY: result = verify(input); break;
        case NODUS_MESH_OPEN: result = open(input); break;
        default: return 0;
        }
    } catch (...) {
        return 0;
    }
    if (result.size() > cap || (!out && !result.empty())) return 0;
    if (!result.empty()) std::memcpy(out, result.data(), result.size());
    if (written) *written = result.size();
    return 1;
#else
    (void)op; (void)in; (void)n; (void)out; (void)cap;
    return 0;
#endif
}

extern "C" NODUS_HEADLESS_API int32_t nodus_mesh_operator(int32_t argc, const char* const* argv, char* out, size_t cap) {
    using namespace nodus::mesh;
#if NODUS_MESH_CRYPTO
    std::vector<std::string> a;
    for (int32_t i = 0; i < argc; ++i) a.emplace_back(argv[i] ? argv[i] : "");
    bool ok = false;
    std::string text;
    try {
        if (a.size() == 2 && (a[0] == "keygen" || a[0] == "root-keygen")) {
            text = keygen(Folder(a[1]), ok);
            if (ok && a[0] == "root-keygen")
                text = "site root made in " + a[1] + ": copy epochs/0/node.pub to every node folder as root.pub; keep the folder offline";
        } else if ((a.size() == 2 || a.size() == 3 || a.size() == 4) && a[0] == "rotate") {
            const double grace = a.size() >= 3 ? std::stod(a[2]) : 3600.0;
            const int stop = a.size() == 4 && a[3].rfind("--drill-stop=", 0) == 0 ? std::stoi(a[3].substr(13)) : 0;
            text = rotate(Folder(a[1]), grace, stop, ok);
        } else if (a.size() >= 3 && a[0] == "force") {
            std::vector<Bytes> revoked;
            std::vector<uint16_t> allow;
            bool has_allow = false;
            Bytes replacement;
            bool has_replacement = false, bad = false;
            for (size_t i = 3; i < a.size() && !bad; ++i) {
                if (a[i] == "--replace" && i + 1 < a.size()) { has_replacement = read_all(a[++i], replacement); bad = !has_replacement; }
                else if (a[i] == "--suites" && i + 1 < a.size()) {
                    has_allow = true;
                    const std::string list = a[++i];
                    size_t at = 0;
                    while (at <= list.size()) {
                        const size_t end = list.find(',', at);
                        const std::string w = list.substr(at, end == std::string::npos ? std::string::npos : end - at);
                        if (!w.empty()) allow.push_back(static_cast<uint16_t>(std::stoul(w)));
                        if (end == std::string::npos) break;
                        at = end + 1;
                    }
                } else {
                    Bytes kid;
                    bad = !unhex(a[i], kid) || kid.size() != kKid;
                    if (!bad) revoked.push_back(kid);
                }
            }
            text = bad ? std::string("force: a revoked kid is 64 hex digits; --replace names a public block file")
                       : force(Folder(a[1]), a[2], revoked, has_allow ? &allow : nullptr, has_replacement ? &replacement : nullptr, ok);
        } else if (a.size() == 3 && a[0] == "accept") {
            Bytes record;
            text = read_all(a[2], record) ? accept(Folder(a[1]), record, ok) : "could not read " + a[2];
        } else {
            text = "usage: keygen <folder> | root-keygen <folder> | rotate <folder> [grace s] [--drill-stop=1|2] | "
                   "force <root folder> <out file> [<revoked kid hex>...] [--suites 1,2] [--replace <node.pub>] | "
                   "accept <folder> <record file>";
        }
    } catch (const std::exception& e) {
        ok = false;
        text = e.what();
    }
    report(out, cap, text);
    return ok ? 1 : 0;
#else
    (void)argc; (void)argv;
    report(out, cap, "this Nodus was built without the pinned OpenSSL (cmake -P scripts/build_openssl.cmake)");
    return 0;
#endif
}
