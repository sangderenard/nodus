// native_console.h: the canvas console, native (stage B of docs/NODUS_NATIVE_HOST_PLAN_2026-11-23.md).
//
// The protocol of docs/NODUS_CONSOLE_READ_PROTOCOL_2026-11-10.md, served byte for byte as tools/run_canvas.py's
// ConsoleServer serves it and with the documents nodus_graph/console_views.py builds, so its clients
// (nodus_service.py, demo_canvas_tables.py --attach-console, frontend_shell --attach, console_client.py) are unchanged:
//   HELLO nodus-console/1|2 <token>; SNAPSHOT; DIGEST GRAPH TOOLS ACTIVITY META; v2 ACTION <n> + JSON.
// One verb run_canvas.py had not (it never rasterized): FRAME, the canvas as the rasterizer draws it,
//   "FRAME <revision> <width> <height> <n>\n" + n bytes of RGBA (n = width * height * 4), built on the stepping thread
// like every view, so a client without the engine sees what the window shows. The size is the viewport's (ACTION
// viewport sets it).
// Client threads never touch the graph: they ask, and the stepping thread builds what was asked between steps
// (service()), coalesced for 100 ms. An ACTION is applied on the stepping thread too, and a change of revision saves the
// canvas through the headless file path.
//
// Links (stage C): link_host opens a link declared in the canvas (nodus_headless_open_link_host: a LINK_HOST record the
// next save writes; port 0 lets the system pick; "route": true adds the front route row); link_dial and link_udp open a
// dialed link and compose its tables (nodus_headless_compose_link_tables: the four rows in their own tables); link_send
// pushes a tensor into the declared port <id>.send. run_canvas.py placed <id>.bulk and <id>.control, rows that no longer
// exist since the four-row split.
#pragma once

#include "contract_listener.h"   // ensure_sockets, SOCKET and the socket includes (contract_socket.h)
#include "nodus_link.h"          // nodus::link::sha256
#include "nodus_headless_abi.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace nodus::console {

// ======================================================================================================== JSON
// Python's json module, as the console uses it: dumps with ensure_ascii, sort_keys or insertion order, compact
// (",", ":") or default (", ", ": ") separators, floats as Python's repr; loads with ints, floats, NaN and Infinity.
struct Json {
    enum class T { Null, Bool, Int, UInt, Float, Str, Arr, Obj } t = T::Null;
    bool b = false;
    long long i = 0;
    unsigned long long u = 0;
    double f = 0.0;
    bool big = false;   // an integer literal too large for 64 bits (Python keeps it; nothing here can use it)
    std::string s;
    std::vector<Json> a;
    std::vector<std::pair<std::string, Json>> o;

    static Json null() { return Json{}; }
    static Json boolean(bool v) { Json j; j.t = T::Bool; j.b = v; return j; }
    static Json integer(long long v) { Json j; j.t = T::Int; j.i = v; return j; }
    static Json uinteger(unsigned long long v) { Json j; j.t = T::UInt; j.u = v; return j; }
    static Json number(double v) { Json j; j.t = T::Float; j.f = v; return j; }
    static Json string(std::string v) { Json j; j.t = T::Str; j.s = std::move(v); return j; }
    static Json array() { Json j; j.t = T::Arr; return j; }
    static Json object() { Json j; j.t = T::Obj; return j; }

    Json& set(const std::string& key, Json v) {   // insertion order kept; a key set again keeps its place (as a dict)
        for (auto& p : o) if (p.first == key) { p.second = std::move(v); return *this; }
        o.emplace_back(key, std::move(v));
        return *this;
    }
    Json& push(Json v) { a.push_back(std::move(v)); return *this; }
    const Json* get(const std::string& key) const {
        for (const auto& p : o) if (p.first == key) return &p.second;
        return nullptr;
    }
    bool is_int() const { return t == T::Int || t == T::UInt; }   // Python `type(v) is int` (bool is not)
    bool is_number() const { return t == T::Int || t == T::UInt || t == T::Float || t == T::Bool; }   // isinstance(v, (int, float))
    double as_double() const { return t == T::Int ? double(i) : t == T::UInt ? double(u) : t == T::Bool ? (b ? 1.0 : 0.0) : f; }
    std::set<std::string> keys() const { std::set<std::string> k; for (const auto& p : o) k.insert(p.first); return k; }
};

// Python's float repr: the shortest digits that round-trip, fixed for decimal exponents -4..15, else d.ddde+XX.
inline std::string float_repr(double x) {
    if (std::isnan(x)) return "NaN";
    if (std::isinf(x)) return x > 0 ? "Infinity" : "-Infinity";
    if (x == 0.0) return std::signbit(x) ? "-0.0" : "0.0";
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof(buf), x, std::chars_format::scientific);
    std::string sci(buf, r.ptr);
    std::string sign;
    if (sci[0] == '-') { sign = "-"; sci.erase(0, 1); }
    const size_t e = sci.find('e');
    std::string digits;
    for (size_t k = 0; k < e; ++k) if (sci[k] != '.') digits += sci[k];
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int exp = std::stoi(sci.substr(e + 1));
    const int n = static_cast<int>(digits.size());
    if (exp >= -4 && exp < 16) {
        const int point = exp + 1;
        if (point <= 0) return sign + "0." + std::string(size_t(-point), '0') + digits;
        if (point >= n) return sign + digits + std::string(size_t(point - n), '0') + ".0";
        return sign + digits.substr(0, size_t(point)) + "." + digits.substr(size_t(point));
    }
    std::string out = sign + digits.substr(0, 1);
    if (n > 1) out += "." + digits.substr(1);
    char ex[16];
    std::snprintf(ex, sizeof(ex), "e%c%02d", exp < 0 ? '-' : '+', exp < 0 ? -exp : exp);
    return out + ex;
}

namespace detail {
inline void append_codepoint(std::string& out, uint32_t cp) {
    if (cp < 0x80) out += char(cp);
    else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
    else { out += char(0xF0 | (cp >> 18)); out += char(0x80 | ((cp >> 12) & 0x3F)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
}
// One UTF-8 code point at s[k] (U+FFFD for a malformed byte, as a lenient decode does); advances k.
inline uint32_t next_codepoint(const std::string& s, size_t& k) {
    const unsigned char c = static_cast<unsigned char>(s[k]);
    int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (!len || k + size_t(len) > s.size()) { ++k; return 0xFFFD; }
    uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
    for (int j = 1; j < len; ++j) {
        const unsigned char d = static_cast<unsigned char>(s[k + size_t(j)]);
        if ((d & 0xC0) != 0x80) { ++k; return 0xFFFD; }
        cp = (cp << 6) | (d & 0x3F);
    }
    k += size_t(len);
    return cp;
}
}  // namespace detail

// json.dumps of a str with ensure_ascii: printable ASCII as is, the rest \u escaped (lower-case hex, surrogate pairs).
inline std::string json_quote(const std::string& s) {
    std::string out = "\"";
    for (size_t k = 0; k < s.size();) {
        const uint32_t cp = detail::next_codepoint(s, k);
        switch (cp) {
            case '"': out += "\\\""; continue;
            case '\\': out += "\\\\"; continue;
            case '\n': out += "\\n"; continue;
            case '\r': out += "\\r"; continue;
            case '\t': out += "\\t"; continue;
            case '\b': out += "\\b"; continue;
            case '\f': out += "\\f"; continue;
            default: break;
        }
        char b[16];
        if (cp >= 0x20 && cp <= 0x7E) out += char(cp);
        else if (cp < 0x10000) { std::snprintf(b, sizeof(b), "\\u%04x", cp); out += b; }
        else {
            const uint32_t v = cp - 0x10000;
            std::snprintf(b, sizeof(b), "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF));
            out += b;
        }
    }
    return out + "\"";
}

inline void dump_into(std::string& out, const Json& j, bool sorted, bool compact) {
    const char* item = compact ? "," : ", ";
    const char* key = compact ? ":" : ": ";
    switch (j.t) {
        case Json::T::Null: out += "null"; break;
        case Json::T::Bool: out += j.b ? "true" : "false"; break;
        case Json::T::Int: out += std::to_string(j.i); break;
        case Json::T::UInt: out += std::to_string(j.u); break;
        case Json::T::Float: out += float_repr(j.f); break;
        case Json::T::Str: out += json_quote(j.s); break;
        case Json::T::Arr:
            out += "[";
            for (size_t k = 0; k < j.a.size(); ++k) { if (k) out += item; dump_into(out, j.a[k], sorted, compact); }
            out += "]";
            break;
        case Json::T::Obj: {
            std::vector<const std::pair<std::string, Json>*> order;
            for (const auto& p : j.o) order.push_back(&p);
            if (sorted) std::sort(order.begin(), order.end(), [](auto* x, auto* y) { return x->first < y->first; });
            out += "{";
            for (size_t k = 0; k < order.size(); ++k) {
                if (k) out += item;
                out += json_quote(order[k]->first);
                out += key;
                dump_into(out, order[k]->second, sorted, compact);
            }
            out += "}";
            break;
        }
    }
}
// console_views.dumps: sort_keys, compact.
inline std::string dumps_view(const Json& j) { std::string s; dump_into(s, j, true, true); return s; }
// json.dumps(state): insertion order, default separators.
inline std::string dumps_default(const Json& j) { std::string s; dump_into(s, j, false, false); return s; }
// json.dumps(reply, separators=(",", ":")): insertion order, compact.
inline std::string dumps_compact(const Json& j) { std::string s; dump_into(s, j, false, true); return s; }

// json.loads for an ACTION body. False on anything json.loads (or the UTF-8 decode before it) refuses.
class Parser {
public:
    explicit Parser(const std::string& s) : s_(s) {}
    bool parse(Json& out) {
        for (size_t k = 0; k < s_.size();) if (detail::next_codepoint(s_, k) == 0xFFFD) {
            // a malformed byte: refused, as bytes.decode("utf-8") refuses it (a real U+FFFD encodes as EF BF BD)
            if (!(k >= 3 && static_cast<unsigned char>(s_[k - 3]) == 0xEF && static_cast<unsigned char>(s_[k - 2]) == 0xBF &&
                  static_cast<unsigned char>(s_[k - 1]) == 0xBD)) return false;
        }
        ws();
        if (!value(out, 0)) return false;
        ws();
        return p_ == s_.size();
    }

private:
    void ws() { while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\n' || s_[p_] == '\r')) ++p_; }
    bool lit(const char* w) { const size_t n = std::strlen(w); if (s_.compare(p_, n, w) != 0) return false; p_ += n; return true; }
    bool value(Json& out, int depth) {
        if (depth > 200 || p_ >= s_.size()) return false;
        const char c = s_[p_];
        if (c == '{') return object(out, depth);
        if (c == '[') return array(out, depth);
        if (c == '"') { out = Json::string(""); return string(out.s); }
        if (lit("true")) { out = Json::boolean(true); return true; }
        if (lit("false")) { out = Json::boolean(false); return true; }
        if (lit("null")) { out = Json::null(); return true; }
        if (lit("NaN")) { out = Json::number(std::nan("")); return true; }
        if (lit("Infinity")) { out = Json::number(HUGE_VAL); return true; }
        if (lit("-Infinity")) { out = Json::number(-HUGE_VAL); return true; }
        return number(out);
    }
    bool object(Json& out, int depth) {
        out = Json::object();
        ++p_; ws();
        if (p_ < s_.size() && s_[p_] == '}') { ++p_; return true; }
        for (;;) {
            ws();
            if (p_ >= s_.size() || s_[p_] != '"') return false;
            std::string key;
            if (!string(key)) return false;
            ws();
            if (p_ >= s_.size() || s_[p_] != ':') return false;
            ++p_; ws();
            Json v;
            if (!value(v, depth + 1)) return false;
            out.set(key, std::move(v));   // a repeated key: the last value wins, at the first place
            ws();
            if (p_ < s_.size() && s_[p_] == ',') { ++p_; continue; }
            if (p_ < s_.size() && s_[p_] == '}') { ++p_; return true; }
            return false;
        }
    }
    bool array(Json& out, int depth) {
        out = Json::array();
        ++p_; ws();
        if (p_ < s_.size() && s_[p_] == ']') { ++p_; return true; }
        for (;;) {
            ws();
            Json v;
            if (!value(v, depth + 1)) return false;
            out.push(std::move(v));
            ws();
            if (p_ < s_.size() && s_[p_] == ',') { ++p_; continue; }
            if (p_ < s_.size() && s_[p_] == ']') { ++p_; return true; }
            return false;
        }
    }
    bool hex4(uint32_t& v) {
        if (p_ + 4 > s_.size()) return false;
        v = 0;
        for (int k = 0; k < 4; ++k) {
            const char h = s_[p_++];
            v <<= 4;
            if (h >= '0' && h <= '9') v |= uint32_t(h - '0');
            else if (h >= 'a' && h <= 'f') v |= uint32_t(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') v |= uint32_t(h - 'A' + 10);
            else return false;
        }
        return true;
    }
    bool string(std::string& out) {
        ++p_;
        while (p_ < s_.size()) {
            const unsigned char c = static_cast<unsigned char>(s_[p_]);
            if (c == '"') { ++p_; return true; }
            if (c < 0x20) return false;   // json.loads (strict) refuses control characters in a string
            if (c != '\\') { out += char(c); ++p_; continue; }
            if (++p_ >= s_.size()) return false;
            const char e = s_[p_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp < 0xDC00 && p_ + 6 <= s_.size() && s_[p_] == '\\' && s_[p_ + 1] == 'u') {
                        const size_t save = p_;
                        p_ += 2;
                        uint32_t lo = 0;
                        if (hex4(lo) && lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        else p_ = save;
                    }
                    detail::append_codepoint(out, cp);
                    break;
                }
                default: return false;
            }
        }
        return false;
    }
    bool number(Json& out) {
        const size_t start = p_;
        if (p_ < s_.size() && s_[p_] == '-') ++p_;
        if (p_ >= s_.size()) return false;
        if (s_[p_] == '0') ++p_;
        else if (s_[p_] >= '1' && s_[p_] <= '9') { while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_; }
        else return false;
        bool real = false;
        if (p_ < s_.size() && s_[p_] == '.') {
            const size_t d = ++p_;
            while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
            if (p_ == d) { p_ = d - 1; }   // "1." is "1" followed by "." (refused by what follows)
            else real = true;
        }
        if (p_ < s_.size() && (s_[p_] == 'e' || s_[p_] == 'E')) {
            size_t q = p_ + 1;
            if (q < s_.size() && (s_[q] == '+' || s_[q] == '-')) ++q;
            const size_t d = q;
            while (q < s_.size() && std::isdigit(static_cast<unsigned char>(s_[q]))) ++q;
            if (q > d) { p_ = q; real = true; }
        }
        const std::string text = s_.substr(start, p_ - start);
        if (real) { out = Json::number(std::strtod(text.c_str(), nullptr)); return true; }
        long long v = 0;
        auto r = std::from_chars(text.data(), text.data() + text.size(), v);
        if (r.ec == std::errc() && r.ptr == text.data() + text.size()) { out = Json::integer(v); return true; }
        unsigned long long uv = 0;
        auto ru = std::from_chars(text.data(), text.data() + text.size(), uv);
        if (ru.ec == std::errc() && ru.ptr == text.data() + text.size()) { out = Json::uinteger(uv); return true; }
        out = Json::integer(0);
        out.big = true;
        return true;
    }
    const std::string& s_;
    size_t p_ = 0;
};

// ======================================================================================================== documents
// Exceptions as the Python views raise them: what() is str(exc).
struct ValueError : std::runtime_error { using std::runtime_error::runtime_error; };
struct RuntimeError : std::runtime_error { using std::runtime_error::runtime_error; };

// Python's repr of a str, for the few messages that print one.
inline std::string py_repr(const std::string& s) {
    const bool dq = s.find('\'') != std::string::npos && s.find('"') == std::string::npos;
    const char q = dq ? '"' : '\'';
    std::string out(1, q);
    for (char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == q) { out += '\\'; out += c; }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out + q;
}

enum class Builtin : int { NONE = 0, ADD, SUBTRACT, MULTIPLY, DIVIDE, MODULO, KEYBOARD_LISTENER, MOUSE_LISTENER, STACK_DISPLAY,
                           RECT_RGBA, TABLE_NUMBER, CLONE, TENSOR_ALLOCATOR, TENSOR_TOOL };
inline const char* builtin_name(int v) {   // nodus_graph/model.py BuiltinTool
    static const char* const names[] = {"NONE", "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "MODULO", "KEYBOARD_LISTENER",
        "MOUSE_LISTENER", "STACK_DISPLAY", "RECT_RGBA", "TABLE_NUMBER", "CLONE", "TENSOR_ALLOCATOR", "TENSOR_TOOL"};
    return v >= 0 && v <= 13 ? names[v] : nullptr;
}
inline bool builtin_stack_effect(int v, int& consumes, int& produces) {   // model.py _SCALAR_STACK_EFFECT
    switch (v) {
        case 1: case 2: case 3: case 4: case 5: consumes = 2; produces = 1; return true;
        case 8: consumes = 0; produces = 0; return true;
        case 10: consumes = 0; produces = 1; return true;
        case 11: consumes = 2; produces = 0; return true;
        default: return false;
    }
}
inline const char* contract_name(int32_t v) {   // native.py _contract_from_native
    switch (v) {
        case 0: return "scalar_f32";
        case 1: return "abstract_tensor_pointer";
        case 2: return "opaque";
        default: throw RuntimeError("unknown native module stack contract " + std::to_string(v));
    }
}

// What the stepping thread knows besides the graph (run_canvas's ConsoleServer fields).
struct Context {
    NodusHeadlessGraph* g = nullptr;
    std::string canvas_name;   // basename
    std::string canvas_path;
    double rate = 0.0;
    double started = 0.0;      // unix seconds, at the console's start
    std::vector<std::pair<std::string, uint64_t>> hosts;   // attached compile hosts
    std::vector<std::string> link_ids;
    std::function<long long()> ticks;
    // The rasterized canvas (RGBA, width * height * 4) for FRAME; empty: no frames.
    std::function<bool(std::string& rgba, int& width, int& height)> frame;
};

inline int32_t count_of(int32_t n) { return std::max(0, n); }

inline std::string module_label(const GP_CanvasModuleDesc& d) {
    return std::string(d.label, strnlen(d.label, sizeof(d.label)));
}

// _fifo_from_native + FifoConfig's checks + _fifo_to_mapping.
inline Json fifo_mapping(const NodusHeadlessFifoConfig& c) {
    const std::string storage = c.backend_type == 0 ? "host" : c.backend_type == 3 ? "vulkan" : "unsupported:" + std::to_string(c.backend_type);
    if (c.token_bytes <= 0) throw ValueError("token_bytes must be positive");
    if (c.slots <= 0) throw ValueError("slots must be positive; use None for no FIFO configuration");
    if (!(0 <= c.top_k && c.top_k <= c.slots)) throw ValueError("top_k must be between zero and slots");
    if (storage != "host" && storage != "vulkan") throw ValueError("FIFO storage must be 'host' or 'vulkan'");
    return Json::object().set("token_bytes", Json::integer(c.token_bytes)).set("slots", Json::integer(c.slots))
        .set("top_k", Json::integer(c.top_k)).set("type_id", Json::integer(c.type_id)).set("layout", Json::integer(c.layout))
        .set("dtype", Json::integer(c.dtype)).set("storage", Json::string(storage));
}

struct ContractPort { std::string prefix; int32_t capacity = 0, edge = 0, declared = 0; };
inline std::vector<ContractPort> contract_ports(NodusHeadlessGraph* g) {   // Graph.contract_ports
    std::vector<ContractPort> out;
    const int32_t n = nodus_headless_get_contract_port_count(g);
    for (int32_t i = 0; i < n; ++i) {
        char name[1024] = {};
        ContractPort p;
        if (!nodus_headless_get_contract_port_at(g, i, name, sizeof(name), &p.capacity, &p.edge, &p.declared))
            throw RuntimeError("native contract port " + std::to_string(i) + " query failed");
        p.prefix = name;
        out.push_back(p);
    }
    return out;
}
struct BoundPort { bool ok = false; int32_t port = 0, hosts = 0, engines = 0; };
inline BoundPort contract_port(NodusHeadlessGraph* g, const std::string& prefix) {   // Graph.contract_port
    BoundPort b;
    b.ok = nodus_headless_get_contract_port(g, prefix.c_str(), &b.port, &b.hosts, &b.engines) != 0;
    return b;
}

inline Json contract_port_documents(NodusHeadlessGraph* g) {
    Json out = Json::array();
    for (const auto& p : contract_ports(g)) {
        const BoundPort b = contract_port(g, p.prefix);
        const uint64_t loan = nodus_headless_contract_pair_loan_ticks(p.prefix.c_str());
        out.push(Json::object().set("prefix", Json::string(p.prefix)).set("capacity", Json::integer(p.capacity))
                     .set("edge", Json::integer(p.edge)).set("listen_port", Json::integer(b.ok ? b.port : p.declared))
                     .set("loan_ticks", loan ? Json::uinteger(loan) : Json::null()));
    }
    return out;
}

inline Json digest_document(const Context& c, long long revision, const std::string& data, long long tick) {
    const auto hash = nodus::link::sha256(data.data(), data.size());
    static const char* hex = "0123456789abcdef";
    std::string sha;
    for (uint8_t b : hash) { sha += hex[b >> 4]; sha += hex[b & 15]; }
    const int32_t modules = count_of(nodus_headless_get_module_count(c.g));
    long long rows = 0;
    for (int32_t m = 0; m < modules; ++m) rows += count_of(nodus_headless_get_module_row_count(c.g, m));
    return Json::object().set("protocol", Json::integer(2)).set("canvas", Json::string(c.canvas_name))
        .set("revision", Json::integer(revision)).set("sha256", Json::string(sha)).set("bytes", Json::integer((long long)data.size()))
        .set("compile_hosts", Json::integer((long long)c.hosts.size())).set("tick", Json::integer(tick))
        .set("rate", Json::number(c.rate)).set("started", Json::number(c.started))
        .set("modules", Json::integer(modules)).set("rows", Json::integer(rows))
        .set("edges", Json::integer(count_of(nodus_headless_get_edge_count(c.g))))
        .set("contract_ports", Json::integer((long long)contract_ports(c.g).size()));
}

// Graph.definition() + GraphDefinition.validate() + to_mapping() + graph_document's additions. Throws ValueError /
// RuntimeError with the messages the Python raises, so the fallback's definition_error is the same text.
inline Json strict_graph_document(NodusHeadlessGraph* g, long long revision) {
    int32_t scheduler_value = 0;
    if (!nodus_headless_get_scheduler_mode(g, &scheduler_value)) throw RuntimeError("native scheduler mode query failed");
    if (scheduler_value != 0 && scheduler_value != 1) throw RuntimeError("unknown native scheduler mode " + std::to_string(scheduler_value));
    const int32_t module_count = nodus_headless_get_module_count(g);
    if (module_count < 0) throw RuntimeError("native module count query failed");
    struct Row { int kind = 0; int count = 0; int tool = 0; std::string plugin; int consumes = 0, produces = 0; };
    struct Module { uint64_t stable = 0; std::string label; int x = 0, y = 0, w = 0, h = 0; std::string contract; std::vector<Row> rows; int inputs = 0, outputs = 0; };
    std::vector<Module> modules;
    for (int32_t mi = 0; mi < module_count; ++mi) {
        GP_CanvasModuleDesc desc{};
        if (!nodus_headless_get_module_desc(g, mi, &desc)) throw RuntimeError("native module " + std::to_string(mi) + " descriptor query failed");
        Module m;
        m.stable = nodus_headless_get_module_uuid(g, mi);
        if (!m.stable) throw RuntimeError("native module " + std::to_string(mi) + " has no stable identity");
        int32_t contract = 0;
        if (!nodus_headless_get_module_stack_contract(g, mi, &contract)) throw RuntimeError("native module " + std::to_string(mi) + " contract query failed");
        m.contract = contract_name(contract);
        const int32_t row_count = nodus_headless_get_module_row_count(g, mi);
        if (row_count < 0) throw RuntimeError("native module " + std::to_string(mi) + " row count query failed");
        for (int32_t ri = 0; ri < row_count; ++ri) {
            GP_CanvasModuleRowInfo info{};
            char plugin_id[1024] = {};
            if (!nodus_headless_get_module_row(g, mi, ri, &info, plugin_id, sizeof(plugin_id)))
                throw RuntimeError("native module " + std::to_string(mi) + " row " + std::to_string(ri) + " query failed");
            Row r;
            r.kind = info.kind;
            if (info.kind == 0) { r.count = info.attachment_count; m.inputs += info.attachment_count; }
            else if (info.kind == 2) { r.count = info.attachment_count; m.outputs += info.attachment_count; }
            else if (info.kind == 1 && info.tool_origin == 0) {
                if (!builtin_name(info.tool)) throw ValueError(std::to_string(info.tool) + " is not a valid BuiltinTool");
                r.kind = 10; r.tool = info.tool;
            } else if (info.kind == 1 && info.tool_origin == 1) {
                r.kind = 11; r.plugin = plugin_id; r.consumes = info.consumes; r.produces = info.produces;
                if (r.plugin.empty()) throw ValueError("plugin_id must not be empty");
                if (r.consumes < 0 || r.produces < 0) throw ValueError("stack effects must be non-negative");
            } else {
                throw RuntimeError("native module " + std::to_string(mi) + " row " + std::to_string(ri) + " has unknown kind/origin " +
                                   std::to_string(info.kind) + "/" + std::to_string(info.tool_origin));
            }
            m.rows.push_back(r);
        }
        m.label = module_label(desc);
        m.x = desc.x; m.y = desc.y; m.w = desc.w; m.h = desc.h;
        if (m.w <= 0 || m.h <= 0) throw ValueError("module width and height must be positive");
        modules.push_back(std::move(m));
    }
    auto reference = [&](int32_t module_index, int32_t contact_index, const char* direction) {
        if (module_index < 0 || module_index >= int32_t(modules.size()))
            throw RuntimeError("native endpoint names missing module " + std::to_string(module_index));
        const bool input = std::strcmp(direction, "input") == 0;
        const int ordinal = input ? contact_index : contact_index - modules[size_t(module_index)].inputs;
        const int limit = input ? modules[size_t(module_index)].inputs : modules[size_t(module_index)].outputs;
        if (ordinal < 0 || ordinal >= limit)
            throw RuntimeError(std::string("native ") + direction + " endpoint " + std::to_string(module_index) + "." +
                               std::to_string(contact_index) + " is outside materialized row arity");
        return Json::object().set("module_id", Json::uinteger(modules[size_t(module_index)].stable))
            .set("direction", Json::string(direction)).set("ordinal", Json::integer(ordinal));
    };
    const int32_t edge_count = nodus_headless_get_edge_count(g);
    if (edge_count < 0) throw RuntimeError("native edge count query failed");
    Json edges = Json::array();
    for (int32_t ei = 0; ei < edge_count; ++ei) {
        GP_CanvasEdgeDesc d{};
        int32_t type_id = 0;
        if (!nodus_headless_get_edge(g, ei, &d, &type_id)) throw RuntimeError("native edge " + std::to_string(ei) + " query failed");
        NodusHeadlessFifoConfig c{};
        Json fifo = nodus_headless_get_edge_fifo_config(g, ei, &c) ? fifo_mapping(c) : Json::null();
        Json source = reference(d.a_module, d.a_contact_idx, "output");
        Json target = reference(d.b_module, d.b_contact_idx, "input");
        Json token = fifo.t == Json::T::Obj ? *fifo.get("token_bytes") : Json::null();
        Json slots = fifo.t == Json::T::Obj ? *fifo.get("slots") : Json::null();
        edges.push(Json::object().set("source", std::move(source)).set("target", std::move(target)).set("fifo", std::move(fifo))
                       .set("type_id", Json::integer(type_id)).set("index", Json::integer(ei)).set("token_bytes", token).set("slots", slots));
    }
    auto ports = [&](bool is_output) {
        const int32_t count = nodus_headless_get_port_count(g, is_output ? 1 : 0);
        if (count < 0) throw RuntimeError("native port count query failed");
        const char* direction = is_output ? "output" : "input";
        Json out = Json::array();
        for (int32_t k = 0; k < count; ++k) {
            char name[1024] = {};
            int32_t mi = 0, ci = 0, head = 0;
            NodusHeadlessFifoConfig c{};
            if (!nodus_headless_get_port_at(g, is_output ? 1 : 0, k, name, sizeof(name), &mi, &ci, &head, &c))
                throw RuntimeError(std::string("native ") + direction + " port " + std::to_string(k) + " query failed");
            Json port = Json::object().set("name", Json::string(name));
            port.set("connector", reference(mi, ci, direction));
            port.set("fifo", fifo_mapping(c));
            if (!*name) throw ValueError("port name must not be empty");
            if (is_output) port.set("start_at_head", Json::boolean(head != 0));
            out.push(std::move(port));
        }
        return out;
    };
    Json inputs = ports(false);
    Json outputs = ports(true);

    // GraphDefinition.validate()
    std::set<uint64_t> ids;
    for (const auto& m : modules) ids.insert(m.stable);
    if (ids.size() != modules.size()) throw ValueError("module stable identities must be unique");
    for (const auto& m : modules) {
        const std::string id = std::to_string(m.stable);
        int depth = 0;
        std::string contract;
        for (size_t ri = 0; ri < m.rows.size(); ++ri) {
            const Row& r = m.rows[ri];
            const std::string idx = std::to_string(ri);
            if (r.kind == 0) {
                if (r.count < 1 || r.count > 32) throw ValueError("module " + id + " input row " + idx + " count must be in 1..32");
                depth += r.count;
            } else if (r.kind == 2) {
                if (r.count < 1 || r.count > 32) throw ValueError("module " + id + " output row " + idx + " count must be in 1..32");
                if (depth < r.count) throw ValueError("module " + id + " output row " + idx + " underflows the stack");
                depth -= r.count;
            } else if (r.kind == 10) {
                int consumes = 0, produces = 0;
                if (!builtin_stack_effect(r.tool, consumes, produces))
                    throw ValueError(std::string(builtin_name(r.tool)) + " has no ordinary scalar-stack contract");
                if (depth < consumes) throw ValueError("module " + id + " tool row " + idx + " underflows the stack");
                depth += produces - consumes;
                if (!contract.empty() && contract != "scalar_f32") throw ValueError("module " + id + " mixes execution contracts");
                contract = "scalar_f32";
            } else {
                if (depth < r.consumes) throw ValueError("module " + id + " plugin row " + idx + " underflows the stack");
                depth += r.produces - r.consumes;
                if (!contract.empty() && contract != m.contract) throw ValueError("module " + id + " mixes execution contracts");
                contract = m.contract;
            }
        }
        if (depth) throw ValueError("module " + id + " leaves " + std::to_string(depth) + " value(s) on the stack");
        const std::string actual = contract.empty() ? "opaque" : contract;
        if (m.contract != actual)
            throw ValueError("module " + id + " row contract " + actual + " does not match declared " + m.contract);
    }
    std::set<std::pair<std::string, std::string>> names;
    for (const auto* list : {&inputs, &outputs}) {
        const std::string direction = list == &inputs ? "input" : "output";
        for (const auto& p : list->a)
            if (!names.insert({direction, p.get("name")->s}).second)
                throw ValueError("duplicate " + direction + " port name: " + py_repr(p.get("name")->s));
    }

    // to_mapping() + graph_document's additive keys
    Json mods = Json::array();
    for (size_t mi = 0; mi < modules.size(); ++mi) {
        const Module& m = modules[mi];
        Json rows = Json::array();
        for (size_t ri = 0; ri < m.rows.size(); ++ri) {
            const Row& r = m.rows[ri];
            Json row = Json::object();
            if (r.kind == 0) row.set("kind", Json::string("input")).set("count", Json::integer(r.count));
            else if (r.kind == 2) row.set("kind", Json::string("output")).set("count", Json::integer(r.count));
            else if (r.kind == 10) row.set("kind", Json::string("builtin")).set("tool", Json::string(builtin_name(r.tool)));
            else row.set("kind", Json::string("plugin")).set("plugin_id", Json::string(r.plugin)).set("consumes", Json::integer(r.consumes))
                     .set("produces", Json::integer(r.produces)).set("contract", Json::string(m.contract));
            row.set("index", Json::integer((long long)ri));
            rows.push(std::move(row));
        }
        Json contacts = Json::array();
        for (int k = 0; k < m.inputs; ++k)
            contacts.push(Json::object().set("index", Json::integer(k)).set("ordinal", Json::integer(k)).set("direction", Json::string("input")).set("name", Json::null()));
        for (int k = 0; k < m.outputs; ++k)
            contacts.push(Json::object().set("index", Json::integer(m.inputs + k)).set("ordinal", Json::integer(k)).set("direction", Json::string("output")).set("name", Json::null()));
        mods.push(Json::object().set("stable_id", Json::uinteger(m.stable)).set("label", Json::string(m.label)).set("x", Json::integer(m.x))
                      .set("y", Json::integer(m.y)).set("width", Json::integer(m.w)).set("height", Json::integer(m.h))
                      .set("contract", Json::string(m.contract)).set("rows", std::move(rows)).set("index", Json::integer((long long)mi))
                      .set("exec_mode", Json::null()).set("contacts", std::move(contacts)));
    }
    return Json::object().set("schema", Json::string("nodus.graph-definition.v1"))
        .set("scheduler", Json::string(scheduler_value == 1 ? "SCHEDULED" : "FREE_SPINNING")).set("modules", std::move(mods))
        .set("edges", std::move(edges)).set("connectors", Json::array()).set("inputs", std::move(inputs)).set("outputs", std::move(outputs))
        .set("revision", Json::integer(revision));   // contract_ports: added by graph_document, outside the fallback
}

// console_views._native_document: the tolerant read for a canvas that cannot be a strict v1 definition.
inline Json native_graph_document(NodusHeadlessGraph* g, long long revision, const std::string& reason) {
    Json modules = Json::array();
    const int32_t module_count = count_of(nodus_headless_get_module_count(g));
    for (int32_t mi = 0; mi < module_count; ++mi) {
        GP_CanvasModuleDesc desc{};
        nodus_headless_get_module_desc(g, mi, &desc);
        const uint64_t stable = nodus_headless_get_module_uuid(g, mi);
        int32_t contract = 0;
        nodus_headless_get_module_stack_contract(g, mi, &contract);
        Json rows = Json::array();
        int inputs = 0, outputs = 0;
        const int32_t row_count = count_of(nodus_headless_get_module_row_count(g, mi));
        for (int32_t ri = 0; ri < row_count; ++ri) {
            GP_CanvasModuleRowInfo info{};
            char plugin_id[1024] = {};
            if (!nodus_headless_get_module_row(g, mi, ri, &info, plugin_id, sizeof(plugin_id))) continue;
            Json row = Json::object();
            if (info.kind == 0) { row.set("kind", Json::string("input")).set("count", Json::integer(info.attachment_count)); inputs += info.attachment_count; }
            else if (info.kind == 2) { row.set("kind", Json::string("output")).set("count", Json::integer(info.attachment_count)); outputs += info.attachment_count; }
            else if (info.tool_origin == 1) row.set("kind", Json::string("plugin")).set("plugin_id", Json::string(plugin_id))
                                               .set("consumes", Json::integer(info.consumes)).set("produces", Json::integer(info.produces));
            else row.set("kind", Json::string("builtin")).set("tool", builtin_name(info.tool) ? Json::string(builtin_name(info.tool)) : Json::integer(info.tool));
            row.set("index", Json::integer(ri));
            rows.push(std::move(row));
        }
        Json contacts = Json::array();
        for (int k = 0; k < inputs; ++k)
            contacts.push(Json::object().set("index", Json::integer(k)).set("ordinal", Json::integer(k)).set("direction", Json::string("input")).set("name", Json::null()));
        for (int k = 0; k < outputs; ++k)
            contacts.push(Json::object().set("index", Json::integer(inputs + k)).set("ordinal", Json::integer(k)).set("direction", Json::string("output")).set("name", Json::null()));
        modules.push(Json::object().set("index", Json::integer(mi)).set("stable_id", stable ? Json::uinteger(stable) : Json::null())
                         .set("label", Json::string(module_label(desc))).set("x", Json::integer(desc.x)).set("y", Json::integer(desc.y))
                         .set("width", Json::integer(desc.w)).set("height", Json::integer(desc.h)).set("contract", Json::string(contract_name(contract)))
                         .set("exec_mode", Json::null()).set("rows", std::move(rows)).set("contacts", std::move(contacts)));
    }
    Json edges = Json::array();
    const int32_t edge_count = count_of(nodus_headless_get_edge_count(g));
    for (int32_t ei = 0; ei < edge_count; ++ei) {
        GP_CanvasEdgeDesc d{};
        int32_t type_id = 0;
        if (!nodus_headless_get_edge(g, ei, &d, &type_id)) continue;
        NodusHeadlessFifoConfig c{};
        Json fifo = nodus_headless_get_edge_fifo_config(g, ei, &c) ? fifo_mapping(c) : Json::null();
        Json token = fifo.t == Json::T::Obj ? *fifo.get("token_bytes") : Json::null();
        Json slots = fifo.t == Json::T::Obj ? *fifo.get("slots") : Json::null();
        edges.push(Json::object().set("index", Json::integer(ei))
                       .set("source", Json::object().set("module_index", Json::integer(d.a_module)).set("contact_index", Json::integer(d.a_contact_idx)))
                       .set("target", Json::object().set("module_index", Json::integer(d.b_module)).set("contact_index", Json::integer(d.b_contact_idx)))
                       .set("fifo", std::move(fifo)).set("type_id", Json::integer(type_id)).set("token_bytes", token).set("slots", slots));
    }
    Json ports[2] = {Json::array(), Json::array()};
    for (int is_output = 0; is_output < 2; ++is_output) {
        const int32_t count = count_of(nodus_headless_get_port_count(g, is_output));
        for (int32_t k = 0; k < count; ++k) {
            char name[1024] = {};
            int32_t mi = 0, ci = 0, head = 0;
            NodusHeadlessFifoConfig c{};
            if (nodus_headless_get_port_at(g, is_output, k, name, sizeof(name), &mi, &ci, &head, &c))
                ports[is_output].push(Json::object().set("name", Json::string(name))
                                          .set("connector", Json::object().set("module_index", Json::integer(mi)).set("contact_index", Json::integer(ci)))
                                          .set("fifo", fifo_mapping(c)).set("start_at_head", Json::boolean(head != 0)));
        }
    }
    return Json::object().set("schema", Json::string("nodus.graph-definition.v1")).set("definition_error", Json::string(reason))
        .set("revision", Json::integer(revision)).set("modules", std::move(modules)).set("edges", std::move(edges))
        .set("connectors", Json::array()).set("inputs", std::move(ports[0])).set("outputs", std::move(ports[1]))
        .set("contract_ports", contract_port_documents(g));
}

// console_views.graph_document: only the definition falls back to native indices; the contract ports are read after it,
// and their failure fails the view.
inline Json graph_document(NodusHeadlessGraph* g, long long revision) {
    Json document;
    try {
        document = strict_graph_document(g, revision);
    } catch (const ValueError& e) {
        return native_graph_document(g, revision, e.what());
    } catch (const RuntimeError& e) {
        return native_graph_document(g, revision, e.what());
    }
    document.set("contract_ports", contract_port_documents(g));
    return document;
}

inline Json tools_document(NodusHeadlessGraph* g, const Json* graph_doc) {
    std::map<std::string, Json> tools;
    for (int v = 1; v <= 13; ++v)
        tools[builtin_name(v)] = Json::object().set("name", Json::string(builtin_name(v))).set("plugin_id", Json::null())
                                     .set("source", Json::string("builtin")).set("description", Json::null());
    auto plugin = [&](const std::string& id) {
        if (!tools.count(id))
            tools[id] = Json::object().set("name", Json::string(id)).set("plugin_id", Json::string(id))
                            .set("source", Json::string("plugin")).set("description", Json::null());
    };
    for (const auto& p : contract_ports(g)) { plugin(p.prefix + ".market"); plugin(p.prefix + ".worker"); }
    if (graph_doc)
        for (const auto& m : graph_doc->get("modules")->a)
            for (const auto& r : m.get("rows")->a)
                if (r.get("kind") && r.get("kind")->s == "plugin") plugin(r.get("plugin_id")->s);
    Json list = Json::array();
    for (auto& [name, doc] : tools) list.push(std::move(doc));
    return Json::object().set("tools", std::move(list)).set("complete", Json::boolean(false))
        .set("missing", Json::string("registered plugin tools that are neither placed on this canvas nor a contract pair, and the "
                                     "module library: the headless ABI has no tool registry enumeration and the host has no "
                                     "library path (needs new C++)"));
}

inline Json activity_document(const Context& c, long long tick) {
    Json edges = Json::array();
    const int32_t edge_count = count_of(nodus_headless_get_edge_count(c.g));
    for (int32_t ei = 0; ei < edge_count; ++ei) {
        NodusHeadlessFifoConfig probe{};
        int32_t unread = 0;
        const bool ok = nodus_headless_get_edge_fifo_config(c.g, ei, &probe) && nodus_headless_get_edge_unread(c.g, ei, &unread);
        edges.push(Json::object().set("index", Json::integer(ei)).set("fill", ok ? Json::integer(unread) : Json::null()).set("seq", Json::null()));
    }
    Json ports = Json::array();
    for (const auto& p : contract_ports(c.g)) {
        const BoundPort b = contract_port(c.g, p.prefix);
        ports.push(Json::object().set("prefix", Json::string(p.prefix)).set("pending", Json::null())
                       .set("engines", b.ok ? Json::integer(b.engines) : Json::null()).set("hosts", b.ok ? Json::integer(b.hosts) : Json::null()));
    }
    Json hosts = Json::array();
    for (const auto& [prefix, id] : c.hosts) {
        uint64_t s[5] = {};
        if (!nodus_headless_compile_host_stats(prefix.c_str(), id, s)) continue;
        hosts.push(Json::object().set("prefix", Json::string(prefix)).set("host_id", Json::uinteger(id)).set("accepted", Json::uinteger(s[0]))
                       .set("completed", Json::uinteger(s[1])).set("refused", Json::uinteger(s[2])).set("acknowledged", Json::uinteger(s[3]))
                       .set("in_flight", Json::uinteger(s[4])));
    }
    return Json::object().set("tick", Json::integer(tick)).set("edges", std::move(edges)).set("contract_ports", std::move(ports))
        .set("compile_hosts", std::move(hosts));
}

// Graph.meta_group: the group's snapshot with each member's latest record by field name, or null.
inline Json meta_group(NodusHeadlessGraph* g, uint64_t id) {
    static const char* const fields[16] = {"tick", "time_ns", "dt_ns", "ordinal", "edge_idx", "rope_id", "write_seq", "write_delta",
        "read_seq", "read_delta", "occupancy", "slots", "refused_total", "refused_delta", "bytes_delta", "flags"};
    const int32_t count = nodus_headless_meta_member_count(g, id);
    const int32_t capacity = 8 + std::max(count, 1) * 16;
    std::vector<uint64_t> out(size_t(capacity), 0);
    if (!nodus_headless_meta_snapshot(g, id, out.data(), capacity)) return Json::null();
    Json members = Json::array();
    for (uint64_t m = 0; m < out[3]; ++m) {
        const size_t base = 8 + size_t(m) * 16;
        Json record = Json::object();
        for (int k = 0; k < 16; ++k) record.set(fields[k], Json::uinteger(out[base + size_t(k)]));
        int32_t canvas = -1, table = -1;
        uint64_t rope = 0;
        if (nodus_headless_meta_member(g, id, int32_t(m), &canvas, &table, &rope)) record.set("canvas_edge_idx", Json::integer(canvas));
        members.push(std::move(record));
    }
    return Json::object().set("id", Json::uinteger(out[0])).set("channel_group", Json::integer(static_cast<long long>(static_cast<int64_t>(out[1]))))
        .set("vertices", Json::uinteger(out[2])).set("records", Json::uinteger(out[4])).set("feed_refused", Json::uinteger(out[5]))
        .set("slots", Json::uinteger(out[6])).set("members", std::move(members));
}
inline std::vector<uint64_t> meta_group_ids(NodusHeadlessGraph* g) {
    std::vector<uint64_t> ids;
    const int32_t n = nodus_headless_meta_count(g);
    for (int32_t k = 0; k < n; ++k) ids.push_back(nodus_headless_meta_id_at(g, k));
    return ids;
}
// Graph.link_report: the 16 counters, the state, and the reason when there is one; null for an unknown link.
inline Json link_report(const std::string& id) {
    static const char* const counters[16] = {"listen_port", "account", "control_sent", "control_received", "bulk_bytes_sent",
        "bulk_bytes_received", "bulk_tensors_sent", "bulk_tensors_received", "control_refused", "bulk_refused", "bulk_dropped",
        "peers_refused", "control_read_stalls", "bulk_read_stalls", "protocol_violations", "refused_full"};
    static const char* const states[4] = {"waiting", "linked", "closed", "failed"};
    int32_t state = 0;
    uint64_t out[16] = {};
    if (!nodus_headless_link_report(id.c_str(), &state, out)) return Json::null();
    Json report = Json::object();
    for (int k = 0; k < 16; ++k) report.set(counters[k], Json::uinteger(out[k]));
    report.set("state", Json::string(state >= 0 && state < 4 ? states[state] : std::to_string(state)));
    char reason[512] = {};
    if (nodus_headless_link_reason(id.c_str(), reason, sizeof(reason)) > 0) report.set("reason", Json::string(reason));
    return report;
}
inline Json meta_document(const Context& c, long long tick) {
    Json groups = Json::array();
    for (uint64_t id : meta_group_ids(c.g)) {
        Json group = meta_group(c.g, id);
        if (group.t != Json::T::Null) groups.push(std::move(group));
    }
    Json links = Json::array();
    for (const auto& id : std::set<std::string>(c.link_ids.begin(), c.link_ids.end())) {
        Json report = link_report(id);
        if (report.t == Json::T::Null) continue;
        Json entry = Json::object().set("id", Json::string(id));
        for (auto& [k, v] : report.o) entry.set(k, std::move(v));
        links.push(std::move(entry));
    }
    return Json::object().set("tick", Json::integer(tick)).set("groups", std::move(groups)).set("links", std::move(links));
}

// =================================================================================== PLATFORM REGION BEGIN: sockets
namespace platform {
#if defined(_WIN32)
using socklen = int;
inline int last_error() { return WSAGetLastError(); }
#else
using socklen = socklen_t;
#endif
// A loopback listener on `port` (0: the system picks); the bound port in `bound`.
inline SOCKET listen_loopback(uint16_t port, uint16_t& bound) {
    nodus::contracts::ensure_sockets();
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return INVALID_SOCKET;
    sockaddr_in a{};
    a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons(port);
    socklen len = sizeof(a);
    if (!nodus::contracts::WireSocket::reuse_listen_port(s) || bind(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 ||
        listen(s, 16) != 0 || getsockname(s, reinterpret_cast<sockaddr*>(&a), &len) != 0) {
        nodus::contracts::WireSocket::close_socket(s);
        return INVALID_SOCKET;
    }
    bound = ntohs(a.sin_port);
    return s;
}
// One accepted connection within `ms`, or INVALID_SOCKET.
inline SOCKET accept_within(SOCKET listener, int ms) {
    fd_set r;
    FD_ZERO(&r);
    FD_SET(listener, &r);
    timeval tv{ms / 1000, (ms % 1000) * 1000};
    if (select(static_cast<int>(listener + 1), &r, nullptr, nullptr, &tv) <= 0) return INVALID_SOCKET;
    return accept(listener, nullptr, nullptr);
}
inline void receive_timeout(SOCKET s, int seconds) {
#if defined(_WIN32)
    DWORD ms = DWORD(seconds) * 1000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
#else
    timeval tv{seconds, 0};
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
}
// Bytes received (>0), 0 at the end of the stream, -1 on an error or the idle timeout.
inline int receive(SOCKET s, char* buf, int n) { const int r = ::recv(s, buf, n, 0); return r > 0 ? r : r == 0 ? 0 : -1; }
inline bool send_all(SOCKET s, const std::string& data) {
    size_t at = 0;
    while (at < data.size()) {
#if defined(_WIN32)
        const int n = ::send(s, data.data() + at, int(data.size() - at), 0);
#else
        const ssize_t n = ::send(s, data.data() + at, data.size() - at, MSG_NOSIGNAL);
#endif
        if (n <= 0) return false;
        at += size_t(n);
    }
    return true;
}
inline void close(SOCKET s) { nodus::contracts::WireSocket::close_socket(s); }
}  // namespace platform
// ===================================================================================== PLATFORM REGION END

// ======================================================================================================== the server
class Server {
public:
    static constexpr int kMaxClients = 8;
    static constexpr int kIdleSeconds = 60;
    static constexpr double kCoalesceSeconds = 0.1;
    static constexpr int kStepWaitSeconds = 5;

    Server(std::string token, Context context) : token_(std::move(token)), c_(std::move(context)) {
        std::random_device rd;
        tmp_ = std::filesystem::temp_directory_path() / ("nodus_console_" + std::to_string(rd()) + std::to_string(rd()));
        std::filesystem::create_directories(tmp_);
    }
    ~Server() { close(); }
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    uint16_t start(uint16_t port) {
        uint16_t bound = 0;
        listener_ = platform::listen_loopback(port, bound);
        if (listener_ == INVALID_SOCKET) throw std::runtime_error("console port " + std::to_string(port) + " could not be bound on 127.0.0.1");
        accept_thread_ = std::thread([this] { accept_loop(); });
        return bound;
    }
    // Stepping thread: the graph appears, and the hosts and links the startup attached.
    void set_context(NodusHeadlessGraph* g, std::vector<std::pair<std::string, uint64_t>> hosts, std::vector<std::string> links) {
        c_.g = g;
        c_.hosts = std::move(hosts);
        for (auto& id : links) if (std::find(c_.link_ids.begin(), c_.link_ids.end(), id) == c_.link_ids.end()) c_.link_ids.push_back(id);
    }
    void close() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (closing_) return;
            closing_ = true;
        }
        cv_.notify_all();
        if (listener_ != INVALID_SOCKET) platform::close(listener_);
        if (accept_thread_.joinable()) accept_thread_.join();
        std::unique_lock<std::mutex> lk(mu_);
        // Client threads end at once when their socket closes; each socket is closed once, here or by its thread.
        for (SOCKET s : client_sockets_) platform::close(s);
        client_sockets_.clear();
        cv_.wait_for(lk, std::chrono::seconds(2), [this] { return clients_ == 0; });
        lk.unlock();
        std::error_code ec;
        std::filesystem::remove_all(tmp_, ec);
    }

    // Stepping thread, after each step (run_canvas ConsoleServer.service).
    void service() {
        if (!wanted_.load()) return;
        std::set<std::string> wanted;
        std::vector<std::shared_ptr<Action>> actions;
        {
            std::lock_guard<std::mutex> lk(mu_);
            wanted.swap(want_views_);
            actions.swap(actions_);
            wanted_.store(false);
        }
        for (auto& a : actions) {
            Json reply;
            try {
                reply = apply_action(a->request);
            } catch (const std::exception& e) {
                reply = Json::object().set("error", Json::string(std::string(e.what()).substr(0, 200)));
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                a->reply = std::move(reply);
                a->done = true;
            }
            cv_.notify_all();
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (actions.empty() && snap_ && seconds_since(snap_->at) < kCoalesceSeconds && covers(*snap_, wanted)) return;
        }
        std::shared_ptr<Snap> snap;
        try {
            snap = build(wanted);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "nodus: console snapshot failed: %s\n", e.what());
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (snap) snap_ = snap;
            ++generation_;
        }
        cv_.notify_all();
    }

private:
    struct Snap {
        std::chrono::steady_clock::time_point at;
        std::string data, state;
        long long revision = 0;
        std::map<std::string, std::string> views;
        std::set<std::string> attempted;   // the views this build was asked for (a view asked for and absent failed)
        int frame_width = 0, frame_height = 0;   // views["frame"]: the RGBA raster
    };
    struct Action { Json request; Json reply; bool done = false; };

    static double seconds_since(std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    }
    static bool covers(const Snap& s, const std::set<std::string>& wanted) {
        for (const auto& v : wanted) if (!s.views.count(v)) return false;
        return true;
    }

    // run_canvas ConsoleServer._build: the canvas bytes (saved through the headless file path into a private folder),
    // the revision, STATE, DIGEST and the asked-for views.
    std::shared_ptr<Snap> build(const std::set<std::string>& wanted) {
        const auto path = tmp_ / "snapshot.nodus.txt";
        if (!nodus_headless_save_file(c_.g, path.string().c_str())) throw std::runtime_error("native save failed: " + path.string());
        std::string data;
        {
            std::ifstream in(path, std::ios::binary);
            data.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        std::error_code ec;
        std::filesystem::remove(path, ec);
        std::filesystem::remove_all(path.string() + ".modules", ec);
        if (data != last_bytes_) { last_bytes_ = data; ++revision_; }
        Json modules = Json::array();
        const int32_t module_count = count_of(nodus_headless_get_module_count(c_.g));
        for (int32_t mi = 0; mi < module_count; ++mi) {
            Json tail = Json::array();
            int32_t n = 0;
            if (nodus_headless_get_stack_tail(c_.g, mi, nullptr, 0, &n) && n > 0) {
                std::vector<float> v(size_t(n), 0.0f);
                if (nodus_headless_get_stack_tail(c_.g, mi, v.data(), n, &n))
                    for (int32_t k = 0; k < n; ++k) tail.push(Json::number(double(v[size_t(k)])));
            }
            modules.push(Json::object().set("index", Json::integer(mi)).set("rows", Json::integer(nodus_headless_get_module_row_count(c_.g, mi)))
                             .set("stack_tail", std::move(tail)));
        }
        Json cports = Json::array();
        for (const auto& p : contract_ports(c_.g)) {
            const BoundPort b = contract_port(c_.g, p.prefix);
            cports.push(Json::object().set("prefix", Json::string(p.prefix)).set("listen_port", Json::integer(b.ok ? b.port : p.declared))
                            .set("hosts", Json::integer(b.ok ? b.hosts : 0)).set("engines", Json::integer(b.ok ? b.engines : 0)));
        }
        Json ports = Json::array();
        for (int is_output = 0; is_output < 2; ++is_output) {
            const int32_t count = count_of(nodus_headless_get_port_count(c_.g, is_output));
            for (int32_t k = 0; k < count; ++k) {
                char name[1024] = {};
                int32_t mi = 0, ci = 0, head = 0;
                NodusHeadlessFifoConfig cfg{};
                if (!nodus_headless_get_port_at(c_.g, is_output, k, name, sizeof(name), &mi, &ci, &head, &cfg)) continue;
                int32_t unread = 0;
                if (!nodus_headless_get_port_unread(c_.g, name, is_output, &unread)) unread = -1;
                ports.push(Json::object().set("name", Json::string(name)).set("is_output", Json::boolean(is_output != 0)).set("unread", Json::integer(unread)));
            }
        }
        const long long tick = c_.ticks();
        Json state = Json::object().set("revision", Json::integer(revision_)).set("ticks", Json::integer(tick))
                         .set("canvas", Json::string(c_.canvas_name)).set("modules", std::move(modules)).set("contract_ports", std::move(cports))
                         .set("ports", std::move(ports)).set("events", Json::array());
        auto snap = std::make_shared<Snap>();
        snap->attempted = wanted;
        snap->attempted.insert("digest");
        snap->data = data;
        snap->state = dumps_default(state);
        snap->revision = revision_;
        snap->views["digest"] = dumps_view(digest_document(c_, revision_, data, tick));
        try {
            std::optional<Json> graph_doc;
            if (wanted.count("graph") || wanted.count("tools")) graph_doc = graph_document(c_.g, revision_);
            if (wanted.count("graph")) snap->views["graph"] = dumps_view(*graph_doc);
            if (wanted.count("tools")) snap->views["tools"] = dumps_view(tools_document(c_.g, graph_doc ? &*graph_doc : nullptr));
            if (wanted.count("meta")) snap->views["meta"] = dumps_view(meta_document(c_, tick));
            if (wanted.count("activity")) snap->views["activity"] = dumps_view(activity_document(c_, tick));
            if (wanted.count("frame") && c_.frame) {
                std::string rgba;
                if (c_.frame(rgba, snap->frame_width, snap->frame_height)) snap->views["frame"] = std::move(rgba);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "nodus: console view failed: %s\n", e.what());
        }
        snap->at = std::chrono::steady_clock::now();
        return snap;
    }

    // ---------------------------------------------------------------------------------------- ACTION (stepping thread)
    std::string module_label_at(int32_t mi) {
        GP_CanvasModuleDesc d{};
        if (!nodus_headless_get_module_desc(c_.g, mi, &d)) return "";
        return module_label(d);
    }
    // run_canvas _canvas_edge: an index, or "<from module label> > <to module label>" naming exactly one edge.
    int32_t canvas_edge(const Json& name) {
        const int32_t count = nodus_headless_get_edge_count(c_.g);
        if (name.t == Json::T::Int) {
            if (name.i < 0 || name.i >= count) throw ValueError("no canvas edge " + std::to_string(name.i));
            return int32_t(name.i);
        }
        if (name.t != Json::T::Str || name.s.find('>') == std::string::npos)
            throw ValueError("an edge is a canvas edge index or '<from module label> > <to module label>'");
        auto strip = [](std::string s) {
            const char* ws = " \t\n\r\f\v";
            s.erase(0, s.find_first_not_of(ws));
            const size_t e = s.find_last_not_of(ws);
            s.erase(e == std::string::npos ? 0 : e + 1);
            return s;
        };
        const size_t gt = name.s.find('>');
        const std::string source = strip(name.s.substr(0, gt)), target = strip(name.s.substr(gt + 1));
        std::vector<int32_t> found;
        for (int32_t e = 0; e < count; ++e) {
            GP_CanvasEdgeDesc d{};
            int32_t type_id = 0;
            if (nodus_headless_get_edge(c_.g, e, &d, &type_id) && module_label_at(d.a_module) == source && module_label_at(d.b_module) == target)
                found.push_back(e);
        }
        if (found.size() != 1) throw ValueError("edge '" + name.s + "' names " + std::to_string(found.size()) + " canvas edges");
        return found[0];
    }

    static bool decode_hex(const std::string& s, std::string& out) {
        if (s.size() % 2) return false;
        out.clear();
        for (size_t k = 0; k < s.size(); k += 2) {
            auto nib = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
            const int hi = nib(s[k]), lo = nib(s[k + 1]);
            if (hi < 0 || lo < 0) return false;
            out += char(hi * 16 + lo);
        }
        return true;
    }
    // base64.b64decode(..., validate=True): the standard alphabet and padding, nothing else.
    static bool decode_base64(const std::string& s, std::string& out) {
        static const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        if (s.size() % 4) return false;
        out.clear();
        for (size_t k = 0; k < s.size(); k += 4) {
            int v[4];
            int pad = 0;
            for (int j = 0; j < 4; ++j) {
                const char c = s[k + size_t(j)];
                if (c == '=' && k + 4 == s.size() && j >= 2) { v[j] = 0; ++pad; continue; }
                if (pad) return false;
                const size_t p = alphabet.find(c);
                if (p == std::string::npos) return false;
                v[j] = int(p);
            }
            const uint32_t n = uint32_t(v[0] << 18 | v[1] << 12 | v[2] << 6 | v[3]);
            out += char(n >> 16);
            if (pad < 2) out += char((n >> 8) & 0xFF);
            if (pad < 1) out += char(n & 0xFF);
        }
        return true;
    }

    // The link ops that open or feed a link (run_canvas _apply_link), with the link declared in the canvas
    // (nodus_headless_open_link_host records a host; a dial composes its tables, which bind again when it is reopened).
    template <class Subset, class ValueOr>
    bool apply_link(const std::string& op, const std::string& id, const Json& request, Subset subset, ValueOr value_or) {
        static const auto name_ok = [](const std::string& s) {
            if (s.empty() || s.size() > 60) return false;
            for (char ch : s) if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '.' || ch == '_' || ch == '-')) return false;
            return true;
        };
        auto metrics_options = [&]() {   // run_canvas _metrics_options: the channel is set before the row is made
            const Json metrics = value_or("metrics", Json::boolean(false));
            const Json channel = value_or("metrics_channel", Json::integer(0));
            if (metrics.t != Json::T::Bool || !channel.is_int() || channel.big || channel.t != Json::T::Int || channel.i < 0 || channel.i > 0x7FFFFFFF)
                throw ValueError("metrics is true or false; metrics_channel is a meta channel group 0..2147483647");
            if (channel.i && !metrics.b) throw ValueError("metrics_channel needs metrics: true");
            if (metrics.b) nodus_headless_set_link_metrics_channel(id.c_str(), int32_t(channel.i));
            return metrics.b ? NODUS_LINK_METRICS : 0;
        };
        if (op == "link_send") {
            if (!request.get("bytes") || !subset({"op", "expected_revision", "id", "bytes", "fill"}))
                throw ValueError("link_send takes op, expected_revision, id, bytes[, fill]");
            const Json size = *request.get("bytes");
            const Json fill = value_or("fill", Json::integer(0));
            if (!size.is_int() || size.big || size.t != Json::T::Int || size.i < 1 || size.i > (64ll << 20) || !fill.is_int() || fill.big ||
                fill.t != Json::T::Int || fill.i < 0 || fill.i > 255)
                throw ValueError("link_send: bytes 1..64 MiB, fill 0..255");
            const uint64_t shape[1] = {uint64_t(size.i)};
            const uint64_t handle = nodus_tensor_create(NODUS_DTYPE_U8, 1, shape);
            if (!handle) throw RuntimeError("the tensor core refused the tensor");
            const std::string data(size_t(size.i), char(fill.i));
            if (nodus_tensor_write(handle, 0, data.data(), data.size()) != int64_t(data.size())) {
                nodus_tensor_destroy(handle);
                throw RuntimeError("the tensor could not be written");
            }
            NodusHeadlessPublishResult result{};
            if (!nodus_headless_push_token_ex(c_.g, (id + ".send").c_str(), &handle, sizeof(handle), &result) || !result.accepted) {
                nodus_tensor_destroy(handle);
                throw RuntimeError(id + ".send is full");
            }
            return true;
        }
        if (op == "link_udp") {
            static const std::set<std::string> keys = {"op", "expected_revision", "id", "broker_host", "broker_port", "prefix", "role", "link_id", "key", "expires_at"};
            for (const auto& k : keys) if (!request.get(k)) throw ValueError("link_udp takes op, expected_revision, id, broker_host, broker_port, prefix, role, link_id, key, expires_at[, account, metrics, metrics_channel]");
            if (!subset({"op", "expected_revision", "id", "broker_host", "broker_port", "prefix", "role", "link_id", "key", "expires_at", "account", "metrics", "metrics_channel"}))
                throw ValueError("link_udp takes op, expected_revision, id, broker_host, broker_port, prefix, role, link_id, key, expires_at[, account, metrics, metrics_channel]");
            const Json& prefix = *request.get("prefix");
            const Json& port = *request.get("broker_port");
            const Json& role = *request.get("role");
            const Json& expires = *request.get("expires_at");
            const Json& host = *request.get("broker_host");
            if (prefix.t != Json::T::Str || !name_ok(prefix.s) || port.t != Json::T::Int || port.i <= 0 || port.i >= 65536 ||
                role.t != Json::T::Int || (role.i != 1 && role.i != 2) || expires.t != Json::T::Int || host.t != Json::T::Str)
                throw ValueError("link_udp: a prefix, a broker port 1-65535, role 1 or 2, an integer expiry and a broker host");
            std::string link_id, key;
            const Json& lid = *request.get("link_id");
            const Json& k = *request.get("key");
            if (lid.t != Json::T::Str || k.t != Json::T::Str || !decode_hex(lid.s, link_id) || !decode_hex(k.s, key)) throw ValueError("link_id and key are hex");
            if (link_id.size() != 16 || key.size() != 32) throw ValueError("role is 1 or 2, the link id 16 bytes and the key 32");
            const Json account = value_or("account", Json::integer(0));
            const int flags = metrics_options();
            if (!nodus_headless_register_link_udp_dialer(c_.g, id.c_str(), host.s.c_str(), uint16_t(port.i), prefix.s.c_str(), int32_t(role.i),
                                                         reinterpret_cast<const uint8_t*>(link_id.data()), uint64_t(expires.i),
                                                         reinterpret_cast<const uint8_t*>(key.data()),
                                                         account.t == Json::T::Int ? uint64_t(account.i) : 0, 15000, 0))
                throw RuntimeError("native UDP link registration refused: " + id);
            if (!nodus_headless_compose_link_tables(c_.g, id.c_str(), flags)) throw RuntimeError("the link tables of " + id + " could not be placed");
            return true;
        }
        const Json prefix = value_or("prefix", Json::null());
        const Json port = value_or("port", Json::null());
        if (prefix.t != Json::T::Str || !name_ok(prefix.s)) throw ValueError("link prefix: 1-60 of A-Z a-z 0-9 . _ -");
        if (op == "link_host") {
            if (!(port.t == Json::T::Int && port.i >= 0 && port.i < 65536)) throw ValueError("link port is 0-65535 (0: the system picks it)");
            for (const char* k : {"key_file"}) if (!request.get(k)) throw ValueError("link_host takes op, expected_revision, id, prefix, port, key_file[, metrics, metrics_channel, route]");
            if (!subset({"op", "expected_revision", "id", "prefix", "port", "key_file", "metrics", "metrics_channel", "route"}))
                throw ValueError("link_host takes op, expected_revision, id, prefix, port, key_file[, metrics, metrics_channel, route]");
            const Json& key_file = *request.get("key_file");
            const Json route = value_or("route", Json::boolean(false));
            if (key_file.t != Json::T::Str || key_file.s.empty()) throw ValueError("key_file is the path of the 32-byte link key");
            if (route.t != Json::T::Bool) throw ValueError("route is true or false");
            const int flags = metrics_options() | (route.b ? NODUS_LINK_ROUTE : 0);
            if (nodus_headless_open_link_host(c_.g, id.c_str(), prefix.s.c_str(), uint16_t(port.i), flags, key_file.s.c_str()) <= 0)
                throw RuntimeError("native link host registration refused: " + id);
            return true;
        }
        if (op != "link_dial") throw ValueError("unknown canvas UI action");
        if (!(port.t == Json::T::Int && port.i > 0 && port.i < 65536)) throw ValueError("link port is 1-65535");
        for (const char* k : {"host", "control_ticket", "bulk_ticket"})
            if (!request.get(k)) throw ValueError("link_dial takes op, expected_revision, id, host, port, prefix, control_ticket, bulk_ticket[, secure, pin, metrics, metrics_channel]");
        if (!subset({"op", "expected_revision", "id", "host", "port", "prefix", "control_ticket", "bulk_ticket", "secure", "pin", "metrics", "metrics_channel"}))
            throw ValueError("link_dial takes op, expected_revision, id, host, port, prefix, control_ticket, bulk_ticket[, secure, pin, metrics, metrics_channel]");
        const Json& host = *request.get("host");
        if (host.t != Json::T::Str || host.s.empty() || host.s.size() > 253) throw ValueError("link host is a name or an address");
        std::string control, bulk, pin;
        const Json& ct = *request.get("control_ticket");
        const Json& bt = *request.get("bulk_ticket");
        const Json pinv = value_or("pin", Json::null());
        if (ct.t != Json::T::Str || bt.t != Json::T::Str || !decode_base64(ct.s, control) || !decode_base64(bt.s, bulk) ||
            (pinv.t != Json::T::Null && (pinv.t != Json::T::Str || !decode_hex(pinv.s, pin))))
            throw ValueError("link tickets are base64, a pin is hex");
        const Json secure = value_or("secure", Json::boolean(true));
        if (secure.t != Json::T::Bool) throw ValueError("secure is true or false");
        if (control.size() != 128 || bulk.size() != 128 || (pinv.t != Json::T::Null && pin.size() != 32)) throw ValueError("tickets are 128 bytes and a pin is 32");
        const int flags = metrics_options();
        if (!nodus_headless_register_link_wss_dialer(c_.g, id.c_str(), host.s.c_str(), uint16_t(port.i), "/ws/link/", prefix.s.c_str(), secure.b ? 1 : 0,
                                                     pinv.t == Json::T::Null ? nullptr : reinterpret_cast<const uint8_t*>(pin.data()),
                                                     reinterpret_cast<const uint8_t*>(control.data()), reinterpret_cast<const uint8_t*>(bulk.data()), 10000, 0))
            throw RuntimeError("native wss link dialer registration refused: " + id);
        if (!nodus_headless_compose_link_tables(c_.g, id.c_str(), flags)) throw RuntimeError("the link tables of " + id + " could not be placed");
        return true;
    }

    // run_canvas ConsoleServer._apply_action, on the stepping thread.
    Json apply_action(const Json& request) {
        const Json* expected = request.t == Json::T::Obj ? request.get("expected_revision") : nullptr;
        if (!expected || !expected->is_int()) throw ValueError("expected_revision is required");
        if (expected->big || expected->t != Json::T::Int || expected->i != revision_)
            return Json::object().set("error", Json::string("canvas revision changed")).set("revision", Json::integer(revision_));
        const Json* opv = request.get("op");
        const std::string op = opv && opv->t == Json::T::Str ? opv->s : std::string();
        const auto keys = request.keys();
        auto subset = [&](std::initializer_list<const char*> allowed) {
            std::set<std::string> a(allowed.begin(), allowed.end());
            for (const auto& k : keys) if (!a.count(k)) return false;
            return true;
        };
        auto exactly = [&](std::initializer_list<const char*> allowed) { return subset(allowed) && keys.size() == allowed.size(); };
        auto value_or = [&](const char* k, Json d) { const Json* v = request.get(k); return v ? *v : d; };
        bool handled = false;
        Json meta;
        bool with_meta = false;
        if (op == "viewport" && exactly({"op", "expected_revision", "width", "height"})) {
            const Json& w = *request.get("width");
            const Json& h = *request.get("height");
            if (!w.is_int() || !h.is_int()) throw ValueError("viewport width and height must be integers");
            if (w.big || h.big || w.t != Json::T::Int || h.t != Json::T::Int || w.i < INT32_MIN || w.i > INT32_MAX || h.i < INT32_MIN || h.i > INT32_MAX ||
                !nodus_headless_ui_viewport(c_.g, int32_t(w.i), int32_t(h.i)))
                throw ValueError("canvas viewport must be 64..4096 pixels on each axis");
            handled = true;
        } else if (op == "mouse" && subset({"op", "expected_revision", "kind", "x", "y", "dx", "dy", "button", "scroll"})) {
            static const std::map<std::string, int32_t> kinds = {{"click", 0}, {"down", 1}, {"up", 2}, {"move", 3}, {"scroll", 4}};
            const Json kind = value_or("kind", Json::null());
            const Json button = value_or("button", Json::integer(0));
            if (kind.t != Json::T::Str || !kinds.count(kind.s) || !button.is_int() || button.big || button.t != Json::T::Int || button.i < 0 || button.i > 5)
                throw ValueError("invalid canvas mouse action");
            const Json x = value_or("x", Json::null()), y = value_or("y", Json::null()), dx = value_or("dx", Json::integer(0)),
                       dy = value_or("dy", Json::integer(0)), scroll = value_or("scroll", Json::integer(0));
            for (const Json* v : {&x, &y, &dx, &dy, &scroll})
                if (!v->is_number() || v->big || !std::isfinite(v->as_double()) || std::fabs(v->as_double()) > 4096)
                    throw ValueError("canvas mouse coordinates must be finite and bounded");
            const int32_t r = nodus_headless_ui_mouse(c_.g, kinds.at(kind.s), float(x.as_double()), float(y.as_double()), float(dx.as_double()),
                                                      float(dy.as_double()), int32_t(button.i), float(scroll.as_double()));
            if (r < 0) throw RuntimeError("native canvas UI action could not synchronize edges");
            handled = r != 0;
        } else if (op == "key" && subset({"op", "expected_revision", "key", "scancode", "action", "mods"})) {
            const Json values[4] = {value_or("key", Json::null()), value_or("scancode", Json::integer(0)), value_or("action", Json::integer(1)),
                                    value_or("mods", Json::integer(0))};
            for (const auto& v : values)
                if (!v.is_int() || v.big || v.t != Json::T::Int || v.i > 65535 || v.i < -65535) throw ValueError("invalid canvas key action");
            const int32_t r = nodus_headless_ui_key(c_.g, int32_t(values[0].i), int32_t(values[1].i), int32_t(values[2].i), int32_t(values[3].i));
            if (r < 0) throw RuntimeError("native canvas UI key could not synchronize edges");
            handled = r != 0;
        } else if (op == "link_host" || op == "link_dial" || op == "link_udp" || op == "link_send" || op == "link_report") {
            const Json* idv = request.get("id");
            static const auto valid = [](const std::string& s) {
                if (s.empty() || s.size() > 60) return false;
                for (char ch : s) if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '.' || ch == '_' || ch == '-')) return false;
                return true;
            };
            if (!idv || idv->t != Json::T::Str || !valid(idv->s)) throw ValueError("link id: 1-60 of A-Z a-z 0-9 . _ -");
            const std::string id = idv->s;
            if (op == "link_report") {
                if (!exactly({"op", "expected_revision", "id"})) throw ValueError("link_report takes op, expected_revision, id");
                return Json::object().set("handled", Json::boolean(true)).set("revision", Json::integer(revision_))
                    .set("canvas", Json::string(c_.canvas_name)).set("link", link_report(id));
            }
            handled = apply_link(op, id, request, subset, value_or);
            if (std::find(c_.link_ids.begin(), c_.link_ids.end(), id) == c_.link_ids.end() && op != "link_send") c_.link_ids.push_back(id);
        } else if (op == "meta_group" || op == "meta_report" || op == "meta_destroy") {
            if (op == "meta_group") {
                if (!exactly({"op", "expected_revision", "edges", "channel_group"}))
                    throw ValueError("meta_group takes op, expected_revision, edges, channel_group");
                const Json& edges = *request.get("edges");
                const Json& channel = *request.get("channel_group");
                if (edges.t != Json::T::Arr || edges.a.empty() || edges.a.size() > 64 || !channel.is_int() || channel.big ||
                    channel.t != Json::T::Int || channel.i < 0 || channel.i > 0x7FFFFFFF)
                    throw ValueError("meta_group: 1-64 edges and a channel group 0..2147483647");
                std::vector<int32_t> indices;
                for (const auto& e : edges.a) indices.push_back(canvas_edge(e));
                const uint64_t group = nodus_headless_meta_create(c_.g, indices.data(), int32_t(indices.size()), int32_t(channel.i));
                if (!group) {
                    std::string list = "[";
                    for (size_t k = 0; k < indices.size(); ++k) list += (k ? ", " : "") + std::to_string(indices[k]);
                    throw RuntimeError("native meta group refused over edges " + list + "]");
                }
                meta = Json::object().set("id", Json::uinteger(group)).set("group", meta_group(c_.g, group));
            } else if (op == "meta_destroy") {
                const Json* idv = request.get("id");
                if (!exactly({"op", "expected_revision", "id"}) || !idv || !idv->is_int() || idv->big)
                    throw ValueError("meta_destroy takes op, expected_revision, id");
                const uint64_t id = idv->t == Json::T::UInt ? idv->u : uint64_t(idv->i);
                if (!nodus_headless_meta_destroy(c_.g, id)) throw RuntimeError("no meta group " + (idv->t == Json::T::UInt ? std::to_string(idv->u) : std::to_string(idv->i)));
                meta = Json::object().set("id", *idv).set("destroyed", Json::boolean(true));
            } else {
                if (!subset({"op", "expected_revision", "id"})) throw ValueError("meta_report takes op, expected_revision[, id]");
                std::vector<uint64_t> ids;
                if (const Json* idv = request.get("id")) {
                    if (!idv->is_int() || idv->big) throw ValueError("meta_report id is a meta group id");
                    ids.push_back(idv->t == Json::T::UInt ? idv->u : uint64_t(idv->i));
                } else {
                    ids = meta_group_ids(c_.g);
                }
                Json list = Json::array();
                for (uint64_t id : ids) { Json gdoc = meta_group(c_.g, id); if (gdoc.t != Json::T::Null) list.push(std::move(gdoc)); }
                return Json::object().set("handled", Json::boolean(true)).set("revision", Json::integer(revision_))
                    .set("canvas", Json::string(c_.canvas_name)).set("meta", std::move(list));
            }
            handled = true;
            with_meta = true;
        } else {
            throw ValueError("unknown canvas UI action");
        }
        const long long previous = revision_;
        auto snap = build({"graph"});
        if (revision_ != previous && !nodus_headless_save_file(c_.g, c_.canvas_path.c_str()))
            throw RuntimeError("native save failed: " + c_.canvas_path);
        {
            std::lock_guard<std::mutex> lk(mu_);
            snap_ = snap;
            ++generation_;
        }
        cv_.notify_all();
        Json reply = Json::object().set("handled", Json::boolean(handled)).set("revision", Json::integer(revision_))
                         .set("canvas", Json::string(c_.canvas_name));
        if (with_meta) reply.set("meta", std::move(meta));
        return reply;
    }

    // ---------------------------------------------------------------------------------------- client threads
    void accept_loop() {
        for (;;) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (closing_) return;
            }
            SOCKET conn = platform::accept_within(listener_, 200);
            if (conn == INVALID_SOCKET) continue;
            bool admit = false;
            {
                std::lock_guard<std::mutex> lk(mu_);
                admit = !closing_ && clients_ < kMaxClients;
                if (admit) { ++clients_; client_sockets_.insert(conn); }
            }
            if (!admit) {
                platform::send_all(conn, "REFUSED too many clients\n");
                platform::close(conn);
                continue;
            }
            std::thread([this, conn] { client(conn); }).detach();
        }
    }

    // Python's file.readline(4096): up to and including '\n', at most 4096 bytes; "" at the end of the stream.
    struct Reader {
        SOCKET s;
        std::string buf;
        bool eof = false, failed = false;
        bool fill() {
            char tmp[4096];
            const int n = platform::receive(s, tmp, sizeof(tmp));
            if (n > 0) { buf.append(tmp, size_t(n)); return true; }
            if (n == 0) eof = true; else failed = true;
            return false;
        }
        bool line(std::string& out) {
            for (;;) {
                const size_t nl = buf.find('\n');
                if (nl != std::string::npos && nl < 4096) { out = buf.substr(0, nl + 1); buf.erase(0, nl + 1); return true; }
                if (buf.size() >= 4096) { out = buf.substr(0, 4096); buf.erase(0, 4096); return true; }
                if (!fill()) { if (failed) return false; out = buf; buf.clear(); return true; }
            }
        }
        bool read(size_t n, std::string& out) {   // up to n bytes; fewer only at the end of the stream
            while (buf.size() < n) if (!fill()) { if (failed) return false; break; }
            out = buf.substr(0, std::min(n, buf.size()));
            buf.erase(0, out.size());
            return true;
        }
    };

    // Python int() of the text after "ACTION ": surrounding whitespace, an optional sign, digits with single underscores.
    static bool parse_size(const std::string& text, long long& out) {
        const char* ws = " \t\n\r\f\v";
        const size_t b = text.find_first_not_of(ws);
        if (b == std::string::npos) return false;
        std::string t = text.substr(b, text.find_last_not_of(ws) - b + 1);
        bool neg = false;
        if (!t.empty() && (t[0] == '+' || t[0] == '-')) { neg = t[0] == '-'; t.erase(0, 1); }
        if (t.empty() || t.front() == '_' || t.back() == '_') return false;
        long long v = 0;
        for (size_t k = 0; k < t.size(); ++k) {
            if (t[k] == '_') { if (t[k - 1] == '_') return false; continue; }
            if (!std::isdigit(static_cast<unsigned char>(t[k]))) return false;
            if (v > 1000000000000LL) return false;
            v = v * 10 + (t[k] - '0');
        }
        out = neg ? -v : v;
        return true;
    }

    // A build that has `view`, or one that tried it and failed, within 5 s. run_canvas returned whatever build came next:
    // a build already under way for another client's view (asked for before this one) answered "REFUSED unavailable"
    // for a view nobody had tried yet -- seen in both hosts, TOOLS right after GRAPH (2026-11-23). Here the view is asked
    // for again until a build has tried it.
    std::shared_ptr<Snap> snapshot(const std::string& view) {
        std::unique_lock<std::mutex> lk(mu_);
        if (snap_ && seconds_since(snap_->at) < kCoalesceSeconds && snap_->views.count(view)) return snap_;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(kStepWaitSeconds);
        for (;;) {
            want_views_.insert(view);
            const long long generation = generation_;
            wanted_.store(true);
            if (!cv_.wait_until(lk, deadline, [&] { return generation_ != generation || closing_; }) || closing_) return snap_;
            if (snap_ && (snap_->views.count(view) || snap_->attempted.count(view))) return snap_;
        }
    }

    bool closing() { std::lock_guard<std::mutex> lk(mu_); return closing_; }

    void client(SOCKET conn) {
        platform::receive_timeout(conn, kIdleSeconds);
        Reader reader{conn};
        bool authed = false, actions = false;
        static const std::map<std::string, std::string> verbs = {
            {"DIGEST", "digest"}, {"GRAPH", "graph"}, {"TOOLS", "tools"}, {"ACTIVITY", "activity"}, {"META", "meta"}};
        auto send = [&](const std::string& data) { return platform::send_all(conn, data); };
        while (!closing()) {
            std::string raw;
            if (!reader.line(raw) || raw.empty()) break;
            std::string line = raw;
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
            if (!authed) {
                const size_t s1 = line.find(' ');
                const size_t s2 = s1 == std::string::npos ? std::string::npos : line.find(' ', s1 + 1);
                const std::string verb = line.substr(0, s1), version = s2 == std::string::npos ? "" : line.substr(s1 + 1, s2 - s1 - 1);
                if (s2 == std::string::npos || verb != "HELLO" || (version != "nodus-console/1" && version != "nodus-console/2")) {
                    send("REFUSED bad hello\n");
                    break;
                }
                const std::string presented = line.substr(s2 + 1);
                unsigned diff = presented.size() == token_.size() ? 0u : 1u;   // compare_digest: constant time over the token
                for (size_t k = 0; k < token_.size(); ++k) diff |= unsigned(uint8_t(token_[k]) ^ uint8_t(k < presented.size() ? presented[k] : 0));
                if (diff) { send("REFUSED bad token\n"); break; }
                authed = true;
                actions = version == "nodus-console/2";
                if (!send("OK console " + c_.canvas_name + (actions ? " actions\n" : " read-only\n"))) break;
                continue;
            }
            if (line == "SNAPSHOT") {
                auto snap = snapshot("digest");
                if (!snap) { if (!send("REFUSED unavailable\n")) break; continue; }
                if (!send("CANVAS " + std::to_string(snap->revision) + " " + std::to_string(snap->data.size()) + "\n" + snap->data +
                          "\nSTATE " + snap->state + "\n")) break;
            } else if (line == "FRAME") {   // the rasterized canvas: "FRAME <revision> <width> <height> <n>\n" + n RGBA bytes
                auto snap = snapshot("frame");
                if (!snap || !snap->views.count("frame")) { if (!send("REFUSED unavailable\n")) break; continue; }
                const std::string& rgba = snap->views.at("frame");
                if (!send("FRAME " + std::to_string(snap->revision) + " " + std::to_string(snap->frame_width) + " " +
                          std::to_string(snap->frame_height) + " " + std::to_string(rgba.size()) + "\n" + rgba)) break;
            } else if (verbs.count(line)) {
                const std::string& view = verbs.at(line);
                auto snap = snapshot(view);
                if (!snap || !snap->views.count(view)) { if (!send("REFUSED unavailable\n")) break; continue; }
                const std::string& body = snap->views.at(view);
                if (!send("JSON " + line + " " + std::to_string(body.size()) + "\n" + body)) break;
            } else if (actions && line.rfind("ACTION ", 0) == 0) {
                long long size = 0;
                std::string body;
                Json request;
                if (!parse_size(line.substr(7), size) || size < 1 || size > 4096) { send("REFUSED invalid action\n"); break; }
                if (!reader.read(size_t(size), body)) break;   // the idle timeout ended the read: the connection closes
                if (!Parser(body).parse(request)) { send("REFUSED invalid action\n"); break; }
                auto a = std::make_shared<Action>();
                a->request = std::move(request);
                bool done = false;
                {
                    std::unique_lock<std::mutex> lk(mu_);
                    actions_.push_back(a);
                    wanted_.store(true);
                    done = cv_.wait_for(lk, std::chrono::seconds(kStepWaitSeconds), [&] { return a->done || closing_; }) && a->done;
                }
                if (!done) { send("REFUSED action outcome uncertain\n"); break; }
                const std::string reply = dumps_compact(a->reply);
                if (!send("JSON ACTION " + std::to_string(reply.size()) + "\n" + reply)) break;
            } else if (!send("REFUSED read-only\n")) {
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (client_sockets_.erase(conn)) platform::close(conn);
            --clients_;
        }
        cv_.notify_all();
    }

    const std::string token_;
    Context c_;
    std::filesystem::path tmp_;
    SOCKET listener_ = INVALID_SOCKET;
    std::thread accept_thread_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<bool> wanted_{false};
    std::set<std::string> want_views_;
    std::vector<std::shared_ptr<Action>> actions_;
    std::shared_ptr<Snap> snap_;
    long long generation_ = 0;
    int clients_ = 0;
    std::set<SOCKET> client_sockets_;
    bool closing_ = false;

    // stepping thread only
    std::string last_bytes_;
    long long revision_ = 0;
};

}  // namespace nodus::console
