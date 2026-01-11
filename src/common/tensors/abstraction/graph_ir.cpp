#include "common/tensors/abstraction/graph_ir.h"

#include "common/tensors/abstraction/graph_sparse.h"

#include <cctype>
#include <charconv>
#include <cstdlib>
#include <sstream>

namespace nodus::tensors {

namespace {

struct Lexer final {
  std::string_view src;
  size_t i = 0;
  size_t line = 1;
  size_t col = 1;

  char peek() const { return (i < src.size()) ? src[i] : '\0'; }

  char take() {
    char ch = peek();
    if (ch == '\0') return ch;
    ++i;
    if (ch == '\n') {
      ++line;
      col = 1;
    } else {
      ++col;
    }
    return ch;
  }

  GraphIrSpan span() const { return GraphIrSpan{line, col}; }

  void skip_ws_and_comments() {
    for (;;) {
      while (std::isspace(static_cast<unsigned char>(peek()))) take();
      if (peek() == '#') {
        while (peek() != '\0' && peek() != '\n') take();
        continue;
      }
      if (peek() == '/' && i + 1 < src.size() && src[i + 1] == '/') {
        while (peek() != '\0' && peek() != '\n') take();
        continue;
      }
      break;
    }
  }

  GraphIrToken next(GraphIrError* out_error) {
    skip_ws_and_comments();

    GraphIrToken t;
    t.at = span();

    const char ch = peek();
    if (ch == '\0') {
      t.kind = GraphIrTokenKind::End;
      return t;
    }

    if (ch == '(') {
      take();
      t.kind = GraphIrTokenKind::LParen;
      return t;
    }
    if (ch == ')') {
      take();
      t.kind = GraphIrTokenKind::RParen;
      return t;
    }
    if (ch == ',') {
      take();
      t.kind = GraphIrTokenKind::Comma;
      return t;
    }
    if (ch == ';') {
      take();
      t.kind = GraphIrTokenKind::Semi;
      return t;
    }
    if (ch == '=') {
      take();
      t.kind = GraphIrTokenKind::Assign;
      return t;
    }

    if (ch == '"') {
      take();
      t.kind = GraphIrTokenKind::String;
      std::string s;
      while (peek() != '\0' && peek() != '"') {
        char c = take();
        if (c == '\\' && peek() != '\0') {
          // minimal escaping: \" and \\.
          const char n = take();
          if (n == '"' || n == '\\') s.push_back(n);
          else s.push_back(n);
          continue;
        }
        s.push_back(c);
      }
      if (peek() != '"') {
        if (out_error) *out_error = GraphIrError{t.at, "unterminated string"};
        return GraphIrToken{};
      }
      take();
      t.text = std::move(s);
      return t;
    }

    if (std::isalpha(static_cast<unsigned char>(ch)) || ch == '_') {
      t.kind = GraphIrTokenKind::Ident;
      std::string s;
      while (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_') {
        s.push_back(take());
      }
      t.text = std::move(s);
      return t;
    }

    // number: [-]?[0-9]+(.[0-9]+)?
    if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '-' || ch == '.') {
      t.kind = GraphIrTokenKind::Number;
      std::string s;
      if (ch == '-') s.push_back(take());
      bool saw_digit = false;
      while (std::isdigit(static_cast<unsigned char>(peek()))) {
        saw_digit = true;
        s.push_back(take());
      }
      if (peek() == '.') {
        s.push_back(take());
        while (std::isdigit(static_cast<unsigned char>(peek()))) {
          saw_digit = true;
          s.push_back(take());
        }
      }
      if (!saw_digit) {
        if (out_error) *out_error = GraphIrError{t.at, "invalid number"};
        return GraphIrToken{};
      }
      char* endp = nullptr;
      t.number = std::strtod(s.c_str(), &endp);
      return t;
    }

    if (out_error) {
      std::ostringstream oss;
      oss << "unexpected character '" << ch << "'";
      *out_error = GraphIrError{t.at, oss.str()};
    }
    return GraphIrToken{};
  }
};

struct Parser final {
  Lexer lex;
  GraphIrToken cur;
  GraphIrError* err = nullptr;
  bool ok = true;

  explicit Parser(std::string_view src, GraphIrError* out_error) : err(out_error) {
    lex.src = src;
    cur = lex.next(err);
    ok = (cur.kind != GraphIrTokenKind::End) || true;
  }

  void bump() {
    if (!ok) return;
    cur = lex.next(err);
    if (err && !err->message.empty()) ok = false;
  }

  bool expect(GraphIrTokenKind kind, const char* msg) {
    if (!ok) return false;
    if (cur.kind == kind) return true;
    if (err) *err = GraphIrError{cur.at, msg};
    ok = false;
    return false;
  }

  GraphIrExpr parse_expr() {
    if (!ok) return GraphIrExpr{};

    if (cur.kind == GraphIrTokenKind::Number) {
      GraphIrNumber n;
      n.value = cur.number;
      n.at = cur.at;
      bump();
      return GraphIrExpr{n};
    }
    if (cur.kind == GraphIrTokenKind::String) {
      GraphIrString s;
      s.value = cur.text;
      s.at = cur.at;
      bump();
      return GraphIrExpr{s};
    }
    if (cur.kind == GraphIrTokenKind::Ident) {
      const std::string name = cur.text;
      const GraphIrSpan at = cur.at;
      bump();
      if (cur.kind == GraphIrTokenKind::LParen) {
        bump();
        GraphIrCall c;
        c.op = name;
        c.at = at;
        if (cur.kind != GraphIrTokenKind::RParen) {
          for (;;) {
            c.args.push_back(parse_expr());
            if (!ok) return GraphIrExpr{};
            if (cur.kind == GraphIrTokenKind::Comma) {
              bump();
              continue;
            }
            break;
          }
        }
        if (!expect(GraphIrTokenKind::RParen, "expected ')'") ) return GraphIrExpr{};
        bump();
        return GraphIrExpr{c};
      }
      GraphIrIdent id;
      id.name = name;
      id.at = at;
      return GraphIrExpr{id};
    }

    if (err) *err = GraphIrError{cur.at, "expected expression"};
    ok = false;
    return GraphIrExpr{};
  }

  GraphIrStmt parse_stmt() {
    GraphIrStmt s;
    s.at = cur.at;

    // assignment: Ident '=' expr
    if (cur.kind == GraphIrTokenKind::Ident) {
      GraphIrToken look = cur;
      // take ident
      const std::string name = cur.text;
      const GraphIrSpan at = cur.at;
      bump();

      if (cur.kind == GraphIrTokenKind::Assign) {
        bump();
        s.assign = name;
        s.expr = parse_expr();
      } else {
        // rewind by constructing expr as either call or ident.
        // We already consumed the ident; rebuild a minimal expr by simulating what parse_expr would have done.
        if (cur.kind == GraphIrTokenKind::LParen) {
          bump();
          GraphIrCall c;
          c.op = name;
          c.at = at;
          if (cur.kind != GraphIrTokenKind::RParen) {
            for (;;) {
              c.args.push_back(parse_expr());
              if (!ok) break;
              if (cur.kind == GraphIrTokenKind::Comma) {
                bump();
                continue;
              }
              break;
            }
          }
          expect(GraphIrTokenKind::RParen, "expected ')'");
          bump();
          s.expr = GraphIrExpr{c};
        } else {
          s.expr = GraphIrExpr{GraphIrIdent{name, at}};
        }
      }
      return s;
    }

    s.expr = parse_expr();
    return s;
  }
};

static std::optional<uint32_t> as_u32(const GraphIrValue& v) {
  if (const auto* p = std::get_if<uint32_t>(&v)) return *p;
  if (const auto* d = std::get_if<double>(&v)) {
    if (*d >= 0.0 && *d <= 4294967295.0) return static_cast<uint32_t>(*d);
  }
  return std::nullopt;
}

static std::optional<double> as_f64(const GraphIrValue& v) {
  if (const auto* p = std::get_if<double>(&v)) return *p;
  if (const auto* u = std::get_if<uint32_t>(&v)) return static_cast<double>(*u);
  if (const auto* b = std::get_if<bool>(&v)) return *b ? 1.0 : 0.0;
  return std::nullopt;
}

static std::optional<std::string> as_string(const GraphIrValue& v) {
  if (const auto* p = std::get_if<std::string>(&v)) return *p;
  return std::nullopt;
}

static bool eval_expr(const GraphIrExpr& e,
                      const GraphIrOperatorSet& ops,
                      GraphIrContext& ctx,
                      GraphIrValue& out,
                      GraphIrError* out_error);

static bool eval_call(const GraphIrCall& c,
                      const GraphIrOperatorSet& ops,
                      GraphIrContext& ctx,
                      GraphIrValue& out,
                      GraphIrError* out_error) {
  std::vector<GraphIrValue> args;
  args.reserve(c.args.size());

  for (const auto& a : c.args) {
    GraphIrValue av;
    if (!eval_expr(a, ops, ctx, av, out_error)) return false;
    args.push_back(std::move(av));
  }

  std::string err;
  if (!ops.call(c.op, args, out, ctx, err)) {
    if (out_error) *out_error = GraphIrError{c.at, err.empty() ? "operator call failed" : err};
    return false;
  }
  return true;
}

static bool eval_expr(const GraphIrExpr& e,
                      const GraphIrOperatorSet& ops,
                      GraphIrContext& ctx,
                      GraphIrValue& out,
                      GraphIrError* out_error) {
  if (std::holds_alternative<GraphIrNumber>(e.v)) {
    out = std::get<GraphIrNumber>(e.v).value;
    return true;
  }
  if (std::holds_alternative<GraphIrString>(e.v)) {
    out = std::get<GraphIrString>(e.v).value;
    return true;
  }
  if (std::holds_alternative<GraphIrIdent>(e.v)) {
    const auto& id = std::get<GraphIrIdent>(e.v);
    auto it = ctx.symbols.find(id.name);
    if (it == ctx.symbols.end()) {
      if (out_error) *out_error = GraphIrError{id.at, "unknown identifier: " + id.name};
      return false;
    }
    out = it->second;
    return true;
  }
  if (std::holds_alternative<GraphIrCall>(e.v)) {
    return eval_call(std::get<GraphIrCall>(e.v), ops, ctx, out, out_error);
  }
  if (out_error) *out_error = GraphIrError{GraphIrSpan{}, "invalid expression"};
  return false;
}

} // namespace

uint32_t GraphEditBuilder::alloc_node_id() {
  return graph_make_id(GraphObjectKind::Node, next_node_++);
}

uint32_t GraphEditBuilder::alloc_port_id() {
  return graph_make_id(GraphObjectKind::Port, next_port_++);
}

uint32_t GraphEditBuilder::alloc_edge_id() {
  return graph_make_id(GraphObjectKind::Edge, next_edge_++);
}

uint32_t GraphEditBuilder::add_node(std::string_view kind) {
  const uint32_t id = alloc_node_id();
  GraphEdit e;
  e.kind = GraphEditKind::AddNode;
  e.a = id;
  e.key = std::string(kind);
  edits_.push_back(std::move(e));
  return id;
}

uint32_t GraphEditBuilder::add_edge(std::string_view kind) {
  const uint32_t id = alloc_edge_id();
  GraphEdit e;
  e.kind = GraphEditKind::AddEdge;
  e.a = id;
  e.key = std::string(kind);
  edits_.push_back(std::move(e));
  return id;
}

uint32_t GraphEditBuilder::add_port(uint32_t node_id, uint32_t flags, uint32_t type_bits) {
  const uint32_t id = alloc_port_id();

  {
    GraphEdit e;
    e.kind = GraphEditKind::AddPort;
    e.a = node_id;
    e.b = id;
    e.value = static_cast<uint32_t>(flags);
    edits_.push_back(std::move(e));
  }

  if (type_bits != 0) {
    GraphEdit e;
    e.kind = GraphEditKind::SetAttr;
    e.a = id;
    e.key = "port.type_bits";
    e.value = static_cast<uint32_t>(type_bits);
    edits_.push_back(std::move(e));
  }

  return id;
}

void GraphEditBuilder::connect(uint32_t src_port, uint32_t dst_port) {
  GraphEdit e;
  e.kind = GraphEditKind::Connect;
  e.a = src_port;
  e.b = dst_port;
  edits_.push_back(std::move(e));
}

void GraphEditBuilder::set_attr(uint32_t obj_id, std::string_view key, GraphIrValue value) {
  GraphEdit e;
  e.kind = GraphEditKind::SetAttr;
  e.a = obj_id;
  e.key = std::string(key);
  e.value = std::move(value);
  edits_.push_back(std::move(e));
}

void GraphIrOperatorSet::add(GraphIrOpSpec spec, Handler handler) {
  // Avoid unspecified evaluation order between map key construction and moving `spec`.
  const std::string key = spec.name;
  ops_[key] = Entry{std::move(spec), std::move(handler)};
}

void GraphIrOperatorSet::add_all(const GraphIrOperatorSet& other, bool overwrite) {
  for (const auto& kv : other.ops_) {
    if (!overwrite) {
      if (ops_.find(kv.first) != ops_.end()) continue;
    }
    ops_[kv.first] = kv.second;
  }
}

const GraphIrOpSpec* GraphIrOperatorSet::find(std::string_view name) const {
  auto it = ops_.find(std::string(name));
  if (it == ops_.end()) return nullptr;
  return &it->second.spec;
}

bool GraphIrOperatorSet::call(std::string_view name,
                              const std::vector<GraphIrValue>& args,
                              GraphIrValue& out,
                              GraphIrContext& ctx,
                              std::string& out_error) const {
  auto it = ops_.find(std::string(name));
  if (it == ops_.end()) {
    out_error = "unknown operator: " + std::string(name);
    return false;
  }

  const auto& entry = it->second;
  const uint32_t argc = static_cast<uint32_t>(args.size());
  if (argc < entry.spec.min_args || argc > entry.spec.max_args) {
    std::ostringstream oss;
    oss << "operator '" << entry.spec.name << "' expects " << entry.spec.min_args;
    if (entry.spec.max_args == 0xFFFFFFFFu) {
      oss << "+ args";
    } else if (entry.spec.max_args != entry.spec.min_args) {
      oss << ".." << entry.spec.max_args;
    }
    oss << ", got " << argc;
    out_error = oss.str();
    return false;
  }

  return entry.handler(entry.spec, args, out, ctx, out_error);
}

std::vector<GraphIrOpSpec> GraphIrOperatorSet::list_specs() const {
  std::vector<GraphIrOpSpec> out;
  out.reserve(ops_.size());
  for (const auto& kv : ops_) out.push_back(kv.second.spec);
  return out;
}

bool graph_ir_parse(std::string_view src, GraphIrProgram& out_program, GraphIrError* out_error) {
  out_program = GraphIrProgram{};
  Parser p(src, out_error);

  while (p.cur.kind != GraphIrTokenKind::End && p.ok) {
    GraphIrStmt s = p.parse_stmt();
    if (!p.ok) break;

    // optional semicolons
    if (p.cur.kind == GraphIrTokenKind::Semi) p.bump();

    out_program.statements.push_back(std::move(s));

    // tolerate extra semicolons
    while (p.cur.kind == GraphIrTokenKind::Semi) p.bump();
  }

  return p.ok;
}

bool graph_ir_eval(const GraphIrProgram& program,
                   const GraphIrOperatorSet& ops,
                   GraphIrContext& ctx,
                   GraphIrError* out_error) {
  for (const auto& s : program.statements) {
    GraphIrValue v;
    if (!eval_expr(s.expr, ops, ctx, v, out_error)) return false;
    if (s.assign) ctx.symbols[*s.assign] = v;
  }
  return true;
}

GraphIrOperatorSet make_core_graph_ir_ops() {
  GraphIrOperatorSet set;

  set.add(GraphIrOpSpec{"node", 1, 1, "node(kind: string) -> u32"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "node(): ctx.edits is null";
              return false;
            }
            auto kind = as_string(args[0]);
            if (!kind) {
              err = "node(): kind must be a string";
              return false;
            }
            out = ctx.edits->add_node(*kind);
            return true;
          });

  set.add(GraphIrOpSpec{"edge", 1, 1, "edge(kind: string) -> u32"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "edge(): ctx.edits is null";
              return false;
            }
            auto kind = as_string(args[0]);
            if (!kind) {
              err = "edge(): kind must be a string";
              return false;
            }
            out = ctx.edits->add_edge(*kind);
            return true;
          });

  set.add(GraphIrOpSpec{"port", 3, 3, "port(node_id: u32, flags: u32, type_bits: u32) -> u32"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "port(): ctx.edits is null";
              return false;
            }
            auto node_id = as_u32(args[0]);
            auto flags = as_u32(args[1]);
            auto type_bits = as_u32(args[2]);
            if (!node_id || !flags || !type_bits) {
              err = "port(): expected (u32,u32,u32)";
              return false;
            }
            out = ctx.edits->add_port(*node_id, *flags, *type_bits);
            return true;
          });

  set.add(GraphIrOpSpec{"connect", 2, 2, "connect(src_port: u32, dst_port: u32) -> void"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "connect(): ctx.edits is null";
              return false;
            }
            auto a = as_u32(args[0]);
            auto b = as_u32(args[1]);
            if (!a || !b) {
              err = "connect(): expected (u32,u32)";
              return false;
            }
            ctx.edits->connect(*a, *b);
            out = std::monostate{};
            return true;
          });

  set.add(GraphIrOpSpec{"attr", 3, 3, "attr(obj_id: u32, key: string, value: scalar) -> void"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "attr(): ctx.edits is null";
              return false;
            }
            auto obj = as_u32(args[0]);
            auto key = as_string(args[1]);
            if (!obj || !key) {
              err = "attr(): expected (u32,string,scalar)";
              return false;
            }
            ctx.edits->set_attr(*obj, *key, args[2]);
            out = std::monostate{};
            return true;
          });

  return set;
}

} // namespace nodus::tensors
