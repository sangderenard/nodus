#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace nodus::tensors {

// A tiny, embeddable "function-call" IR that parses simple statements like:
//   a = pt(0, 0);
//   b = lerp(a, pt(10, 0), 0.5);
//   contour(a, b, "open");
//
// The intent is to keep syntax boring and easy to embed, while letting callers
// supply an operator set that defines how calls mutate some target graph/state.

struct GraphIrSpan final {
  size_t line = 1;
  size_t col = 1;
};

struct GraphIrError final {
  GraphIrSpan at;
  std::string message;
};

enum class GraphIrTokenKind : uint8_t {
  End,
  Ident,
  Number,
  String,
  LParen,
  RParen,
  Comma,
  Semi,
  Assign,
};

struct GraphIrToken final {
  GraphIrTokenKind kind = GraphIrTokenKind::End;
  GraphIrSpan at;
  std::string text;  // for Ident/String
  double number = 0; // for Number
};

struct GraphIrExpr;

struct GraphIrCall final {
  std::string op;
  std::vector<GraphIrExpr> args;
  GraphIrSpan at;
};

struct GraphIrIdent final {
  std::string name;
  GraphIrSpan at;
};

struct GraphIrNumber final {
  double value = 0.0;
  GraphIrSpan at;
};

struct GraphIrString final {
  std::string value;
  GraphIrSpan at;
};

struct GraphIrExpr final {
  std::variant<GraphIrCall, GraphIrIdent, GraphIrNumber, GraphIrString> v;
};

struct GraphIrStmt final {
  // If set, assigns the expression result into symbols[name].
  std::optional<std::string> assign;
  GraphIrExpr expr;
  GraphIrSpan at;
};

struct GraphIrProgram final {
  std::vector<GraphIrStmt> statements;
};

// Values passed into and returned from operators.
using GraphIrValue = std::variant<std::monostate, bool, uint32_t, double, std::string>;

// A generic "edit log" that operator sets can emit. This intentionally does NOT
// force materialization into `SparseGraph` yet; it is meant as a stable, easy
// intermediate for future lowering.

enum class GraphEditKind : uint8_t {
  AddNode,
  AddPort,
  AddEdge,
  Connect,
  SetAttr,
};

struct GraphEdit final {
  GraphEditKind kind = GraphEditKind::AddNode;

  // Primary ids (meaning depends on kind).
  uint32_t a = 0;
  uint32_t b = 0;

  // Optional payload.
  std::string key;
  GraphIrValue value;
};

class GraphEditBuilder final {
 public:
  uint32_t alloc_node_id();
  uint32_t alloc_port_id();
  uint32_t alloc_edge_id();

  const std::vector<GraphEdit>& edits() const { return edits_; }

  // Convenience helpers that both allocate ids and emit edits.
  uint32_t add_node(std::string_view kind);
  uint32_t add_edge(std::string_view kind);
  uint32_t add_port(uint32_t node_id, uint32_t flags, uint32_t type_bits);
  void connect(uint32_t src_port, uint32_t dst_port);
  void set_attr(uint32_t obj_id, std::string_view key, GraphIrValue value);

 private:
  std::vector<GraphEdit> edits_;
  uint32_t next_node_ = 1;
  uint32_t next_port_ = 1;
  uint32_t next_edge_ = 1;
};

struct GraphIrOpSpec final {
  std::string name;
  uint32_t min_args = 0;
  uint32_t max_args = 0; // inclusive, 0xFFFFFFFF for variadic
  std::string doc;
};

struct GraphIrContext final {
  std::unordered_map<std::string, GraphIrValue> symbols;
  GraphEditBuilder* edits = nullptr; // optional
  void* user = nullptr;              // optional (operator-set specific state)
};

class GraphIrOperatorSet final {
 public:
  using Handler = std::function<bool(const GraphIrOpSpec& spec,
                                     const std::vector<GraphIrValue>& args,
                                     GraphIrValue& out,
                                     GraphIrContext& ctx,
                                     std::string& out_error)>;

  void add(GraphIrOpSpec spec, Handler handler);
  // Merge another operator set into this one.
  // If overwrite is false, existing operators keep precedence.
  void add_all(const GraphIrOperatorSet& other, bool overwrite = true);
  const GraphIrOpSpec* find(std::string_view name) const;
  bool call(std::string_view name,
            const std::vector<GraphIrValue>& args,
            GraphIrValue& out,
            GraphIrContext& ctx,
            std::string& out_error) const;

  std::vector<GraphIrOpSpec> list_specs() const;

 private:
  struct Entry {
    GraphIrOpSpec spec;
    Handler handler;
  };
  std::unordered_map<std::string, Entry> ops_;
};

// Parse and evaluate.
bool graph_ir_parse(std::string_view src, GraphIrProgram& out_program, GraphIrError* out_error);
bool graph_ir_eval(const GraphIrProgram& program,
                   const GraphIrOperatorSet& ops,
                   GraphIrContext& ctx,
                   GraphIrError* out_error);

// A minimal, generic operator table that emits graph edits into ctx.edits.
// - node(kind: string) -> u32 node_id
// - edge(kind: string) -> u32 edge_id
// - port(node_id: u32, flags: u32, type_bits: u32) -> u32 port_id
// - connect(src_port: u32, dst_port: u32) -> void
// - attr(obj_id: u32, key: string, value: any-scalar) -> void
GraphIrOperatorSet make_core_graph_ir_ops();

} // namespace nodus::tensors
