#pragma once

#include "kpath_ids.h"
#include "kpath_shaper.h"

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace nodus::tensors::kpath {

enum class AtlasNodeKind : uint8_t { Codepoint, Cluster, Glyph, Symbol, Opcode, ArmatureJoint };

struct NodeMetadata final {
  GlyphOutline outline;
  RotDir winding{RotDir::Zero};
};

struct AtlasNode final {
  AtlasNodeKind kind{};
  uint32_t payload{};
  uint32_t metadata_index{std::numeric_limits<uint32_t>::max()};
  static constexpr uint32_t kNoMetadata = std::numeric_limits<uint32_t>::max();
};

struct Posting final { TokenId token{}; uint32_t offset{}; uint32_t span{}; };

struct AtlasEdge final {
  NodeId src{};
  NodeId dst{};
  uint32_t label{};
  float advance_x{};
  float advance_y{};
  float offset_x{};
  float offset_y{};
  uint32_t cluster_id{};
  uint32_t postings_begin{};
  uint32_t postings_count{};
};

struct TokenPath final {
  TokenId id{};
  uint32_t edges_begin{};
  uint32_t edges_count{};
};

class Atlas final {
 public:
  const AtlasNode& node(NodeId n) const { return nodes_.at(n.v); }
  const AtlasEdge& edge(EdgeId e) const { return edges_.at(e.v); }
  std::span<const EdgeId> token_edges(TokenId t) const;
  std::span<const Posting> edge_postings(EdgeId e) const;
  const NodeMetadata* node_metadata(NodeId n) const;

 private:
  friend class AtlasBuilder;
  std::vector<AtlasNode> nodes_;
  std::vector<AtlasEdge> edges_;
  std::vector<EdgeId> token_edge_store_;
  std::vector<TokenPath> tokens_;
  std::vector<Posting> postings_store_;
  std::vector<NodeMetadata> node_metadata_;
};

class AtlasBuilder final {
 public:
  NodeId add_node(AtlasNode n) {
    n.payload = static_cast<uint32_t>(nodes_.size());
    nodes_.push_back(n);
    return NodeId{static_cast<uint32_t>(nodes_.size() - 1)};
  }

  uint32_t add_node_metadata(NodeMetadata meta) {
    node_metadata_.push_back(std::move(meta));
    return static_cast<uint32_t>(node_metadata_.size() - 1);
  }

  EdgeId add_edge(AtlasEdge e) {
    e.postings_begin = static_cast<uint32_t>(postings_store_.size());
    e.postings_count = 0;
    edges_.push_back(e);
    return EdgeId{static_cast<uint32_t>(edges_.size() - 1)};
  }

  TokenId add_token(std::span<const EdgeId> edges) {
    uint32_t begin = static_cast<uint32_t>(token_edge_store_.size());
    for (auto e : edges) token_edge_store_.push_back(e);
    TokenPath path{TokenId{static_cast<uint32_t>(tokens_.size())}, begin, static_cast<uint32_t>(edges.size())};
    tokens_.push_back(path);
    return path.id;
  }

  void add_posting(EdgeId e, Posting p) {
    if (e.v >= postings_store_.capacity()) postings_store_.reserve(e.v + 1);
    postings_store_.push_back(p);
    auto& edge = edges_.at(e.v);
    if (edge.postings_count == 0) edge.postings_begin = static_cast<uint32_t>(postings_store_.size() - 1);
    edge.postings_count++;
  }

  Atlas finalize() {
    Atlas atlas;
    atlas.nodes_ = std::move(nodes_);
    atlas.edges_ = std::move(edges_);
    atlas.tokens_ = std::move(tokens_);
    atlas.token_edge_store_ = std::move(token_edge_store_);
    atlas.postings_store_ = std::move(postings_store_);
    atlas.node_metadata_ = std::move(node_metadata_);
    return atlas;
  }

 private:
  std::vector<AtlasNode> nodes_;
  std::vector<AtlasEdge> edges_;
  std::vector<EdgeId> token_edge_store_;
  std::vector<TokenPath> tokens_;
  std::vector<Posting> postings_store_;
  std::vector<NodeMetadata> node_metadata_;
};

inline std::span<const EdgeId> Atlas::token_edges(TokenId t) const {
  if (t.v >= tokens_.size()) return {};
  const auto& path = tokens_[t.v];
  return std::span<const EdgeId>(token_edge_store_.data() + path.edges_begin, path.edges_count);
}

inline std::span<const Posting> Atlas::edge_postings(EdgeId e) const {
  if (e.v >= edges_.size()) return {};
  const auto& edge = edges_[e.v];
  return std::span<const Posting>(postings_store_.data() + edge.postings_begin, edge.postings_count);
}

inline const NodeMetadata* Atlas::node_metadata(NodeId n) const {
  if (n.v >= nodes_.size()) return nullptr;
  const auto& node = nodes_[n.v];
  if (node.metadata_index == AtlasNode::kNoMetadata) return nullptr;
  if (node.metadata_index >= node_metadata_.size()) return nullptr;
  return &node_metadata_[node.metadata_index];
}

} // namespace nodus::tensors::kpath
