#pragma once

#include "kpath_ids.h"
#include "kpath_shaper.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <unordered_map>
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

struct AtlasStorage final {
  std::vector<AtlasNode> nodes_;
  std::vector<AtlasEdge> edges_;
  std::vector<EdgeId> token_edge_store_;
  std::vector<TokenPath> tokens_;
  std::vector<Posting> postings_store_;
  std::vector<NodeMetadata> node_metadata_;
};

class Atlas final {
 public:
  Atlas() : storage_(std::make_shared<AtlasStorage>()) {}

  size_t node_count() const noexcept { return storage_ ? storage_->nodes_.size() : 0; }
  size_t edge_count() const noexcept { return storage_ ? storage_->edges_.size() : 0; }
  size_t token_count() const noexcept { return storage_ ? storage_->tokens_.size() : 0; }

  const AtlasNode& node(NodeId n) const { return storage_->nodes_.at(n.v); }
  const AtlasEdge& edge(EdgeId e) const { return storage_->edges_.at(e.v); }
  std::span<const EdgeId> token_edges(TokenId t) const;
  std::span<const Posting> edge_postings(EdgeId e) const;
  const NodeMetadata* node_metadata(NodeId n) const;

 private:
  friend class AtlasBuilder;
  explicit Atlas(std::shared_ptr<AtlasStorage> storage) : storage_(std::move(storage)) {}

  std::shared_ptr<AtlasStorage> storage_;
};

class AtlasBuilder final {
 public:
  AtlasBuilder() : storage_(std::make_shared<AtlasStorage>()) {}

  // O(1) view of the current, stateful atlas contents. The returned Atlas shares
  // storage with this builder and will observe subsequent mutations.
  Atlas snapshot() const { return Atlas(storage_); }

  NodeId add_node(AtlasNode n) {
    storage_->nodes_.push_back(n);
    return NodeId{static_cast<uint32_t>(storage_->nodes_.size() - 1)};
  }

  uint32_t add_node_metadata(NodeMetadata meta) {
    storage_->node_metadata_.push_back(std::move(meta));
    return static_cast<uint32_t>(storage_->node_metadata_.size() - 1);
  }

  EdgeId add_edge(AtlasEdge e) {
    e.postings_begin = static_cast<uint32_t>(storage_->postings_store_.size());
    e.postings_count = 0;
    storage_->edges_.push_back(e);
    return EdgeId{static_cast<uint32_t>(storage_->edges_.size() - 1)};
  }

  TokenId add_token(std::span<const EdgeId> edges) {
    return add_token_impl(edges);
  }

  // Dedupe helper: returns an existing token if an identical edge list has
  // already been registered, otherwise creates a new token.
  TokenId add_token_dedup(std::span<const EdgeId> edges) {
    if (edges.empty()) return add_token_impl(edges);

    const uint64_t h = hash_edges(edges);
    auto it = token_index_.find(h);
    if (it != token_index_.end()) {
      for (TokenId existing : it->second) {
        if (token_edges_equal(existing, edges)) return existing;
      }
    }

    TokenId created = add_token_impl(edges);
    token_index_[h].push_back(created);
    return created;
  }

  void add_posting(EdgeId e, Posting p) {
    if (e.v >= storage_->postings_store_.capacity()) storage_->postings_store_.reserve(e.v + 1);
    storage_->postings_store_.push_back(p);
    auto& edge = storage_->edges_.at(e.v);
    if (edge.postings_count == 0) edge.postings_begin = static_cast<uint32_t>(storage_->postings_store_.size() - 1);
    edge.postings_count++;
  }

  Atlas finalize() {
    return snapshot();
  }

 private:
  static uint64_t hash_edges(std::span<const EdgeId> edges) {
    // FNV-1a 64-bit
    uint64_t h = 14695981039346656037ull;
    for (const EdgeId e : edges) {
      uint32_t v = e.v;
      for (int i = 0; i < 4; ++i) {
        uint8_t b = static_cast<uint8_t>((v >> (i * 8)) & 0xFFu);
        h ^= static_cast<uint64_t>(b);
        h *= 1099511628211ull;
      }
    }
    // Include length to reduce collisions for different concatenations.
    h ^= static_cast<uint64_t>(edges.size());
    h *= 1099511628211ull;
    return h;
  }

  TokenId add_token_impl(std::span<const EdgeId> edges) {
    uint32_t begin = static_cast<uint32_t>(storage_->token_edge_store_.size());
    for (auto e : edges) storage_->token_edge_store_.push_back(e);
    TokenPath path{TokenId{static_cast<uint32_t>(storage_->tokens_.size())}, begin, static_cast<uint32_t>(edges.size())};
    storage_->tokens_.push_back(path);
    return path.id;
  }

  bool token_edges_equal(TokenId token, std::span<const EdgeId> edges) const {
    if (token.v >= storage_->tokens_.size()) return false;
    const auto& path = storage_->tokens_[token.v];
    if (path.edges_count != edges.size()) return false;
    const EdgeId* begin = storage_->token_edge_store_.data() + path.edges_begin;
    for (uint32_t i = 0; i < path.edges_count; ++i) {
      if (begin[i].v != edges[i].v) return false;
    }
    return true;
  }

  std::shared_ptr<AtlasStorage> storage_;

  // Edge-list identity map: hash(edges) -> candidate tokens.
  std::unordered_map<uint64_t, std::vector<TokenId>> token_index_;
};

inline std::span<const EdgeId> Atlas::token_edges(TokenId t) const {
  if (!storage_ || t.v >= storage_->tokens_.size()) return {};
  const auto& path = storage_->tokens_[t.v];
  return std::span<const EdgeId>(storage_->token_edge_store_.data() + path.edges_begin, path.edges_count);
}

inline std::span<const Posting> Atlas::edge_postings(EdgeId e) const {
  if (!storage_ || e.v >= storage_->edges_.size()) return {};
  const auto& edge = storage_->edges_[e.v];
  return std::span<const Posting>(storage_->postings_store_.data() + edge.postings_begin, edge.postings_count);
}

inline const NodeMetadata* Atlas::node_metadata(NodeId n) const {
  if (!storage_ || n.v >= storage_->nodes_.size()) return nullptr;
  const auto& node = storage_->nodes_[n.v];
  if (node.metadata_index == AtlasNode::kNoMetadata) return nullptr;
  if (node.metadata_index >= storage_->node_metadata_.size()) return nullptr;
  return &storage_->node_metadata_[node.metadata_index];
}

} // namespace nodus::tensors::kpath
