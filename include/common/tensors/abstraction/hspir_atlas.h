#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace nodus::tensors::hspir {

using TokenId = uint32_t;
using NodeId = uint32_t;
using EdgeId = uint32_t;

struct AtlasNode {
  NodeId id = 0;
  uint32_t payload = 0;
  uint8_t kind = 0;
};

struct AtlasEdge {
  EdgeId id = 0;
  NodeId src = 0;
  NodeId dst = 0;
  uint32_t label = 0;
};

struct Posting {
  TokenId token = 0;
  uint32_t offset = 0;
  uint32_t span = 0;
};

struct TokenPath {
  TokenId token = 0;
  std::vector<EdgeId> edges;
};

class Atlas {
 public:
  Atlas() = default;
  NodeId add_node(const AtlasNode& node);
  EdgeId add_edge(const AtlasEdge& edge);
  void append_posting(EdgeId edge, Posting posting);
  void consume_token(TokenPath path);
  std::span<const EdgeId> token_edges(TokenId token) const;
  std::span<const Posting> postings_for_edge(EdgeId edge) const;

 private:
  std::vector<AtlasNode> nodes_;
  std::vector<AtlasEdge> edges_;
  std::vector<std::vector<Posting>> postings_;
  std::vector<TokenPath> tokens_;
};

inline NodeId Atlas::add_node(const AtlasNode& node) {
  AtlasNode stored = node;
  stored.id = static_cast<NodeId>(nodes_.size());
  nodes_.push_back(stored);
  return stored.id;
}

inline EdgeId Atlas::add_edge(const AtlasEdge& edge) {
  AtlasEdge stored = edge;
  stored.id = static_cast<EdgeId>(edges_.size());
  edges_.push_back(stored);
  if (stored.id >= postings_.size()) postings_.resize(stored.id + 1);
  return stored.id;
}

inline void Atlas::append_posting(EdgeId edge, Posting posting) {
  if (edge >= postings_.size()) postings_.resize(edge + 1);
  postings_[edge].push_back(posting);
}

inline void Atlas::consume_token(TokenPath path) {
  tokens_.push_back(path);
}

inline std::span<const EdgeId> Atlas::token_edges(TokenId token) const {
  if (token >= tokens_.size()) return {};
  return tokens_[token].edges;
}

inline std::span<const Posting> Atlas::postings_for_edge(EdgeId edge) const {
  if (edge >= postings_.size()) return {};
  return postings_[edge];
}

}  // namespace nodus::tensors::hspir
