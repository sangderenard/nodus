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

}  // namespace nodus::tensors::hspir
