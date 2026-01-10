#pragma once

#include "common/tensors/abstraction/hspir_atlas.h"

#include <unordered_map>

namespace nodus::tensors::hspir {

class AtlasBuilder {
 public:
  AtlasBuilder() = default;

  NodeId add_node(const AtlasNode& node) { return atlas_.add_node(node); }

  EdgeId add_edge(const AtlasEdge& edge) {
    auto id = atlas_.add_edge(edge);
    postings_[id] = {};
    return id;
  }

  void append_posting(EdgeId edge, Posting posting) {
    postings_[edge].push_back(posting);
    atlas_.append_posting(edge, posting);
  }

  TokenId consume_token(const TokenPath& path) {
    tokens_.push_back(path);
    atlas_.consume_token(path);
    return path.token;
  }

  Atlas finalize() {
    return std::move(atlas_);
  }

 private:
  Atlas atlas_;
  std::unordered_map<EdgeId, std::vector<Posting>> postings_;
  std::vector<TokenPath> tokens_;
};

}  // namespace nodus::tensors::hspir
