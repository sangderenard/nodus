#pragma once

#include "common/tensors/abstraction/hspir_schema.h"

#include <vector>

namespace nodus::tensors::hspir {

class MetricSchemaBuilder {
 public:
  explicit MetricSchemaBuilder(uint32_t id) : schema_(id) {}

  MetricSchemaBuilder& add_channel(ChannelDef def) {
    schema_.add_channel(std::move(def));
    return *this;
  }

  MetricSchema finalize() { return std::move(schema_); }

 private:
  MetricSchema schema_;
};

}  // namespace nodus::tensors::hspir
