#pragma once
// Process-owned ContractNegotiator instances keyed by caller-supplied prefix,
// registered as `<prefix>.market` / `<prefix>.worker` rows in
// tool_registry_global() so a saved canvas naming those plugin ids comes up
// live in any image linking canvas_tables_static. Engines, hosts and the
// result transport are bound later by whatever participant connects.
#include "contract_negotiator_tool.h"
#include <cstddef>
#include <string>
namespace nodus::contracts {
// Creates and registers on first call; returns the existing instance after.
// Throws std::runtime_error naming the prefix if registration fails.
ContractNegotiator& acquire(const std::string& prefix, size_t capacity);
ContractNegotiator* find(const std::string& prefix);
// Unregisters both tool ids, destroys the negotiator; false if unknown.
// No bound row instance may execute after release.
bool release(const std::string& prefix);
}
