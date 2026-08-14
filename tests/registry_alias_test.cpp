// registry_alias_test -- E13 resolution semantics (boundary error register).
//
// Canvas ROWS and definition modules name bare catalog ids; PluginLoader
// registers versioned unique ids. The alias layer connects them under a
// stated policy; this test pins every clause of that policy:
//   1. an alias resolves find()/create() to its target entry;
//   2. a real entry always beats an alias of the same name (in-image
//      reference implementations keep priority over loaded witnesses);
//   3. aliases to unknown targets are refused (one-hop rule);
//   4. re-registering an alias re-points it (latest load wins);
//   5. unregistering a target drops aliases that pointed at it -- no
//      dangling names that quietly stop meaning anything.
#include "tool_api.h"
#include "tool_registry.h"

#include <iostream>
#include <string>

namespace {

int g_failures = 0;

bool check(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "[ALIAS] FAILED: " << what << "\n";
        ++g_failures;
    }
    return condition;
}

class StubTool final : public ITool {
public:
    explicit StubTool(const char* id) : id_(id) {}
    const char* id_cstr() const noexcept override { return id_.c_str(); }
    const char* name_cstr() const noexcept override { return id_.c_str(); }
    ToolCaps caps() const override { return static_cast<ToolCaps>(0); }
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
private:
    std::string id_;
};

ToolRegistry::Entry make_entry(const char* id) {
    ToolRegistry::Entry e;
    e.id = id;
    e.name = id;
    e.factory = [id]() {
        return std::unique_ptr<ITool, std::function<void(ITool*)>>(
            new StubTool(id), [](ITool* p) { delete p; });
    };
    return e;
}

} // namespace

int main() {
    ToolRegistry& reg = tool_registry_global();

    // 1) Alias resolves to its target.
    check(reg.register_tool(make_entry("aliastest.real_v1")), "register real_v1");
    check(reg.register_alias("aliastest.bare", "aliastest.real_v1"),
          "register alias bare -> real_v1");
    {
        auto tool = reg.create("aliastest.bare");
        check(tool != nullptr, "create through alias");
        check(tool && std::string(tool->id_cstr()) == "aliastest.real_v1",
              "alias creates the target implementation");
        check(reg.resolve("aliastest.bare") == "aliastest.real_v1",
              "resolve reports the target id");
    }

    // 2) A real entry beats an alias of the same name.
    check(reg.register_tool(make_entry("aliastest.owned")), "register owned");
    check(!reg.register_alias("aliastest.owned", "aliastest.real_v1"),
          "alias shadowing a real entry is refused");
    {
        auto tool = reg.create("aliastest.owned");
        check(tool && std::string(tool->id_cstr()) == "aliastest.owned",
              "real entry still creates itself");
    }

    // 3) One-hop rule: unknown target refused.
    check(!reg.register_alias("aliastest.dangling", "aliastest.no_such_target"),
          "alias to unknown target is refused");
    check(reg.create("aliastest.dangling") == nullptr, "refused alias resolves to nothing");

    // 4) Latest wins: re-point to a second version.
    check(reg.register_tool(make_entry("aliastest.real_v2")), "register real_v2");
    check(reg.register_alias("aliastest.bare", "aliastest.real_v2"),
          "re-point alias to real_v2");
    {
        auto tool = reg.create("aliastest.bare");
        check(tool && std::string(tool->id_cstr()) == "aliastest.real_v2",
              "alias now creates the latest version");
        auto pinned = reg.create("aliastest.real_v1");
        check(pinned && std::string(pinned->id_cstr()) == "aliastest.real_v1",
              "earlier version stays reachable by unique id");
    }

    // 5) Unregistering the target drops the alias.
    check(reg.unregister_tool("aliastest.real_v2"), "unregister real_v2");
    check(reg.create("aliastest.bare") == nullptr,
          "alias dropped with its target (no dangling name)");
    check(reg.resolve("aliastest.bare").empty(), "resolve reports unknown");

    // cleanup
    reg.unregister_tool("aliastest.real_v1");
    reg.unregister_tool("aliastest.owned");

    if (g_failures) {
        std::cerr << "[ALIAS] " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "[ALIAS] OK: alias policy holds (target resolution, real-entry "
                 "priority, one-hop, latest-wins, drop-on-unregister)\n";
    return 0;
}
