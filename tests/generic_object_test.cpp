// generic_object_test.cpp
//
// A foreign class becomes a working C++ object without being compiled into a
// bespoke type and without passing through an SSA class table. The schema is
// the only thing that travels; behaviour is bound at the destination, and the
// same schema binds against different backends.
#include "generic_object.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace nodus::oop;

namespace {

int failures = 0;

bool check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "[OOP] FAILED: " << what << "\n";
        ++failures;
    }
    return condition;
}

// Document mode: a schema produced by a frontend (turing's ingestion capture,
// or any other producer) arrives as a file, and the SAME shell takes it on.
// Nothing in here knows which class is coming -- that is the proof: a class
// this binary has never heard of parses, binds, instantiates, and dispatches.
int run_schema_document(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!check(stream.good(), "open schema document: " + path)) return 1;
    std::ostringstream buffer;
    buffer << stream.rdbuf();

    ClassSchema schema;
    try {
        schema = parse_class_schema(buffer.str());
    } catch (const std::exception& parse_error) {
        check(false, std::string("parse schema document: ") + parse_error.what());
        return 1;
    }
    check(!schema.identity.empty(), "document declares a class identity");
    check(!schema.fields.empty() || !schema.methods.empty(),
          "document carries members, not just a name");

    ClassBinding binding(schema);
    // Bind every declared method to an implementation that records which
    // body reference the destination would dispatch into. The reference is
    // the destination's own tool/graph id -- behaviour is bound, not shipped.
    int dispatched = 0;
    std::string last_reference;
    for (const MethodSchema& method : schema.methods) {
        const std::string reference = method.body_reference;
        check(binding.bind(method.name,
                           [&dispatched, &last_reference, reference](
                               GenericObject&, const std::vector<SlotValue>&,
                               SlotValue* result, std::string*) {
                               ++dispatched;
                               last_reference = reference;
                               if (result) {
                                   result->kind = SlotKind::Scalar;
                                   result->scalar = 1.0;
                               }
                               return true;
                           }),
              "bind declared method " + method.name);
    }
    check(binding.unbound_methods().empty(), "every declared method bound");

    // Two instances, per-instance state on the first declared field.
    GenericObject first(binding), second(binding);
    if (!schema.fields.empty()) {
        const std::string& field_name = schema.fields.front().name;
        SlotValue* a = first.field(field_name);
        SlotValue* b = second.field(field_name);
        check(a != nullptr && b != nullptr, "declared field addressable: " + field_name);
        if (a && b) {
            a->kind = SlotKind::Scalar; a->scalar = 3.0;
            b->kind = SlotKind::Scalar; b->scalar = 10.0;
            check(first.field(field_name)->scalar == 3.0 &&
                  second.field(field_name)->scalar == 10.0,
                  "instances hold their own state");
        }
    }

    // Every declared method dispatches through the object's own table.
    SlotValue out{};
    std::string error;
    for (const MethodSchema& method : schema.methods) {
        check(first.call(method.name, {}, &out, &error),
              "dispatch " + method.name + ": " + error);
    }
    check(dispatched == static_cast<int>(schema.methods.size()),
          "each method ran exactly once");

    if (failures) {
        std::cerr << "[OOP] " << failures << " failure(s) in document mode\n";
        return 1;
    }
    std::cout << "[OOP] OK: class '" << schema.identity << "' ("
              << schema.origin_language << ") from " << path << " -- "
              << schema.fields.size() << " field(s), " << schema.methods.size()
              << " method(s) -- lives as a C++ object; last dispatch bound to '"
              << last_reference << "'\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1) return run_schema_document(argv[1]);
    // A class as it arrives: text, no compilation, no class table.
    const std::string document =
        "schema 1\n"
        "class Linear python\n"
        "base Module\n"
        "field W float32 0\n"
        "field b float32 1\n"
        "field frozen float32 2 readonly\n"
        "method forward abstract_tensor.matmul 1\n"
        "method reset abstract_tensor.neg 0\n";

    const ClassSchema schema = parse_class_schema(document);
    check(schema.identity == "Linear", "identity parsed");
    check(schema.origin_language == "python", "origin parsed");
    check(schema.bases.size() == 1 && schema.bases[0] == "Module", "base parsed");
    check(schema.fields.size() == 3, "three fields");
    check(schema.fields[2].readonly, "readonly flag parsed");
    check(schema.methods.size() == 2, "two methods");
    check(schema.methods[0].body_reference == "abstract_tensor.matmul",
          "method names the implementation the destination owns");

    // Round trip: the text form is stable, so a schema survives being stored
    // and re-read without drifting.
    const ClassSchema reparsed = parse_class_schema(serialize_class_schema(schema));
    check(reparsed.identity == schema.identity, "round trip identity");
    check(reparsed.fields.size() == schema.fields.size(), "round trip fields");
    check(reparsed.methods.size() == schema.methods.size(), "round trip methods");

    ClassBinding binding(schema);
    check(binding.slot_of("W") == 0 && binding.slot_of("b") == 1, "slots indexed");
    check(binding.slot_of("nope") == -1, "unknown field has no slot");
    check(binding.unbound_methods().size() == 2, "nothing bound yet");

    // A method calling into whatever the destination decided implements it.
    // Two bindings of the SAME schema, standing for two backend groups: the
    // schema is backend-agnostic, the binding is where a target is chosen.
    int inmemory_calls = 0, eigen_calls = 0;
    binding.bind("forward", [&](GenericObject& self,
                                const std::vector<SlotValue>& args,
                                SlotValue* result, std::string* error) {
        ++inmemory_calls;
        const SlotValue* weight = self.field("W");
        if (weight == nullptr) {
            if (error) *error = "W missing";
            return false;
        }
        if (args.size() != 1) {
            if (error) *error = "forward takes one argument";
            return false;
        }
        if (result) {
            result->kind = SlotKind::Scalar;
            result->scalar = weight->scalar * args[0].scalar;
        }
        return true;
    });
    check(binding.unbound_methods().size() == 1, "one method still unbound");

    // Refusing to bind a member the class never declared.
    check(!binding.bind("invented", [](GenericObject&, const std::vector<SlotValue>&,
                                       SlotValue*, std::string*) { return true; }),
          "a binder cannot invent members");

    // Instances: many objects, one binding.
    GenericObject a(binding), b(binding);
    a.field("W")->kind = SlotKind::Scalar;
    a.field("W")->scalar = 3.0;
    b.field("W")->kind = SlotKind::Scalar;
    b.field("W")->scalar = 10.0;
    check(a.field("W")->scalar == 3.0 && b.field("W")->scalar == 10.0,
          "instances hold their own state");
    check(&a.binding() == &b.binding(), "instances share one class binding");

    // Dispatch is a lookup on the object, not a resolution of a name in source.
    SlotValue out{};
    std::string error;
    SlotValue argument{};
    argument.kind = SlotKind::Scalar;
    argument.scalar = 4.0;
    check(a.call("forward", {argument}, &out, &error), "dispatch succeeds: " + error);
    check(out.scalar == 12.0, "a.forward used a's own field");
    check(b.call("forward", {argument}, &out, &error), "second instance dispatches");
    check(out.scalar == 40.0, "b.forward used b's own field");
    check(inmemory_calls == 2, "the bound implementation ran twice");

    // Faults name themselves, and the two failure kinds are distinguished.
    check(!a.call("reset", {}, &out, &error), "declared-but-unbound refuses");
    check(error.find("not bound") != std::string::npos,
          "unbound method says so: " + error);
    check(!a.call("nonexistent", {}, &out, &error), "undeclared method refuses");
    check(error.find("declares no method") != std::string::npos,
          "undeclared method says so: " + error);

    // The same schema bound against a different target: only the binding
    // changes, the class and its instances are untouched.
    ClassBinding eigen_binding(schema);
    eigen_binding.bind("forward", [&](GenericObject& self,
                                      const std::vector<SlotValue>& args,
                                      SlotValue* result, std::string*) {
        ++eigen_calls;
        if (result) {
            result->kind = SlotKind::Scalar;
            result->scalar = self.field("W")->scalar * args[0].scalar + 1.0;
        }
        return true;
    });
    GenericObject c(eigen_binding);
    c.field("W")->kind = SlotKind::Scalar;
    c.field("W")->scalar = 3.0;
    check(c.call("forward", {argument}, &out, &error), "second backend dispatches");
    check(out.scalar == 13.0 && eigen_calls == 1,
          "the same schema ran against a different implementation");

    if (failures) {
        std::cerr << "[OOP] " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "[OOP] OK: a foreign class parsed, round-tripped, bound, and "
                 "dispatched as a C++ object -- one shell type, no per-class "
                 "codegen, no SSA class table, two backends from one schema\n";
    return 0;
}
