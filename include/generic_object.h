// generic_object.h
//
// One C++ object type that any foreign class becomes.
//
// A class arriving from Python, Java, or C++ does not get compiled into a
// bespoke C++ type, and does not get flattened into an SSA class-object
// table. It arrives as a schema -- identity, a field layout, a method list --
// and this shell takes that shape on: fields become addressable slots,
// methods become entries in a table of callables that is rebound per class.
// Behaviourally it is the clone of its source; structurally it is one type.
//
// Nothing here infers anything. An object carries its own class description,
// so `object.call("forward", ...)` is a lookup in that object's own method
// table -- not a static resolution of the name `forward` against source. That
// distinction is the whole reason this layer exists: the moment dispatch
// depends on re-deriving a receiver's type from source text, objects have
// stopped being transportable.
//
// Method bodies are NOT held here. A method names an implementation the
// destination already owns -- a registered tool, a canonical operator, an
// imported artifact -- so behaviour is bound, never transported. Which
// backend group supplies it (inmemory, eigen, torch, onnx, a compiled
// artifact) is the binder's choice, and the same schema binds against any of
// them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nodus::oop {

// Mirrors the transportable schema (turing: src/compiler/oop_schema.py).
// Text-carried, like every other interchange format in this repo, so a
// producer needs nothing but the ability to write lines.
struct FieldSchema {
    std::string name;
    std::string type_name;      // canonical spelling, or "unknown"
    std::int32_t slot = -1;     // -1 => the destination assigns one
    bool readonly = false;
};

struct MethodSchema {
    std::string name;
    // What implements this method at the destination. A registered tool id
    // (which may be a canonical operator name), or an imported symbol.
    std::string body_reference;
    std::uint32_t parameter_count = 0;
    bool is_constructor = false;
};

struct ClassSchema {
    std::string identity;
    std::vector<FieldSchema> fields;
    std::vector<MethodSchema> methods;
    std::vector<std::string> bases;
    std::string origin_language;   // "python", "c++", "java", ...
};

// A field's runtime value. Deliberately a small closed set: the value types
// this ecosystem already moves across its own boundaries. A tensor field
// holds a handle, never an owned buffer, so a field costs 8 bytes and an
// object never becomes a place data secretly lives.
enum class SlotKind : std::uint8_t { Empty, Tensor, Scalar, Object };

struct SlotValue {
    SlotKind kind = SlotKind::Empty;
    std::uint64_t tensor = 0;   // AbstractTensorHandle::id
    double scalar = 0.0;
    std::uint64_t object = 0;   // instance id of another GenericObject
};

// The bound behaviour of one method: whatever the destination decided
// implements it. Arguments and results travel as slot values, which is the
// same currency the fields use.
using BoundMethod =
    std::function<bool(class GenericObject&, const std::vector<SlotValue>&,
                       SlotValue*, std::string*)>;

// The per-class binding: a schema plus the method table built for it once.
// Instances share it, so binding cost is paid per class, not per object --
// which is what makes instantiating many objects of one class cheap.
class ClassBinding {
public:
    explicit ClassBinding(ClassSchema schema);

    const ClassSchema& schema() const { return schema_; }
    const std::string& identity() const { return schema_.identity; }
    std::size_t field_count() const { return schema_.fields.size(); }

    // Slot index for a field name, or -1. Resolved once at binding time.
    std::int32_t slot_of(const std::string& field_name) const;

    // Install the implementation for a method. Returns false if the schema
    // declares no such method -- a binder cannot invent members.
    bool bind(const std::string& method_name, BoundMethod implementation);

    const BoundMethod* method(const std::string& method_name) const;

    // Method names the schema declares but nothing has bound yet. This is the
    // honest report a destination owes its caller: an object with unbound
    // methods is usable, and says exactly where it is not.
    std::vector<std::string> unbound_methods() const;

private:
    ClassSchema schema_;
    std::unordered_map<std::string, std::int32_t> slots_;
    std::unordered_map<std::string, BoundMethod> methods_;
};

// An instance: a reference to its class binding plus its own slot storage.
// This is the "mildly clever shell" -- one type, whose behaviour is entirely
// determined by the binding it points at.
class GenericObject {
public:
    explicit GenericObject(const ClassBinding& binding);

    const ClassBinding& binding() const { return *binding_; }
    const std::string& identity() const { return binding_->identity(); }

    SlotValue& at(std::int32_t slot) { return slots_[static_cast<std::size_t>(slot)]; }
    const SlotValue& at(std::int32_t slot) const {
        return slots_[static_cast<std::size_t>(slot)];
    }

    // Field access by name, for callers that have a name rather than a slot.
    // Returns nullptr when the class declares no such field -- never a
    // default-constructed slot, which would read as "the field exists and is
    // empty".
    SlotValue* field(const std::string& name);
    const SlotValue* field(const std::string& name) const;

    // Dispatch: a lookup in this object's own class binding. `error` receives
    // a named reason on failure (unknown method, unbound method, or the
    // implementation's own message).
    bool call(const std::string& method_name,
              const std::vector<SlotValue>& arguments,
              SlotValue* result,
              std::string* error);

private:
    const ClassBinding* binding_;
    std::vector<SlotValue> slots_;
};

// -- schema interchange (SCHEMA V1) ---------------------------------------
//
// One record per line, whitespace separated, matching the KIRTEXT/NODUSPKG
// idiom already used here:
//
//   schema 1
//   class Linear python
//   base Module
//   field W float32 0
//   field b float32 1 readonly
//   method forward abstract_tensor.matmul 1
//
// Parsing refuses a malformed document by line rather than guessing.
ClassSchema parse_class_schema(const std::string& text);
std::string serialize_class_schema(const ClassSchema& schema);

} // namespace nodus::oop
