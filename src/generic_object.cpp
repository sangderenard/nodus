#include "generic_object.h"

#include <sstream>
#include <stdexcept>

namespace nodus::oop {

namespace {

[[noreturn]] void fail(std::size_t line_number, const std::string& reason) {
    throw std::runtime_error(
        "schema line " + std::to_string(line_number) + ": " + reason);
}

} // namespace

ClassBinding::ClassBinding(ClassSchema schema) : schema_(std::move(schema)) {
    // Assign any slot the producer left to the destination, then index every
    // field once so instances never search by name at access time.
    std::int32_t next = 0;
    for (auto& member : schema_.fields) {
        if (member.slot < 0) {
            member.slot = next;
        }
        next = (member.slot >= next) ? member.slot + 1 : next;
    }
    for (const auto& member : schema_.fields) {
        slots_[member.name] = member.slot;
    }
}

std::int32_t ClassBinding::slot_of(const std::string& field_name) const {
    auto found = slots_.find(field_name);
    return found == slots_.end() ? -1 : found->second;
}

bool ClassBinding::bind(const std::string& method_name,
                        BoundMethod implementation) {
    // A binder may only implement what the class declares. Accepting an
    // unknown name would let a destination grow members the source never had,
    // which is precisely the divergence this layer exists to prevent.
    for (const auto& member : schema_.methods) {
        if (member.name == method_name) {
            methods_[method_name] = std::move(implementation);
            return true;
        }
    }
    return false;
}

const BoundMethod* ClassBinding::method(const std::string& method_name) const {
    auto found = methods_.find(method_name);
    return found == methods_.end() ? nullptr : &found->second;
}

std::vector<std::string> ClassBinding::unbound_methods() const {
    std::vector<std::string> pending;
    for (const auto& member : schema_.methods) {
        if (methods_.find(member.name) == methods_.end()) {
            pending.push_back(member.name);
        }
    }
    return pending;
}

GenericObject::GenericObject(const ClassBinding& binding) : binding_(&binding) {
    std::size_t extent = 0;
    for (const auto& member : binding.schema().fields) {
        extent = std::max(extent, static_cast<std::size_t>(member.slot) + 1);
    }
    slots_.assign(extent, SlotValue{});
}

SlotValue* GenericObject::field(const std::string& name) {
    const std::int32_t slot = binding_->slot_of(name);
    if (slot < 0 || static_cast<std::size_t>(slot) >= slots_.size()) return nullptr;
    return &slots_[static_cast<std::size_t>(slot)];
}

const SlotValue* GenericObject::field(const std::string& name) const {
    const std::int32_t slot = binding_->slot_of(name);
    if (slot < 0 || static_cast<std::size_t>(slot) >= slots_.size()) return nullptr;
    return &slots_[static_cast<std::size_t>(slot)];
}

bool GenericObject::call(const std::string& method_name,
                         const std::vector<SlotValue>& arguments,
                         SlotValue* result,
                         std::string* error) {
    const BoundMethod* implementation = binding_->method(method_name);
    if (implementation == nullptr) {
        if (error) {
            // Distinguish "this class has no such method" from "declared but
            // nothing implements it yet" -- different faults, different fixes.
            bool declared = false;
            for (const auto& member : binding_->schema().methods) {
                declared = declared || member.name == method_name;
            }
            *error = declared
                ? ("method '" + method_name + "' is declared by " +
                   binding_->identity() + " but not bound to an implementation")
                : (binding_->identity() + " declares no method '" +
                   method_name + "'");
        }
        return false;
    }
    return (*implementation)(*this, arguments, result, error);
}

ClassSchema parse_class_schema(const std::string& text) {
    ClassSchema schema;
    std::istringstream stream(text);
    std::string line;
    std::size_t line_number = 0;
    bool saw_header = false;

    while (std::getline(stream, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::istringstream fields(line);
        std::string record;
        fields >> record;

        if (!saw_header) {
            std::string version;
            if (record != "schema" || !(fields >> version) || version != "1") {
                fail(line_number, "expected 'schema 1' header");
            }
            saw_header = true;
            continue;
        }

        if (record == "class") {
            if (!(fields >> schema.identity)) {
                fail(line_number, "class needs an identity");
            }
            fields >> schema.origin_language; // optional
        } else if (record == "base") {
            std::string base;
            if (!(fields >> base)) fail(line_number, "base needs a name");
            schema.bases.push_back(base);
        } else if (record == "field") {
            FieldSchema member;
            if (!(fields >> member.name >> member.type_name)) {
                fail(line_number, "field needs a name and a type");
            }
            std::int32_t slot = -1;
            if (fields >> slot) member.slot = slot;
            std::string flag;
            while (fields >> flag) {
                if (flag == "readonly") member.readonly = true;
                else fail(line_number, "unknown field flag " + flag);
            }
            schema.fields.push_back(std::move(member));
        } else if (record == "method") {
            MethodSchema member;
            if (!(fields >> member.name >> member.body_reference)) {
                fail(line_number, "method needs a name and a body reference");
            }
            std::uint32_t count = 0;
            if (fields >> count) member.parameter_count = count;
            std::string flag;
            while (fields >> flag) {
                if (flag == "constructor") member.is_constructor = true;
                else fail(line_number, "unknown method flag " + flag);
            }
            schema.methods.push_back(std::move(member));
        } else {
            fail(line_number, "unknown record " + record);
        }
    }
    if (!saw_header) fail(0, "empty document");
    if (schema.identity.empty()) fail(line_number, "no class record");
    return schema;
}

std::string serialize_class_schema(const ClassSchema& schema) {
    std::ostringstream out;
    out << "schema 1\n";
    out << "class " << schema.identity;
    if (!schema.origin_language.empty()) out << " " << schema.origin_language;
    out << "\n";
    for (const auto& base : schema.bases) out << "base " << base << "\n";
    for (const auto& member : schema.fields) {
        out << "field " << member.name << " " << member.type_name << " "
            << member.slot;
        if (member.readonly) out << " readonly";
        out << "\n";
    }
    for (const auto& member : schema.methods) {
        out << "method " << member.name << " " << member.body_reference << " "
            << member.parameter_count;
        if (member.is_constructor) out << " constructor";
        out << "\n";
    }
    return out.str();
}

} // namespace nodus::oop
