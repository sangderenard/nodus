#include "generic_object.h"
#include <fstream>
#include <iostream>
#include <sstream>
int main(int argc, char** argv) {
    std::ifstream in(argv[1]);
    std::stringstream ss; ss << in.rdbuf();
    const auto s = nodus::oop::parse_class_schema(ss.str());
    std::cout << "C++ parsed: " << s.identity << " origin=" << s.origin_language
              << " bases=" << s.bases.size() << " fields=" << s.fields.size()
              << " methods=" << s.methods.size() << "\n";
    for (const auto& f : s.fields)
        std::cout << "   field " << f.name << " " << f.type_name << " slot " << f.slot << "\n";
    for (const auto& m : s.methods)
        std::cout << "   method " << m.name << " -> " << m.body_reference
                  << " arity " << m.parameter_count << "\n";
    nodus::oop::ClassBinding binding(s);
    std::cout << "bound class " << binding.identity() << ", W slot="
              << binding.slot_of("W") << ", unbound methods="
              << binding.unbound_methods().size() << "\n";
    nodus::oop::GenericObject obj(binding);
    std::cout << "instantiated an object of a class authored in Python\n";
    return 0;
}
