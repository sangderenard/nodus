#include "module_library_actualizer.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: modlib_actualize <lib> <root>\n";
        return 2;
    }
    return gp_module_library_actualize_from_file(argv[1], argv[2]) ? 0 : 1;
}
