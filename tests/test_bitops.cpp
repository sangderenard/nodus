#include "bitops.h"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace nodus::bitops;

// Example: Gray code table builder (optional)
std::vector<uint32_t> build_gray_table(uint32_t width) {
    std::vector<uint32_t> table(1u << width);
    for (uint32_t i = 0; i < (1u << width); ++i) {
        table[i] = BitOps::int_to_gray(i);
    }
    return table;
}

// Example: Validation function for bitwise ops
bool validate_bitops(uint32_t bit_width) {
    uint32_t max_val = (1u << bit_width);
    for (uint32_t x = 0; x < max_val; ++x) {
        for (uint32_t y = 0; y < max_val; ++y) {
            if ((BitOps::bit_add(x, y) & ((1u << bit_width) - 1)) != ((x + y) & ((1u << bit_width) - 1))) return false;
            if ((BitOps::bit_sub(x, y) & ((1u << bit_width) - 1)) != ((x - y) & ((1u << bit_width) - 1))) return false;
            if ((BitOps::bit_and(x, y)) != (x & y)) return false;
            if ((BitOps::bit_or(x, y)) != (x | y)) return false;
            if ((BitOps::bit_xor(x, y)) != (x ^ y)) return false;
            if (y != 0 && (BitOps::bit_div(x, y)) != (x / y)) return false;
            if (y != 0 && (BitOps::bit_mod(x, y)) != (x % y)) return false;
        }
        if (BitOps::bit_not(x) != ~x) return false;
        for (uint32_t s = 0; s < bit_width; ++s) {
            if (BitOps::bit_shift_left(x, s) != (x << s)) return false;
            if (BitOps::bit_shift_right(x, s) != (x >> s)) return false;
        }
        if (BitOps::gray_to_int(BitOps::int_to_gray(x)) != x) return false;
    }
    return true;
}

int main() {
    // Test bitops validation for several bit widths
    for (uint32_t width : {1, 2, 4, 8, 16, 32}) {
        bool ok = validate_bitops(width);
        std::cout << "BitOps validation for width " << width << ": " << (ok ? "PASS" : "FAIL") << std::endl;
        assert(ok);
    }
    std::cout << "All BitOps tests passed.\n";
    return 0;
}
