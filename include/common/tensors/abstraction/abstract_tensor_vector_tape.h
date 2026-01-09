#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace nodus::tensors {

enum class FontVectorOp : uint8_t {
    MoveTo = 0,
    LineTo,
    QuadTo,
    CubicTo,
    ClosePath,
};

inline constexpr uint32_t kVectorInvalidIndex = std::numeric_limits<uint32_t>::max();

struct VectorPoint2 {
    float x = 0.0f;
    float y = 0.0f;
};

// Pieces are the smallest drawable instruction units.
struct VectorPiece {
    FontVectorOp op = FontVectorOp::MoveTo;
    uint32_t origin = kVectorInvalidIndex;
    uint32_t destination = kVectorInvalidIndex;
    uint32_t control0 = kVectorInvalidIndex;
    uint32_t control1 = kVectorInvalidIndex;
};

// Items group pieces; typically a glyph, contour, or logical run.
struct VectorItem {
    uint32_t piece_start = 0;
    uint32_t piece_count = 0;
    uint32_t span_id = 0;
};

// Segments group items; typically a whole string or higher-level section.
struct VectorSegment {
    uint32_t item_start = 0;
    uint32_t item_count = 0;
    uint32_t flags = 0;
};

// Sparse, table-oriented tape of vector instructions for abstract tensor workflows.
struct ParametricVectorTape {
    std::vector<VectorSegment> segments;
    std::vector<VectorItem> items;
    std::vector<VectorPiece> pieces;
    std::vector<VectorPoint2> points;

    uint32_t append_point(float x, float y) {
        points.push_back({x, y});
        return static_cast<uint32_t>(points.size() - 1);
    }

    uint32_t append_piece(FontVectorOp op,
                          uint32_t origin,
                          uint32_t destination,
                          uint32_t control0 = kVectorInvalidIndex,
                          uint32_t control1 = kVectorInvalidIndex) {
        VectorPiece piece;
        piece.op = op;
        piece.origin = origin;
        piece.destination = destination;
        piece.control0 = control0;
        piece.control1 = control1;
        pieces.push_back(piece);
        return static_cast<uint32_t>(pieces.size() - 1);
    }

    uint32_t append_item(uint32_t piece_start, uint32_t piece_count, uint32_t span_id = 0) {
        items.push_back({piece_start, piece_count, span_id});
        return static_cast<uint32_t>(items.size() - 1);
    }

    uint32_t append_segment(uint32_t item_start, uint32_t item_count, uint32_t flags = 0) {
        segments.push_back({item_start, item_count, flags});
        return static_cast<uint32_t>(segments.size() - 1);
    }
};

} // namespace nodus::tensors
