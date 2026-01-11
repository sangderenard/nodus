#include "common/tensors/abstraction/kpath/kpath_program.h"
#include "common/tensors/abstraction/kpath/kpath_shaper.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <filesystem>

namespace fs = std::filesystem;
using namespace nodus::tensors::kpath;

struct Options {
  std::string text = "kpath";
  std::string text_file;
  std::string font_path;
  std::string outline_dir;
  size_t outline_limit = 4;
  float point_size = 18.0f;
  bool outline_all = false;
  bool help = false;
};

void print_help(const char* program) {
  std::cout << "Usage: " << program << " [options]\n"
            << "Options:\n"
            << "  --text \"utf8 text\"     text to shape and inspect (default: \"kpath\")\n"
            << "  --text-file PATH            read text from a file instead of the command line\n"
            << "  --font PATH                 explicitly pick a font file (takes precedence over environment vars)\n"
            << "  --font-size VALUE           font size in points (default 18)\n"
            << "  --outline-dir DIR           dump per-glyph outlines into the given directory\n"
            << "  --outline-limit N           print outlines for at most N glyphs (0 disables, default 4)\n"
            << "  --outline-all               ignore the outline limit and print every glyph that was shaped\n"
            << "  --help, -h                  show this help message\n"
            << "Environment:\n"
            << "  KPATH_SHAPER_FONT           highest priority font that will be reshaped\n"
            << "  KPATH_TEST_FONT             fallback font used by the test suite\n"
            << "Default font candidates: C:/Windows/Fonts/arial.ttf, DejaVuSans.ttf, seguisb.ttf\n";
}

bool parse_options(int argc, char** argv, Options& opts) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      opts.help = true;
      return true;
    }
    if (arg == "--text" && i + 1 < argc) {
      opts.text = argv[++i];
      continue;
    }
    if (arg == "--text-file" && i + 1 < argc) {
      opts.text_file = argv[++i];
      continue;
    }
    if (arg == "--font" && i + 1 < argc) {
      opts.font_path = argv[++i];
      continue;
    }
    if (arg == "--font-size" && i + 1 < argc) {
      try {
        size_t consumed = 0;
        opts.point_size = std::stof(argv[++i], &consumed);
        if (consumed == 0) {
          std::cerr << "Invalid font-size value\n";
          return false;
        }
      } catch (const std::exception&) {
        std::cerr << "Invalid font-size value\n";
        return false;
      }
      continue;
    }
    if (arg == "--outline-dir" && i + 1 < argc) {
      opts.outline_dir = argv[++i];
      continue;
    }
    if (arg == "--outline-limit" && i + 1 < argc) {
      try {
        size_t value = std::stoul(argv[++i]);
        opts.outline_limit = value;
      } catch (const std::exception&) {
        std::cerr << "Invalid outline-limit value\n";
        return false;
      }
      continue;
    }
    if (arg == "--outline-all") {
      opts.outline_all = true;
      continue;
    }
    std::cerr << "Unknown option: " << arg << "\n";
    return false;
  }
  return true;
}

std::string read_text_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Failed to open text file: " + path);
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

std::optional<std::string> locate_font(const Options& opts) {
  if (!opts.font_path.empty()) {
    if (fs::exists(opts.font_path)) {
      return opts.font_path;
    }
    std::cerr << "Requested font not found: " << opts.font_path << "\n";
    return std::nullopt;
  }

  const char* env_candidates[] = {"KPATH_SHAPER_FONT", "KPATH_TEST_FONT"};
  for (const char* env_name : env_candidates) {
    if (const char* env_value = std::getenv(env_name)) {
      fs::path candidate(env_value);
      if (fs::exists(candidate)) {
        std::cout << "Found font via " << env_name << ": " << candidate << "\n";
        return candidate.string();
      }
    }
  }

  const std::vector<fs::path> default_fonts{
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/DejaVuSans.ttf",
    "C:/Windows/Fonts/seguisb.ttf"
  };
  for (const auto& candidate : default_fonts) {
    if (fs::exists(candidate)) {
      return candidate.string();
    }
  }
  return std::nullopt;
}

const char* outline_op_name(OutlineOp op) {
  switch (op) {
    case OutlineOp::MoveTo:
      return "MoveTo";
    case OutlineOp::LineTo:
      return "LineTo";
    case OutlineOp::QuadTo:
      return "QuadTo";
    case OutlineOp::CubicTo:
      return "CubicTo";
    case OutlineOp::Arc:
      return "Arc";
    case OutlineOp::Sin:
      return "Sin";
    case OutlineOp::Close:
      return "Close";
    default:
      return "Unknown";
  }
}

std::string describe_segment(size_t index, const OutlineSegment& seg) {
  std::ostringstream ss;
  ss << "    [" << index << "] " << outline_op_name(seg.op);
  ss << " start=(" << std::fixed << std::setprecision(3) << seg.x1 << "," << seg.y1 << ")";
  ss << " mid=(" << seg.x2 << "," << seg.y2 << ")";
  ss << " end=(" << seg.x3 << "," << seg.y3 << ")";
  ss << "";
  return ss.str();
}

bool dump_outline_to_file(const GlyphOutline& outline,
                          const fs::path& directory,
                          size_t cluster_index,
                          size_t glyph_index) {
  try {
    if (!directory.empty()) {
      fs::create_directories(directory);
    }
    std::ostringstream filename;
    filename << "glyph_" << cluster_index << "_" << glyph_index << "_" << outline.glyph_id << ".txt";
    fs::path target = directory / filename.str();
    std::ofstream out(target);
    if (!out) {
      std::cerr << "Failed to write outline file: " << target << "\n";
      return false;
    }
    out << "GlyphOutline glyph_id=" << outline.glyph_id << "\n";
    for (size_t i = 0; i < outline.segments.size(); ++i) {
      out << describe_segment(i, outline.segments[i]) << "\n";
    }
    return true;
  } catch (const std::exception& err) {
    std::cerr << "Error dumping outline to " << directory << ": " << err.what() << "\n";
    return false;
  }
}

int main(int argc, char** argv) {
  Options opts;
  if (!parse_options(argc, argv, opts)) {
    print_help(argv[0]);
    return 1;
  }
  if (opts.help) {
    print_help(argv[0]);
    return 0;
  }

  std::string text;
  try {
    if (!opts.text_file.empty()) {
      text = read_text_file(opts.text_file);
    } else {
      text = opts.text;
    }
  } catch (const std::exception& err) {
    std::cerr << "Failed to read text: " << err.what() << "\n";
    return 1;
  }

  if (text.empty()) {
    std::cerr << "Input text is empty" << std::endl;
    return 1;
  }

  auto font_path = locate_font(opts);
  if (!font_path) {
    std::cerr << "No font available for shaping (set KPATH_SHAPER_FONT or install one of the candidate fonts).\n";
    return 1;
  }
  std::cout << "Using font: " << *font_path << " size=" << opts.point_size << "\n";

  Shaper shaper;
  if (!shaper.load_font(*font_path, opts.point_size)) {
    std::cerr << "Unable to load font for shaping" << std::endl;
    return 1;
  }

  auto sequence = codepoints_from_utf8(text);
  if (sequence.codepoints.empty()) {
    std::cerr << "Decoded text contains no codepoints" << std::endl;
    return 1;
  }

  std::vector<Cluster> clusters;
  if (!shaper.shape(sequence, clusters)) {
    std::cerr << "Shaping failed" << std::endl;
    return 1;
  }

  if (clusters.empty()) {
    std::cerr << "Shaper returned no clusters" << std::endl;
    return 1;
  }

  size_t total_glyphs = 0;
  for (const auto& cluster : clusters) {
    total_glyphs += cluster.glyph_ids.size();
  }

  std::cout << "Processed text: " << std::quoted(text) << "\n";
  std::cout << "Clusters: " << clusters.size() << " glyphs: " << total_glyphs << "\n";

  for (size_t cluster_index = 0; cluster_index < clusters.size(); ++cluster_index) {
    const Cluster& cluster = clusters[cluster_index];
    std::ostringstream glyphs;
    for (size_t j = 0; j < cluster.glyph_ids.size(); ++j) {
      if (j) glyphs << ",";
      glyphs << cluster.glyph_ids[j];
    }
    std::cout << "  cluster " << cluster_index
              << " start=" << cluster.start
              << " count=" << cluster.count
              << " glyphs=" << glyphs.str()
              << " advance=(" << std::fixed << std::setprecision(3)
              << cluster.advance_x << "," << cluster.advance_y << ")"
              << std::defaultfloat << "\n";
  }

  size_t outlines_to_emit = opts.outline_all ? std::numeric_limits<size_t>::max() : opts.outline_limit;
  size_t outlines_emitted = 0;

  for (size_t cluster_index = 0; cluster_index < clusters.size(); ++cluster_index) {
    const Cluster& cluster = clusters[cluster_index];
    for (size_t glyph_idx = 0; glyph_idx < cluster.glyph_ids.size(); ++glyph_idx) {
      if (!opts.outline_all && outlines_emitted >= outlines_to_emit) {
        break;
      }
      uint32_t glyph_id = cluster.glyph_ids[glyph_idx];
      GlyphOutline outline;
      if (!shaper.extract_outline(glyph_id, outline)) {
        std::cerr << "Failed to extract outline for glyph " << glyph_id << "\n";
        continue;
      }
      std::cout << "Glyph outline " << glyph_id << " (cluster " << cluster_index << ") has "
                << outline.segments.size() << " segments" << std::endl;
      for (size_t seg_idx = 0; seg_idx < outline.segments.size(); ++seg_idx) {
        std::cout << describe_segment(seg_idx, outline.segments[seg_idx]) << "\n";
      }
      if (!opts.outline_dir.empty()) {
        dump_outline_to_file(outline, opts.outline_dir, cluster_index, glyph_idx);
      }
      ++outlines_emitted;
    }
    if (!opts.outline_all && outlines_emitted >= outlines_to_emit) {
      break;
    }
  }

  std::cout << "Outlined " << outlines_emitted << " glyph(s)" << std::endl;
  if (!opts.outline_dir.empty()) {
    std::cout << "Outlines stored (one file per reported glyph) in " << opts.outline_dir << std::endl;
  }
  return 0;
}
