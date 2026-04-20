#pragma once

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

namespace ds_progress {

inline std::string path_basename(const std::string& p) {
  auto pos = p.find_last_of("/\\");
  if (pos == std::string::npos) return p;
  return p.substr(pos + 1);
}

inline void render_row_progress(int64_t done, int64_t total, const std::string& file_tag,
                                const std::string& line_prefix = {}) {
  constexpr int kBarW = 40;
  std::cerr << "\r\033[2K";
  if (!line_prefix.empty()) std::cerr << line_prefix;
  std::cerr << "[";
  if (total > 0) {
    int filled = static_cast<int>(kBarW * done / total);
    if (filled > kBarW) filled = kBarW;
    int pct = static_cast<int>(100 * done / total);
    if (pct > 100) pct = 100;
    for (int i = 0; i < kBarW; ++i) std::cerr << (i < filled ? '#' : '.');
    std::cerr << "] " << std::setw(3) << pct << "%  " << file_tag << "  rows " << done << "/" << total;
  } else {
    for (int i = 0; i < kBarW; ++i) std::cerr << '.';
    std::cerr << "   " << file_tag << "  rows " << done;
  }
  std::cerr << std::flush;
}

inline void finish_progress_line() { std::cerr << "\n"; }

inline void render_phase_progress(int step_1based, int total_steps, const std::string& detail,
                                  const std::string& line_prefix = {}) {
  constexpr int kBarW = 40;
  std::cerr << "\r\033[2K";
  if (!line_prefix.empty()) std::cerr << line_prefix;
  std::cerr << "[";
  if (total_steps > 0) {
    int filled = static_cast<int>(kBarW * step_1based / total_steps);
    if (filled > kBarW) filled = kBarW;
    int pct = static_cast<int>(100 * step_1based / total_steps);
    if (pct > 100) pct = 100;
    for (int i = 0; i < kBarW; ++i) std::cerr << (i < filled ? '#' : '.');
    std::cerr << "] " << std::setw(3) << pct << "%  (" << step_1based << "/" << total_steps << ")  " << detail;
  } else {
    std::cerr << detail;
  }
  std::cerr << std::flush;
}

}  
