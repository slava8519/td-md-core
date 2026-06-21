#include "tdmd/cli/dashboard.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

// M7 dashboard renderer (see dashboard.hpp). A pure formatter: every method
// turns a finished Snapshot into text, with zero side effects and no terminal
// or thread dependency. render_plain is the canonical, escape-free form the
// unit test pins; render_ansi wraps it with cursor control + color for a TTY.
namespace tdmd::cli {

namespace {

// 8 levels — the sparkline alphabet (▁▂▃▄▅▆▇█), ordered low..high.
const char* kSpark[8] = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};

// SGR helpers — no-ops when color is off (non-TTY); kept tiny on purpose.
std::string sgr(bool on, const char* code) {
  return on ? (std::string("\033[") + code + "m") : std::string();
}
const char* kReset = "\033[0m";

std::string fmt(const char* f, double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), f, v);
  return std::string(buf);
}
std::string fmt_l(const char* f, long v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), f, v);
  return std::string(buf);
}

}  // namespace

std::string Dashboard::sparkline(const std::vector<double>& v) {
  if (v.empty()) return std::string();
  double lo = v[0], hi = v[0];
  for (double x : v) {
    lo = std::min(lo, x);
    hi = std::max(hi, x);
  }
  const double span = hi - lo;
  std::string out;
  out.reserve(v.size() * 3);
  for (double x : v) {
    // Flat history => all mid-level; else scale into [0,7].
    int lvl = (span > 0.0) ? int((x - lo) / span * 7.0 + 0.5) : 4;
    lvl = std::clamp(lvl, 0, 7);
    out += kSpark[lvl];
  }
  return out;
}

std::string Dashboard::render_logline(const Snapshot& s) const {
  std::string out = "pass=";
  out += fmt_l("%ld", s.pass);
  out += "/" + (s.total > 0 ? std::to_string(s.total) : std::string("?"));
  out += " E=" + fmt("%.10g", s.e_total());
  out += " rel=" + fmt("%.3e", s.rel_drift());
  out += " dt=" + fmt("%.5g", s.dt);
  out += " T=" + fmt("%.2f", s.temperature());
  out += " vmax=" + fmt("%.4g", s.v_max);
  const double hr = s.buffer_headroom();
  out += " Rbuf_hr=" + (hr >= 0.0 ? fmt("%.3f", hr) : std::string("n/a"));
  if (s.halted) out += " HALT=" + s.halt_msg;
  return out;
}

std::string Dashboard::render_plain(const Snapshot& s) const {
  const bool c = tty_;  // color only on a real terminal
  std::string out;

  // Title line.
  out += sgr(c, "1;36") + "=== TD-MD Core — live ===" + (c ? kReset : "") + "\n";

  // Progress: pass / total.
  out += "progress : " + fmt_l("%ld", s.pass) + "/" +
         (s.total > 0 ? std::to_string(s.total) : std::string("?")) + " passes\n";

  // Energy + NVE drift (the headline invariant).
  out += "energy   : E=" + fmt("%.10f", s.e_total()) + " eV" + "   (pe=" + fmt("%.6f", s.pe) +
         " ke=" + fmt("%.6f", s.ke) + ")\n";
  out += "NVE drift: rel=" + fmt("%.3e", s.rel_drift()) +
         "   (E-e0)/|e0|,  e0=" + fmt("%.6f", s.e0) + " eV\n";

  // Timestep + temperature + v_max.
  out += "dt       : " + fmt("%.5g", s.dt) + " ps\n";
  out += "T        : " + fmt("%.2f", s.temperature()) + " K\n";
  out += "v_max    : " + fmt("%.4g", s.v_max) + " A/ps\n";

  // The TD-distinctive readout: causality-buffer headroom.
  const double hr = s.buffer_headroom();
  out += "Rbuf hr  : ";
  if (hr >= 0.0) {
    // Color the headroom: green healthy (<0.8), yellow tightening, red ->1.
    const char* col = (hr < 0.8) ? "32" : (hr < 0.95 ? "33" : "31");
    out +=
        sgr(c, col) + fmt("%.3f", hr) + (c ? kReset : "") + "   (v_max*dt / R_buf,  <1 healthy)\n";
  } else {
    out += "n/a\n";
  }

  // Decomposition.
  out +=
      "ring     : zones=" + std::to_string(s.zones) + "  nodes=" + std::to_string(s.nodes) + "\n";

  // HALT status (green running / red halted).
  if (s.halted) {
    out += sgr(c, "1;31") + "STATUS   : HALT — " + s.halt_msg + (c ? kReset : "") + "\n";
    if (!s.rescue_path.empty()) out += "rescue   : " + s.rescue_path + "\n";
  } else {
    out += sgr(c, "1;32") + "STATUS   : running" + (c ? kReset : "") + "\n";
  }

  // E(t) sparkline.
  out += "E(t)     : " + sparkline(s.spark) + "\n";
  return out;
}

std::string Dashboard::render_ansi(const Snapshot& s) const {
  if (!tty_) return render_plain(s);
  // Home the cursor, then emit each line with a clear-to-eol so a shorter
  // frame does not leave stale tails (no full-screen clear => no flicker).
  std::string body = render_plain(s);
  std::string out = "\033[H";  // cursor home
  std::size_t pos = 0;
  while (pos < body.size()) {
    std::size_t nl = body.find('\n', pos);
    std::string line = body.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
    out += line + "\033[K\n";  // clear to end of line
    if (nl == std::string::npos) break;
    pos = nl + 1;
  }
  return out;
}

}  // namespace tdmd::cli
