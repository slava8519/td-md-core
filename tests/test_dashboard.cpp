// M7 (UX) — dashboard acceptance (A6). The renderer is a PURE function of a
// Snapshot, so it is tested with NO real run, NO GPU, NO terminal — exactly the
// cloud-CI posture. We feed a synthetic deterministic Snapshot stream and assert
// exact substrings (progress, E, rel-drift, T, the Rbuf-headroom readout, the
// HALT red marker, sparkline ordering); then we hammer ProgressMonitor::on_pass
// from many threads to exercise the concurrent-z-node thread-safety contract.
#include <gtest/gtest.h>

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "tdmd/cli/dashboard.hpp"
#include "tdmd/cli/progress_monitor.hpp"
#include "tdmd/core/conveyor.hpp"

using namespace tdmd;

namespace {

// A deterministic synthetic snapshot: e0=1.5, this pass E=1.6 => rel = +0.1/1.5
// = 6.667e-02. n_dof chosen so T is a clean check.
cli::Snapshot make_snap() {
  cli::Snapshot s;
  s.pass = 50;
  s.total = 100;
  s.pe = -2.0;
  s.ke = 3.6;  // E = pe+ke = 1.6
  s.e0 = 1.5;
  s.dt = 0.002;
  s.v_max = 5.0;
  s.r_buf = 0.02;  // v_max*dt / r_buf = 5*0.002/0.02 = 0.5
  s.n_dof = 1;     // T = 2*ke/(1*kB)
  s.zones = 4;
  s.nodes = 2;
  s.spark = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  return s;
}

}  // namespace

// --- the pure renderer: exact substrings on a known snapshot (A6) ---

TEST(Dashboard, RenderPlainSubstrings) {
  cli::Dashboard d(/*tty=*/false);
  const std::string out = d.render_plain(make_snap());

  EXPECT_NE(out.find("50/100 passes"), std::string::npos) << out;
  EXPECT_NE(out.find("E=1.6000000000 eV"), std::string::npos) << out;
  // rel = (1.6-1.5)/1.5 = 0.06666... -> %.3e
  EXPECT_NE(out.find("rel=6.667e-02"), std::string::npos) << out;
  // T = 2*3.6/(1*8.617333262e-5) ~ 83552 K
  EXPECT_NE(out.find("T        :"), std::string::npos) << out;
  EXPECT_NE(out.find(" K"), std::string::npos) << out;
  // the TD-distinctive readout
  EXPECT_NE(out.find("Rbuf hr  : 0.500"), std::string::npos) << out;
  EXPECT_NE(out.find("v_max*dt / R_buf"), std::string::npos) << out;
  EXPECT_NE(out.find("zones=4  nodes=2"), std::string::npos) << out;
  EXPECT_NE(out.find("STATUS   : running"), std::string::npos) << out;
  // plain (non-TTY) form carries NO ANSI escapes
  EXPECT_EQ(out.find("\033["), std::string::npos) << "plain must be escape-free";
}

// --- temperature value is the thermal.hpp formula T = 2 ke/(n_dof kB) ---

TEST(Dashboard, TemperatureMatchesFormula) {
  cli::Snapshot s = make_snap();
  const double T = 2.0 * s.ke / (double(s.n_dof) * units::kB);
  char want[64];
  std::snprintf(want, sizeof(want), "%.2f", T);
  cli::Dashboard d(false);
  EXPECT_NE(d.render_plain(s).find(std::string("T        : ") + want), std::string::npos);
}

// --- HALT: red marker present in TTY mode, message surfaced; rescue path ---

TEST(Dashboard, HaltRedMarkerAndRescue) {
  cli::Snapshot s = make_snap();
  s.halted = true;
  s.halt_msg = "causality (INV-4) at step 51";
  s.rescue_path = "rescue.xyz";

  cli::Dashboard tty(/*tty=*/true);
  const std::string a = tty.render_ansi(s);
  EXPECT_NE(a.find("HALT — causality (INV-4) at step 51"), std::string::npos);
  EXPECT_NE(a.find("\033[1;31m"), std::string::npos) << "expected red SGR";
  EXPECT_NE(a.find("rescue   : rescue.xyz"), std::string::npos);

  // plain form: the message is there, but no escapes.
  cli::Dashboard plain(/*tty=*/false);
  const std::string p = plain.render_plain(s);
  EXPECT_NE(p.find("HALT — causality (INV-4) at step 51"), std::string::npos);
  EXPECT_EQ(p.find("\033["), std::string::npos);
}

// --- sparkline: monotone ascending input yields ascending block levels ---

TEST(Dashboard, SparklineOrdering) {
  const std::string sp = cli::Dashboard::sparkline({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
  // first glyph is the lowest block, last is the full block
  EXPECT_EQ(sp.substr(0, 3), std::string("\xe2\x96\x81"));  // ▁
  EXPECT_NE(sp.find("\xe2\x96\x88"), std::string::npos);    // █ present
  // last 3 bytes are the full block (max value last)
  EXPECT_EQ(sp.substr(sp.size() - 3, 3), std::string("\xe2\x96\x88"));
  // empty input => empty string (no crash)
  EXPECT_EQ(cli::Dashboard::sparkline({}), std::string());
  // flat input => no crash, fixed mid-level glyphs
  EXPECT_FALSE(cli::Dashboard::sparkline({2.0, 2.0, 2.0}).empty());
}

// --- the ANSI frame homes the cursor and clears each line ---

TEST(Dashboard, AnsiCursorControl) {
  cli::Dashboard d(/*tty=*/true);
  const std::string a = d.render_ansi(make_snap());
  EXPECT_EQ(a.substr(0, 3), std::string("\033[H")) << "cursor-home prefix";
  EXPECT_NE(a.find("\033[K"), std::string::npos) << "clear-to-eol per line";
}

// --- one-line log fallback is parse-friendly key=value ---

TEST(Dashboard, LogLineFormat) {
  cli::Dashboard d(false);
  const std::string l = d.render_logline(make_snap());
  EXPECT_NE(l.find("pass=50/100"), std::string::npos) << l;
  EXPECT_NE(l.find("rel=6.667e-02"), std::string::npos) << l;
  EXPECT_NE(l.find("Rbuf_hr=0.500"), std::string::npos) << l;
  EXPECT_EQ(l.find('\n'), std::string::npos) << "single line";
}

// --- ProgressMonitor thread-safety: hammer on_pass_stats from many threads ---
// (covers the concurrent-z-node contract; run under TSan in CI for the race net)

TEST(Dashboard, ProgressMonitorThreadSafe) {
  // Discard frames into a null sink (the monitor writes to a FILE*).
  std::FILE* dev = std::fopen("/dev/null", "w");
  ASSERT_NE(dev, nullptr);

  cli::Snapshot ctx;
  ctx.e0 = 1.5;
  ctx.n_dof = 1;
  ctx.zones = 4;
  ctx.nodes = 5;
  cli::ProgressMonitor mon(dev, /*tty=*/true, /*total=*/100000, ctx);
  mon.set_c_buf(1.5);
  mon.start();

  constexpr int kThreads = 5;  // five "node" threads
  constexpr int kPer = 20000;
  std::vector<std::thread> ts;
  std::atomic<long> next{1};
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&] {
      for (int i = 0; i < kPer; ++i) {
        core::PassStats st;
        const long h = next.fetch_add(1);
        st.pe = -2.0;
        st.ke = 3.6;
        st.dt = 0.002;
        st.v_max = 5.0;
        mon.on_pass_stats(h, st);
      }
    });
  }
  for (auto& th : ts) th.join();
  mon.stop();
  std::fclose(dev);
  SUCCEED();  // no data race / no crash under TSan == pass
}
