#include <gtest/gtest.h>
#include <array>
#include "tdmd/core/zone.hpp"
#include "tdmd/core/fsm.hpp"

using namespace tdmd::core;

// M1 acceptance (Roadmap M1 / ZoneFSM §4, §6, §7).

TEST(ZoneFsm, CanonicalCycle) {
  Zone z;  // starts c
  ASSERT_EQ(z.type, ZoneType::c);
  ZoneFSM::apply(z, ZoneEvent::RECV);   EXPECT_EQ(z.type, ZoneType::d); EXPECT_FALSE(z.force_complete);
  ZoneFSM::apply(z, ZoneEvent::SPHERE); EXPECT_EQ(z.type, ZoneType::w); EXPECT_FALSE(z.force_complete);
  ZoneFSM::apply(z, ZoneEvent::START);  EXPECT_EQ(z.type, ZoneType::p); EXPECT_FALSE(z.force_complete);
  ZoneFSM::apply(z, ZoneEvent::END);    EXPECT_EQ(z.type, ZoneType::o); EXPECT_TRUE(z.force_complete);
  ZoneFSM::apply(z, ZoneEvent::SEND);   EXPECT_EQ(z.type, ZoneType::c); EXPECT_FALSE(z.force_complete);
}

// Each state allows exactly one event; everything else is forbidden and throws.
TEST(ZoneFsm, OnlyCanonicalTransitionsAllowed) {
  const std::array<ZoneType, 5> states = {ZoneType::c, ZoneType::d, ZoneType::w,
                                          ZoneType::p, ZoneType::o};
  const std::array<ZoneEvent, 5> events = {ZoneEvent::RECV, ZoneEvent::SPHERE,
                                           ZoneEvent::START, ZoneEvent::END,
                                           ZoneEvent::SEND};
  // canonical (state -> its single legal event)
  auto legal = [](ZoneType s, ZoneEvent e) {
    return (s == ZoneType::c && e == ZoneEvent::RECV) ||
           (s == ZoneType::d && e == ZoneEvent::SPHERE) ||
           (s == ZoneType::w && e == ZoneEvent::START) ||
           (s == ZoneType::p && e == ZoneEvent::END) ||
           (s == ZoneType::o && e == ZoneEvent::SEND);
  };
  for (auto s : states)
    for (auto e : events) {
      EXPECT_EQ(ZoneFSM::allowed(s, e), legal(s, e));
      Zone z; z.type = s;
      if (legal(s, e)) {
        EXPECT_NO_THROW(ZoneFSM::apply(z, e));
      } else {
        EXPECT_THROW(ZoneFSM::apply(z, e), std::logic_error);  // INV / §4
      }
    }
}

// INV-3: force_complete is true iff the zone is in state o.
TEST(ZoneFsm, Inv3ForceCompleteIffReady) {
  Zone z;
  ZoneFSM::apply(z, ZoneEvent::RECV);
  ZoneFSM::apply(z, ZoneEvent::SPHERE);
  ZoneFSM::apply(z, ZoneEvent::START);
  EXPECT_FALSE(z.force_complete);  // p: still partial
  ZoneFSM::apply(z, ZoneEvent::END);
  EXPECT_TRUE(z.force_complete);   // o only
  EXPECT_EQ(z.type, ZoneType::o);
}

// §7.1 — first zone S1 seeded with an artificial SPHERE (d -> w).
TEST(ZoneFsm, FirstZoneSeed) {
  Zone s1; s1.type = ZoneType::d;
  seed_first_zone(s1);
  EXPECT_EQ(s1.type, ZoneType::w);
}

// §7.3 — at t0 node P1 holds data (d), the rest are free (c).
TEST(ZoneFsm, InitialZoneTypes) {
  EXPECT_EQ(initial_zone_type(1), ZoneType::d);
  EXPECT_EQ(initial_zone_type(2), ZoneType::c);
  EXPECT_EQ(initial_zone_type(4), ZoneType::c);
}

// §7.4 — anti-deadlock: odd node sends-then-receives, even receives-then-sends.
TEST(ZoneFsm, AntiDeadlockIoOrder) {
  EXPECT_EQ(node_io_order(1), (std::array<IoOp, 2>{IoOp::SEND, IoOp::RECV}));
  EXPECT_EQ(node_io_order(2), (std::array<IoOp, 2>{IoOp::RECV, IoOp::SEND}));
  EXPECT_EQ(node_io_order(3), (std::array<IoOp, 2>{IoOp::SEND, IoOp::RECV}));
}
