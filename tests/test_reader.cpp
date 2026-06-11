// M3/B10: reader hardening — missing/zero mass, malformed numbers, duplicate
// ids and count mismatches are rejected with a message instead of NaN/terminate.
#include <gtest/gtest.h>

#include <fstream>
#include <filesystem>
#include <string>

#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/core/soa.hpp"

using namespace tdmd;
namespace fs = std::filesystem;

static std::string project_root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}

namespace {

class TempData {
 public:
  explicit TempData(const std::string& content) {
    static int n = 0;
    path_ = fs::temp_directory_path() /
            ("tdmd_reader_test_" + std::to_string(n++) + ".data");
    std::ofstream f(path_);
    f << content;
  }
  ~TempData() { std::error_code ec; fs::remove(path_, ec); }
  std::string path() const { return path_.string(); }

 private:
  fs::path path_;
};

// minimal valid 2-atom file builder
std::string two_atoms(const std::string& masses_section,
                      const std::string& atoms_rows) {
  return "test\n\n2 atoms\n1 atom types\n\n"
         "0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n\n" +
         masses_section + "Atoms # atomic\n\n" + atoms_rows;
}

bool read(const std::string& content) {
  TempData d(content);
  core::AtomSoA<double> atoms;
  core::Box box;
  return io::read_lammps_data(d.path(), atoms, box);
}

}  // namespace

TEST(Reader, GoldenFileReads) {
  core::AtomSoA<double> atoms;
  core::Box box;
  ASSERT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_72.data", atoms, box));
  EXPECT_EQ(atoms.n, 72);
  EXPECT_NEAR(box.len(0), 12.15, 1e-9);
  EXPECT_NEAR(box.len(2), 8.10, 1e-9);
  for (int i = 0; i < atoms.n; ++i) {
    EXPECT_EQ(atoms.type[i], 1);
    EXPECT_NEAR(atoms.mass[i], 26.9815, 1e-9);
  }
}

// M3 dataset: 3x3x4 cells = 144 atoms, box z=16.2 Å -> >=4 zones of width r_cut
TEST(Reader, Fcc144DatasetReads) {
  core::AtomSoA<double> atoms;
  core::Box box;
  ASSERT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_144.data", atoms, box));
  EXPECT_EQ(atoms.n, 144);
  EXPECT_NEAR(box.len(2), 16.20, 1e-9);
  EXPECT_GE(box.len(2) / 4.05, 4.0);  // >=4 zones of width >= r_cut=4.0
}

TEST(Reader, ValidTwoAtomFileReads) {
  EXPECT_TRUE(read(two_atoms("Masses\n\n1 26.98\n\n",
                             "1 1 1.0 2.0 3.0\n2 1 5.0 5.0 5.0\n")));
}

TEST(Reader, MissingMassesSectionFails) {  // was: mass=0 -> NaN in integrator
  testing::internal::CaptureStderr();
  EXPECT_FALSE(read(two_atoms("", "1 1 1.0 2.0 3.0\n2 1 5.0 5.0 5.0\n")));
  EXPECT_NE(testing::internal::GetCapturedStderr().find("mass"),
            std::string::npos);
}

TEST(Reader, ZeroMassFails) {
  EXPECT_FALSE(read(two_atoms("Masses\n\n1 0.0\n\n",
                              "1 1 1.0 2.0 3.0\n2 1 5.0 5.0 5.0\n")));
}

TEST(Reader, MassForWrongTypeFails) {  // atoms are type 2, mass given for type 1
  EXPECT_FALSE(read(two_atoms("Masses\n\n1 26.98\n\n",
                              "1 2 1.0 2.0 3.0\n2 2 5.0 5.0 5.0\n")));
}

TEST(Reader, MalformedNumberFails) {  // was: std::stod -> terminate
  testing::internal::CaptureStderr();
  EXPECT_FALSE(read(two_atoms("Masses\n\n1 26.98\n\n",
                              "1 1 abc 2.0 3.0\n2 1 5.0 5.0 5.0\n")));
  EXPECT_NE(testing::internal::GetCapturedStderr().find("parse error"),
            std::string::npos);
}

TEST(Reader, DuplicateAtomIdFails) {
  testing::internal::CaptureStderr();
  EXPECT_FALSE(read(two_atoms("Masses\n\n1 26.98\n\n",
                              "1 1 1.0 2.0 3.0\n1 1 5.0 5.0 5.0\n")));
  EXPECT_NE(testing::internal::GetCapturedStderr().find("duplicate"),
            std::string::npos);
}

TEST(Reader, AtomCountMismatchFails) {
  EXPECT_FALSE(read(two_atoms("Masses\n\n1 26.98\n\n", "1 1 1.0 2.0 3.0\n")));
}

TEST(Reader, MissingFileFails) {
  core::AtomSoA<double> atoms;
  core::Box box;
  EXPECT_FALSE(io::read_lammps_data("/nonexistent/no.data", atoms, box));
}
