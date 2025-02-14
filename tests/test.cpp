#include <gtest/gtest.h>

#include <zlib.h>

#include <array>
#include <string>
#include <iostream>

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

TEST(ZLIB, BasicCompression) {
  std::string input = "Conan is a MIT-licensed, Open Source package manager for C and C++ development, "
                      "allowing development teams to easily and efficiently manage their packages and "
                      "dependencies across platforms and build systems.";
  std::array<char, 256> buffer_out = {0};

  z_stream defstream;
  defstream.zalloc = Z_NULL;
  defstream.zfree = Z_NULL;
  defstream.opaque = Z_NULL;
  defstream.avail_in = static_cast<uInt>(input.size());
  defstream.next_in = reinterpret_cast<Bytef*>(input.data());
  defstream.avail_out = static_cast<uInt>(buffer_out.size());
  defstream.next_out = reinterpret_cast<Bytef*>(buffer_out.data());

  deflateInit(&defstream, Z_BEST_COMPRESSION);
  deflate(&defstream, Z_FINISH);
  deflateEnd(&defstream);

  std::cout << "Uncompressed size is: " << input.size() << '\n';
  std::cout << "Compressed size is: " << defstream.total_out << '\n';
  std::cout << "ZLIB VERSION: " << zlibVersion() << '\n';

  EXPECT_GT(input.size(), defstream.total_out);
}