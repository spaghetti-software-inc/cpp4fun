#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

TEST_CASE("Example test case", "[example]") {
    REQUIRE(1 + 1 == 2);
    REQUIRE(2 * 2 == 4);
}