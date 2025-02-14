#include <iostream>
#include <string>
#include <vector>
#include <zlib.h>

int main() {
    std::string input = "Conan is a MIT-licensed, Open Source package manager for C and C++ development, "
                        "allowing development teams to easily and efficiently manage their packages and "
                        "dependencies across platforms and build systems.";
    std::vector<char> output(256);

    z_stream defstream;
    defstream.zalloc = Z_NULL;
    defstream.zfree = Z_NULL;
    defstream.opaque = Z_NULL;
    defstream.avail_in = static_cast<uInt>(input.size());
    defstream.next_in = reinterpret_cast<Bytef*>(input.data());
    defstream.avail_out = static_cast<uInt>(output.size());
    defstream.next_out = reinterpret_cast<Bytef*>(output.data());

    deflateInit(&defstream, Z_BEST_COMPRESSION);
    deflate(&defstream, Z_FINISH);
    deflateEnd(&defstream);

    std::cout << "Uncompressed size is: " << input.size() << '\n';
    std::cout << "Compressed size is: " << defstream.total_out << '\n';
    std::cout << "ZLIB VERSION: " << zlibVersion() << '\n';

    return EXIT_SUCCESS;
}
