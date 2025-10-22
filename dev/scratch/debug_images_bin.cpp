#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

// POD read helpers
static inline uint64_t read_u64(const char*& p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    p += 8;
    return v;
}

static inline uint32_t read_u32(const char*& p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    p += 4;
    return v;
}

static inline double read_f64(const char*& p) {
    double v;
    std::memcpy(&v, p, 8);
    p += 8;
    return v;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <images.bin>" << std::endl;
        return 1;
    }

    std::string path = argv[1];
    
    // Read file
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "Failed to open: " << path << std::endl;
        return 1;
    }

    auto sz = static_cast<std::size_t>(f.tellg());
    std::vector<char> buf(sz);
    f.seekg(0, std::ios::beg);
    f.read(buf.data(), sz);
    
    if (!f) {
        std::cerr << "Failed to read file" << std::endl;
        return 1;
    }

    std::cout << "File size: " << sz << " bytes" << std::endl;

    const char* cur = buf.data();
    const char* end = cur + sz;

    uint64_t n_images = read_u64(cur);
    std::cout << "Number of images: " << n_images << std::endl;
    std::cout << "Current position: " << (cur - buf.data()) << std::endl;

    for (uint64_t i = 0; i < n_images; ++i) {
        std::cout << "\n--- Image " << (i+1) << "/" << n_images << " ---" << std::endl;
        std::cout << "  Position before: " << (cur - buf.data()) << " bytes" << std::endl;
        
        // Check bounds
        if (cur + 4 > end) {
            std::cerr << "ERROR: Not enough bytes for image_id" << std::endl;
            return 1;
        }
        
        uint32_t id = read_u32(cur);
        std::cout << "  Image ID: " << id << std::endl;

        // Quaternion
        if (cur + 32 > end) {
            std::cerr << "ERROR: Not enough bytes for quaternion" << std::endl;
            return 1;
        }
        for (int k = 0; k < 4; ++k) {
            double q = read_f64(cur);
            std::cout << "  q[" << k << "] = " << q << std::endl;
        }

        // Translation
        if (cur + 24 > end) {
            std::cerr << "ERROR: Not enough bytes for translation" << std::endl;
            return 1;
        }
        for (int k = 0; k < 3; ++k) {
            double t = read_f64(cur);
            std::cout << "  t[" << k << "] = " << t << std::endl;
        }

        // Camera ID
        if (cur + 4 > end) {
            std::cerr << "ERROR: Not enough bytes for camera_id" << std::endl;
            return 1;
        }
        uint32_t camera_id = read_u32(cur);
        std::cout << "  Camera ID: " << camera_id << std::endl;

        // Name (null-terminated string)
        const char* name_start = cur;
        while (cur < end && *cur != '\0') {
            cur++;
        }
        if (cur >= end) {
            std::cerr << "ERROR: Unterminated string" << std::endl;
            return 1;
        }
        std::string name(name_start, cur - name_start);
        cur++; // Skip null terminator
        std::cout << "  Name: " << name << std::endl;

        // Number of 2D points
        if (cur + 8 > end) {
            std::cerr << "ERROR: Not enough bytes for npts" << std::endl;
            return 1;
        }
        uint64_t npts = read_u64(cur);
        std::cout << "  Num 2D points: " << npts << std::endl;

        // Skip 2D points
        size_t points_size = npts * (sizeof(double) * 2 + sizeof(uint64_t));
        std::cout << "  Points data size: " << points_size << " bytes" << std::endl;
        
        if (cur + points_size > end) {
            std::cerr << "ERROR: Not enough bytes for 2D points data" << std::endl;
            std::cerr << "  Remaining bytes: " << (end - cur) << std::endl;
            std::cerr << "  Required bytes: " << points_size << std::endl;
            return 1;
        }
        
        cur += points_size;
        std::cout << "  Position after: " << (cur - buf.data()) << " bytes" << std::endl;
    }

    std::cout << "\n✓ Successfully read all images" << std::endl;
    std::cout << "Final position: " << (cur - buf.data()) << " bytes" << std::endl;
    std::cout << "File size: " << sz << " bytes" << std::endl;
    
    if (cur != end) {
        std::cout << "WARNING: " << (end - cur) << " trailing bytes" << std::endl;
    }

    return 0;
}
