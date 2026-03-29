#pragma once

#include <string>
#include <vector>
#include <cstdint>

class SpiDevice {
public:
    SpiDevice();
    ~SpiDevice();

    bool open(const std::string& device, uint32_t speed_hz, uint8_t mode);
    bool transfer(const std::vector<uint8_t>& tx_data);
    void close();

private:
    int fd_ = -1;
    std::string device_path_;
};