#include "spi_device.h"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>
#include <cstring>

SpiDevice::SpiDevice() : fd_(-1) {}

SpiDevice::~SpiDevice() {
    close();
}

bool SpiDevice::open(const std::string& device, uint32_t speed_hz, uint8_t mode) {
    device_path_ = device;
    fd_ = ::open(device_path_.c_str(), O_RDWR);
    if (fd_ < 0) {
        std::cerr << "ERROR: Can't open SPI device " << device << std::endl;
        return false;
    }

    if (ioctl(fd_, SPI_IOC_WR_MODE, &mode) == -1) {
        std::cerr << "ERROR: Can't set SPI mode" << std::endl;
        return false;
    }
    
    if (ioctl(fd_, SPI_IOC_WR_MAX_SPEED_HZ, &speed_hz) == -1) {
        std::cerr << "ERROR: Can't set SPI speed" << std::endl;
        return false;
    }

    return true;
}

bool SpiDevice::transfer(const std::vector<uint8_t>& tx_data) {
    if (fd_ < 0) return false;

    struct spi_ioc_transfer tr = {};
    tr.tx_buf = (unsigned long)tx_data.data();
    tr.len = tx_data.size();

    if (ioctl(fd_, SPI_IOC_MESSAGE(1), &tr) < 1) {
        std::cerr << "ERROR: SPI transfer failed" << std::endl;
        return false;
    }
    return true;
}

void SpiDevice::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}