#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>
#include <iomanip>
#include <libcamera/libcamera.h>
#include <sys/mman.h>
#include <linux/spi/spidev.h>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <cerrno>
#include <cstring>
#include <libcamera/libcamera/ipa/raspberrypi_ipa_interface.h>
#include <cstdlib>
#include <array>

#include "utils/thread_safe_queue.h"
#include "detector.h"
#include "transform.h"
#include "spi_device.h"

using namespace libcamera;
using namespace std::chrono_literals;

// --- Globals ---
struct FrameData {
    cv::Mat frame0_rgb; 
    cv::Mat frame1_rgb;
    uint64_t timestamp0;
    uint64_t timestamp1;
    Size frame_size;
};

std::atomic<bool> stop_flag(false);

ThreadSafeQueue<FrameData>* g_data_queue = nullptr;
CameraConfiguration *g_config0 = nullptr;
CameraConfiguration *g_config1 = nullptr;
std::shared_ptr<Camera> camera0 = nullptr;
std::shared_ptr<Camera> camera1 = nullptr;
FrameBufferAllocator *g_allocator0 = nullptr;
FrameBufferAllocator *g_allocator1 = nullptr;
std::vector<std::unique_ptr<Request>> g_requests0;
std::vector<std::unique_ptr<Request>> g_requests1;
std::mutex g_frame_mutex;
FrameData g_pending_frame;
bool g_cam0_ready = false;
bool g_cam1_ready = false;

// (智能指针的声明，用于黄金关机)
std::unique_ptr<CameraManager> cm = nullptr;
std::unique_ptr<CameraConfiguration> config0_ptr = nullptr;
std::unique_ptr<CameraConfiguration> config1_ptr = nullptr;
std::unique_ptr<FrameBufferAllocator> allocator0_ptr = nullptr;
std::unique_ptr<FrameBufferAllocator> allocator1_ptr = nullptr;


// --- Signal Handler ---
void signal_handler(int signum) {
    std::cout << ">>> Interrupt signal (" << signum << ") received." << std::endl;
    stop_flag = true;
}

// --- Parity Helper ---
uint8_t add_parity(uint8_t byte) {
    int ones = 0;
    for (int i = 0; i < 7; ++i) { if ((byte >> i) & 1) ones++; }
    return (byte << 1) | (ones % 2 != 0 ? 1 : 0);
}

// --- Consumer Thread  ---
void processing_worker(ThreadSafeQueue<FrameData>& queue) {
    Transformer transformer0, transformer1;
    if (!transformer0.load_matrix("homography_matrix0.yml") || !transformer1.load_matrix("homography_matrix1.yml")) {
        std::cerr << "[Worker] ERROR: Failed to load homography matrices. Exiting." << std::endl;
        stop_flag = true; return;
    }

    SpiDevice spi;
    if (!spi.open("/dev/spidev1.0", 500000, SPI_MODE_0)) {
        std::cerr << "[Worker] ERROR: Failed to open SPI device. Exiting." << std::endl;
        stop_flag = true; return;
    }
    std::cout << "[Worker] Homography and SPI initialized." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    int processed_count = 0;
    cv::Mat bgr_frame0, bgr_frame1;

    while (!stop_flag) {
        FrameData data;
        auto t_wait_start = std::chrono::high_resolution_clock::now();
        queue.wait_and_pop(data);
        auto t_wait_end = std::chrono::high_resolution_clock::now();
        if (stop_flag) break;

        if (data.frame0_rgb.empty() || data.frame1_rgb.empty()) {
             if (!stop_flag) std::cerr << "[Worker] WARN: Received empty RGB/BGR frame. Skipping." << std::endl;
            continue;
        }

        int expected_h = data.frame_size.height;
        int expected_w = data.frame_size.width;
        if (data.frame0_rgb.rows != expected_h || data.frame0_rgb.cols != expected_w ||
            data.frame1_rgb.rows != expected_h || data.frame1_rgb.cols != expected_w) {
             std::cerr << "[Worker] ERROR: BGR Mat dimensions mismatch! Skipping." << std::endl;
             continue;
        }

        long long copy_time_0 = 0, copy_time_1 = 0; // V19.5: 计时 Copy
        long long detect_time_0 = 0, detect_time_1 = 0;
        long long spi_time = 0;
        long long wait_time = std::chrono::duration_cast<std::chrono::microseconds>(t_wait_end - t_wait_start).count();

        try {
            // V19.5 关键修复： 移除 cvtColor, 替换为 copyTo
            // (我们假设 libcamera:RGB888 实际上是 BGR888)
            auto t1 = std::chrono::high_resolution_clock::now();
            data.frame0_rgb.copyTo(bgr_frame0);
            auto t2 = std::chrono::high_resolution_clock::now();
            data.frame1_rgb.copyTo(bgr_frame1);
            auto t3 = std::chrono::high_resolution_clock::now();
            
            copy_time_0 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            copy_time_1 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        } catch (const cv::Exception& e) { std::cerr << "[Worker] ERROR: cv::copyTo failed: " << e.what() << std::endl; continue; }

        cv::Point2f p0 = cv::Point2f(-1,-1), p1 = cv::Point2f(-1,-1);
         try { 
            // V19.5: bgr_frame0 现在是正确的 BGR 格式了
            auto t4 = std::chrono::high_resolution_clock::now();
            p0 = detect_circle(bgr_frame0); 
            auto t5 = std::chrono::high_resolution_clock::now();
            detect_time_0 = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
         }
         catch (const cv::Exception& e) { std::cerr << "[Worker] ERROR: detect_circle (Cam0) failed: " << e.what() << std::endl; }
         
         try { 
            auto t6 = std::chrono::high_resolution_clock::now();
            p1 = detect_circle(bgr_frame1); 
            auto t7 = std::chrono::high_resolution_clock::now();
            detect_time_1 = std::chrono::duration_cast<std::chrono::microseconds>(t7 - t6).count();
         }
         catch (const cv::Exception& e) { std::cerr << "[Worker] ERROR: detect_circle (Cam1) failed: " << e.what() << std::endl; }

        // (画方框的代码我们仍然保留，它几乎不耗时，但不会显示)
        //const int box_radius = 15;
        //const cv::Scalar color_green(0, 255, 0); 
        //if (p0.x > 0) {
        //    cv::rectangle(bgr_frame0, 
        //                  cv::Point(p0.x - box_radius, p0.y - box_radius), 
        //                  cv::Point(p0.x + box_radius, p0.y + box_radius), 
        //                  color_green, 2);
        //}
        //if (p1.x > 0) {
        //    cv::rectangle(bgr_frame1, 
        //                  cv::Point(p1.x - box_radius, p1.y - box_radius), 
        //                  cv::Point(p1.x + box_radius, p1.y + box_radius), 
        //                  color_green, 2);
        //}

        double world_x = NAN, world_y = NAN;
        if (p0.x > 0 && p1.x > 0) {
            try {
                cv::Point2f w0 = transformer0.camera_to_world(p0);
                cv::Point2f w1 = transformer1.camera_to_world(p1);
                world_x = (w0.x + w1.x) / 2.0; world_y = (w0.y + w1.y) / 2.0;
            } catch (const cv::Exception& e) { std::cerr << "[Worker] ERROR: camera_to_world (fusion) failed: " << e.what() << std::endl; }
        } else if (p0.x > 0) {
            try { cv::Point2f w0 = transformer0.camera_to_world(p0); world_x = w0.x; world_y = w0.y; }
            catch (const cv::Exception& e) { std::cerr << "[Worker] ERROR: camera_to_world (Cam0) failed: " << e.what() << std::endl; }
        } else if (p1.x > 0) {
            try { cv::Point2f w1 = transformer1.camera_to_world(p1); world_x = w1.x; world_y = w1.y; }
            catch (const cv::Exception& e) { std::cerr << "[Worker] ERROR: camera_to_world (Cam1) failed: " << e.what() << std::endl; }
        }

        processed_count++;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed_chrono = now - start_time;
        long long elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed_chrono).count();
        if (elapsed_sec == 0) elapsed_sec = 1;
        double fps = static_cast<double>(processed_count) / elapsed_sec;
        
        double sync_error_us = 0;
        if (data.timestamp1 > data.timestamp0) {
            sync_error_us = static_cast<double>(data.timestamp1 - data.timestamp0) / 1000.0;
        } else {
            sync_error_us = -static_cast<double>(data.timestamp0 - data.timestamp1) / 1000.0;
        }

        // 只有在检测到小球时才打印日志
        if (!std::isnan(world_x)) 
        {
            std::cout << std::fixed << std::setprecision(2);
            
            // 第 1 行 (状态):
            std::cout << "[Worker] Sync: " << sync_error_us << "us | FPS: " << fps
                      << " | P0: (" << p0.x << "," << p0.y << "), P1: (" << p1.x << "," << p1.y << ") | "
                      << "World: (" << world_x << "," << world_y << ") mm" << std::endl;
            
            // (SPI 逻辑)
            int x_int = static_cast<int>(world_x * 10); int y_int = static_cast<int>(world_y * 10);
            x_int = std::max(std::min(x_int, 8191), -8191); y_int = std::max(std::min(y_int, 8191), -8191);
            uint8_t x_sign = (x_int < 0) ? 0x40 : 0x00; uint8_t y_sign = (y_int < 0) ? 0x40 : 0x00;
            int x_abs = std::abs(x_int); int y_abs = std::abs(y_int);
            uint8_t x_high7 = ((x_abs >> 7) & 0x3F) | x_sign; uint8_t x_low7 = x_abs & 0x7F;
            uint8_t y_high7 = ((y_abs >> 7) & 0x3F) | y_sign; uint8_t y_low7 = y_abs & 0x7F;
            std::vector<uint8_t> spi_data = { 0xAA, add_parity(x_high7), add_parity(x_low7), add_parity(y_high7), add_parity(y_low7) };
            
            auto t8 = std::chrono::high_resolution_clock::now();
            spi.transfer(spi_data);
            auto t9 = std::chrono::high_resolution_clock::now();
            spi_time = std::chrono::duration_cast<std::chrono::microseconds>(t9 - t8).count();

            // 第 2 行 (性能分析):
            std::cout << "         [Timing] Wait: " << (wait_time / 1000.0) << "ms"
                      << " | Copy0: " << (copy_time_0 / 1000.0) << "ms"
                      << " | Copy1: " << (copy_time_1 / 1000.0) << "ms"
                      << " | Det0: " << (detect_time_0 / 1000.0) << "ms"
                      << " | Det1: " << (detect_time_1 / 1000.0) << "ms"
                      << " | SPI: "  << (spi_time / 1000.0) << "ms"
                      << std::endl;
        }
        // --- 结束 V19.5 修复 (移除了 "else" 块) ---
        // if (!bgr_frame0.empty()) cv::imshow("Cam0 (BGR)", bgr_frame0);
        // if (!bgr_frame1.empty()) cv::imshow("Cam1 (BGR)", bgr_frame1);
        // cv::waitKey(1); 


        if (processed_count >= 100) { start_time = now; processed_count = 0; }
    }
    spi.close();
    
    // 注释掉销毁窗口
    // cv::destroyAllWindows();
    std::cout << "[Worker] Thread finished." << std::endl;
}

// --- Producer Callbacks (V19.5: RGB) ---
cv::Mat cloneFrame(FrameBuffer *buffer, const Size &size) {
    if (!buffer || buffer->planes().empty()) { return cv::Mat(); }
    const FrameBuffer::Plane &plane = buffer->planes()[0];
    
    size_t expected_length = static_cast<size_t>(size.width * size.height * 3);
    size_t map_length = expected_length;
    
    if (plane.length < expected_length) {
         // std::cerr << "[Callback] WARN: Plane length (" << plane.length ...
    }
    
    void *data_ptr = mmap(NULL, map_length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
    if (data_ptr == MAP_FAILED) {
        std::cerr << "[Callback] ERROR: mmap failed! Error: " << strerror(errno) << std::endl; return cv::Mat();
    }
    
    cv::Mat rgb_mat; // (这个变量名现在有误导性，它实际上是 BGR)
    cv::Mat cloned_frame;
    try {
        rgb_mat = cv::Mat(size.height, size.width, CV_8UC3, data_ptr);
        if (rgb_mat.empty()) { munmap(data_ptr, map_length); return cv::Mat(); }
        cloned_frame = rgb_mat.clone();
    } catch (const cv::Exception& e) {
        std::cerr << "[Callback] ERROR: OpenCV exception during Mat creation/clone: " << e.what() << std::endl;
        cloned_frame = cv::Mat();
    }
    munmap(data_ptr, map_length);
    return cloned_frame;
}

void requestComplete0(Request *request) {
    if (!request || !camera0 || stop_flag) return;
    Request::Status status = request->status();
    if (status == Request::Status::RequestCancelled) { return; }
    if (status != Request::Status::RequestComplete) { std::cerr << "[Callback Cam0] WARN: Request non-complete status..." << std::endl; }
    
    if (request->buffers().find(g_config0->at(0).stream()) == request->buffers().end()) {
        std::cerr << "[Callback Cam0] ERROR: Buffer map does not contain main stream!" << std::endl;
        request->reuse(Request::ReuseBuffers); camera0->queueRequest(request); return;
    }
    FrameBuffer *buffer = request->buffers().at(g_config0->at(0).stream()); 
    
    if (!buffer) { std::cerr << "[Callback Cam0] ERROR: Failed to get buffer" << std::endl; request->reuse(Request::ReuseBuffers); camera0->queueRequest(request); return; }

    std::lock_guard<std::mutex> lock(g_frame_mutex);
    g_pending_frame.frame0_rgb = cloneFrame(buffer, g_config0->at(0).size);
    if (!g_pending_frame.frame0_rgb.empty()) {
         g_pending_frame.timestamp0 = *request->metadata().get(controls::SensorTimestamp);
         g_cam0_ready = true;
    } else {
         std::cerr << "[Callback Cam0] WARN: Cloning frame 0 failed." << std::endl;
    }

    if (g_cam0_ready && g_cam1_ready) {
        g_pending_frame.frame_size = g_config0->at(0).size;
        if (g_data_queue) g_data_queue->push(std::move(g_pending_frame));
        g_cam0_ready = false; g_cam1_ready = false; g_pending_frame = {};
    }

    request->reuse(Request::ReuseBuffers);
    if (camera0 && !stop_flag) { camera0->queueRequest(request); }
}

void requestComplete1(Request *request) {
    if (!request || !camera1 || stop_flag) return;
    Request::Status status = request->status();
    if (status == Request::Status::RequestCancelled) { return; }
    if (status != Request::Status::RequestComplete) { std::cerr << "[Callback Cam1] WARN: Request non-complete status..." << std::endl; }

    if (request->buffers().find(g_config1->at(0).stream()) == request->buffers().end()) {
        std::cerr << "[Callback Cam1] ERROR: Buffer map does not contain main stream!" << std::endl;
        request->reuse(Request::ReuseBuffers); camera1->queueRequest(request); return;
    }
    FrameBuffer *buffer = request->buffers().at(g_config1->at(0).stream());

    if (!buffer) { std::cerr << "[Callback Cam1] ERROR: Failed to get buffer" << std::endl; request->reuse(Request::ReuseBuffers); camera1->queueRequest(request); return; }

    std::lock_guard<std::mutex> lock(g_frame_mutex);
    g_pending_frame.frame1_rgb = cloneFrame(buffer, g_config1->at(0).size);
    if (!g_pending_frame.frame1_rgb.empty()) {
         g_pending_frame.timestamp1 = *request->metadata().get(controls::SensorTimestamp);
         g_cam1_ready = true;
    } else {
         std::cerr << "[Callback Cam1] WARN: Cloning frame 1 failed." << std::endl;
    }

    if (g_cam0_ready && g_cam1_ready) {
        g_pending_frame.frame_size = g_config1->at(0).size;
        if (g_data_queue) g_data_queue->push(std::move(g_pending_frame));
        g_cam0_ready = false; g_cam1_ready = false; g_pending_frame = {};
    }

    request->reuse(Request::ReuseBuffers);
    if (camera1 && !stop_flag) { camera1->queueRequest(request); }
}


// --- Main Function (V19.5: Viewfinder + FrameDurationLimits) ---
int main() {
    // *** 我们不再设置 .yaml 环境变量 ***

    signal(SIGINT, signal_handler); signal(SIGTERM, signal_handler);
    cm = std::make_unique<CameraManager>(); cm->start();
    if (cm->cameras().size() < 2) {
        std::cerr << "Error: Found " << cm->cameras().size() << " cameras, but 2 are required." << std::endl;
        cm->stop(); return -1;
    }
    camera0 = cm->cameras()[0];
    camera1 = cm->cameras()[1];

    if (camera0->acquire()) { std::cerr << "Failed to acquire camera 0" << std::endl; cm->stop(); return -1; }
    std::cout << ">>> Camera 0 acquired." << std::endl;
    if (camera1->acquire()) { std::cerr << "Failed to acquire camera 1" << std::endl; camera0->release(); cm->stop(); return -1; }
    std::cout << ">>> Camera 1 acquired." << std::endl;

    // --- V19.5: Viewfinder + RGB888 ---
    std::cout << ">>> Requesting Viewfinder stream role for Cam0..." << std::endl;
    config0_ptr = camera0->generateConfiguration({ StreamRole::Viewfinder }); 
    if (!config0_ptr) { std::cerr << "Failed to generate config 0" << std::endl; /* ... */ }
    g_config0 = config0_ptr.get();
    g_config0->at(0).pixelFormat = formats::RGB888; // [0] 是 Viewfinder 流
    g_config0->at(0).size = Size(640, 480);
    if (camera0->configure(g_config0) < 0) {
        std::cerr << "Failed to configure camera 0." << std::endl; camera0->release(); camera1->release(); cm->stop(); return -1;
    }
    std::cout << ">>> Camera 0 configured: " << g_config0->at(0).pixelFormat.toString() << " " << g_config0->at(0).size.toString() << std::endl;

    std::cout << ">>> Requesting Viewfinder stream role for Cam1..." << std::endl;
    config1_ptr = camera1->generateConfiguration({ StreamRole::Viewfinder }); 
    if (!config1_ptr) { std::cerr << "Failed to generate config 1" << std::endl; /* ... */ }
    g_config1 = config1_ptr.get();
    g_config1->at(0).pixelFormat = formats::RGB888; // [0] 是 Viewfinder 流
    g_config1->at(0).size = Size(640, 480);
    if (camera1->configure(g_config1) < 0) {
        std::cerr << "Failed to configure camera 1." << std::endl; camera0->release(); camera1->release(); cm->stop(); return -1;
    }
    std::cout << ">>> Camera 1 configured: " << g_config1->at(0).pixelFormat.toString() << " " << g_config1->at(0).size.toString() << std::endl;
    // --- 结束 ---

    // (分配 camera0)
    allocator0_ptr = std::make_unique<FrameBufferAllocator>(camera0);
    g_allocator0 = allocator0_ptr.get();
    if (g_allocator0->allocate(g_config0->at(0).stream()) < 0) { std::cerr << "Failed to allocate buffers 0 (Stream 0)." << std::endl; /* ... */ }
    std::cout << ">>> Buffers allocated (Cam0)." << std::endl;
    g_requests0.clear();
    for (const std::unique_ptr<FrameBuffer>& buffer : g_allocator0->buffers(g_config0->at(0).stream())) {
        auto request = camera0->createRequest(); if (!request) { std::cerr << "Failed to create request 0." << std::endl; /* ... */ }
        request->addBuffer(g_config0->at(0).stream(), buffer.get()); // RGB 流
        g_requests0.push_back(std::move(request));
    }
    std::cout << ">>> " << g_requests0.size() << " requests created (Cam0)." << std::endl;

    // (分配 camera1)
    allocator1_ptr = std::make_unique<FrameBufferAllocator>(camera1);
    g_allocator1 = allocator1_ptr.get();
    if (g_allocator1->allocate(g_config1->at(0).stream()) < 0) { std::cerr << "Failed to allocate buffers 1 (Stream 0)." << std::endl; /* ... */ }
    std::cout << ">>> Buffers allocated (Cam1)." << std::endl;
    g_requests1.clear();
    for (const std::unique_ptr<FrameBuffer>& buffer : g_allocator1->buffers(g_config1->at(0).stream())) {
        auto request = camera1->createRequest(); if (!request) { std::cerr << "Failed to create request 1." << std::endl; /* ... */ }
        request->addBuffer(g_config1->at(0).stream(), buffer.get()); // RGB 流
        g_requests1.push_back(std::move(request));
    }
    std::cout << ">>> " << g_requests1.size() << " requests created (Cam1)." << std::endl;

    ThreadSafeQueue<FrameData> data_queue(10); g_data_queue = &data_queue;
    std::cout << ">>> Starting worker thread..." << std::endl;
    std::thread worker(processing_worker, std::ref(data_queue));
    std::cout << ">>> Worker thread started." << std::endl;

    std::cout << ">>> Connecting camera signals..." << std::endl;
    camera0->requestCompleted.connect(requestComplete0);
    camera1->requestCompleted.connect(requestComplete1);
    std::cout << ">>> Signals connected." << std::endl;

    // --- V19.5: 保持 V19 的 60 FPS ControlList 修复 ---
    int64_t frame_duration = 16666; // 60 FPS
    static std::array<int64_t, 2> duration_limits = { frame_duration, frame_duration };

    ControlList controls0;
    controls0.set(controls::rpi::SyncMode, controls::rpi::SyncModeServer);
    controls0.set(controls::FrameDurationLimits, duration_limits);
    controls0.set(controls::AeEnable, true); 

    ControlList controls1;
    controls1.set(controls::rpi::SyncMode, controls::rpi::SyncModeClient);
    controls1.set(controls::FrameDurationLimits, duration_limits);
    controls1.set(controls::AeEnable, true); 
    
    std::cout << ">>> Starting cameras (Master/Slave, 60 FPS, Constrained AE)..." << std::endl;
    // --- 结束 V19.5 修复 ---

    // 必须先启动从机 (Client)
    if (camera1->start(&controls1) < 0) { std::cerr << "Failed to start camera 1 (Client)" << std::endl; /* ... */ }
    std::cout << ">>> Camera 1 (Client) started." << std::endl;
    
    std::this_thread::sleep_for(500ms); 

    // 再启动主机 (Server)
    if (camera0->start(&controls0) < 0) { std::cerr << "Failed to start camera 0 (Server)" << std::endl; /* ... */ }
     std::cout << ">>> Camera 0 (Server) started." << std::endl;


    std::cout << ">>> Queueing initial requests..." << std::endl;
    int queued_count0 = 0, queued_count1 = 0;
    for (auto& req : g_requests0) { if (camera0->queueRequest(req.get()) < 0) { stop_flag = true; break; } queued_count0++; }
    for (auto& req : g_requests1) { if (camera1->queueRequest(req.get()) < 0) { stop_flag = true; break; } queued_count1++; }
    std::cout << ">>> " << queued_count0 << " (Cam0) and " << queued_count1 << " (Cam1) initial requests queued." << std::endl;
    
    if (!stop_flag) {
        std::cout << ">>> Starting capture loop (main thread idle)..." << std::endl;
        while (!stop_flag) { std::this_thread::sleep_for(100ms); }
    }
    
    // --- 最终的“黄金”关机顺序 ---
    std::cout << ">>> Stop signal received. Initiating shutdown..." << std::endl;
    if (camera0) { camera0->stop(); std::cout << ">>> Camera 0 stopped." << std::endl; }
    if (camera1) { camera1->stop(); std::cout << ">>> Camera 1 stopped." << std::endl; }
    if (camera0) { camera0->requestCompleted.disconnect(requestComplete0); std::cout << ">>> Signal 0 disconnected." << std::endl; }
    if (camera1) { camera1->requestCompleted.disconnect(requestComplete1); std::cout << ">>> Signal 1 disconnected." << std::endl; }

    if (g_data_queue) { std::cout << ">>> Pushing stop signal to worker..." << std::endl; g_data_queue->push({}); }
    if (worker.joinable()) { 
        std::cout << ">>> Joining worker thread..." << std::endl; 
        worker.join(); 
        std::cout << ">>> Worker thread joined." << std::endl; 
    }

    // 步骤 1: 释放缓冲区
    if (g_allocator0 && g_config0) { 
        std::cout << ">>> Freeing buffers 0..." << std::endl; 
        g_allocator0->free(g_config0->at(0).stream()); // RGB 流
        std::cout << ">>> Buffers 0 freed." << std::endl; 
    }
    if (g_allocator1 && g_config1) { 
        std::cout << ">>> Freeing buffers 1..." << std::endl; 
        g_allocator1->free(g_config1->at(0).stream()); // RGB 流
        std::cout << ">>> Buffers 1 freed." << std::endl; 
    }
    
    // 步骤 2: 销毁请求
    std::cout << ">>> Clearing requests..." << std::endl;
    g_requests0.clear(); 
    g_requests1.clear();
    
    // 步骤 3: 释放相机硬件
    if (camera0) { 
        std::cout << ">>> Releasing camera 0..." << std::endl; 
        camera0->release(); 
        std::cout << ">>> Camera 0 released." << std::endl; 
    }
    if (camera1) { 
        std::cout << ">>> Releasing camera 1..." << std::endl; 
        camera1->release(); 
        std::cout << ">>> Camera 1 released." << std::endl; 
    }
    
    // 步骤 4: 销毁所有 libcamera *对象*
    std::cout << ">>> Destroying libcamera objects..." << std::endl;
    camera0.reset(); 
    camera1.reset();
    allocator0_ptr.reset(); 
    allocator1_ptr.reset(); 
    config0_ptr.reset();    
    config1_ptr.reset();    
    
    // 步骤 5: 重置所有全局裸指针
    g_allocator0 = nullptr; g_allocator1 = nullptr;
    g_config0 = nullptr; g_config1 = nullptr;
    g_data_queue = nullptr;

    // 步骤 6: 安全地停止和销毁 CameraManager
    if (cm) { 
        std::cout << ">>> Stopping CameraManager..." << std::endl; 
        cm->stop(); 
        cm.reset(); 
    }
    
    std::cout << ">>> System stopped safely." << std::endl;
    return 0;
}
