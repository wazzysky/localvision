/*
 * 这是一个 C++ 程序，用于从 libcamera 捕获视频流，
 * 在一个单独的线程中处理图像（检测一个圆并进行坐标变换），
 * 最终通过 SPI 将计算出的坐标发送出去。
 * 它使用了多线程的“生产者-消费者”模型。
 */

// --- 核心头文件 ---
#include <iostream>     // 用于标准I/O (std::cout, std::cerr)
#include <thread>       // 用于创建和管理线程 (std::thread)
#include <chrono>       // 用于时间和帧率计算 (std::chrono)
#include <csignal>      // 用于捕获系统信号 (如 SIGINT, 即 Ctrl+C)
#include <atomic>       // 用于线程安全的原子变量 (std::atomic<bool>)
#include <iomanip>      // 用于格式化输出 (std::setprecision)
#include <libcamera/libcamera.h> // libcamera 核心库

// --- 系统和硬件交互头文件 ---
#include <sys/mman.h>     // 用于 mmap 内存映射，实现高效的缓冲区访问
#include <linux/spi/spidev.h> // 用于 SPI 设备常量 (SPI_MODE_0)
#include <memory>         // 用于智能指针 (std::shared_ptr, std::unique_ptr)
#include <mutex>          // (虽然在这个文件中没直接用，但在 thread_safe_queue.h 中肯定用了)
#include <opencv2/opencv.hpp> // OpenCV 库，用于图像处理
#include <cerrno>         // 用于 C 风格的错误码 (errno)
#include <cstring>        // 用于 C 风格的字符串操作 (strerror)

// --- 自定义模块头文件 ---
#include "utils/thread_safe_queue.h" // 包含自定义的线程安全队列
#include "detector.h"    // 包含 `detect_circle` 函数的声明
#include "transform.h"   // 包含 `Transformer` 类的声明
#include "spi_device.h"  // 包含 `SpiDevice` 类的声明

// --- 命名空间 ---
using namespace libcamera; // 使用 libcamera 的命名空间，避免写 libcamera::
using namespace std::chrono_literals; // 允许使用时间后缀，如 100ms

// --- 全局变量 ---
// 定义一个结构体，用于在生产者和消费者线程之间传递数据
struct FrameData {
    cv::Mat frame0_yuv;  // 存储从相机复制出来的 YUV 格式图像
    uint64_t timestamp0; // 帧的时间戳
    Size frame_size;     // 帧的尺寸 (如 640x480)
};

// 全局停止标志，`std::atomic` 保证了它在多线程中的读写是安全的
std::atomic<bool> stop_flag(false); 

// 全局指针，指向生产者-消费者队列。在 main() 中初始化
ThreadSafeQueue<FrameData>* g_data_queue = nullptr; 

// 全局指针，指向相机配置。在 main() 中初始化
CameraConfiguration *g_config0 = nullptr; 

// 全局智能指针，指向相机对象。使用 shared_ptr 方便管理生命周期
std::shared_ptr<Camera> camera0 = nullptr; 

// 全局指针，指向帧缓冲区分配器。在 main() 中初始化
FrameBufferAllocator *g_allocator0 = nullptr; 

// 全局向量，存储所有 libcamera 的请求(Request)对象
// 使用 unique_ptr 确保 Request 对象在向量销毁时被正确释放
std::vector<std::unique_ptr<Request>> g_requests0;


// --- 信号处理器 ---
// 当用户按下 Ctrl+C (SIGINT) 或系统发送终止信号 (SIGTERM) 时，此函数会被调用
void signal_handler(int signum) {
    std::cout << ">>> 捕获到中断信号 (" << signum << ")。" << std::endl;
    // 设置全局停止标志为 true，通知所有循环停止
    stop_flag = true; 
}

// --- 奇偶校验辅助函数 ---
// 为一个7位的数据字节添加一个偶校验位，使其成为一个8位字节
// 偶校验：确保最终的8位字节中 '1' 的总数是偶数
uint8_t add_parity(uint8_t byte) {
    int ones = 0; // 用来统计 '1' 的个数
    // 循环遍历前 7 位 (bit 0 到 bit 6)
    for (int i = 0; i < 7; ++i) { 
        if ((byte >> i) & 1) ones++; // 如果第 i 位是 '1'，计数器加 1
    }
    // (byte << 1) 将7位数据左移一位，空出第 0 位 (LSB)
    // (ones % 2 != 0 ? 1 : 0) 检查 '1' 的个数：
    //   - 如果 ones 是奇数 (odd)，则 ( != 0 ) 为 true，返回 1。总 '1' 数变为偶数。
    //   - 如果 ones 是偶数 (even)，则 ( != 0 ) 为 false，返回 0。总 '1' 数保持偶数。
    // 使用 | (按位或) 将校验位设置到第 0 位
    return (byte << 1) | (ones % 2 != 0 ? 1 : 0);
}

// --- 消费者线程 ---
// 这个函数在一个单独的线程 (worker) 中运行，负责处理所有耗时的计算
void processing_worker(ThreadSafeQueue<FrameData>& queue) {
    // 1. 初始化
    Transformer transformer0; // 创建坐标变换对象
    // 加载用于透视变换的单应性矩阵
    if (!transformer0.load_matrix("homography_matrix0.yml")) {
        std::cerr << "[Worker] 错误: 加载 homography_matrix0.yml 失败。正在退出。" << std::endl;
        stop_flag = true; return; // 失败则设置停止标志并退出线程
    }
    
    SpiDevice spi; // 创建 SPI 通信对象
    // 打开 SPI 设备 (总线1，设备0)，速率 500kHz，模式 0
    if (!spi.open("/dev/spidev1.0", 500000, SPI_MODE_0)) {
        std::cerr << "[Worker] 错误: 打开 SPI 设备失败。正在退出。" << std::endl;
        stop_flag = true; return; // 失败则退出
    }
    std::cout << "[Worker] 单应性矩阵和 SPI 初始化完成。" << std::endl;

    // 用于 FPS (每秒帧数) 计算
    auto start_time = std::chrono::high_resolution_clock::now(); // 记录开始时间
    int processed_count = 0; // 记录已处理的帧数
    
    cv::Mat bgr_frame; // 在循环外声明 Mat，避免重复分配内存

    // 2. 主处理循环
    while (!stop_flag) { // 只要 stop_flag 没被设置
        FrameData data; // 创建一个空的数据包
        // 关键：从队列中取数据。如果队列为空，此函数会阻塞（等待）
        queue.wait_and_pop(data); 
        
        // 唤醒后再次检查 stop_flag，因为可能是为了停止线程而被唤醒的
        if (stop_flag) break; 

        // 检查取出的数据是否为空（可能是停止信号或无效数据）
        if (data.frame0_yuv.empty()) {
             if (!stop_flag) std::cerr << "[Worker] 警告: 收到空 YUV 帧。跳过。" << std::endl;
            continue; // 跳过此次循环
        }

        // 健全性检查：YUV I420 格式的高度应为 1.5 倍
        int expected_h = data.frame_size.height * 3 / 2;
        int expected_w = data.frame_size.width;
        if (data.frame0_yuv.rows != expected_h || data.frame0_yuv.cols != expected_w) {
             std::cerr << "[Worker] 错误: YUV Mat 尺寸不匹配！跳过。" << std::endl;
             continue;
        }

        // 3. 图像处理
        // (耗时) 将 YUV I420 格式转换为 BGR 格式，以便 OpenCV 处理
        try { cv::cvtColor(data.frame0_yuv, bgr_frame, cv::COLOR_YUV2BGR_I420); }
        catch (const cv::Exception& e) { std::cerr << "[Worker] 错误: cv::cvtColor 失败: " << e.what() << std::endl; continue; }

        cv::Point2f p0 = cv::Point2f(-1,-1); // 初始化为无效坐标
         // (耗时) 调用 detector.h 中的函数检测圆心
         try { p0 = detect_circle(bgr_frame); }
         catch (const cv::Exception& e) { std::cerr << "[Worker] 错误: detect_circle 失败: " << e.what() << std::endl; }

        // 4. 坐标变换
        double world_x = NAN, world_y = NAN; // 初始化为 "非数字"
        if (p0.x > 0) { // 检查是否检测到了有效的圆心
            try { 
                // (耗时) 使用单应性矩阵将像素坐标 (p0) 转换为真实世界坐标 (w0)
                cv::Point2f w0 = transformer0.camera_to_world(p0); 
                world_x = w0.x; 
                world_y = w0.y; 
            }
            catch (const cv::Exception& e) { std::cerr << "[Worker] 错误: camera_to_world 失败: " << e.what() << std::endl; }
        }
        
        // 5. FPS 计算和打印
        processed_count++; // 处理帧数+1
        auto now = std::chrono::high_resolution_clock::now(); // 获取当前时间
        auto elapsed_chrono = now - start_time; // 计算经过的时间
        long long elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed_chrono).count();
        if (elapsed_sec == 0) elapsed_sec = 1; // 避免除以零
        double fps = static_cast<double>(processed_count) / elapsed_sec; // 计算 FPS
        std::cout << std::fixed << std::setprecision(2); // 设置输出格式
        std::cout << "[Worker] FPS: " << fps << " | 像素坐标 P0: (" << p0.x << "," << p0.y << ") | ";

        // 6. SPI 数据打包和发送
        if (!std::isnan(world_x)) { // 如果世界坐标有效
            std::cout << "世界坐标: (" << world_x << "," << world_y << ") mm" << std::endl;
            
            // --- 数据编码 ---
            // 乘以 10，将毫米 (mm) 变为 0.1 毫米，转为整数，实现定点数
            int x_int = static_cast<int>(world_x * 10); 
            int y_int = static_cast<int>(world_y * 10);
            
            // 将值限制在 ±8191 之间 ( 8191 = 2^13 - 1 )
            // 这意味着我们使用 14 位有符号整数 (1个符号位, 13个数据位)
            x_int = std::max(std::min(x_int, 8191), -8191); 
            y_int = std::max(std::min(y_int, 8191), -8191);
            
            // 提取符号位。如果为负，设置 x_sign 的第 6 位 (0x40 = 0100 0000)
            uint8_t x_sign = (x_int < 0) ? 0x40 : 0x00; 
            uint8_t y_sign = (y_int < 0) ? 0x40 : 0x00;
            
            int x_abs = std::abs(x_int); int y_abs = std::abs(y_int); // 取绝对值
            
            // 将 13 位的数据拆分为 7 位高位和 7 位低位
            // 高 7 位: (x_abs >> 7) & 0x3F (取高 6 位)，然后 | x_sign (或上符号位)
            uint8_t x_high7 = ((x_abs >> 7) & 0x3F) | x_sign; 
            uint8_t x_low7 = x_abs & 0x7F; // 低 7 位: (x_abs & 0x7F) (取低 7 位)
            uint8_t y_high7 = ((y_abs >> 7) & 0x3F) | y_sign; 
            uint8_t y_low7 = y_abs & 0x7F;
            
            // 构建 5 字节的 SPI 数据包
            std::vector<uint8_t> spi_data = { 
                0xAA,                           // 1. 帧头 (起始字节)
                add_parity(x_high7),            // 2. X 坐标高位 + 校验
                add_parity(x_low7),             // 3. X 坐标低位 + 校验
                add_parity(y_high7),            // 4. Y 坐标高位 + 校验
                add_parity(y_low7)              // 5. Y 坐标低位 + 校验
            };
            
            // (耗时) 通过 SPI 发送数据
            spi.transfer(spi_data); 
        } else {
            // 如果没有检测到目标
            std::cout << "世界坐标: 未检测到" << std::endl;
        }

        // 每处理 100 帧，重置一次 FPS 计数器，以获得近期的 FPS
        if (processed_count >= 100) { 
            start_time = now; 
            processed_count = 0; 
        }
    }
    
    // 7. 退出
    spi.close(); // 循环结束，关闭 SPI 设备
    std::cout << "[Worker] 工作线程结束。" << std::endl;
}

// --- 生产者回调 ---
// 辅助函数：从 libcamera 的 FrameBuffer 中克隆一份图像数据
// 这是必要的，因为 FrameBuffer 的内存在回调返回后会被相机驱动重用
cv::Mat cloneFrame(FrameBuffer *buffer, const Size &size) {
    if (!buffer || buffer->planes().empty()) { return cv::Mat(); } // 安全检查
    
    // 获取缓冲区的第一个数据平面 (YUV 数据通常存在一个平面)
    const FrameBuffer::Plane &plane = buffer->planes()[0]; 

    // YUV I420 格式的期望长度 = 宽 * 高 * 1.5
    size_t expected_length = static_cast<size_t>(size.width * size.height * 3 / 2);
    size_t map_length = expected_length; // 默认映射长度
    
    // 检查实际长度是否小于期望长度（在某些平台上可能发生）
    if (plane.length < expected_length) {
         // (注释掉了警告，因为它可能频繁出现但无害)
         // std::cerr << "[cloneFrame] 警告: ... " << std::endl;
    }

    // 关键：mmap (内存映射)
    // 将缓冲区的文件描述符 (plane.fd.get()) 映射到本程序的虚拟地址空间
    // PROT_READ = 只读, MAP_SHARED = 共享内存
    // data_ptr 是一个指向相机 DMA 缓冲区的 *直接指针*
    void *data_ptr = mmap(NULL, map_length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
    if (data_ptr == MAP_FAILED) {
        std::cerr << "[Callback] 错误: mmap 失败! 错误: " << strerror(errno) << std::endl; return cv::Mat();
    }

    cv::Mat yuv_mat;
    cv::Mat cloned_frame;
    try {
        // 创建一个 cv::Mat "头" (Mat wrapper)
        // 它 *指向* data_ptr 所指向的内存，但 *不复制* 数据
        yuv_mat = cv::Mat(size.height * 3 / 2, size.width, CV_8UC1, data_ptr);
        
        if (yuv_mat.empty()) { munmap(data_ptr, map_length); return cv::Mat(); }
        
        // 关键：克隆 (clone)
        // 这会分配一块新的内存，并将 yuv_mat (即 mmap 区域) 的数据
        // *完整地复制* 到新内存 (cloned_frame) 中。
        cloned_frame = yuv_mat.clone(); 
    } catch (const cv::Exception& e) {
        std::cerr << "[Callback] 错误: Mat 创建/克隆时 OpenCV 异常: " << e.what() << std::endl;
        cloned_frame = cv::Mat(); // 确保返回空 Mat
    }
    
    // 关键：解除映射
    // 无论成功与否，都必须解除内存映射
    munmap(data_ptr, map_length); 
    
    return cloned_frame; // 返回这份独立的数据副本
}

// 相机请求完成回调函数
// !!! 此函数在 libcamera 自己的内部线程上运行，必须极快地完成并返回 !!!
void requestComplete0(Request *request) {
    // 安全检查：如果正在停止，或 request/camera 无效，则直接返回
    if (!request || !camera0 || stop_flag) return;

    Request::Status status = request->status(); // 获取请求的状态
    
    // 如果请求是在关机时被取消的，直接返回，不再重新排队
    if (status == Request::Status::RequestCancelled) {
        return; 
    } else if (status != Request::Status::RequestComplete) {
        // 如果请求因其他原因失败
        std::cerr << "[Callback Cam0] 警告: 请求状态非完成: " << static_cast<int>(status) << std::endl;
    }

    // 从完成的请求中获取缓冲区
    if (request->buffers().empty()) {
         std::cerr << "[Callback Cam0] 错误: 没有缓冲区!" << std::endl;
         request->reuse(Request::ReuseBuffers); camera0->queueRequest(request); // 尝试重新排队
         return;
    }
    FrameBuffer *buffer = request->buffers().begin()->second; // 获取缓冲区指针
    if (!buffer) {
        std::cerr << "[Callback Cam0] 错误: 获取缓冲区失败" << std::endl;
        request->reuse(Request::ReuseBuffers); camera0->queueRequest(request); // 尝试重新排队
        return;
    }

    // --- 生产者核心逻辑 ---
    FrameData packet; // 创建一个数据包
    if (!g_config0) { std::cerr << "[Callback Cam0] 错误: g_config0 为空!" << std::endl; request->reuse(Request::ReuseBuffers); camera0->queueRequest(request); return; }
    
    // 从全局配置中获取帧大小
    packet.frame_size = g_config0->at(0).size; 
    
    // 调用辅助函数，深拷贝一份图像数据
    packet.frame0_yuv = cloneFrame(buffer, packet.frame_size); 

    if (!packet.frame0_yuv.empty()) { // 如果克隆成功
         // 获取硬件时间戳
         packet.timestamp0 = *request->metadata().get(controls::SensorTimestamp);
         if (g_data_queue) { // 检查队列指针是否有效
             // 将数据包 "移动" (std::move) 到线程安全队列中
             // std::move 比复制更高效，它转移了数据的所有权
             g_data_queue->push(std::move(packet)); 
         }
    }
    // 如果克隆失败 (frame0_yuv 为空)，我们选择静默丢弃这一帧，不警告

    // --- 关键：重新排队 ---
    // 告诉 libcamera 这个请求的缓冲区可以重用了
    request->reuse(Request::ReuseBuffers); 
    if (camera0 && !stop_flag) { // 再次检查相机是否有效且未停止
        // 将这个请求重新提交给相机队列，以便用于捕获 *未来* 的某一帧
        camera0->queueRequest(request); 
    }
}

// --- 主函数 ---
int main() {
    // 注册信号处理器，捕获 Ctrl+C 和终止信号
    signal(SIGINT, signal_handler); 
    signal(SIGTERM, signal_handler);
    
    // 1. 初始化相机管理器 (CameraManager)
    std::unique_ptr<CameraManager> cm = std::make_unique<CameraManager>(); 
    cm->start(); // 启动管理器
    if (cm->cameras().empty()) { std::cerr << "错误: 未找到相机。" << std::endl; cm->stop(); return -1; }
    
    // 2. 获取并配置相机
    camera0 = cm->cameras()[0]; // 获取系统中的第一个相机

    if (camera0->acquire()) { std::cerr << "获取相机 0 失败" << std::endl; cm->stop(); return -1; }
    std::cout << ">>> 相机 0 已获取。" << std::endl;

    // 生成一个配置，指定我们需要一个 "Viewfinder" (取景器) 角色的流
    std::unique_ptr<CameraConfiguration> config0_ptr = camera0->generateConfiguration({ StreamRole::Viewfinder });
    g_config0 = config0_ptr.get(); // 将裸指针存到全局变量
    
    // 手动设置流的参数
    g_config0->at(0).pixelFormat = formats::YUV420; // 像素格式 YUV420
    g_config0->at(0).size = Size(640, 480);       // 分辨率 640x480
    
    // 验证并应用配置
    if (camera0->configure(g_config0) < 0) {
        std::cerr << "配置相机 0 失败。" << std::endl; camera0->release(); cm->stop(); return -1;
    }
    std::cout << ">>> 相机 0 配置完成: " << g_config0->at(0).pixelFormat.toString() << " " << g_config0->at(0).size.toString() << std::endl;

    // 3. 分配缓冲区
    // 创建一个与相机 0 关联的缓冲区分配器
    auto allocator0_ptr = std::make_unique<FrameBufferAllocator>(camera0);
    g_allocator0 = allocator0_ptr.get(); // 存到全局变量
    
    // 为配置好的流分配内存缓冲区 (libcamera 会决定分配几个)
    if (g_allocator0->allocate(g_config0->at(0).stream()) < 0) {
        std::cerr << "分配缓冲区失败。" << std::endl; camera0->release(); cm->stop(); return -1;
    }
    std::cout << ">>> 缓冲区已分配 (" << g_allocator0->buffers(g_config0->at(0).stream()).size() << ") 个。" << std::endl; // 显示分配了多少个缓冲区
    
    // 4. 创建请求 (Request)
    g_requests0.clear(); // 清空全局请求向量
    // 遍历分配器中的 *每一个* 缓冲区
    for (const std::unique_ptr<FrameBuffer>& buffer : g_allocator0->buffers(g_config0->at(0).stream())) {
        auto request = camera0->createRequest(); // 创建一个请求
        if (!request) { std::cerr << "创建请求失败。" << std::endl; camera0->release(); cm->stop(); return -1; }
        // 将 *这个* 缓冲区添加到 *这个* 请求中
        request->addBuffer(g_config0->at(0).stream(), buffer.get()); 
        // 将这个请求存入全局向量
        g_requests0.push_back(std::move(request)); 
    }
     std::cout << ">>> " << g_requests0.size() << " 个请求已创建。" << std::endl;

    // 5. 启动消费者线程
    // 创建一个容量为 10 的线程安全队列
    ThreadSafeQueue<FrameData> data_queue(10); 
    g_data_queue = &data_queue; // 将其地址存到全局指针
    std::cout << ">>> 正在启动工作线程..." << std::endl;
    // 创建线程，执行 processing_worker 函数，并传入队列的引用
    std::thread worker(processing_worker, std::ref(data_queue)); 
    std::cout << ">>> 工作线程已启动。" << std::endl;

    // 6. 启动相机 (生产者)
    std::cout << ">>> 正在连接相机信号..." << std::endl;
    // 关键：将 "请求完成" 信号连接到我们的回调函数 `requestComplete0`
    camera0->requestCompleted.connect(requestComplete0); 
    std::cout << ">>> 信号已连接。" << std::endl;

    // 设置相机控制参数：帧持续时间 33333 微秒 (约 30 FPS)
    ControlList controls0; controls0.set(controls::FrameDuration, 33333); 
    std::cout << ">>> 正在启动相机..." << std::endl;
    if (camera0->start(&controls0) < 0) { // 启动相机并应用控制参数
         std::cerr << "启动相机 0 失败" << std::endl;
         // 启动失败的清理
         camera0->requestCompleted.disconnect(requestComplete0);
         camera0->release(); cm->stop();
         if (g_data_queue) g_data_queue->push({}); // 通知工作线程退出
         worker.join(); return -1;
    }
     std::cout << ">>> 相机 0 已启动。" << std::endl;

    // 7. 提交初始请求 (Prime the pump)
    int queued_count = 0;
    std::cout << ">>> 正在提交初始请求..." << std::endl;
    for (auto& req : g_requests0) { // 遍历所有创建的请求
        if (!camera0) break; 
        // 将请求提交给相机驱动
        int ret = camera0->queueRequest(req.get()); 
        if (ret < 0) { // 如果提交失败
            std::cerr << "提交请求 " << queued_count << " 失败。错误码: " << ret << std::endl;
            stop_flag = true; break; // 设置停止标志
        }
        queued_count++;
    }
    std::cout << ">>> " << queued_count << " 个初始请求已提交。" << std::endl;

    // 8. 主线程等待
    if (!stop_flag) {
        std::cout << ">>> 开始捕获循环 (主线程空闲)..." << std::endl;
        // 主线程进入睡眠循环，不占用 CPU，直到 stop_flag 变为 true
        while (!stop_flag) { 
            std::this_thread::sleep_for(100ms); 
        }
    }

    // --- V25 改进的关机序列 ---
    // (当用户按下 Ctrl+C, stop_flag 变为 true, 上面的循环退出，代码执行到这里)
    std::cout << ">>> 收到停止信号。开始关机..." << std::endl;
    if (camera0) {
        std::cout << ">>> 正在停止相机..." << std::endl;
        camera0->stop(); // 停止相机流
        std::cout << ">>> 相机 0 已停止。" << std::endl;
        std::cout << ">>> 正在断开信号..." << std::endl;
        // 断开回调连接，这样就不会再有 requestComplete0 被调用
        camera0->requestCompleted.disconnect(requestComplete0); 
        std::cout << ">>> 信号已断开。" << std::endl;
    }

    if (g_data_queue) {
        std::cout << ">>> 正在向工作线程发送停止信号..." << std::endl;
        // 向队列中 push 一个空数据包
        // 正在 `wait_and_pop` 的工作线程会收到这个包，然后退出循环
        g_data_queue->push({}); 
    }
    if (worker.joinable()) {
        std::cout << ">>> 正在等待工作线程结束..." << std::endl;
        worker.join(); // 阻塞，直到工作线程 (worker) 完全退出
        std::cout << ">>> 工作线程已结束。" << std::endl;
    }

    // 现在工作线程和相机回调都已停止，可以安全地释放资源
    if (camera0) {
         std::cout << ">>> 正在释放相机..." << std::endl;
         camera0->release(); // 释放对相机的占用
         std::cout << ">>> 相机 0 已释放。" << std::endl;
    }
    if (g_allocator0 && g_config0) { // 检查指针有效性
         std::cout << ">>> 正在释放缓冲区..." << std::endl;
         int ret = g_allocator0->free(g_config0->at(0).stream()); // 释放所有缓冲区内存
         std::cout << ">>> 缓冲区已释放 (返回码: " << ret << ")。" << std::endl;
    }
     g_requests0.clear(); // 清空请求向量 (销毁所有 unique_ptr)
     
     // 显式地将所有全局指针设为 null，防止悬挂指针
     camera0.reset();
     g_allocator0 = nullptr;
     g_config0 = nullptr;
     g_data_queue = nullptr;

    if (cm) {
        std::cout << ">>> 正在停止相机管理器..." << std::endl;
        cm->stop(); // 停止相机管理器
    }
    std::cout << ">>> 系统安全停止。" << std::endl;
    return 0; // 程序正常退出
}