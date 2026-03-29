import numpy as np
import cv2

DATA_SET = 1  # 选择数据集编号（0或1）

# 原始数据
data0 = np.array([
    [72, 166, -74.25, 525],
    [113, 177, -74.25, 315],
    [72, 77, 74.25, 525],
    [113, 45, 74.25, 315],
    [60, 198, -148.5, 630],
    [72, 209, -148.5, 525],
    [89, 225, -148.5, 420],
    [114, 246, -148.5, 315],
    [59, 47, 148.5, 630],
    [71, 29, 148.5, 525],
    [87, 9, 148.5, 420],
    [60, 123, 0, 630],
    [72, 120, 0, 525],
    [89, 118, 0, 420],
    [114, 113, 0, 315],
    [155, 105, 0, 210],
    [242, 87, 0, 105]
])

data1 = np.array([
    [62, 197, -74.25, 525],
    [106, 228, -74.25, 315],
    [62, 109, 74.25, 525],
    [106, 94, 74.25, 315],
    [50, 231, -148.5, 630],
    [63, 247, -148.5, 525],
    [51, 75, 148.5, 630],
    [62, 63, 148.5, 525],
    [80, 49, 148.5, 420],
    [107, 23, 148.5, 315],
    [152, 4, 148.5, 210],
    [51, 150, 0, 630],
    [64, 152, 0, 525],
    [81, 156, 0, 420],
    [106, 160, 0, 315],
    [150, 168, 0, 210],
    [237, 188, 0, 105]
])

# 分离坐标
if DATA_SET == 0:
    data = data0
else:
    data = data1
img_points = data[:, :2].astype(np.float32)
phys_points = data[:, 2:].astype(np.float32)

# 计算单应性矩阵
image_pts = img_points.reshape(-1, 1, 2)
object_pts = phys_points.reshape(-1, 1, 2)
H, _ = cv2.findHomography(image_pts, object_pts)

def camera_to_world(x0, y0):
    """将摄像头坐标(x0, y0)转换为物理坐标(x, y)"""
    src_pt = np.array([[x0, y0]], dtype=np.float32)
    dst_pt = cv2.perspectiveTransform(src_pt.reshape(1, -1, 2), H)
    return dst_pt[0][0]

# 验证所有点精度
print("=== 透视变换精度验证 ===")
errors = []
for i, (img_pt, phys_pt) in enumerate(zip(img_points, phys_points)):
    pred = camera_to_world(img_pt[0], img_pt[1])
    error = np.sqrt((pred[0]-phys_pt[0])**2 + (pred[1]-phys_pt[1])**2)
    errors.append(error)
    print(f"点{i+1}: 预测({pred[0]:.2f},{pred[1]:.2f}) | 实际({phys_pt[0]},{phys_pt[1]}) | 误差:{error:.2f}mm")

print(f"\n平均误差: {np.mean(errors):.2f}mm, 最大误差: {np.max(errors):.2f}mm")
'''
# 保存变换矩阵
if DATA_SET == 0:
    np.save("homography_matrix0.yml", H)
    print("\n单应性矩阵已保存为 'homography_matrix0.yml'")
else:
    np.save("homography_matrix1.yml", H)
    print("\n单应性矩阵已保存为 'homography_matrix1.yml'")
'''
# 保存变换矩阵
if DATA_SET == 0:
    filename = "homography_matrix0.yml"
    print(f"\n单应性矩阵已保存为 '{filename}'")
else:
    filename = "homography_matrix1.yml"
    print(f"\n单应性矩阵已保存为 '{filename}'")

# 使用OpenCV的FileStorage保存为YML格式
fs = cv2.FileStorage(filename, cv2.FileStorage_WRITE)
fs.write("homography_matrix", H)
fs.release()

# 使用示例
test_points = [
    (71, 160),   # 已知点
    (100, 150),  # 新点
    (200, 100)   # 新点
]

print("\n=== 坐标映射示例 ===")
for pt in test_points:
    world_pt = camera_to_world(pt[0], pt[1])
    print(f"摄像头坐标 ({pt[0]}, {pt[1]}) -> 物理坐标 ({world_pt[0]:.2f}, {world_pt[1]:.2f}) mm")
