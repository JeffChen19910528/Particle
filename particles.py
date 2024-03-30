import numpy as np
import matplotlib.pyplot as plt
import os

# 創建存儲圖片的目錄
if not os.path.exists('particle_movement'):
    os.makedirs('particle_movement')

# 定義函數來計算兩個點之間的歐氏距離
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# 初始化粒子數量和平面大小
num_particles = 100
plane_size = 100

# 隨機初始化粒子坐標
particles = np.random.rand(num_particles, 2) * plane_size

# 定義函數來更新粒子位置
def update_particles(particles, k=3):
    updated_particles = np.copy(particles)
    for i, particle in enumerate(particles):
        # 找到最近的 k 個鄰居
        distances = [distance(particle, p) for p in particles]
        nearest_neighbors_indices = np.argsort(distances)[1:k+1]
        centroid = np.mean(particles[nearest_neighbors_indices], axis=0)
        # 將粒子移向鄰居的質心
        updated_particles[i] += (centroid - particle) * 0.1  # 移動速度因子可調整
        # 確保與其他粒子保持最小距離
        for j, other_particle in enumerate(particles):
            if i != j:
                dist = distance(updated_particles[i], other_particle)
                if dist < 1.0:
                    updated_particles[i] += (updated_particles[i] - other_particle) * 0.1
    return updated_particles

# 迭代更新位置並繪製圖形
iterations = 10
for i in range(iterations):
    particles = update_particles(particles)
    plt.figure(figsize=(8, 8))
    plt.scatter(particles[:, 0], particles[:, 1], color='b', alpha=0.6)
    plt.title(f"Iteration {i+1}")
    plt.xlim(0, plane_size)
    plt.ylim(0, plane_size)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)
    plt.savefig(f'particle_movement/iteration_{i+1}.png')  # 另存圖片
    plt.close()

print("Particle movement images saved successfully.")