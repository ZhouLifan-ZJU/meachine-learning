import numpy as np
import matplotlib.pyplot as plt

# 方法 1：使用系统中的中文字体（推荐 SimHei 或 Microsoft YaHei）
plt.rcParams['font.sans-serif'] = ['SimHei']   # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题


def visualize_distribution_changes():
    # 模拟不同层的输入分布变化
    np.random.seed(42)

    # 模拟4个不同层的输入分布
    distributions = [
        np.random.normal(0.1, 0.5, 1000),  # 第1层输入
        np.random.normal(2.8, 3.2, 1000),  # 第2层输入
        np.random.normal(-0.5, 0.8, 1000),  # 第3层输入
        np.random.normal(1.2, 2.5, 1000)  # 第4层输入
    ]

    labels = ['第1层输入', '第2层输入', '第3层输入', '第4层输入']

    plt.figure(figsize=(12, 8))

    # 绘制原始分布
    plt.subplot(2, 2, 1)
    for i, dist in enumerate(distributions):
        plt.hist(dist, bins=50, alpha=0.6, label=labels[i])
    plt.title('没有BN: 各层输入分布差异巨大')
    plt.xlabel('输入值')
    plt.ylabel('频次')
    plt.legend()

    # 绘制BN后的分布
    plt.subplot(2, 2, 2)
    for i, dist in enumerate(distributions):
        # 模拟BN操作
        mean = np.mean(dist)
        std = np.std(dist)
        bn_dist = (dist - mean) / (std + 1e-8)
        plt.hist(bn_dist, bins=50, alpha=0.6, label=labels[i])
    plt.title('有BN: 各层输入分布基本一致')
    plt.xlabel('标准化后的输入值')
    plt.ylabel('频次')
    plt.legend()

    # 统计量对比
    plt.subplot(2, 2, 3)
    original_stats = []
    bn_stats = []

    for dist in distributions:
        original_stats.append([np.mean(dist), np.std(dist)])
        bn_dist = (dist - np.mean(dist)) / (np.std(dist) + 1e-8)
        bn_stats.append([np.mean(bn_dist), np.std(bn_dist)])

    x = range(len(distributions))
    original_means = [s[0] for s in original_stats]
    bn_means = [s[0] for s in bn_stats]

    plt.plot(x, original_means, 'ro-', label='原始均值', markersize=8)
    plt.plot(x, bn_means, 'bo-', label='BN后均值', markersize=8)
    plt.title('均值变化对比')
    plt.xlabel('网络层')
    plt.ylabel('均值')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

visualize_distribution_changes()