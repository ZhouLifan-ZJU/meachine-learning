import numpy as np
import matplotlib.pyplot as plt


def whetherLinearSeparable(X):
    """
    判断点集是否线性可分

    参数:
    X: numpy数组，每行前n-1列为特征坐标，最后一列为类别标签(1或-1)

    返回:
    Y: 1表示线性可分，-1表示线性不可分
    """
    # 分离特征和标签
    features = X[:, :-1]  # 所有行，除最后一列外的所有列
    labels = X[:, -1]  # 所有行，最后一列

    # 添加偏置项
    n_samples, n_features = features.shape
    X_with_bias = np.c_[np.ones(n_samples), features]  # 添加一列1作为偏置

    # 初始化权重
    weights = np.zeros(n_features + 1)

    # 使用感知机算法
    max_iterations = 1000
    for iteration in range(max_iterations):
        misclassified = False
        for i in range(n_samples):
            # 计算预测值
            prediction = np.sign(np.dot(X_with_bias[i], weights))
            # 如果预测为0，视为错误分类（因为标签是±1）
            if prediction == 0:
                prediction = -1

            # 如果分类错误，更新权重
            if prediction != labels[i]:
                weights += labels[i] * X_with_bias[i]
                misclassified = True

        # 如果没有错误分类，则线性可分
        if not misclassified:
            return 1

    # 如果达到最大迭代次数仍有错误分类，则线性不可分
    return -1


# 创建单个函数绘制所有子图
def plot_all_datasets(X1, X2, X3):
    """
    将三个数据集绘制在同一张图的子图中
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    datasets = [X1, X2, X3]
    titles = ["Example 1", "Example 2", "Example 3"]

    for i, (X, title) in enumerate(zip(datasets, titles)):
        # 分离正负样本
        positive = X[X[:, 2] == 1]
        negative = X[X[:, 2] == -1]

        # 绘制散点图
        axes[i].scatter(positive[:, 0], positive[:, 1], c='red', marker='o', label='Class +1', s=50)
        axes[i].scatter(negative[:, 0], negative[:, 1], c='blue', marker='x', label='Class -1', s=50)

        # 设置子图标题和标签
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
        axes[i].set_title(f'{title} (Linear Separable: {whetherLinearSeparable(X)})')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].axis('equal')

    plt.tight_layout()
    plt.savefig('linear_separability.png')
    plt.close()


# 测试示例
if __name__ == "__main__":
    # 示例1：线性不可分
    X1 = np.array([[-0.5, -0, 1], [3.5, 4.1, -1], [4.5, 6, -1], [-2, -2.0, -1], [-4.1, -2.8, -1],
                   [1, 3, -1], [-7.1, -4.2, 1], [-6.1, -2.2, 1], [-4.1, 2.2, 1], [1.4, 4.3, 1],
                   [-2.4, 4.0, 1], [-8.4, -5, 1]])

    # 示例2：线性不可分
    X2 = np.array([[-0.5, -0, -1], [3.5, 4.1, -1], [4.5, 6, 1], [-2, -2.0, -1], [-4.1, -2.8, -1],
                   [1, 3, -1], [-7.1, -4.2, 1], [-6.1, -2.2, 1], [-4.1, 2.2, 1], [1.4, 4.3, 1],
                   [-2.4, 4.0, 1], [-8.4, -5, 1]])

    # 示例3：线性可分
    X3 = np.array([[-0.5, -0, -1], [3.5, 4.1, -1], [4.5, 6, -1], [-2, -2.0, -1], [-4.1, -2.8, -1],
                   [1, 3, -1], [-7.1, -4.2, 1], [-6.1, -2.2, 1], [-4.1, 2.2, 1], [1.4, 4.3, -1],
                   [-2.4, 4.0, 1], [-8.4, -5, 1]])

    print("示例1结果:", whetherLinearSeparable(X1))  # 应该输出-1
    print("示例2结果:", whetherLinearSeparable(X2))  # 应该输出-1
    print("示例3结果:", whetherLinearSeparable(X3))  # 应该输出1

    # 绘制三个示例在同一张图中
    plot_all_datasets(X1, X2, X3)