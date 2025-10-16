import numpy as np
from sklearn.preprocessing import LabelEncoder
import time


# ===========================================================
# svm_train — 实现基于 SMO 的 SVM 训练函数
# ===========================================================
def svm_train(X, y, C=1.0, tol=1e-3, max_passes=5, kernel='linear', gamma=0.05):
    """
    使用 SMO 算法训练 SVM 模型
    参数:
        X: 训练特征矩阵 (N, d)
        y: 标签向量 (-1 或 +1)
        C: 惩罚系数
        tol: 容忍误差
        max_passes: 连续无更新的迭代次数上限
        kernel: 核函数类型 ('linear' 或 'rbf')
        gamma: RBF核参数
    返回:
        model: dict，包含训练得到的模型参数
    """

    # 定义核函数
    def kernel_func(x1, x2):
        if kernel == 'linear':
            return np.dot(x1, x2)
        elif kernel == 'rbf':
            return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("未知核函数类型")

    n_samples = X.shape[0]
    alphas = np.zeros(n_samples)
    b = 0
    passes = 0

    print("计算核矩阵 K(x_i, x_j)... (使用向量化加速)")
    if kernel == 'linear':
        # 线性核可以直接用矩阵乘法计算
        K = X @ X.T
    else:
        # 使用 sklearn 的高效 RBF kernel 计算（底层 C 实现，速度快）
        from sklearn.metrics.pairwise import rbf_kernel
        K = rbf_kernel(X, X, gamma=gamma)

    print("开始训练 SVM 模型（SMO算法）...")
    start_time = time.time()

    # SMO 主循环
    while passes < max_passes:
        num_changed = 0
        for i in range(n_samples):
            Ei = np.dot(alphas * y, K[:, i]) + b - y[i]

            if (y[i] * Ei < -tol and alphas[i] < C) or (y[i] * Ei > tol and alphas[i] > 0):
                j = np.random.choice([x for x in range(n_samples) if x != i])
                Ej = np.dot(alphas * y, K[:, j]) + b - y[j]

                ai_old, aj_old = alphas[i], alphas[j]

                # 计算 L, H
                if y[i] != y[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                if L == H:
                    continue

                # 计算 eta
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                # 更新 α_j
                alphas[j] -= y[j] * (Ei - Ej) / eta
                alphas[j] = np.clip(alphas[j], L, H)
                if abs(alphas[j] - aj_old) < 1e-5:
                    continue

                # 更新 α_i
                alphas[i] += y[i] * y[j] * (aj_old - alphas[j])

                # 更新 b
                b1 = b - Ei - y[i]*(alphas[i]-ai_old)*K[i, i] - y[j]*(alphas[j]-aj_old)*K[i, j]
                b2 = b - Ej - y[i]*(alphas[i]-ai_old)*K[i, j] - y[j]*(alphas[j]-aj_old)*K[j, j]

                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                num_changed += 1

        print(f"迭代完成一次: passes={passes}, 本轮更新了 {num_changed} 个 α")

        if num_changed == 0:
            passes += 1
        else:
            passes = 0

    print(f"训练完成，总耗时: {time.time() - start_time:.2f} 秒")
    print(f"支持向量数量: {(alphas > 1e-5).sum()}")

    model = {
        'alphas': alphas,
        'b': b,
        'X': X,
        'y': y,
        'C': C,
        'kernel': kernel,
        'gamma': gamma
    }
    return model


# ===========================================================
# svm_predict — 使用训练好的模型进行预测
# ===========================================================
def svm_predict(model, X):
    alphas = model['alphas']
    b = model['b']
    X_train = model['X']
    y_train = model['y']
    kernel = model['kernel']
    gamma = model['gamma']

    def kernel_func(x1, x2):
        if kernel == 'linear':
            return np.dot(x1, x2)
        elif kernel == 'rbf':
            return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

    print("开始预测测试集...")
    y_pred = []
    for idx, x in enumerate(X):
        f = np.sum(alphas * y_train * np.array([kernel_func(x, xj) for xj in X_train])) + b
        y_pred.append(np.sign(f))
        if idx % 20 == 0:
            print(f"预测进度：{idx+1}/{len(X)}")
    return np.array(y_pred)


# ===========================================================
# 加载 krkopt 数据集
# ===========================================================
def load_data_krkopt(path):
    print("加载 krkopt.data 数据集...")
    data = np.loadtxt(path, dtype=str, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]

    # 将象棋坐标字母数字化
    X_encoded = np.array([LabelEncoder().fit_transform(col) for col in X.T]).T

    # 将 draw 类作为 +1，其余作为 -1（二分类）
    y_encoded = np.where(y == 'draw', 1, -1)
    print(f"数据集加载完成，共 {len(X)} 条样本。")
    print(f"标签类别分布: +1={np.sum(y_encoded==1)}, -1={np.sum(y_encoded==-1)}")

    return X_encoded, y_encoded


# ===========================================================
# 主程序
# ===========================================================
if __name__ == "__main__":
    X, y = load_data_krkopt("krkopt.data")

    # 划分训练与测试集
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"训练集: {len(X_train)} 条，测试集: {len(X_test)} 条")

    # 训练模型
    model = svm_train(X_train, y_train, C=1.0, kernel='linear')

    # 测试模型
    y_pred = svm_predict(model, X_test)

    # 输出准确率
    acc = np.mean(y_pred == y_test)
    print(f"测试集准确率: {acc:.4f}")
