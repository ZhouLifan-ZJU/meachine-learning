import numpy as np
import random
import time
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from collections import Counter


# =========================
# 数据预处理模块
# =========================
def load_krkopt_data(file_path):
    """
    加载并解析 krkopt.data 数据集
    将字母特征映射为数字
    """
    letter_map = {c: i + 1 for i, c in enumerate("abcdefgh")}
    X, y = [], []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 7:
                continue
            features = []
            for i, val in enumerate(parts[:-1]):
                if val.isdigit():
                    features.append(float(val))
                else:
                    features.append(float(letter_map.get(val, 0)))
            label = parts[-1].strip()
            y_val = 1 if label == 'fifteen' else -1  # 简化为二分类
            X.append(features)
            y.append(y_val)

    X = np.array(X)
    y = np.array(y)
    print(f"数据集加载完成，共 {len(X)} 条样本。")
    print(f"标签类别分布: {Counter(y)}")
    return X, y


# =========================
# 核心模块：SMO 实现
# =========================
def svm_train(X, y, C=1.0, tol=1e-3, max_passes=5, kernel='rbf', gamma=0.05, max_iter=10):
    """
    使用简化版 SMO 算法训练 SVM 模型（可收敛优化版）
    """

    n_samples = X.shape[0]
    alphas = np.zeros(n_samples)
    b = 0
    passes = 0

    print("计算核矩阵 K(x_i, x_j)... (使用向量化加速)")
    if kernel == 'linear':
        K = X @ X.T
    else:
        K = rbf_kernel(X, X, gamma=gamma)

    print("开始训练 SVM 模型（优化版 SMO）...")
    start_time = time.time()

    for iteration in range(max_iter):
        num_changed = 0
        for i in range(n_samples):
            Ei = np.dot(alphas * y, K[:, i]) + b - y[i]
            if (y[i] * Ei < -tol and alphas[i] < C) or (y[i] * Ei > tol and alphas[i] > 0):
                j = random.choice([jj for jj in range(n_samples) if jj != i])
                Ej = np.dot(alphas * y, K[:, j]) + b - y[j]

                alpha_i_old, alpha_j_old = alphas[i], alphas[j]
                if y[i] != y[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                if L == H:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alphas[j] -= y[j] * (Ei - Ej) / eta
                alphas[j] = np.clip(alphas[j], L, H)

                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue

                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                b1 = b - Ei - y[i] * (alphas[i] - alpha_i_old) * K[i, i] - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                b2 = b - Ej - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) * K[j, j]
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                num_changed += 1

            # 每处理1000个样本打印进度
            if i % 1000 == 0 and i > 0:
                print(f"  已处理 {i}/{n_samples} 样本, 当前更新 α 数: {num_changed}")

        print(f"迭代 {iteration+1}/{max_iter} 完成, 本轮更新 α: {num_changed}")
        if num_changed == 0:
            passes += 1
        else:
            passes = 0

        # 早停条件
        if passes >= 2:
            print("连续两轮无 α 更新，提前收敛。")
            break

    print(f"训练完成，用时 {time.time() - start_time:.2f} 秒")
    return {"alphas": alphas, "b": b, "X": X, "y": y, "kernel": kernel, "gamma": gamma}


def svm_predict(model, X_test):
    """
    预测函数
    """
    alphas, b, X_train, y_train = model["alphas"], model["b"], model["X"], model["y"]
    kernel = model["kernel"]
    gamma = model["gamma"]

    if kernel == 'linear':
        w = np.dot(alphas * y_train, X_train)
        preds = np.sign(X_test @ w + b)
    else:
        K = rbf_kernel(X_test, X_train, gamma=gamma)
        preds = np.sign(np.dot(K, alphas * y_train) + b)
    return preds


# =========================
# 主程序入口
# =========================
if __name__ == "__main__":
    print("加载 krkopt.data 数据集...")
    X, y = load_krkopt_data("krkopt.data")

    # ✅ 按类别均衡抽样，保留代表性样本
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == -1)[0]
    sample_size = min(len(pos_idx), len(neg_idx), 3000)
    sel_idx = np.concatenate([
        np.random.choice(pos_idx, sample_size, replace=False),
        np.random.choice(neg_idx, sample_size, replace=False)
    ])
    X, y = X[sel_idx], y[sel_idx]
    print(f"采样后样本数: {len(X)} (平衡集)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = svm_train(X_train, y_train, C=1.0, kernel='rbf', gamma=0.05, max_iter=10)
    y_pred = svm_predict(model, X_test)

    acc = np.mean(y_pred == y_test)
    print(f"测试集准确率: {acc:.4f}")
