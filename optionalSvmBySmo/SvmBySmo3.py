# my_svm_smo.py
# -*- coding: utf-8 -*-
"""
Simple SMO-based SVM implementation (C-SVC + RBF kernel)
Implements svm_train / svm_predict similar to LIBSVM style.

Author: Lifan Zhou
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any


# ===========================
# 1. 内核函数
# ===========================

def linear_kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """线性核 K(x,z) = x^T z"""
    return X1 @ X2.T


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """RBF 核 K(x,z) = exp(-gamma * ||x - z||^2)"""
    X1_sq = np.sum(X1 ** 2, axis=1)[:, None]
    X2_sq = np.sum(X2 ** 2, axis=1)[None, :]
    sq_dists = X1_sq + X2_sq - 2 * (X1 @ X2.T)
    return np.exp(-gamma * sq_dists)


# ===========================
# 2. 二分类 SVM（SMO 求解）
# ===========================

class BinarySMO_SVM:
    """
    使用 Platt 简化 SMO 训练二分类 SVM。
    标签必须是 y ∈ {+1, -1}

    参考：李航《统计学习方法》第 7.4 节。
    """

    def __init__(self, C=1.0, kernel='rbf', gamma=0.5,
                 tol=1e-3, max_passes=3, max_epochs=5, random_state=0):
        """
        :param C: 惩罚系数
        :param kernel: 'linear' or 'rbf'
        :param gamma: RBF 核参数
        :param tol: KKT 容忍度
        :param max_passes: 连续几轮没有 alpha 更新则认为收敛
        :param max_epochs: 最多完整扫描训练集的轮数（硬上限，防止过长运行时间）
        :param random_state: 随机种子
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.max_epochs = max_epochs
        self.random_state = random_state

        # 训练好后存储的量
        self.alphas = None
        self.b = 0.0
        self.X = None
        self.y = None
        self.K = None  # 核矩阵

    # ----------- 内部工具 ------------

    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """预计算训练样本之间的核矩阵 K_ij = K(x_i, x_j)"""
        if self.kernel == 'linear':
            return linear_kernel(X, X)
        elif self.kernel == 'rbf':
            return rbf_kernel(X, X, self.gamma)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    # ----------- 训练函数（SMO 主循环） ------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        使用 SMO 训练 SVM。
        :param X: (m, n) 训练样本
        :param y: (m,) 标签，必须为 +1/-1
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m, n = X.shape

        self.X = X
        self.y = y
        self.alphas = np.zeros(m)
        self.b = 0.0
        self.K = self._compute_kernel_matrix(X)

        rng = np.random.default_rng(self.random_state)
        passes = 0
        epoch = 0  # 完整扫描次数

        print(f"[BinarySMO] Start training, m={m}, C={self.C}, "
              f"kernel={self.kernel}, max_epochs={self.max_epochs}")

        # SMO 外层循环：不断扫描所有样本
        while passes < self.max_passes and epoch < self.max_epochs:
            num_changed_alphas = 0

            for i in range(m):
                # f(x_i) = sum_j alpha_j y_j K_ji + b
                f_i = (self.alphas * y) @ self.K[:, i] + self.b
                E_i = f_i - y[i]

                # 检查 KKT 条件是否被严重违反
                if ((y[i] * E_i < -self.tol and self.alphas[i] < self.C) or
                        (y[i] * E_i > self.tol and self.alphas[i] > 0)):

                    # 随机选择 j != i 作为配对变量
                    j = i
                    while j == i:
                        j = rng.integers(0, m)

                    f_j = (self.alphas * y) @ self.K[:, j] + self.b
                    E_j = f_j - y[j]

                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]

                    # 计算 [L, H]
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])

                    if L == H:
                        continue

                    # eta = 2*K_ij - K_ii - K_jj
                    eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    # eta >= 0 时，对偶目标不是严格凹，跳过
                    if eta >= 0:
                        continue

                    # 更新 alpha_j
                    self.alphas[j] -= y[j] * (E_i - E_j) / eta

                    # 截断到 [L, H]
                    if self.alphas[j] > H:
                        self.alphas[j] = H
                    elif self.alphas[j] < L:
                        self.alphas[j] = L

                    # 更新幅度太小，跳过
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        self.alphas[j] = alpha_j_old
                        continue

                    # 根据约束更新 alpha_i
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])

                    # 计算两个候选 b
                    b1 = (self.b - E_i
                          - y[i] * (self.alphas[i] - alpha_i_old) * self.K[i, i]
                          - y[j] * (self.alphas[j] - alpha_j_old) * self.K[i, j])

                    b2 = (self.b - E_j
                          - y[i] * (self.alphas[i] - alpha_i_old) * self.K[i, j]
                          - y[j] * (self.alphas[j] - alpha_j_old) * self.K[j, j])

                    # 根据 alpha_i / alpha_j 是否在 (0, C) 内来选择 b
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = 0.5 * (b1 + b2)

                    num_changed_alphas += 1

            print(f"[BinarySMO] epoch={epoch}, pass={passes}, "
                  f"num_changed_alphas={num_changed_alphas}")

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

            epoch += 1

        print(f"[BinarySMO] Stop training at epoch={epoch}, passes={passes}")

        # 只保留支持向量
        sv = self.alphas > 1e-6
        self.X = self.X[sv]
        self.y = self.y[sv]
        self.alphas = self.alphas[sv]

        print(f"[BinarySMO] Done. #SV = {self.alphas.shape[0]}")
        return self

    # ----------- 预测相关 ------------

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """计算 f(x) = sum_i alpha_i y_i K(x_i, x) + b"""
        X = np.asarray(X, dtype=float)
        if self.kernel == 'linear':
            K = linear_kernel(X, self.X)
        else:
            K = rbf_kernel(X, self.X, self.gamma)

        # (N, m) * (m,) 逐元素乘以 alpha_i y_i 再求和
        return (K * (self.alphas * self.y)).sum(axis=1) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        f = self._decision_function(X)
        y_pred = np.sign(f)
        y_pred[y_pred == 0] = 1  # 在边界上的点归为 +1
        return y_pred


# ===========================
# 3. 多分类 One-vs-Rest 封装
# ===========================

class OneVsRestSVM:
    """
    用 BinarySMO_SVM 做 One-vs-Rest 多分类。
    每个类别训练一个“该类 vs 其他类”的二分类器。
    """

    def __init__(self, C=1.0, kernel='rbf', gamma=0.5,
                 tol=1e-3, max_passes=3, max_epochs=5, random_state=0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.max_epochs = max_epochs
        self.random_state = random_state

        self.classes_ = None
        self.models_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self.models_ = []

        num_classes = len(self.classes_)
        print(f"[OneVsRest] Start training, num_classes={num_classes}")

        for idx, c in enumerate(self.classes_):
            print(f"[OneVsRest] Training classifier {idx + 1}/{num_classes} for class={c}")
            # y_binary = +1 (当前类), -1 (其他类)
            y_binary = np.where(y == c, 1.0, -1.0)
            model = BinarySMO_SVM(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                tol=self.tol,
                max_passes=self.max_passes,
                max_epochs=self.max_epochs,
                random_state=self.random_state + idx
            )
            model.fit(X, y_binary)
            self.models_.append(model)
            print(f"[OneVsRest] Finished classifier {idx + 1}/{num_classes}")

        print("[OneVsRest] All classifiers trained.")
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        scores = []
        for model in self.models_:
            scores.append(model._decision_function(X))
        return np.vstack(scores)  # (num_classes, N)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        best = np.argmax(scores, axis=0)
        return self.classes_[best]


# ===========================
# 4. LIBSVM 风格接口
# ===========================

def svm_train(X: np.ndarray, y: np.ndarray,
              C=1.0, kernel='rbf', gamma=0.5,
              tol=1e-3, max_passes=3, random_state=0) -> Dict[str, Any]:
    """
    LIBSVM 风格的训练接口。
    - 若 y 只有两个不同取值，则训练二分类 SVM
    - 否则使用 One-vs-Rest 多分类
    返回的 model 是一个字典，内部封装上述类实例。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    unique = np.unique(y)
    if len(unique) == 2:
        # 二分类：先把标签映射到 +1/-1
        y_pos = unique[0]
        y_mapped = np.where(y == y_pos, 1.0, -1.0)
        print(f"[svm_train] Binary classification, labels={unique}")
        binary_model = BinarySMO_SVM(
            C=C, kernel=kernel, gamma=gamma,
            tol=tol, max_passes=max_passes,
            max_epochs=5,           # 硬上限，可按需要改
            random_state=random_state
        ).fit(X, y_mapped)
        return {"type": "binary",
                "label_pos": unique[0],
                "label_neg": unique[1],
                "model": binary_model}
    else:
        # 多分类 One-vs-Rest
        classes_, y_indices = np.unique(y, return_inverse=True)
        print(f"[svm_train] Multiclass classification, num_classes={len(classes_)}")
        ovr = OneVsRestSVM(
            C=C, kernel=kernel, gamma=gamma,
            tol=tol, max_passes=max_passes,
            max_epochs=5,           # 硬上限，可按需要改
            random_state=random_state
        ).fit(X, y_indices)
        return {"type": "multiclass",
                "classes": classes_,
                "ovr": ovr}


def svm_predict(model: Dict[str, Any], X: np.ndarray):
    """
    LIBSVM 风格的预测接口。
    返回 (y_pred, decision_values)
    """
    X = np.asarray(X, dtype=float)
    if model["type"] == "binary":
        binary_model: BinarySMO_SVM = model["model"]
        f = binary_model._decision_function(X)
        y_sign = np.sign(f)
        y_sign[y_sign == 0] = 1
        # 将 +1/-1 映射回原始标签
        y_pred = np.where(y_sign == 1, model["label_pos"], model["label_neg"])
        return y_pred, f

    else:
        ovr: OneVsRestSVM = model["ovr"]
        scores = ovr.decision_function(X)  # (K, N)
        best = np.argmax(scores, axis=0)
        y_pred = model["classes"][best]
        return y_pred, scores


# ===========================
# 5. 在 krkopt.data 上的示例
# ===========================

def load_krkopt(path: str):
    """读取并数值化 krkopt.data 数据集"""
    cols = ["wk_file", "wk_rank",
            "wr_file", "wr_rank",
            "bk_file", "bk_rank",
            "label"]
    df = pd.read_csv(path, header=None, names=cols)

    file_map = {c: i for i, c in enumerate("abcdefgh", start=1)}
    rank_map = {str(i): i for i in range(1, 9)}

    def encode_row(row):
        return [
            file_map[row["wk_file"]],
            rank_map[str(row["wk_rank"])],
            file_map[row["wr_file"]],
            rank_map[str(row["wr_rank"])],
            file_map[row["bk_file"]],
            rank_map[str(row["bk_rank"])],
        ]

    X = np.array([encode_row(r) for _, r in df.iterrows()], dtype=float)
    y = df["label"].values  # 多分类标签（字符串）
    return X, y


def train_test_split_manual(X, y, test_ratio=0.3, random_state=0):
    """简单手写版 train/test 划分（打乱后前一部分为 train）"""
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    indices = rng.permutation(n)
    test_size = int(n * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


if __name__ == "__main__":
    # === 路径改成你本地 krkopt.data 的路径 ===
    data_path = "krkopt.data"

    # 1. 读取数据并归一化
    X_all, y_all = load_krkopt(data_path)
    X_all = X_all / 8.0  # 简单缩放到 [0,1]

    n_total = X_all.shape[0]
    print(f"[MAIN] Loaded {n_total} samples from {data_path}")

    # 为了控制 SMO 的计算量，这里对大规模数据集做一次随机抽样。
    # 报告里可以说明：使用 krkopt 数据集的随机子集进行训练。
    N_SUBSET = 2000  # 比 4000 更轻，方便 5 分钟内跑完
    if N_SUBSET < n_total:
        rng = np.random.default_rng(0)
        subset_idx = rng.permutation(n_total)[:N_SUBSET]
        X = X_all[subset_idx]
        y = y_all[subset_idx]
        print(f"[MAIN] Use subset of {N_SUBSET} samples for training/evaluation.")
    else:
        X, y = X_all, y_all
        print(f"[MAIN] Use all samples for training/evaluation.")

    # 2. 划分训练 / 测试集
    X_train, X_test, y_train, y_test = train_test_split_manual(
        X, y, test_ratio=0.3, random_state=0
    )
    print(f"[MAIN] Train size = {X_train.shape[0]}, Test size = {X_test.shape[0]}")

    # 3. 训练 SVM（多分类）
    C = 5.0
    gamma = 1.0
    kernel = 'rbf'  # 如需进一步加速，可以自己改成 'linear'

    print(f"[MAIN] Start training SVM with SMO (C={C}, kernel={kernel}, gamma={gamma})")
    t0 = time.time()
    model = svm_train(
        X_train, y_train,
        C=C, kernel=kernel, gamma=gamma,
        tol=1e-2,   # 放宽 KKT 容忍度，加速收敛
        max_passes=2,
        random_state=0
    )
    t1 = time.time()
    print(f"[MAIN] Training finished, elapsed = {t1 - t0:.2f} seconds")

    # 4. 在训练集和测试集上评估精度
    print("[MAIN] Evaluating on train/test set ...")
    y_pred_train, _ = svm_predict(model, X_train)
    y_pred_test, _ = svm_predict(model, X_test)

    train_acc = (y_pred_train == y_train).mean()
    test_acc = (y_pred_test == y_test).mean()

    print(f"[RESULT] Train accuracy: {train_acc:.4f}")
    print(f"[RESULT] Test  accuracy: {test_acc:.4f}")
