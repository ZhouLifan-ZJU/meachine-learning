class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    def backward(self, dout):
        return dout * self.mask

import numpy as np

# 创建Dropout层，丢弃率50%
dropout = Dropout(dropout_ratio=0.5)

# 模拟输入数据（批量大小=2，特征数=4）
x = np.array([[1.0, 2.0, 3.0, 4.0],
              [5.0, 6.0, 7.0, 8.0]])

print("原始输入:")
print(x)
print()

# 训练时的前向传播
output_train = dropout.forward(x, train_flg=True)
print("训练模式输出 (随机丢弃50%神经元):")
print(output_train)
print("使用的掩码:")
print(dropout.mask.astype(int))
print()

# 测试时的前向传播
output_test = dropout.forward(x, train_flg=False)
print("测试模式输出 (所有神经元，但缩放50%):")
print(output_test)