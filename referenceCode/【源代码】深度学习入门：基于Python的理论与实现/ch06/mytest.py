import numpy as np
def numerical_example():
    print("=== 具体数值例子 ===")

    # 假设一个简单的网络层
    input_data = np.array([1.0, 1.0, 1.0, 1.0])  # 4个神经元，每个输出1.0
    dropout_ratio = 0.5

    print(f"输入数据: {input_data}")
    print(f"Dropout比例: {dropout_ratio}")
    print()

    # 模拟多次训练（随机丢弃）
    print("训练阶段（随机性演示）:")
    train_outputs = []
    for i in range(5):
        mask = np.random.rand(4) > dropout_ratio
        output = input_data * mask
        train_outputs.append(output)
        print(f"第{i + 1}次: 掩码={mask.astype(int)}, 输出={output}, 平均值={np.mean(output):.1f}")

    train_avg = np.mean([np.mean(x) for x in train_outputs])
    print(f"训练平均输出: {train_avg:.1f}")
    print()

    # 测试阶段
    test_output = input_data * (1 - dropout_ratio)
    print(f"测试阶段输出: {test_output}")
    print(f"测试平均值: {np.mean(test_output):.1f}")
    print()
    print("结论：测试缩放后的输出与训练平均输出一致！")


numerical_example()