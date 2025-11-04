if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import dezero
# 1️⃣ 检测 CuPy 是否安装
try:
    import cupy
    print(f"CuPy 已安装，版本: {cupy.__version__}")
except ImportError:
    print("CuPy 未安装，请先安装 cupy 或 cupy-cudaXXX")
    sys.exit(1)

# 2️⃣ 检测 GPU 可用性
gpu_count = cupy.cuda.runtime.getDeviceCount()
if gpu_count == 0:
    print("未检测到可用的 NVIDIA GPU")
else:
    print(f"检测到 {gpu_count} 块 GPU")
    for i in range(gpu_count):
        prop = cupy.cuda.runtime.getDeviceProperties(i)
        print(f"GPU {i}: {prop['name'].decode()}  |  显存: {prop['totalGlobalMem'] // 1024**2} MB")

# 3️⃣ 检测 DeZero GPU 模式
try:
    import dezero
    if dezero.cuda.gpu_enable:
        print("DeZero GPU 模式已启用 ✅")
    else:
        print("DeZero GPU 模式未启用 ❌")
except Exception as e:
    print(f"DeZero GPU 检测出错: {e}")

# 4️⃣ 测试 CuPy 运算
try:
    import cupy as cp
    x = cp.arange(10)
    print("CuPy 测试成功:", x)
except Exception as e:
    print(f"CuPy 测试失败: {e}")
