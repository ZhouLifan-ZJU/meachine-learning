import os
import re

# 1. 设置 dezero 源码根目录（修改为你的路径）
DEZERO_DIR = r"E:\course-code\meachine-learning\referenceCode\【源代码】深度学习入门：自制框架\dezero"

# 2. 定义要替换的映射
# key 是正则匹配的旧类型，value 是替换的新类型
replace_map = {
    r"\bnp\.int\b": "int",
    r"\bnp\.float\b": "float",
    r"\bnp\.bool\b": "bool",
}

# 3. 遍历所有 .py 文件并替换
for root, dirs, files in os.walk(DEZERO_DIR):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content
            for old, new in replace_map.items():
                content = re.sub(old, new, content)

            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Updated: {file_path}")
