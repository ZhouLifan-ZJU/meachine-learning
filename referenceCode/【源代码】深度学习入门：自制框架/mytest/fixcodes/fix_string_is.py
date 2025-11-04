import os
import re

# 1. 设置 dezero 源码目录（修改为你本地路径）
DEZERO_DIR = r"E:\course-code\meachine-learning\referenceCode\【源代码】深度学习入门：自制框架\dezero"

# 2. 匹配字符串比较 is / is not 的正则
# 注意：仅匹配形如 `variable is 'string'` 或 `variable is not "string"`
pattern_is = re.compile(r"(\b\w+\b)\s+is\s+(['\"].+?['\"])")
pattern_is_not = re.compile(r"(\b\w+\b)\s+is\s+not\s+(['\"].+?['\"])")

# 3. 遍历所有 .py 文件并替换
for root, dirs, files in os.walk(DEZERO_DIR):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # 替换 is 为 ==
            content = pattern_is.sub(r"\1 == \2", content)
            # 替换 is not 为 !=
            content = pattern_is_not.sub(r"\1 != \2", content)

            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Updated: {file_path}")
