#!/bin/bash

# 获取当前目录路径和项目名
curr_dir=$(pwd)
project_name=$(basename "$curr_dir")
zip_path="$curr_dir/${project_name}.zip"

# 切换到上一级目录打包（但输出 zip 放回当前目录）
cd "$curr_dir/../" && \
zip -r "$zip_path" "$project_name" \
    -x "${project_name}/runs/*" \
       "${project_name}/**/.git/*" \
       "${project_name}/**/__pycache__/*" \
       "${project_name}/**/.vscode/*" \
       "${project_name}/**/.gitignore" 

echo "✅ 已打包到: $zip_path"

