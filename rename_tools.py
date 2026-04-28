import os

def sanitize_filenames(root_dir):
    """
    深度遍历文件夹，将文件名中的敏感字符 '&' 替换为安全字符 'x'
    """
    print(f"🚀 开始扫描并清洗目录: {root_dir}")
    rename_count = 0
    
    # os.walk 会自动遍历该文件夹及其所有子文件夹
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查是否包含危险字符 '&'
            if '&' in filename:
                # 【核心修改】：将 '&' 替换为 'x'
                safe_name = filename.replace('&', 'x')
                
                old_file_path = os.path.join(dirpath, filename)
                new_file_path = os.path.join(dirpath, safe_name)
                
                try:
                    os.rename(old_file_path, new_file_path)
                    rename_count += 1
                except Exception as e:
                    print(f"❌ 无法重命名 {filename}: {e}")
                    
    print(f"✅ 清洗完成！共将 {rename_count} 个文件中的 '&' 成功替换为了 'x'。")

if __name__ == "__main__":
    # ⚠️ 请确保这里的路径指向你本地的 full_cars 文件夹
    # 假设这个 python 文件和 data 文件夹在同一层级
    target_directory = "./data/full_cars" 
    
    if os.path.exists(target_directory):
        sanitize_filenames(target_directory)
    else:
        print(f"❌ 找不到路径: {target_directory}，请检查！")