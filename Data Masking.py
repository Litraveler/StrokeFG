import pandas as pd
import os


def find_max_length(directories):
    """计算所有触摸事件的最大长度L"""
    max_length = 0
    for dir_path in directories:
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            df = pd.read_csv(file_path)
            if 'touch_id' not in df.columns:
                continue
            grouped = df.groupby('touch_id')
            for _, group in grouped:
                current_len = len(group)
                if current_len > max_length:
                    max_length = current_len
    return max_length


def expand_with_mask(directories, output_dirs, L):
    """将每个触摸事件扩展到长度L，并添加掩码"""
    # 定义需要置零的列（包括mask列）
    zero_cols = [
        'ACTION_TYPE', 'Time', 'X', 'Y', 'SizeMajor',
        'SizeMinor', 'Orientation', 'Pressure', 'Size'
    ]

    for dir_idx, dir_path in enumerate(directories):
        output_dir = output_dirs[dir_idx]
        os.makedirs(output_dir, exist_ok=True)

        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            df = pd.read_csv(file_path)
            if 'touch_id' not in df.columns:
                continue

            masked_groups = []
            grouped = df.groupby('touch_id')
            for touch_id, group in grouped:
                current_len = len(group)
                if current_len < L:
                    # 获取当前组的user_id和touch_id
                    user_id_val = group['user_id'].iloc[0]
                    touch_id_val = group['touch_id'].iloc[0]
                    pad_rows = L - current_len

                    # 生成填充数据：user_id和touch_id保留，其他列置0
                    pad_data = {}
                    for col in group.columns:
                        if col == 'user_id':
                            pad_data[col] = user_id_val
                        elif col == 'touch_id':
                            pad_data[col] = touch_id_val
                        elif col in zero_cols:
                            pad_data[col] = 0
                        else:
                            pad_data[col] = 0  # 其他列默认置0

                    # 创建填充行DataFrame
                    pad_df = pd.DataFrame([pad_data] * pad_rows)
                    expanded_group = pd.concat([group, pad_df], ignore_index=True)
                else:
                    expanded_group = group.copy()
                masked_groups.append(expanded_group)

            # 合并所有组并保存
            final_df = pd.concat(masked_groups, ignore_index=True)
            output_path = os.path.join(output_dir, file_name)
            final_df.to_csv(output_path, index=False)

#全局最长触摸序列长度 L = 300
if __name__ == "__main__":
    # 输入目录和输出目录配置
    input_dirs = [
        '../processed_data_sit_normalized',
        '../processed_data_walk_normalized'
    ]
    output_dirs = [
        '../masked_sit_different_postures',
        '../masked_walk_different_postures'
    ]

    # 1. 计算全局最长触摸序列长度L
    L = 221

    # 2. 对每个文件进行掩码扩展
    expand_with_mask(input_dirs, output_dirs, L)
    print("数据处理完成，结果已保存到 masked_sit 和 masked_walk 目录")