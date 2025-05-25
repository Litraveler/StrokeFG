import os
import uuid
import pandas as pd
import numpy as np
import glob
from scipy.interpolate import interp1d

#注释：6位用户没有walk文件
# 配置路径
data_dir = '../data_different_posture'
output_dir_walk = '../processed_data_walk/'
output_dir_sit = '../processed_data_sit/'
os.makedirs(output_dir_walk, exist_ok=True)
os.makedirs(output_dir_sit, exist_ok=True)

def find_file_name(user_path, pattern):
    """查找匹配的文件路径"""
    file_pattern = os.path.join(user_path, pattern)
    matching_files = glob.glob(file_pattern)
    if not matching_files:
        raise FileNotFoundError(f"未找到匹配的文件: {file_pattern}")
    return matching_files[0]

for n in range(1, 31):
    walk_path = os.path.join(output_dir_walk, f'touch_walk_{n}.csv')
    sit_path = os.path.join(output_dir_sit, f'touch_sit_{n}.csv')
    if not os.path.exists(walk_path):
        pd.DataFrame(columns=[
            'ACTION_TYPE', 'Time', 'X', 'Y', 'SizeMajor', 'SizeMinor',
            'Orientation', 'Pressure', 'Size', 'user_id', 'touch_id'
        ]).to_csv(walk_path, index=False)
    if not os.path.exists(sit_path):
        pd.DataFrame(columns=[
            'ACTION_TYPE', 'Time', 'X', 'Y', 'SizeMajor', 'SizeMinor',
            'Orientation', 'Pressure', 'Size', 'user_id', 'touch_id'
        ]).to_csv(sit_path, index=False)
# 处理每个用户
user_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
print(len(user_folders))
for user_folder in user_folders:
    user_path = os.path.join(data_dir, user_folder)
    user_id = str(uuid.uuid4())
    try:
        # 读取触摸数据
        walk_file_path = find_file_name(user_path, '*touch_gesture_walk_touchData*.csv')
        sit_file_path = find_file_name(user_path, '*touch_gesture_sit_touchData*.csv')
        if os.path.exists(sit_file_path):
            pattern = 0
            flag = 0
            sit_data = pd.read_csv(sit_file_path)
            # 处理触摸事件
            current_touch = None

            for _, row in sit_data.iterrows():
                action = str(row['ACTION_TYPE'])
                if action.startswith('Down_'):
                    parts = action.split('_')
                    if len(parts) != 2:
                        continue
                    n = int(parts[1])
                    if pattern != n:
                        pattern = n
                        flag = 0

                    current_touch = {
                        'n': n,
                        'start_time': row['Time'],
                        'end_time': None,
                        'rows': [row],
                        'touch_id': str(uuid.uuid4())
                    }
                elif action.startswith('Move_') and current_touch is not None:
                    parts = action.split('_')
                    if len(parts) != 2 or int(parts[1]) != current_touch['n']:
                        continue
                    current_touch['rows'].append(row)
                elif action.startswith('Up_') and current_touch is not None:
                    parts = action.split('_')
                    if len(parts) != 2 or int(parts[1]) != current_touch['n']:
                        continue
                    current_touch['end_time'] = row['Time']
                    current_touch['rows'].append(row)
                    # 处理触摸数据
                    touch_df = pd.DataFrame(current_touch['rows'])
                    touch_df['user_id'] = user_id
                    touch_df['touch_id'] = current_touch['touch_id']
                    flag += 1
                    if flag > 2:
                        # 保存数据
                        output_sit_path = os.path.join(output_dir_sit, f'touch_sit_{current_touch["n"]}.csv')
                        touch_df.to_csv(output_sit_path, mode='a', header=False, index=False)
                    current_touch = None

        if os.path.exists(walk_file_path):
            pattern = 0
            flag = 0
            walk_data = pd.read_csv(walk_file_path)
            # 处理触摸事件
            current_touch = None
            for _, row in walk_data.iterrows():
                action = str(row['ACTION_TYPE'])
                if action.startswith('Down_'):
                    parts = action.split('_')
                    if len(parts) != 2:
                        continue
                    n = int(parts[1])
                    if pattern != n:
                        pattern = n
                        flag = 0
                    current_touch = {
                        'n': n,
                        'start_time': row['Time'],
                        'end_time': None,
                        'rows': [row],
                        'touch_id': str(uuid.uuid4())
                    }
                elif action.startswith('Move_') and current_touch is not None:
                    parts = action.split('_')
                    if len(parts) != 2 or int(parts[1]) != current_touch['n']:
                        continue
                    current_touch['rows'].append(row)
                elif action.startswith('Up_') and current_touch is not None:
                    parts = action.split('_')
                    if len(parts) != 2 or int(parts[1]) != current_touch['n']:
                        continue
                    current_touch['end_time'] = row['Time']
                    current_touch['rows'].append(row)
                    # 处理触摸数据
                    touch_df = pd.DataFrame(current_touch['rows'])
                    touch_df['user_id'] = user_id
                    touch_df['touch_id'] = current_touch['touch_id']
                    flag += 1
                    if flag > 2:
                        output_touch_path = os.path.join(output_dir_walk, f'touch_walk_{current_touch["n"]}.csv')
                        touch_df.to_csv(output_touch_path, mode='a', header=False, index=False)
                    # 保存数据
                    current_touch = None

    except Exception as e:
        print(f"处理用户 {user_folder} 时出错: {e}")
        continue
print("处理完成！")