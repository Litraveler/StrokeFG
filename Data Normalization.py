import pandas as pd
import os


def process_single_file(file_path, cols):
    """处理单个文件：时间调整 + 全文件归一化"""
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    if df.empty:
        return None

    # Step 1: 按touch_id分组，调整Time列为相对时间
    if 'touch_id' in df.columns and 'Time' in df.columns:
        df['Time'] = df.groupby('touch_id')['Time'].transform(lambda x: x - x.iloc[0])

        # Step 2: 对整个文件的Time列进行归一化（不再分组）
        if df['Time'].nunique() > 1:
            min_t = df['Time'].min()
            max_t = df['Time'].max()
            df['Time'] = (df['Time'] - min_t) / (max_t - min_t)
        else:
            df['Time'] = 0.0  # 全相同则设为0

    # Step 3: 对其他列进行归一化
    for col in cols:
        if col in df.columns and col != 'Time':  # Time已单独处理
            if df[col].nunique() > 1:
                min_val, max_val = df[col].min(), df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.0

    return df


def process_all_files(directory, mode, output_dir=None):
    """处理目录下所有文件，可选择保存到新目录"""
    cols = ['X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size', 'Time']
    for n in range(1, 31):
        file_name = f'touch_{mode}_{n}.csv'
        file_path = os.path.join(directory, file_name)
        processed_df = process_single_file(file_path, cols)

        if processed_df is not None:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, file_name)
                processed_df.to_csv(output_path, index=False)
            else:
                print(f"Processed {file_name}:")
                print(processed_df.head())


if __name__ == "__main__":
    # 处理sit数据并保存到新目录
    process_all_files(
        directory='../processed_data_sit',
        mode='sit',
        output_dir='../processed_data_sit_normalized'
    )

    # 处理walk数据并保存到新目录
    process_all_files(
        directory='../processed_data_walk',
        mode='walk',
        output_dir='../processed_data_walk_normalized'
    )