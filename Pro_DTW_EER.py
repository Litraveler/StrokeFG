import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import re
import warnings
from scipy.interpolate import interp1d
from scipy.stats import alpha

# 忽略特定的警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 设置中文字体
font_path = os.path.abspath('./SIMSUN.ttf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [font_prop.get_name()],
        'axes.unicode_minus': False
    })


def parse_float_array(text):
    """增强型解析函数，处理科学计数法、inf等特殊值"""
    try:
        # 替换换行符为空格，并去除多余字符
        cleaned = re.sub(r"[^0-9\.eE\+\-inf]", " ", str(text))
        # 使用改进的正则表达式匹配所有合法数字（包括科学计数法和inf）
        numbers = re.findall(
            r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|"  # 常规数字和科学计数法
            r"[-+]?inf|"  # 正负无穷大
            r"nan",  # 非数字
            cleaned
        )

        # 转换为浮点数（处理特殊值）
        result = []
        for x in numbers:
            try:
                # 处理特殊值
                if 'inf' in x.lower():
                    value = float('inf') if '-' not in x else float('-inf')
                elif 'nan' in x.lower():
                    value = float('nan')
                else:
                    value = float(x)
                result.append(value)
            except ValueError:
                continue
        return result
    except Exception as e:
        print(f"解析警告：无法解析字符串 '{text}'，错误：{str(e)}")
        return []


def calculate_eer(far, frr):
    """计算等错误率(EER) - 找到FAR和FRR相等的点"""
    # 确保是numpy数组
    far = np.array(far)
    frr = np.array(frr)

    # 找到FAR和FRR最接近的点
    min_idx = np.argmin(np.abs(far - frr))
    eer = (far[min_idx] + frr[min_idx]) / 2
    return eer, min_idx


def compute_metrics(scores, labels, is_distance=True):
    """计算FAR和FRR曲线"""
    # 确保分数和标签是numpy数组
    scores = np.array(scores)
    labels = np.array(labels)

    # 对于距离度量，分数越小表示越相似
    # 对于相似度度量，分数越大表示越相似
    if is_distance:
        # 对于距离，我们需要将分数反转以便计算
        scores_normalized = -scores
    else:
        scores_normalized = scores

    # 获取唯一的分数值作为阈值
    thresholds = np.unique(scores_normalized)
    thresholds = np.sort(thresholds)

    far_values = []
    frr_values = []

    # 对于每个阈值，计算FAR和FRR
    for threshold in thresholds:
        # 预测为正类（相似）的条件：分数 >= 阈值
        predictions = scores_normalized >= threshold

        # 计算FAR（误识率）：负样本被错误接受的比例
        # 负样本：label = 0
        negative_indices = labels == 0
        if np.sum(negative_indices) > 0:
            far = np.sum(predictions[negative_indices]) / np.sum(negative_indices)
        else:
            far = 0

        # 计算FRR（拒识率）：正样本被错误拒绝的比例
        # 正样本：label = 1
        positive_indices = labels == 1
        if np.sum(positive_indices) > 0:
            frr = np.sum(~predictions[positive_indices]) / np.sum(positive_indices)
        else:
            frr = 0

        far_values.append(far)
        frr_values.append(frr)

    return thresholds, far_values, frr_values


def normalize_thresholds(thresholds):
    """将阈值范围归一化到0-1之间"""
    # 过滤掉无效值
    valid_thresholds = thresholds[np.isfinite(thresholds)]
    if len(valid_thresholds) == 0:
        return np.zeros_like(thresholds)

    min_val = np.min(valid_thresholds)
    max_val = np.max(valid_thresholds)
    if max_val - min_val > 0:
        return (thresholds - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(thresholds)


def analyze_results(file_path, method, is_distance=True):
    """分析结果文件并计算EER"""
    # 读取结果文件
    df = pd.read_csv(file_path)

    # 提取特定方法的分数和标签
    if method == 'dtw':
        scores = df['dtw_distance'].values
    elif method == 'protractor':
        scores = df['protractor_similarity'].values
    else:
        raise ValueError("方法必须是 'dtw' 或 'protractor'")

    labels = df['label'].values

    # 计算FAR和FRR曲线
    thresholds, far, frr = compute_metrics(scores, labels, is_distance)

    # 计算EER
    eer, min_idx = calculate_eer(far, frr)

    print(f"{method.upper()} 方法的EER: {eer:.4f}")

    return thresholds, far, frr, (eer, min_idx)


def process_e1_data(file_path, condition):
    """处理E1_results数据"""
    try:
        df = pd.read_csv(file_path)

        # 解析数组列
        df['Fpr'] = df['Fpr'].apply(parse_float_array)
        df['Tpr'] = df['Tpr'].apply(parse_float_array)
        df['Thresholds'] = df['Thresholds'].apply(parse_float_array)

        # 数据有效性检查
        df = df.dropna(subset=['Fpr', 'Tpr', 'Thresholds'])
        df = df[df['Fpr'].apply(len) > 0]
        df = df[df['Tpr'].apply(len) > 0]

        # 筛选对应条件的数据
        condition_data = df[df['File'].str.contains(condition)]

        if condition_data.empty:
            print(f"在E1结果中未找到{condition}条件的数据")
            return None, None, None

        # 汇总所有类别的数据
        all_thresholds = []
        all_far = []
        all_frr = []

        for _, row in condition_data.iterrows():
            if len(row['Thresholds']) > 0 and len(row['Fpr']) > 0 and len(row['Tpr']) > 0:
                # 按阈值降序排列
                order = np.argsort(row['Thresholds'])[::-1]
                thresholds = np.array(row['Thresholds'])[order]
                far = np.array(row['Fpr'])[order]
                frr = 1 - np.array(row['Tpr'])[order]  # FRR = 1 - TPR

                # 过滤掉无效值
                valid_mask = np.isfinite(thresholds) & np.isfinite(far) & np.isfinite(frr)
                thresholds = thresholds[valid_mask]
                far = far[valid_mask]
                frr = frr[valid_mask]

                if len(thresholds) > 0:
                    all_thresholds.append(thresholds)
                    all_far.append(far)
                    all_frr.append(frr)

        if not all_thresholds:
            print(f"E1结果中{condition}条件的数据无效")
            return None, None, None

        # 插值到0-1的阈值范围
        common_thresholds = np.linspace(0, 1, 100)

        interp_far_list = []
        interp_frr_list = []

        for i in range(len(all_thresholds)):
            try:
                # 使用线性插值
                interp_far = interp1d(all_thresholds[i], all_far[i], kind='linear', fill_value="extrapolate")(
                    common_thresholds)
                interp_frr = interp1d(all_thresholds[i], all_frr[i], kind='linear', fill_value="extrapolate")(
                    common_thresholds)
                interp_far_list.append(interp_far)
                interp_frr_list.append(interp_frr)
            except Exception as e:
                print(f"插值过程中出错: {e}")
                continue

        if not interp_far_list:
            print(f"无法对E1结果中{condition}条件的数据进行插值")
            return None, None, None

        # 计算平均曲线
        avg_far = np.mean(interp_far_list, axis=0)
        avg_frr = np.mean(interp_frr_list, axis=0)

        # 计算EER
        eer = calculate_eer(avg_far, avg_frr)[0]

        return common_thresholds, avg_far, avg_frr, eer

    except Exception as e:
        print(f"处理E1数据时出错: {e}")
        return None, None, None, None


def find_threshold_at_fnr(thresholds, frr_values, target_fnr):
    """找到FNR达到指定值时的阈值"""
    # 找到FNR最接近目标值的点
    idx = np.argmin(np.abs(np.array(frr_values) - target_fnr))
    return thresholds[idx], idx


def plot_error_curves(thresholds_dtw, far_dtw, frr_dtw,
                      thresholds_protractor, far_protractor, frr_protractor,
                      e1_thresholds, e1_far, e1_frr, e1_eer,
                      target_fnr_dtw=0.0359, target_fnr_protractor=0.0414):
    """绘制错误率曲线（FPR和FNR），阈值归一化到0-1范围"""
    plt.figure(figsize=(12, 8))

    # 归一化阈值到0-1范围
    norm_thresholds_dtw = normalize_thresholds(thresholds_dtw)
    norm_thresholds_protractor = normalize_thresholds(thresholds_protractor)

    # 定义颜色和线条样式
    colors = ["#A22017", "#00B050", "#5F6A72"]
    line_styles = [
        (0, (5, 5)),  # 虚线
        (0, (5, 2, 1, 2)),  # 点划线
        (0, (5, 2, 1, 2, 1, 2)),  # 双点划线
    ]

    # 找到DTW在目标FNR处的阈值
    dtw_threshold_at_target, dtw_idx = find_threshold_at_fnr(norm_thresholds_dtw, frr_dtw, target_fnr_dtw)
    dtw_far_at_target = far_dtw[dtw_idx]

    # 找到Protractor在目标FNR处的阈值
    protractor_threshold_at_target, protractor_idx = find_threshold_at_fnr(norm_thresholds_protractor, frr_protractor,
                                                                           target_fnr_protractor)
    protractor_far_at_target = far_protractor[protractor_idx]

    # 绘制DTW曲线
    plt.plot(norm_thresholds_dtw, far_dtw, color=colors[0], linestyle=line_styles[0], linewidth=4, alpha=0.6,
             label='DTW FPR')
    plt.plot(norm_thresholds_dtw, frr_dtw, color=colors[1], linestyle=line_styles[1], linewidth=4, alpha=0.6,
             label='DTW FNR')

    # 绘制DTW在目标FNR处的竖线
    plt.axvline(x=dtw_threshold_at_target, color=colors[0], linestyle='--', linewidth=2, alpha=0.6)

    # 绘制交点
    plt.plot(dtw_threshold_at_target, dtw_far_at_target, 'o', color=colors[0], markersize=8, alpha=0.6)
    plt.plot(dtw_threshold_at_target, target_fnr_dtw, 'o', color=colors[1], markersize=8, alpha=0.6)

    # 标注交点的值（黑色字体）
    plt.text(dtw_threshold_at_target + 0.01, dtw_far_at_target + 0.01, f'{dtw_far_at_target:.4f}',
             fontsize=24, color='black', ha='left')
    plt.text(dtw_threshold_at_target - 0.11, target_fnr_dtw + 0.01, f'{target_fnr_dtw:.4f}',
             fontsize=24, color='black', ha='left')

    # 绘制Protractor曲线
    plt.plot(norm_thresholds_protractor, far_protractor, color=colors[0], linestyle=line_styles[0], linewidth=4,
             alpha=0.3, label='Protractor FPR')
    plt.plot(norm_thresholds_protractor, frr_protractor, color=colors[1], linestyle=line_styles[1], linewidth=4,
             alpha=0.3, label='Protractor FNR')

    # 绘制Protractor在目标FNR处的竖线
    plt.axvline(x=protractor_threshold_at_target, color=colors[0], linestyle='--', linewidth=2, alpha=0.3)

    # 绘制交点
    plt.plot(protractor_threshold_at_target, protractor_far_at_target, 'o', color=colors[0], markersize=8, alpha=0.3)
    plt.plot(protractor_threshold_at_target, target_fnr_protractor, 'o', color=colors[1], markersize=8, alpha=0.3)

    # 标注交点的值（黑色字体）
    plt.text(protractor_threshold_at_target + 0.01, protractor_far_at_target + 0.01, f'{protractor_far_at_target:.4f}',
             fontsize=24, color='black', ha='left')
    plt.text(protractor_threshold_at_target + 0.01, target_fnr_protractor + 0.01, f'{target_fnr_protractor:.4f}',
             fontsize=24, color='black', ha='left')

    # 绘制E1_results曲线（如果存在）
    if e1_thresholds is not None and e1_far is not None and e1_frr is not None:
        plt.plot(e1_thresholds, e1_far, color=colors[0], linestyle=line_styles[0], linewidth=4, alpha=1,
                 label='StrokeFG FPR')
        plt.plot(e1_thresholds, e1_frr, color=colors[1], linestyle=line_styles[1], linewidth=4, alpha=1,
                 label='StrokeFG FNR')
        plt.plot(e1_thresholds[np.argmin(np.abs(e1_far - e1_frr))], e1_eer, 'k*', markersize=18,
                 label=f'EER = {e1_eer:.4f}')
        plt.axvline(x=e1_thresholds[np.argmin(np.abs(e1_far - e1_frr))], color='#5F6A72', linestyle='--', linewidth=2)

    # 设置字体大小
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('阈值', fontsize=32)
    plt.ylabel('错误率', fontsize=32)
    plt.legend(fontsize=24, loc='upper left')
    plt.grid(False)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


# 主程序
if __name__ == "__main__":
    # 处理E1数据
    e1_file_path = '../E1_results/test_result_ftt_12800.csv'

    # 分析坐姿条件下的结果
    print("分析坐姿条件下的结果:")
    try:
        sit_dtw_thresholds, sit_dtw_far, sit_dtw_frr, sit_dtw_eer = analyze_results(
            '../DTW_Protractor/sit_results_cross.csv', 'dtw', is_distance=True)
        sit_protractor_thresholds, sit_protractor_far, sit_protractor_frr, sit_protractor_eer = analyze_results(
            '../DTW_Protractor/sit_results_cross.csv', 'protractor', is_distance=False)
        # 处理E1 sit数据
        e1_sit_thresholds, e1_sit_far, e1_sit_frr, e1_sit_eer = process_e1_data(e1_file_path, 'sit')

        # 绘制坐姿条件下的错误率曲线
        plot_error_curves(
            sit_dtw_thresholds, sit_dtw_far, sit_dtw_frr,
            sit_protractor_thresholds, sit_protractor_far, sit_protractor_frr,
            e1_sit_thresholds, e1_sit_far, e1_sit_frr, e1_sit_eer
        )
    except Exception as e:
        print(f"处理坐姿条件时出错: {e}")

    # 分析行走条件下的结果
    print("\n分析行走条件下的结果:")
    try:
        walk_dtw_thresholds, walk_dtw_far, walk_dtw_frr, walk_dtw_eer = analyze_results(
            '../DTW_Protractor/walk_results_cross.csv', 'dtw', is_distance=True)
        walk_protractor_thresholds, walk_protractor_far, walk_protractor_frr, walk_protractor_eer = analyze_results(
            '../DTW_Protractor/walk_results_cross.csv', 'protractor', is_distance=False)

        # 处理E1 walk数据
        e1_walk_thresholds, e1_walk_far, e1_walk_frr, e1_walk_eer = process_e1_data(e1_file_path, 'walk')
        # 绘制行走条件下的错误率曲线
        plot_error_curves(
            walk_dtw_thresholds, walk_dtw_far, walk_dtw_frr,
            walk_protractor_thresholds, walk_protractor_far, walk_protractor_frr,
            e1_walk_thresholds, e1_walk_far, e1_walk_frr, e1_walk_eer
        )
    except Exception as e:
        print(f"处理行走条件时出错: {e}")