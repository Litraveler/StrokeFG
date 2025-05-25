import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from sklearn.metrics import roc_curve
import itertools
import copy

from codes.E1 import csv_files

#！！！！！！！！！！！！！！！测试模型在图案中的迁移能力！！！！！！！！！！！！！！！！！！！
#！！！！！！！！！！！！！！！数据集的大小控制在12800！！！！！！！！！！！！！！！！！！！！
number = 12800
input_data_length = 221
input_feature_length = 8
output_dir = '../E2_results'  # 测试结果保存目录
os.makedirs(output_dir, exist_ok=True)

# ======================== 数据准备与用户划分 ========================
# ======================== 正确 2025-4-21 ========================
def load_user_ids(dirs):
    """从指定目录提取所有用户ID"""
    user_ids = set()
    for dir_path in dirs:
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            df = pd.read_csv(file_path)
            user_ids.update(df['user_id'].unique())
    return list(user_ids)

# 提取用户ID并划分训练/测试用户
all_users = load_user_ids(['../masked_sit', '../masked_walk'])
train_users, test_users = train_test_split(all_users, test_size=0.33, random_state=42)


# ======================== 获取所有CSV文件路径 ========================
# 初始化两个列表分别存储包含“walk”和“sit”的文件路径
csv_files_walk = []
csv_files_sit = []
# 遍历两个目录
for root, _, files in itertools.chain(os.walk('../masked_sit'), os.walk('../masked_walk')):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)
            if 'walk' in file:
                csv_files_walk.append(file_path)
            elif 'sit' in file:
                csv_files_sit.append(file_path)

#======================== 数据生成器 ========================
#==================超参数优化数据生成器======================
class TripletDataGenerator(Sequence):
    """用于三元损失的动态数据生成器"""
    def __init__(self, file_paths, users, number, batch_size):
        self.file_paths = file_paths
        self.users = users
        self.L = input_data_length
        self.batch_size = batch_size  # 可以根据需要调整
        self.number = number
        self.user_data = self._precache_users()
        self.triplet_queue = []
        self.findex = 0
        for file in self.file_paths:
            for user in self.users:
                df = pd.read_csv(file)
                if user in df['user_id'].unique():
                    triplet_pairs = self._generate_triplets_for_user(user, file)
                    self.triplet_queue.extend(triplet_pairs)

    def _precache_users(self):
        """预加载用户数据，并按样本分组"""
        user_data = {}
        for file in self.file_paths:
            df = pd.read_csv(file)
            df = df[df['user_id'].isin(self.users)]
            for user in df['user_id'].unique():
                user_df = df[df['user_id'] == user]
                # 按 touch_id 分组，每个 touch_id 对应一个样本
                grouped = user_df.groupby('touch_id')
                samples = [group for _, group in grouped]
                if len(samples) > 5:
                    samples = samples[-5:]
                user_data.setdefault((file, user), []).extend(samples)
        return user_data

    def __len__(self):
        return int(self.number / self.batch_size)

    def _generate_triplets_for_user(self, user, file):
        """为单个用户生成所有可能的三元组"""
        user_samples = self.user_data.get((file, user), [])
        # 获取该用户的5个样本
        selected_samples = user_samples
        # 生成所有可能的锚点和正样本组合
        anchor_pos_pairs = list(itertools.combinations(selected_samples, 2))
        triplets = []
        # 为每个组合获取负样本
        df = pd.read_csv(file)
        neg_users = [u for u in df['user_id'].unique() if u != user and u in self.users]
        if not neg_users:
            return []
        for anchor_pos in anchor_pos_pairs:
            neg_user_idxs = np.random.choice(len(neg_users),2)
            for neg_user_idx in neg_user_idxs:
                neg_user = neg_users[neg_user_idx]
                neg_samples = self.user_data.get((file, neg_user), [])
                if not neg_samples:
                    continue
                neg_idx = np.random.choice(len(neg_samples))  # 先随机选择索引
                neg = neg_samples[neg_idx]  # 然后通过索引获取样本
                a_features = anchor_pos[0][
                    ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
                p_features = anchor_pos[1][
                    ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
                n_features = neg[
                    ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
                triplets.append((a_features, p_features, n_features))


        return triplets


    def __getitem__(self, idx):
        if idx == 0:
            self.triplet_queue = np.random.permutation(self.triplet_queue)
        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_triplets = self.triplet_queue[start_idx:end_idx]
        anchors, positives, negatives = zip(*batch_triplets)
        # 转换为 NumPy 数组
        anchors_array = np.array(anchors)
        positives_array = np.array(positives)
        negatives_array = np.array(negatives)
        # 打印形状信息
        print(f"Batch {idx} data shape: {batch_triplets.shape}")
        print(f"Anchors shape: {anchors_array.shape}")
        print(f"Positives shape: {positives_array.shape}")
        print(f"Negatives shape: {negatives_array.shape}")

        return (np.array(anchors), np.array(positives), np.array(negatives)), np.zeros(len(batch_triplets))

class VerificationDataGenerator(Sequence):
    """用于认证网络的动态数据生成器"""
    def __init__(self, file_paths, users, number, batch_size=32):
        self.file_paths = file_paths
        self.users = users
        self.batch_size = batch_size
        self.number = number
        self.L = input_data_length
        self.user_data = self._precache_users()
        self.positive_pairs = []  # 存储正样本对
        self.negative_pairs = []  # 存储负样本对
        for file in self.file_paths:
            for user in self.users:
                df = pd.read_csv(file)
                if user in df['user_id'].unique():
                    user_pairs = self._generate_pairs_for_user(user, file)
                    for pair in user_pairs:
                        if pair[2] == 1:  # 根据标签判断正负样本
                            self.positive_pairs.append(pair)
                        else:
                            self.negative_pairs.append(pair)

    def _precache_users(self):
        """预加载用户数据，并按样本分组"""
        user_data = {}
        for file in self.file_paths:
            df = pd.read_csv(file)
            df = df[df['user_id'].isin(self.users)]
            for user in df['user_id'].unique():
                user_df = df[df['user_id'] == user]
                grouped = user_df.groupby('touch_id')
                samples = [group for _, group in grouped]
                if len(samples) > 5:
                    samples = samples[-5:]
                user_data.setdefault((file, user), []).extend(samples)
        return user_data

    def __len__(self):
        # 假设每个用户可以生成 (len(users)-1)*2 个样本对
        return int(self.number / self.batch_size)

    def _generate_pairs_for_user(self, user, file):
        """为单个用户生成所有可能的正负样本对"""
        user_samples = self.user_data.get((file, user), [])
        # 生成所有可能的正样本对
        pos_pairs = list(itertools.combinations(user_samples, 2))
        pairs = []
        for pos_pair in pos_pairs:
            # 正样本对
            a_features = pos_pair[0][['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
            p_features = pos_pair[1][['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
            pairs.append((a_features, p_features, 1))  # 标签为1表示正样本对
            # 负样本对
        df = pd.read_csv(file)
        neg_users = [u for u in self.users if u != user and u in df['user_id'].unique()]
        if not neg_users:
            return []
        for user_sample in user_samples:
            a_features = user_sample[['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
            neg_users = np.random.choice(neg_users, 2)
            for neg_user in neg_users:
                neg_samples = self.user_data.get((file, neg_user), [])
                if not neg_samples:
                    continue
                neg_idx = np.random.choice(len(neg_samples))  # 先随机选择索引
                neg = neg_samples[neg_idx]  # 然后通过索引获取样本
                n_features = neg[
                    ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']].values
                pairs.append((a_features, n_features, 0))  # 标签为0表示负样本对
        return pairs

    def on_epoch_end(self):
        self.positive_pairs = np.random.permutation(self.positive_pairs)
        self.negative_pairs = np.random.permutation(self.negative_pairs)

    def __getitem__(self, idx):
        if idx == 0:
            self.on_epoch_end()
        half_batch_size = self.batch_size // 2  # 每个批次中正负样本的数量
        positive_start = idx * half_batch_size
        positive_end = positive_start + half_batch_size
        negative_start = idx * half_batch_size
        negative_end = negative_start + half_batch_size

        # 获取当前批次的正负样本
        positive_batch = self.positive_pairs[positive_start:positive_end]
        negative_batch = self.negative_pairs[negative_start:negative_end]

        print(f"Positive batch length: {len(positive_batch)}")
        print(f"Negative batch length: {len(negative_batch)}")

        if len(positive_batch) < half_batch_size:
            positive_batch += [(np.zeros((self.L, 8)), np.zeros((self.L, 8)), 1)] * (
                        half_batch_size - len(positive_batch))
        if len(negative_batch) < half_batch_size:
            negative_batch += [(np.zeros((self.L, 8)), np.zeros((self.L, 8)), 0)] * (
                        half_batch_size - len(negative_batch))

        # 合并正负样本
        batch_pairs = np.concatenate([positive_batch, negative_batch], axis=0)
        print(f"Batch length: {len(batch_pairs)}")
        np.random.shuffle(batch_pairs)
        anchors, contrastive, labels = zip(*batch_pairs)
        anchors_array = np.array(anchors)
        contrastive_array = np.array(contrastive)
        labels_array = np.array(labels)
        # 打印形状信息
        print(f"Anchors shape: {anchors_array.shape}")
        print(f"Positives shape: {contrastive_array.shape}")
        print(f"Negatives shape: {labels_array.shape}")

        return (np.array(anchors), np.array(contrastive)), np.array(labels)

# ======================== 双胞胎网络模型 ========================
def build_siamese_cnn(input_shape=(input_data_length, input_feature_length), output_dim=128):
    """参数总量约2.57M的轻量级CNN特征提取器"""
    input_layer = layers.Input(shape=input_shape, name="base_network_input")
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(input_layer)
    x = layers.MaxPooling1D(2)(x)  # 输出形状：(150, 256)
    x = layers.Conv1D(512, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)  # 输出形状：(75, 512)
    x = layers.Conv1D(1024, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)  # 输出形状：(1024,)
    # 特征压缩层
    x = layers.Dense(512, activation='relu')(x)
    output = layers.Dense(output_dim, name="base_network_output")(x)
    return models.Model(inputs=input_layer, outputs=output, name="base_network")

def triplet_loss(y_pred, alpha=0.5):
    """三元组损失函数"""
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(0.0, alpha + positive_dist - negative_dist)
    return tf.reduce_mean(loss)

def build_triplet_model(base_network, alpha=0.5):
    """构建完整的三元组模型"""
    anchor_input = layers.Input(shape=(input_data_length, input_feature_length))
    positive_input = layers.Input(shape=(input_data_length, input_feature_length))
    negative_input = layers.Input(shape=(input_data_length, input_feature_length))

    anchor_embedding = base_network(anchor_input)
    positive_embedding = base_network(positive_input)
    negative_embedding = base_network(negative_input)

    # 直接计算损失值
    loss = triplet_loss([anchor_embedding, positive_embedding, negative_embedding], alpha=alpha)

    model = models.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=loss
    )
    model.add_loss(loss)  # 显式添加损失
    return model

# ======================== 认证网络 ========================
def build_verification_network(input_dim):
    """构建认证网络"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),  # 输入为两个嵌入的差值拼接
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),  # 添加 Dropout 层，丢弃率设为 0.5
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# ======================== 测试流程 ========================
def evaluate_verification_model(model, test_files, test_users):
    """评估认证模型"""
    test_gen = VerificationDataGenerator(test_files, test_users, batch_size=32)
    results = model.evaluate(test_gen, verbose=0)
    return {'loss': results[0], 'accuracy': results[1]}

# ======================== 主流程 ========================
train_files_numbers = [24, 26]
text_files_number = 4

if __name__ == "__main__":
    csv_file = csv_files_sit
    i = 0
    j = 9
    np.random.shuffle(csv_file)
    for train_files_number in train_files_numbers:
        best_siamese_params = {
            'learning_rate': 0.0001,
            'batch_size': 128,
            'alpha': 0.01,
            'output_dim': 256
        }
        best_verification_params = {
            'learning_rate': 0.001,
            'batch_size': 32
        }
        # 构建双胞胎网络
        base_network = build_siamese_cnn(output_dim=best_siamese_params['output_dim'])
        triplet_model = build_triplet_model(base_network, alpha=best_siamese_params['alpha'])
        triplet_model.compile(optimizer=optimizers.Adam(learning_rate=best_siamese_params['learning_rate']))
        train_gen = TripletDataGenerator(csv_file[:train_files_number], train_users, number,
                                         batch_size=best_siamese_params['batch_size'])

        triplet_model.fit(train_gen, epochs=20)
        # 构建保存路径
        save_path = f'../E2_models/{i}/{j}/triplet_model_{train_files_number}.keras'
        # 获取目录路径
        directory = os.path.dirname(save_path)
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(directory):
            os.makedirs(directory)
        # 保存模型
        triplet_model.save(save_path)

        # # 加载模型
        # with custom_object_scope({'triplet_loss': triplet_loss}):
        #     triplet_model = load_model('triplet_model_6000.keras', compile=True)

        base_network = triplet_model.get_layer("base_network")

        # 4. 训练认证网络
        input_a = layers.Input(shape=(input_data_length, input_feature_length))
        input_b = layers.Input(shape=(input_data_length, input_feature_length))

        # 使用预训练的特征提取器（固定参数）
        embedding_a = base_network(input_a)
        embedding_b = base_network(input_b)
        base_network.trainable = False  # 固定特征提取器参数

        # 计算差值并拼接
        diff = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([embedding_a, embedding_b])

        # 认证网络
        verification_model = build_verification_network(best_siamese_params['output_dim'])
        output = verification_model(diff)

        full_model = models.Model(inputs=[input_a, input_b], outputs=output)
        full_model.compile(optimizer=optimizers.Adam(learning_rate=best_verification_params['learning_rate']),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        # 正负数据各35900+条数据
        train_gen = VerificationDataGenerator(csv_file[:train_files_number], train_users, number,
                                              batch_size=best_verification_params['batch_size'])
        full_model.fit(train_gen, epochs=20)

        # 构建保存路径
        save_path = f'../E2_models/{i}/{j}/full_model_{train_files_number}.keras'
        # 获取目录路径
        directory = os.path.dirname(save_path)
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(directory):
            os.makedirs(directory)
        # 保存模型
        full_model.save(save_path)

        # full_model = load_model('full_model.keras', compile=True)

        # 5. 测试每个CSV文件并保存结果
        test_results = {}
        export = {"File": [], "Fpr": [], "Tpr": [], "EER": [], "Thresholds": []}
        for file in csv_file[-text_files_number:]:
            test_gen = VerificationDataGenerator([file], test_users, number=len(test_users) * 20, batch_size=32)
            if len(test_gen) == 0:
                print(f"Skipping {file} (no data).")
                continue

            # 评估模型
            loss, accuracy = full_model.evaluate(test_gen, verbose=0)
            test_results[os.path.basename(file)] = accuracy

            test_gen = VerificationDataGenerator([file], test_users, number=len(test_users) * 20, batch_size=32)
            predictions = []
            labels = []
            for k in range(len(test_gen)):
                (anchors, contrastive), batch_labels = test_gen[k]
                batch_predictions = full_model.predict((anchors, contrastive), verbose=0)
                predictions.extend(batch_predictions.flatten())
                labels.extend(batch_labels)
                print(predictions)
                print(labels)

            # 确保labels不为空
            if len(labels) == 0:
                print(f"Skipping ROC for {file}: No labels collected.")
                continue
            fpr, tpr, thresholds = roc_curve(labels, predictions)
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
            export["File"].append(os.path.basename(file))
            export["Fpr"].append(fpr)
            export["Tpr"].append(tpr)
            export["EER"].append(eer)
            export["Thresholds"].append(thresholds)

        # 构建保存路径
        save_path = f'../E2_results/{i}/{j}/test_results_{train_files_number}.csv'
        # 获取目录路径
        directory = os.path.dirname(save_path)
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(directory):
            os.makedirs(directory)
        # 将字典转换为 DataFrame
        results_df = pd.DataFrame(list(test_results.items()), columns=['File', 'Accuracy'])
        # 保存到 CSV 文件
        results_df.to_csv(save_path, index=False)
        print("Testing completed. Results saved to:", save_path)

        save_path = f'../E2_results/{i}/{j}/test_results_ftt_{train_files_number}.csv'
        # 获取目录路径
        directory = os.path.dirname(save_path)
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(directory):
            os.makedirs(directory)
        df = pd.DataFrame(export)
        df.to_csv(save_path, index=False)












