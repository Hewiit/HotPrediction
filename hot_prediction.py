import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr, pearsonr
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import (
    TRAIN_RATIO, 
    GROUND_TRUTH_FILE, 
    MIN_POSTS, 
    STD_MULTIPLIER, 
    NUM_ITERATIONS, 
    INITIAL_SEARCH_RADIUS,
    AUTH_WEIGHTS,
    MEDIA_BONUS
)
from data_processor import DataProcessor
from hot_calculator import HotCalculator
from transformer_optimizer import TransformerOptimizer
import torch

# 创建结果文件夹
RESULT_DIR = "results"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

class FixedHotPredictor:
    def __init__(self, model_path=None):
        self.data_processor = DataProcessor()
        self.hot_calculator = HotCalculator(AUTH_WEIGHTS, MEDIA_BONUS)
        self.transformer_optimizer = TransformerOptimizer()
        self.filtered_events_data = None
        self.filtered_stats = None
        
        # 如果提供了模型路径，则加载预训练模型
        if model_path:
            self.transformer_optimizer.load_model(model_path)
        
    def train_parameters(self, train_ratio=TRAIN_RATIO, save_model_path='models/best_model.pt'):
        """训练模型参数"""
        try:
            print("开始优化热度预测模型参数...")
            
            # 使用统一的数据处理方法
            if self.filtered_events_data is None:
                self.filtered_events_data, self.filtered_stats = self.data_processor.prepare_event_data(
                    GROUND_TRUTH_FILE,
                    MIN_POSTS,
                    STD_MULTIPLIER
                )
            
            # 保存事件统计信息
            self.filtered_stats.to_csv(os.path.join(RESULT_DIR, "event_stats.csv"), index=False, encoding='utf-8')
            
            # print(f"可用于训练的事件数量：{len(self.filtered_events_data)}")
            
            # 划分训练集和测试集
            train_data, test_data = train_test_split(self.filtered_events_data, train_size=train_ratio, random_state=42)
            # print(f"训练集大小：{len(train_data)}，测试集大小：{len(test_data)}")
            
            # 使用Transformer优化器
            print("使用Transformer模型优化参数...")
            self.transformer_optimizer.train_model(train_data)
            
            # 保存训练好的模型
            if save_model_path:
                # 确保models目录存在
                os.makedirs('models', exist_ok=True)
                self.transformer_optimizer.save_model(save_model_path)
            
            # 评估训练集和测试集效果
            all_predictions = []
            all_true = []
            all_data = [(data, "训练集") for data in train_data] + [(data, "测试集") for data in test_data]
            
            print("\n开始评估模型效果...")
            for (event_df, true_hot, event_name), dataset_type in all_data:
                likes_weight, reposts_weight, comments_weight, sentiment_weight, topic_weight, entity_weight, info_weight, interaction_overall_weight, text_overall_weight = self.transformer_optimizer.optimize(event_df)
                self.hot_calculator.set_weights(likes_weight, reposts_weight, comments_weight, sentiment_weight, topic_weight, entity_weight, info_weight, interaction_overall_weight, text_overall_weight)
                device = self.transformer_optimizer.device
                weights = torch.tensor([likes_weight, reposts_weight, comments_weight, sentiment_weight, topic_weight, entity_weight, info_weight, interaction_overall_weight, text_overall_weight], dtype=torch.float32, device=device)
                pred_hot = self.hot_calculator.calculate_batch_hot_tensor(event_df, weights, device)
                all_predictions.append(pred_hot.item() if torch.is_tensor(pred_hot) else float(pred_hot))
                all_true.append(true_hot)
            
            # 分离训练集和测试集的预测结果
            train_size = len(train_data)
            train_predictions = all_predictions[:train_size]
            train_true = all_true[:train_size]
            test_predictions = all_predictions[train_size:]
            test_true = all_true[train_size:]
            
            # 计算训练集的评估指标
            train_pred_norm = (np.array(train_predictions) - np.min(train_predictions)) / (np.max(train_predictions) - np.min(train_predictions)) * 100
            train_true_norm = (np.array(train_true) - np.min(train_true)) / (np.max(train_true) - np.min(train_true)) * 100
            
            train_linear = pearsonr(train_true_norm, train_pred_norm)[0]
            train_pred_ranks = np.argsort(np.argsort(-train_pred_norm))
            train_true_ranks = np.argsort(np.argsort(-train_true_norm))
            train_rank_correlation = spearmanr(train_true_ranks, train_pred_ranks)[0]
            
            # 计算测试集的评估指标
            test_pred_norm = (np.array(test_predictions) - np.min(test_predictions)) / (np.max(test_predictions) - np.min(test_predictions)) * 100
            test_true_norm = (np.array(test_true) - np.min(test_true)) / (np.max(test_true) - np.min(test_true)) * 100
            
            test_linear = pearsonr(test_true_norm, test_pred_norm)[0]
            test_pred_ranks = np.argsort(np.argsort(-test_pred_norm))
            test_true_ranks = np.argsort(np.argsort(-test_true_norm))
            test_rank_correlation = spearmanr(test_true_ranks, test_pred_ranks)[0]

            return test_rank_correlation, test_linear, train_rank_correlation, train_linear
            
        except Exception as e:
            print(f"训练过程中出错: {str(e)}")
            raise
        
    def predict_hot_scores(self):
        """预测所有事件的热度值"""
        print("开始预测事件热度...")
        try:
            # 使用已经处理好的数据
            if self.filtered_events_data is None:
                self.filtered_events_data, self.filtered_stats = self.data_processor.prepare_event_data(
                    GROUND_TRUTH_FILE,
                    MIN_POSTS,
                    STD_MULTIPLIER
                )
            
            event_scores = {}
            event_origin_scores = {}  # 新增：存储原始热度值
            true_scores = []
            pred_scores = []
            event_params = {}  # 新增：记录每个事件的参数
            
            # 使用过滤后的事件数据进行预测
            for event_df, true_hot, event in self.filtered_events_data:
                try:
                    # 使用Transformer优化器预测参数
                    likes_weight, reposts_weight, comments_weight, sentiment_weight, topic_weight, entity_weight, info_weight, interaction_overall_weight, text_overall_weight = self.transformer_optimizer.optimize(event_df)
                    self.hot_calculator.set_weights(likes_weight, reposts_weight, comments_weight, sentiment_weight, topic_weight, entity_weight, info_weight, interaction_overall_weight, text_overall_weight)
                    device = self.transformer_optimizer.device
                    weights = torch.tensor([likes_weight, reposts_weight, comments_weight, sentiment_weight, topic_weight, entity_weight, info_weight, interaction_overall_weight, text_overall_weight], dtype=torch.float32, device=device)
                    # 使用新的热度计算函数计算事件热度
                    event_hot = self.hot_calculator.calculate_batch_hot_tensor(event_df, weights, device)
                    origin_score = event_hot.item() if torch.is_tensor(event_hot) else float(event_hot)
                    event_origin_scores[event] = origin_score  # 保存原始热度值
                    event_scores[event] = origin_score
                    
                    # 记录该事件的参数，并进行归一化使权重和为1
                    interaction_weights = np.array([float(likes_weight), float(reposts_weight), float(comments_weight)])
                    text_weights = np.array([float(sentiment_weight), float(topic_weight), float(entity_weight), float(info_weight)])
                    overall_weights = np.array([float(interaction_overall_weight), float(text_overall_weight)])
                    
                    # 分别归一化各组权重
                    normalized_interaction = interaction_weights / np.sum(interaction_weights)
                    normalized_text = text_weights / np.sum(text_weights)
                    normalized_overall = overall_weights / np.sum(overall_weights)
                    
                    event_params[event] = {
                        'likes_weight': normalized_interaction[0],
                        'reposts_weight': normalized_interaction[1],
                        'comments_weight': normalized_interaction[2],
                        'sentiment_weight': normalized_text[0],
                        'topic_weight': normalized_text[1],
                        'entity_weight': normalized_text[2],
                        'info_weight': normalized_text[3],
                        'interaction_overall_weight': normalized_overall[0],
                        'text_overall_weight': normalized_overall[1]
                    }
                    
                    true_scores.append(true_hot)
                    pred_scores.append(event_scores[event])
                    
                except Exception as e:
                    print(f"处理事件 {event} 时出错: {str(e)}")
                    event_scores[event] = 0
                    event_origin_scores[event] = 0
            
            if not event_scores:
                raise ValueError("没有成功预测任何事件的热度")
            
            # 归一化预测分数到100分制
            scores = np.array(list(event_scores.values()))
            normalized_scores = (scores / np.max(scores)) * 100
            
            # 更新event_scores为归一化后的分数
            event_scores = {event: score for event, score in zip(event_scores.keys(), normalized_scores)}
            
            # 计算排名
            true_ranks = np.argsort(np.argsort(-np.array(true_scores))) + 1
            pred_ranks = np.argsort(np.argsort(-np.array(normalized_scores))) + 1
            
            # 计算相关系数
            rank_correlation = spearmanr(true_ranks, pred_ranks)[0]
            linear_correlation = pearsonr(true_scores, normalized_scores)[0]
            
            # 创建包含预测结果和真实值的DataFrame
            results_df = pd.DataFrame({
                '事件': list(event_scores.keys()),
                '预测分数': [f"{score:.8f}" for score in normalized_scores],
                '预测排名': pred_ranks,
            })
            
            # 按预测排名排序
            results_df = results_df.sort_values('预测排名')
            
            # 保存排名结果
            results_df.to_csv(os.path.join(RESULT_DIR, 'event_rankings.csv'), index=False, encoding='utf-8')
            
            return event_scores, rank_correlation, linear_correlation, event_params, event_origin_scores
            
        except Exception as e:
            print(f"预测过程中出错: {str(e)}")
            raise

# 初始化预测器，加载模型权重文件
def init_predictor():
    model_path = os.path.join('models', 'best_model.pt')
    if os.path.exists(model_path):
        print("发现预训练模型权重文件，直接加载使用...")
        predictor = FixedHotPredictor(model_path=model_path)
    else:
        print("未找到预训练模型权重文件，开始训练新模型...")
        predictor = FixedHotPredictor()
        # 训练参数
        test_spearman, test_linear, train_spearman, train_linear = predictor.train_parameters()
    return predictor

def predict_single_event(event_name, csv_file_path, image_dir_path=None, output_json_path=None):
    """
    预测单个事件的热度
    
    Args:
        event_name (str): 事件名称
        csv_file_path (str): 事件数据CSV文件路径
        image_dir_path (str, optional): 事件图片文件夹路径
        output_json_path (str, optional): 输出JSON文件路径，如果为None则自动生成
        
    Returns:
        str: 输出JSON文件的路径
    """
    try:
        print(f"开始预测事件 '{event_name}' 的热度...")
        
        # 检查输入文件是否存在
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_file_path}")
        
        # 如果提供了图片文件夹路径，检查是否存在
        if image_dir_path and not os.path.exists(image_dir_path):
            print(f"警告: 图片文件夹不存在: {image_dir_path}")
            image_dir_path = None
        
        # 加载事件数据
        print("加载事件数据...")
        event_df = pd.read_csv(csv_file_path)
        
        # 数据清理
        event_df = DataProcessor.clean_dataframe(event_df)
        
        # 添加事件名称列
        event_df['event_name'] = event_name
        
        # 检查数据是否为空
        if event_df.empty:
            raise ValueError("事件数据为空")
        
        print(f"成功加载 {len(event_df)} 条微博数据")
        
        # 初始化预测器
        predictor = init_predictor()
        
        # 预测热度
        print("计算事件热度...")
        # 使用Transformer优化器预测参数
        likes_weight, reposts_weight, comments_weight, sentiment_weight, topic_weight, entity_weight, info_weight, interaction_overall_weight, text_overall_weight = predictor.transformer_optimizer.optimize(event_df)
        predictor.hot_calculator.set_weights(likes_weight, reposts_weight, comments_weight, sentiment_weight, topic_weight, entity_weight, info_weight, interaction_overall_weight, text_overall_weight)
        device = predictor.transformer_optimizer.device
        weights = torch.tensor([likes_weight, reposts_weight, comments_weight, sentiment_weight, topic_weight, entity_weight, info_weight, interaction_overall_weight, text_overall_weight], dtype=torch.float32, device=device)
        event_hot = predictor.hot_calculator.calculate_batch_hot_tensor(event_df, weights, device)
        origin_score = event_hot.item() if torch.is_tensor(event_hot) else float(event_hot)
        
        # 归一化分数到100分制
        # 这里使用一个简单的归一化方法，实际应用中可能需要根据历史数据调整
        normalized_score = min(100.0, max(0.0, origin_score / 1000.0 * 100))  # 简单的线性归一化
        
        # 计算事件统计信息
        total_posts = len(event_df)
        total_reposts = event_df['转发数'].sum()
        total_comments = event_df['评论数'].sum()
        total_likes = event_df['点赞数'].sum()
        # posts_with_images = event_df['微博图片url'].notna().sum()
        # posts_with_videos = event_df['微博视频url'].notna().sum()
        
        # 计算用户认证分布
        auth_distribution = event_df['认证类型'].value_counts().to_dict()
        
        # 准备结果数据
        result_data = {
            'event_info': {
                'event_name': event_name,
                'csv_file_path': csv_file_path,
                'image_dir_path': image_dir_path,
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'hot_score': {
                'raw_score': float(origin_score),
                'normalized_score': float(normalized_score),
                'score_scale': '0-100'
            },
            'event_statistics': {
                'total_posts': int(total_posts),
                'total_reposts': int(total_reposts),
                'total_comments': int(total_comments),
                'total_likes': int(total_likes),
                'posts_with_images': int(posts_with_images),
                'posts_with_videos': int(posts_with_videos),
                'image_ratio': float(posts_with_images / total_posts) if total_posts > 0 else 0.0,
                'video_ratio': float(posts_with_videos / total_posts) if total_posts > 0 else 0.0
            },
            'user_authentication_distribution': auth_distribution,
            'model_parameters': {}
        }
        
        # 如果使用Transformer，添加模型参数信息
        # 归一化权重参数
        interaction_weights = np.array([float(likes_weight), float(reposts_weight), float(comments_weight)])
        text_weights = np.array([float(sentiment_weight), float(topic_weight), float(entity_weight), float(info_weight)])
        overall_weights = np.array([float(interaction_overall_weight), float(text_overall_weight)])
        
        # 分别归一化各组权重
        normalized_interaction = interaction_weights / np.sum(interaction_weights)
        normalized_text = text_weights / np.sum(text_weights)
        normalized_overall = overall_weights / np.sum(overall_weights)
        
        result_data['model_parameters'] = {
            'interaction_weights': {
                'likes_weight': float(normalized_interaction[0]),
                'reposts_weight': float(normalized_interaction[1]),
                'comments_weight': float(normalized_interaction[2])
            },
            'text_feature_weights': {
                'sentiment_weight': float(normalized_text[0]),
                'topic_weight': float(normalized_text[1]),
                'entity_weight': float(normalized_text[2]),
                'info_weight': float(normalized_text[3])
            },
            'overall_weights': {
                'interaction_overall_weight': float(normalized_overall[0]),
                'text_overall_weight': float(normalized_overall[1])
            }
        }
        
        # 生成输出文件路径
        if output_json_path is None:
            # 自动生成输出路径
            safe_event_name = "".join(c for c in event_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_event_name = safe_event_name.replace(' ', '_')
            output_json_path = os.path.join(RESULT_DIR, f"{safe_event_name}_prediction_result.json")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        
        # 保存结果到JSON文件
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
        
        print(f"事件 '{event_name}' 热度预测完成!")
        print(f"原始热度分数: {origin_score:.2f}")
        print(f"归一化热度分数: {normalized_score:.2f}")
        print(f"结果已保存到: {output_json_path}")
        
        return output_json_path
        
    except Exception as e:
        print(f"预测事件 '{event_name}' 时出错: {str(e)}")
        raise

def main():
    try:
        run_mode = "batch"  # 可选: "batch" 或 "single"
        
        if run_mode == "single":
            # 单个事件预测模式，直接输入事件名称，csv文件路径，图片文件夹路径
            print("=== 单条事件预测模式 ===")
            try:
                event_name = "示例事件"
                csv_file_path = "/path/to/your/event_data.csv"  # 替换为实际的CSV文件路径
                image_dir_path = "/path/to/your/images/"  # 替换为实际的图片文件夹路径（可选）
                
                # 调用单个事件预测函数
                result_json_path = predict_single_event(
                    event_name=event_name,
                    csv_file_path=csv_file_path,
                    image_dir_path=image_dir_path
                )
                
                print(f"预测完成，结果保存在: {result_json_path}")
                
                # 读取并显示结果
                with open(result_json_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                print("\n=== 预测结果摘要 ===")
                print(f"事件名称: {result['event_info']['event_name']}")
                print(f"原始热度分数: {result['hot_score']['raw_score']:.2f}")
                print(f"归一化热度分数: {result['hot_score']['normalized_score']:.2f}")
                print(f"总微博数: {result['event_statistics']['total_posts']}")
                print(f"总转发数: {result['event_statistics']['total_reposts']}")
                print(f"总评论数: {result['event_statistics']['total_comments']}")
                print(f"总点赞数: {result['event_statistics']['total_likes']}")
                
                return result_json_path
                
            except Exception as e:
                print(f"示例预测过程中出错: {str(e)}")
                raise
        else:
            # 批量事件预测模式，事件加载路径在data_processor.py中HOT_EVENTS_BASE_DIR设置，事件数据为csv文件，图片在对应事件文件夹的images文件夹中
            print("=== 批量事件预测模式 ===")
            # 初始化热度预测器
            predictor = init_predictor()
            # 计算分数和指标
            predicted_scores, all_correlation, all_linear_correlation, event_params, event_scores = predictor.predict_hot_scores()
            
            # 保存结果
            results = {
                'event_scores': event_scores,
                'event_parameters': event_params,  # 新增：每个事件的参数
                'metrics': {
                    'all_spearman': all_correlation,
                    'linear_correlation': all_linear_correlation
                }
            }
            
            # 将结果保存到JSON文件
            result_file = os.path.join(RESULT_DIR, "model_metrics.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print("事件预测完成！")
            
    except Exception as e:
        print(f"运行过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()