import numpy as np
from data_processor import DataProcessor
import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.nn.functional import cosine_similarity
import jieba
import math
from collections import Counter
from snownlp import SnowNLP
import re
import os
import json

class TextFeatureExtractor:
    def __init__(self, device=None):
        if device is None:
            # 自动检测最佳设备（仅检测CUDA与CPU）
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        # 加载BERT模型用于主题相关性计算
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')
        self.bert_model.to(device)
        self.bert_model.eval()
        
    def calculate_sentiment_intensity(self, text):
        """计算情感强度"""
        try:
            s = SnowNLP(text)
            # 将情感得分从[0,1]映射到[-1,1]
            sentiment = (s.sentiments - 0.5) * 2
            # 返回情感强度的绝对值
            return abs(sentiment)*len(text)
        except:
            return 0.0
            
    def calculate_topic_relevance(self, text, event_name):
        """计算主题相关性"""
        try:
            with torch.no_grad():
                # 对文本进行编码
                inputs = self.tokenizer(text, return_tensors="pt", 
                                      truncation=True, max_length=512,
                                      padding=True).to(self.device)
                text_embedding = self.bert_model(**inputs).last_hidden_state.mean(dim=1)
                
                # 对事件名进行编码
                event_inputs = self.tokenizer(event_name, return_tensors="pt",
                                            truncation=True, max_length=512,
                                            padding=True).to(self.device)
                event_embedding = self.bert_model(**event_inputs).last_hidden_state.mean(dim=1)
                
                # 计算余弦相似度
                similarity = cosine_similarity(text_embedding, event_embedding).item()
                return max(0, similarity)*len(text)  # 确保相似度非负
        except:
            return 0.0
            
    def calculate_entity_richness(self, text):
        """计算实体丰富度"""
        try:
            # 使用jieba进行分词和词性标注
            import jieba.posseg as pseg
            words = pseg.cut(text)
            
            # 统计不同类型的实体
            entity_count = 0
            for word, flag in words:
                # nr：人名，ns：地名，nt：机构名，nw：作品名
                if flag in ['nr', 'ns', 'nt', 'nw']:
                    entity_count += 1
            
            # 归一化实体数量
            return entity_count
        except:
            return 0.0
            
    def calculate_information_density(self, text):
        """计算文本信息量"""
        try:
            # 分词
            words = list(jieba.cut(text))
            
            if not words:
                return 0.0
                
            # 计算词频
            word_freq = Counter(words)
            total_words = len(words)
            
            # 计算信息熵
            entropy = 0
            for word, freq in word_freq.items():
                prob = freq / total_words
                entropy -= prob * math.log2(prob)
            
            # 计算词汇多样性
            vocabulary_richness = len(word_freq)
            
            # 计算文本长度得分
            length_score = len(text)  
            
            # 综合得分
            info_score = entropy + vocabulary_richness + length_score
            return info_score
        except:
            return 0.0
    
    def calculate_batch_text_features(self, texts, event_name):
        """批量计算文本特征"""
        try:
            # 批量计算情感强度（不需要BERT）
            sentiment_scores = []
            for text in texts:
                sentiment_scores.append(self.calculate_sentiment_intensity(text))
            
            # 批量计算实体丰富度（不需要BERT）
            entity_scores = []
            for text in texts:
                entity_scores.append(self.calculate_entity_richness(text))
            
            # 批量计算信息密度（不需要BERT）
            info_scores = []
            for text in texts:
                info_scores.append(self.calculate_information_density(text))
            # # 优化主题相关性计算 - 批量处理BERT调用
            # topic_scores = self.calculate_batch_topic_relevance(texts, event_name)
            # 简化主题相关性计算 - 使用文本长度作为代理特征，跳过BERT
            topic_scores = [len(text) * 0.1 for text in texts]  # 简单的长度特征
            
            return {
                'sentiment': sentiment_scores,
                'topic': topic_scores,
                'entity': entity_scores,
                'info': info_scores
            }
        except Exception as e:
            print(f"批量计算文本特征时出错: {str(e)}")
            # 返回默认值
            return {
                'sentiment': [0.0] * len(texts),
                'topic': [0.0] * len(texts),
                'entity': [0.0] * len(texts),
                'info': [0.0] * len(texts)
            }
    
    def calculate_batch_topic_relevance(self, texts, event_name):
        """批量计算主题相关性 - 优化BERT调用"""
        try:
            # 对事件名进行一次编码
            event_inputs = self.tokenizer(event_name, return_tensors="pt",
                                        truncation=True, max_length=512,
                                        padding=True).to(self.device)
            with torch.no_grad():
                event_embedding = self.bert_model(**event_inputs).last_hidden_state.mean(dim=1)
            
            # 批量处理所有文本
            batch_size = 8  # 控制批次大小，避免内存溢出
            topic_scores = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # 批量编码文本
                inputs = self.tokenizer(batch_texts, return_tensors="pt", 
                                      truncation=True, max_length=512,
                                      padding=True).to(self.device)
                
                with torch.no_grad():
                    text_embeddings = self.bert_model(**inputs).last_hidden_state.mean(dim=1)
                    
                    # 计算余弦相似度
                    similarities = cosine_similarity(text_embeddings, event_embedding)
                    
                    # 处理每个文本的相似度
                    for j, (text, similarity) in enumerate(zip(batch_texts, similarities)):
                        score = max(0, similarity.item()) * len(text)
                        topic_scores.append(score)
            
            return topic_scores
            
        except Exception as e:
            print(f"批量计算主题相关性时出错: {str(e)}")
            return [0.0] * len(texts)

class HotCalculator:
    def __init__(self, auth_weights=None, media_bonus=None):
        # 初始化参数
        self.auth_weights = auth_weights or {
            '金V': 1.5,
            '蓝V': 1.5,  # 蓝V与金V同等重要
            '个人认证-橙V':1.5,
            '个人认证-达人':1.1,
            '机构认证-媒体':2.0,
            '机构认证-团体':1.5,
            '个人认证-橙V':1.3,
            '个人认证-金V':1.5,
            '红V': 1.3,
            '黄V': 1.1,
            '普通用户': 1.0
        }
        
        # 新权重参数（初始化为合理的默认值，真实使用时会被模型覆盖）
        self.likes_weight = 1.0
        self.reposts_weight = 1.0
        self.comments_weight = 1.0
        self.sentiment_weight = 1.0
        self.topic_weight = 1.0
        self.entity_weight = 1.0
        self.info_weight = 1.0
        # 总体权重
        self.interaction_overall_weight = 0.5
        self.text_overall_weight = 0.5
        
        # 设置默认设备
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # 初始化文本特征提取器
        self.text_extractor = TextFeatureExtractor(self.device)
        
    def set_weights(self, likes_weight, reposts_weight, comments_weight, 
                   sentiment_weight, topic_weight, entity_weight, info_weight, interaction_overall_weight, text_overall_weight):
        """设置新的权重参数结构"""
        # 互动权重
        self.likes_weight = likes_weight      # 点赞权重
        self.reposts_weight = reposts_weight  # 转发权重
        self.comments_weight = comments_weight # 评论权重
        
        # 文本特征权重
        self.sentiment_weight = sentiment_weight  # 情感权重
        self.topic_weight = topic_weight          # 主题权重
        self.entity_weight = entity_weight        # 实体权重
        self.info_weight = info_weight            # 信息量权重
        
        # 总体权重
        self.interaction_overall_weight = interaction_overall_weight  # 互动总体权重
        self.text_overall_weight = text_overall_weight  # 文本总体权重
            
    def get_cache_dir(self):
        """获取缓存目录"""
        cache_dir = os.path.join("cache", "text_features")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
        
    def get_feature_cache_path(self, event_name):
        """获取特定事件的特征缓存文件路径"""
        cache_dir = self.get_cache_dir()
        return os.path.join(cache_dir, f"{event_name}_text_features.json")
        
    def load_cached_features(self, event_name):
        """加载缓存的特征"""
        cache_path = self.get_feature_cache_path(event_name)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载缓存特征失败: {str(e)}")
        return {}
        
    def save_cached_features(self, event_name, features):
        """保存特征到缓存"""
        cache_path = self.get_feature_cache_path(event_name)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(features, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存特征缓存失败: {str(e)}")
            
    def calculate_text_features(self, text, event_name, weibo_id=None):
        """计算文本的四个特征得分
        
        Args:
            text: 微博文本内容
            event_name: 事件名称
            weibo_id: 微博ID，用于缓存索引
            
        Returns:
            tuple: (sentiment_score, topic_score, entity_score, info_score)
        """
        # 如果提供了微博ID，尝试从缓存加载
        if weibo_id is not None:
            cached_features = self.load_cached_features(event_name)
            if str(weibo_id) in cached_features:
                cached = cached_features[str(weibo_id)]
                if isinstance(cached, dict) and len(cached) == 4:
                    return cached['sentiment'], cached['topic'], cached['entity'], cached['info']
        
        # 计算各个文本特征得分
        sentiment_score = self.text_extractor.calculate_sentiment_intensity(text)
        topic_score = self.text_extractor.calculate_topic_relevance(text, event_name)
        entity_score = self.text_extractor.calculate_entity_richness(text)
        info_score = self.text_extractor.calculate_information_density(text)
        
        # 如果提供了微博ID，保存到缓存
        if weibo_id is not None:
            cached_features = self.load_cached_features(event_name)
            cached_features[str(weibo_id)] = {
                'sentiment': sentiment_score,
                'topic': topic_score,
                'entity': entity_score,
                'info': info_score
            }
            self.save_cached_features(event_name, cached_features)
        
        return sentiment_score, topic_score, entity_score, info_score
            
    def calculate_event_hot(self, event_df, device=None):
        """计算事件总热度，使用tensor计算"""
        if device is None:
            device = self.device
            
        # 将所有权重转换为tensor
        weights = torch.tensor([
            self.likes_weight, self.reposts_weight, self.comments_weight,  # 互动权重
            self.sentiment_weight, self.topic_weight, self.entity_weight, self.info_weight,  # 文本特征权重
            self.interaction_overall_weight, self.text_overall_weight  # 总体权重
        ], dtype=torch.float32, device=device)
        
        return self.calculate_batch_hot_tensor(event_df, weights, device)
        
    def calculate_batch_hot_tensor(self, event_df, weights, device):
        """使用新的权重结构计算批量热度 - 优化版本
        
        Args:
            event_df: 事件数据DataFrame
            weights: 权重tensor [likes_weight, reposts_weight, comments_weight, sentiment_weight, topic_weight, entity_weight, info_weight, interaction_overall_weight, text_overall_weight]
            device: 计算设备
        """
        try:
            # 确保weights是一维张量
            if len(weights.shape) > 1:
                weights = weights.squeeze()
            
            event_name = event_df['event_name'].iloc[0]
            
            # 向量化处理所有微博数据
            likes = torch.tensor(event_df['点赞数'].values, dtype=torch.float32, device=device)
            reposts = torch.tensor(event_df['转发数'].values, dtype=torch.float32, device=device)
            comments = torch.tensor(event_df['评论数'].values, dtype=torch.float32, device=device)
            
            # 向量化媒体特征
            # has_image = torch.tensor([
            #     1.2 if pd.notna(url) and str(url).strip() != '' else 1.0 
            #     for url in event_df['微博图片url']
            # ], dtype=torch.float32, device=device)
            
            # has_video = torch.tensor([
            #     1.5 if pd.notna(url) and str(url).strip() != '' else 1.0 
            #     for url in event_df['微博视频url']
            # ], dtype=torch.float32, device=device)
            
            # 向量化认证权重
            auth_weights = torch.tensor([
                float(self.auth_weights.get(auth_type, 1.0))
                for auth_type in event_df['认证类型'].fillna('普通用户')
            ], dtype=torch.float32, device=device)
            
            # 批量计算文本特征 - 优化版本
            texts = (event_df['全文内容'].fillna('') + event_df['原微博内容'].fillna('')).tolist()
            text_features = self.text_extractor.calculate_batch_text_features(texts, event_name)
            
            # 转换为tensor
            sentiment_scores = torch.tensor(text_features['sentiment'], dtype=torch.float32, device=device)
            topic_scores = torch.tensor(text_features['topic'], dtype=torch.float32, device=device)
            entity_scores = torch.tensor(text_features['entity'], dtype=torch.float32, device=device)
            info_scores = torch.tensor(text_features['info'], dtype=torch.float32, device=device)
            
            # 向量化计算互动分数
            interaction_scores = (
                weights[0] * likes +      # w00 * likes
                weights[1] * reposts +    # w01 * reposts  
                weights[2] * comments     # w02 * comments
            ) * auth_weights
            
            # 向量化计算文本分数
            text_scores = (
                weights[3] * sentiment_scores +  # w10 * sentiment
                weights[4] * topic_scores +      # w11 * topic
                weights[5] * entity_scores +     # w12 * entity
                weights[6] * info_scores         # w13 * info
            )
            
            # 向量化计算总体分数
            overall_scores = weights[7] * interaction_scores + weights[8] * text_scores
            
            # 向量化应用媒体权重
            final_scores = overall_scores
            
            # 返回总热度
            return torch.sum(final_scores)
            
        except Exception as e:
            print(f"计算批量热度时出错: {str(e)}")
            print(f"weights shape: {weights.shape}")
            print(f"weights: {weights}")
            raise 