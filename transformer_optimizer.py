import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os
import pickle
import torch.nn.functional as F
# 移除torchsort依赖

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 确保参数类型为整数
        d_model = int(d_model)
        max_len = int(max_len)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerOptimizer(nn.Module):
    def __init__(self, d_model=64, nhead=2, num_encoder_layers=1, dropout=0.1):
        """初始化TransformerOptimizer"""
        super().__init__()
        
        # 参数验证
        assert d_model > 0, "d_model必须大于0"
        assert nhead > 0 and d_model % nhead == 0, "nhead必须大于0且能整除d_model"
        assert num_encoder_layers > 0, "num_encoder_layers必须大于0"
        assert 0 <= dropout < 1, "dropout必须在[0,1)范围内"
        
        # 保存模型维度
        self.d_model = d_model
        
        # 设置设备 - 优先使用CUDA，然后是CPU（不再检测Apple Silicon MPS）
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("发现CUDA GPU，将使用CUDA加速")
        else:
            self.device = torch.device('cpu')
            print("未发现GPU，将使用CPU")
        print(f"当前使用的设备: {self.device}")
        
        # 加载BERT模型
        from transformers import BertTokenizer, BertModel
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')
        self.bert_model.to(self.device)
        
        # 文本特征维度
        self.text_feature_dim = 768  # BERT的输出维度
        
        # 初始化HotCalculator
        from hot_calculator import HotCalculator
        from config import AUTH_WEIGHTS, MEDIA_BONUS
        self.hot_calculator = HotCalculator(AUTH_WEIGHTS, MEDIA_BONUS)
        
        # 特征维度
        self.input_dim = 6  # 基础特征维度
        
        # 文本特征处理层（简化）
        self.text_feature_processor = nn.Sequential(
            nn.Linear(self.text_feature_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 特征嵌入层（简化）
        self.feature_embedding = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 特征融合层（简化，移除中间层和多余归一化）
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # 直接合并基础特征和文本特征
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 添加位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器（简化：减少前馈网络维度）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # 进一步减少前馈网络维度
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers  
        )
        
        # 输出层（简化：直接输出，减少中间层）
        self.interaction_output = nn.Linear(d_model, 3)  # 3个互动权重 (likes, reposts, comments)
        self.text_feature_output = nn.Linear(d_model, 4)  # 4个文本特征权重 (sentiment, topic, entity, info)
        self.overall_weight_output = nn.Linear(d_model, 2)  # 2个总体权重 (w0: 互动权重, w1: 文本权重)
        
        # 添加Softmax层确保权重和为1
        self.interaction_softmax = nn.Softmax(dim=-1)
        self.text_feature_softmax = nn.Softmax(dim=-1)
        self.overall_weight_softmax = nn.Softmax(dim=-1)
        
        # 初始化权重
        self._init_weights()
        
        # 将模型移动到指定设备
        self.to(self.device)
        
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        
        # 初始化损失函数
        self.criterion = nn.MSELoss()
        
        # 初始化学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,
        )
        
        # 创建特征缓存目录
        self.cache_dir = os.path.join("cache", "features")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'norm' not in name:  # 不初始化LayerNorm的权重
                    if len(param.shape) >= 2:
                        # 对于2维及以上的权重使用xavier初始化
                        nn.init.xavier_uniform_(param)
                    else:
                        # 对于1维的权重使用正态分布初始化
                        nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                # 偏置项初始化为0
                nn.init.constant_(param, 0)
                
    def extract_text_features(self, text):
        """使用BERT提取文本特征"""
        # 文本预处理
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # 提取BERT特征
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # 使用[CLS]标记的输出作为文本特征
            text_features = outputs.last_hidden_state[:, 0, :]
        
        return text_features

    def prepare_features(self, event_df):
        """准备模型输入特征"""
        if event_df.empty:
            raise ValueError("输入的DataFrame为空")
        
        if 'event_name' not in event_df.columns:
            raise ValueError("event_df必须包含event_name列")
        
        if event_df['event_name'].isna().all():
            raise ValueError("event_name列不能全为空值")
        
        event_name = event_df['event_name'].iloc[0]
        
        # 尝试从缓存加载特征
        cached_features = self.load_features(event_name)
        if cached_features is not None:
            try:
                # 确保cached_features是tensor且维度正确
                if not isinstance(cached_features, torch.Tensor):
                    print("缓存的特征不是tensor类型，重新计算特征")
                    os.remove(self.get_cache_path(event_name))
                    cached_features = None
                elif not hasattr(cached_features, 'shape') or len(cached_features.shape) == 0:
                    print("缓存的特征没有正确的shape属性，重新计算特征")
                    os.remove(self.get_cache_path(event_name))
                    cached_features = None
                elif cached_features.shape[-1] != self.d_model:
                    print(f"缓存的特征维度不正确: 期望{self.d_model}，实际{cached_features.shape[-1]}，重新计算特征")
                    os.remove(self.get_cache_path(event_name))
                    cached_features = None
                else:
                    # 确保特征有正确的batch维度
                    if len(cached_features.shape) == 1:
                        cached_features = cached_features.unsqueeze(0)
                    return cached_features.unsqueeze(0)
                
            except Exception as e:
                print(f"处理缓存特征时出错: {str(e)}")
                try:
                    os.remove(self.get_cache_path(event_name))
                except:
                    pass
                cached_features = None
            
        try:
            num_posts = len(event_df)
            # 初始化特征矩阵 [num_posts, 6]
            features = np.zeros((num_posts, 6))
            
            # 数值特征的非线性转换和标准化
            for i, col in enumerate(['转发数', '评论数', '点赞数']):
                # 使用更平滑的非线性变换
                raw_values = event_df[col].values
                # 添加时间衰减因子
                time_decay = np.exp(-0.1 * np.arange(len(raw_values)) / len(raw_values))
                
                # 对原始值进行非线性变换
                log_values = np.log1p(raw_values)
                sqrt_values = np.sqrt(raw_values)
                
                # 结合多种变换特征
                combined_values = (log_values + sqrt_values) / 2 * time_decay
                
                # 标准化，但保持一定的非线性特征
                features[:, i] = (combined_values - combined_values.mean()) / (combined_values.std() + 1e-8)
            
            # 准备用户认证类型特征（使用更细致的权重）
            auth_mapping = {
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
            
            # 用户认证权重
            auth_weights = [auth_mapping.get(auth, 1.0) 
                          for auth in event_df['认证类型'].fillna('普通用户')]
            features[:, 3] = auth_weights
            
            # # 增强媒体特征的表达
            # has_image = event_df['微博图片url'].notna().astype(float).values
            # has_video = event_df['微博视频url'].notna().astype(float).values
            
            # 考虑图片数量
            # image_counts = event_df['微博图片url'].fillna('').str.count(',') + has_image
            
            # 归一化媒体特征
            # features[:, 4] = image_counts / (image_counts.max() + 1e-8)
            # features[:, 5] = has_video
            
            # 转换为tensor并确保数值稳定性
            features = torch.FloatTensor(features).to(self.device)
            features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 提取文本特征
            text_features_list = []
            # for text in event_df['微博正文'].fillna(''):
            for text in (event_df['全文内容'].fillna('') + event_df['原微博内容'].fillna('')).tolist():
                text_features = self.extract_text_features(text)
                text_features_list.append(text_features)
            
            # 堆叠所有文本特征
            text_features = torch.cat(text_features_list, dim=0)
            
            # 处理文本特征
            processed_text_features = self.text_feature_processor(text_features)
            
            # 处理基础特征
            processed_base_features = self.feature_embedding(features)
            
            # 特征融合（已简化，移除残差连接）
            fused_features = self.feature_fusion(
                torch.cat([processed_base_features, processed_text_features], dim=-1)
            )
            
            # 保存融合后的特征到缓存（不需要添加batch维度）
            self.save_features(event_name, fused_features)
            
            # 添加batch维度并返回
            return fused_features.unsqueeze(0)
            
        except Exception as e:
            print(f"特征准备过程中出错: {str(e)}")
            print(f"Event DataFrame shape: {event_df.shape}")
            raise

    def forward(self, x, return_attention=False):
        """前向传播"""
        try:
            # 确保输入维度正确
            if len(x.shape) == 2:
                x = x.unsqueeze(0)  # 添加batch维度
                
            # 获取特征
            features = x.squeeze(0)  # 移除batch维度以便处理
            
            # 使用位置编码
            encoded_features = self.pos_encoder(features.unsqueeze(0))
            
            # Transformer编码
            transformer_output = self.transformer_encoder(encoded_features)
            
            # 获取全局特征表示
            global_features = torch.mean(transformer_output, dim=1)  # [batch_size, d_model]
            
            # 预测权重
            interaction_weights = self.interaction_output(global_features)  # [batch_size, 3]
            text_feature_weights = self.text_feature_output(global_features)  # [batch_size, 4]
            overall_weights = self.overall_weight_output(global_features)  # [batch_size, 2]
            
            # 应用Softmax
            interaction_weights = self.interaction_softmax(interaction_weights)
            text_feature_weights = self.text_feature_softmax(text_feature_weights)
            overall_weights = self.overall_weight_softmax(overall_weights)
            
            # 确保所有权重都保持至少1维
            interaction_weights = interaction_weights.view(-1)  # 展平为1维
            text_feature_weights = text_feature_weights.view(-1)  # 展平为1维
            overall_weights = overall_weights.view(-1)  # 展平为1维
            
            # 合并所有权重
            weights = torch.cat([
                interaction_weights,    # 3个互动权重 (likes, reposts, comments)
                text_feature_weights,   # 4个文本特征权重 (sentiment, topic, entity, info)
                overall_weights        # 2个总体权重 (w0: 互动权重, w1: 文本权重)
            ])
            
            if return_attention:
                # Transformer编码器内部已有注意力机制，返回None作为占位符
                return weights, None
            return weights
            
        except Exception as e:
            print(f"前向传播时出错: {str(e)}")
            raise

    def get_cache_path(self, event_name):
        """获取特征缓存文件路径"""
        return os.path.join(self.cache_dir, f"{event_name}_features.pt")
        
    def save_features(self, event_name, features):
        """保存特征到缓存"""
        cache_path = self.get_cache_path(event_name)
        try:
            features_cpu = features.detach().cpu()
            torch.save(features_cpu, cache_path, _use_new_zipfile_serialization=True)
        except Exception as e:
            print(f"保存特征时出错: {str(e)}")
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except:
                    pass
        
    def load_features(self, event_name):
        """从缓存加载融合后的特征"""
        cache_path = self.get_cache_path(event_name)
        if not os.path.exists(cache_path):
            return None
        try:
            # 加载特征
            try:
                features = torch.load(cache_path, map_location=self.device, weights_only=True)
            except RuntimeError as e:
                if "legacy format" in str(e):
                    features = torch.load(cache_path, map_location=self.device, pickle_module=pickle, weights_only=True)
                else:
                    raise e
            
            # 确保特征是tensor类型
            if not isinstance(features, torch.Tensor):
                print(f"加载的特征不是tensor类型: {type(features)}")
                return None
            
            # 确保特征在正确的设备上
            features = features.to(self.device)
            # 检查特征的数值有效性
            if torch.isnan(features).any() or torch.isinf(features).any():
                print("加载的特征包含无效值(NaN或Inf)")
                return None
            
            # 检查特征维度是否正确（应该是融合后的特征维度）
            if features.shape[-1] != self.d_model:
                print(f"加载的特征维度不正确: 期望{self.d_model}，实际{features.shape[-1]}")
                return None
            
            return features
            
        except Exception as e:
            print(f"加载特征缓存时出错: {str(e)}")
            try:
                os.remove(cache_path)
            except:
                pass
            return None
        
    def optimize(self, event_df):
        """使用预训练模型输出热度参数"""
        self.eval()
        with torch.no_grad():
            # 准备特征
            features = self.prepare_features(event_df)
            
            # 使用模型预测权重
            weights, attention_weights = self(features, return_attention=True)
            
            # 提取并处理权重参数
            weights = weights.squeeze()
            if weights.is_cuda:
                weights = weights.cpu()
            
            # 提取所有权重 (总共9个权重)
            # 前3个：互动权重 (likes, reposts, comments)
            likes_weight, reposts_weight, comments_weight = [float(w.item()) for w in weights[:3]]
            # 中间4个：文本特征权重 (sentiment, topic, entity, info)
            sentiment_weight, topic_weight, entity_weight, info_weight = [float(w.item()) for w in weights[3:7]]
            # 最后2个：总体权重 (w0: 互动权重, w1: 文本权重)
            w0, w1 = [float(w.item()) for w in weights[7:9]]
            
            return likes_weight, reposts_weight, comments_weight, sentiment_weight, topic_weight, entity_weight, info_weight, w0, w1
        
    def normalized_mse_loss(self, pred_scores, true_scores):
        """使用标准化后分数的MSE作为损失函数"""
        # pred_scores 和 true_scores 都是 1D 张量
        
        # 标准化预测分数
        pred_max = pred_scores.max() + 1e-8
        pred_normalized = pred_scores/ pred_max * 100
        
        # 标准化真实分数
        true_max = true_scores.max() + 1e-8
        true_normalized = true_scores / true_max
        
        # 计算标准化分数的MSE损失
        mse_loss = F.mse_loss(pred_normalized, true_normalized)
        return mse_loss
    
    def train_model(self, train_data, epochs=20, batch_size=32):
        """训练Transformer模型"""
        print("开始训练，使用AdamW优化器和自适应学习率")
        
        # 预加载所有训练数据的特征，避免在训练循环中重复加载
        print("预加载训练数据特征...")
        preloaded_data = []
        for event_df, true_hot, event_name in train_data:
            try:
                features = self.prepare_features(event_df)
                preloaded_data.append((features, true_hot, event_name, event_df))
            except Exception as e:
                print(f"预加载事件 {event_name} 特征时出错: {str(e)}")
                continue
        
        print(f"成功预加载 {len(preloaded_data)} 个事件的特征")
        
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 10  # 提前停止的耐心值
        
        # 添加L2正则化
        weight_decay = 0.1
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=weight_decay)
        for epoch in range(epochs):
            # 设置为训练模式
            self.train(True)
            total_loss = 0
            total_reg_loss = 0
        

            for i in range(0, len(preloaded_data), batch_size):
                batch_data = preloaded_data[i:i+batch_size]
                self.optimizer.zero_grad()
                
                try:
                    batch_pred_scores = []
                    batch_true_scores = []
                    
                    for features, true_hot, event_name, event_df in batch_data:
                        predicted_weights = self(features)  # 获取预测的权重
                        
                        # 确保predicted_weights是一维张量
                        if len(predicted_weights.shape) > 1:
                            predicted_weights = predicted_weights.squeeze()
                        # 使用tensor计算热度，保持梯度传播
                        hot_score = self.hot_calculator.calculate_batch_hot_tensor(event_df, predicted_weights, self.device)
                        batch_pred_scores.append(hot_score)
                        batch_true_scores.append(true_hot)
                    
                    # 堆叠所有预测分数，保持梯度
                    pred_scores = torch.stack(batch_pred_scores)
                    true_scores = torch.tensor(batch_true_scores, dtype=torch.float32, device=self.device)
                    
                    # 使用标准化MSE损失
                    loss = self.normalized_mse_loss(pred_scores, true_scores)
                    
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                    
                    self.optimizer.step()
                    total_loss += loss.item()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"警告：GPU内存不足，跳过当前批次")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            
            # 计算平均损失
            avg_loss = total_loss / len(preloaded_data)
            
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.7, 
                patience=5,
                min_lr=1e-6
            )
            self.scheduler.step(avg_loss)
            
            # 打印训练信息
            if (epoch + 1) % 2 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')
            
            # 检查是否需要提前停止
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f"Loss 在 {max_patience} 个epoch内没有改善，提前停止训练")
                break
        
        return self 
    
    def save_model(self, model_path):
        """保存模型到指定路径"""
        try:
            # 确保模型保存路径存在
            model_dir = os.path.dirname(model_path)
            if model_dir:  # 如果路径包含目录
                os.makedirs(model_dir, exist_ok=True)
            
            # 获取前馈网络维度
            dim_feedforward = self.transformer_encoder.layers[0].linear1.out_features
            
            # 保存模型状态
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'd_model': self.d_model,
                'nhead': self.transformer_encoder.layers[0].self_attn.num_heads,
                'num_encoder_layers': len(self.transformer_encoder.layers),
                'dim_feedforward': dim_feedforward,  # 保存前馈网络维度
                'dropout': self.transformer_encoder.layers[0].dropout.p
            }, model_path)
            print(f"模型已保存到: {model_path}")
        except Exception as e:
            print(f"保存模型时出错: {str(e)}")
            raise

    def load_model(self, model_path):
        """从指定路径加载模型"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 加载模型状态
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 检查checkpoint中是否包含模型配置信息
            if 'd_model' in checkpoint and 'num_encoder_layers' in checkpoint:
                saved_d_model = checkpoint['d_model']
                saved_num_layers = checkpoint['num_encoder_layers']
                saved_nhead = checkpoint.get('nhead', 4)
                saved_dropout = checkpoint.get('dropout', 0.1)
                
                # 优先使用保存的dim_feedforward，如果没有则从权重形状推断
                saved_dim_feedforward = checkpoint.get('dim_feedforward')
                if saved_dim_feedforward is None:
                    # 尝试从checkpoint推断保存的模型的前馈网络维度
                    # 通过检查第一个linear1层的权重形状
                    saved_state_dict = checkpoint['model_state_dict']
                    if 'transformer_encoder.layers.0.linear1.weight' in saved_state_dict:
                        saved_linear1_weight = saved_state_dict['transformer_encoder.layers.0.linear1.weight']
                        saved_dim_feedforward = saved_linear1_weight.shape[0]
                    else:
                        saved_dim_feedforward = saved_d_model * 8  # 默认值
                
                # 检查当前模型配置是否与保存的配置匹配
                current_num_layers = len(self.transformer_encoder.layers)
                current_d_model = self.d_model
                
                # 获取当前模型的前馈网络维度
                current_dim_feedforward = self.transformer_encoder.layers[0].linear1.out_features
                
                # 如果配置不匹配，需要重新初始化模型
                config_mismatch = (
                    saved_d_model != current_d_model or
                    saved_num_layers != current_num_layers or
                    saved_dim_feedforward != current_dim_feedforward
                )
                
                if config_mismatch:
                    print(f"检测到模型配置不匹配:")
                    print(f"  保存的模型: d_model={saved_d_model}, num_layers={saved_num_layers}, dim_feedforward={saved_dim_feedforward}")
                    print(f"  当前模型: d_model={current_d_model}, num_layers={current_num_layers}, dim_feedforward={current_dim_feedforward}")
                    print(f"正在使用保存的配置重新初始化模型...")
                    
                    # 重新初始化模型组件
                    # 重新创建Transformer编码器层
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=saved_d_model,
                        nhead=saved_nhead,
                        dim_feedforward=saved_dim_feedforward,
                        dropout=saved_dropout,
                        batch_first=True
                    )
                    self.transformer_encoder = nn.TransformerEncoder(
                        encoder_layer,
                        num_layers=saved_num_layers
                    )
                    
                    # 更新d_model
                    self.d_model = saved_d_model
                    
                    # 重新初始化相关层（如果需要调整维度）
                    if saved_d_model != current_d_model:
                        # 重新初始化文本特征处理层（简化版）
                        self.text_feature_processor = nn.Sequential(
                            nn.Linear(self.text_feature_dim, saved_d_model),
                            nn.ReLU(),
                            nn.Dropout(saved_dropout)
                        )
                        
                        # 重新初始化特征嵌入层（简化版）
                        self.feature_embedding = nn.Sequential(
                            nn.Linear(self.input_dim, saved_d_model),
                            nn.ReLU(),
                            nn.Dropout(saved_dropout)
                        )
                        
                        # 重新初始化特征融合层（简化版）
                        self.feature_fusion = nn.Sequential(
                            nn.Linear(saved_d_model * 2, saved_d_model),
                            nn.ReLU(),
                            nn.Dropout(saved_dropout)
                        )
                        
                        # 重新初始化位置编码
                        self.pos_encoder = PositionalEncoding(saved_d_model, saved_dropout)
                        
                        # 重新初始化输出层（简化版，直接输出）
                        self.interaction_output = nn.Linear(saved_d_model, 3)
                        self.text_feature_output = nn.Linear(saved_d_model, 4)
                        self.overall_weight_output = nn.Linear(saved_d_model, 2)
                        
                        # 重新初始化权重
                        self._init_weights()
                        
                        # 重新创建优化器
                        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
                        
                        # 重新创建学习率调度器
                        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            self.optimizer, 
                            mode='min', 
                            factor=0.5, 
                            patience=3,
                        )
                        
                        # 将模型移动到设备
                        self.to(self.device)
            
            # 使用strict=False来加载兼容的权重，忽略不兼容的层
            try:
                self.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("模型权重已加载（部分权重可能因配置不匹配而跳过）")
            except Exception as e:
                print(f"警告：加载模型权重时出现错误: {str(e)}")
                print("尝试使用兼容模式加载...")
                # 尝试手动加载兼容的权重
                saved_state_dict = checkpoint['model_state_dict']
                model_state_dict = self.state_dict()
                
                # 只加载兼容的权重
                compatible_state_dict = {}
                for key, value in saved_state_dict.items():
                    if key in model_state_dict:
                        if model_state_dict[key].shape == value.shape:
                            compatible_state_dict[key] = value
                        else:
                            print(f"跳过不兼容的权重: {key} (形状不匹配)")
                    else:
                        print(f"跳过不存在的权重: {key}")
                
                model_state_dict.update(compatible_state_dict)
                self.load_state_dict(model_state_dict)
                print(f"已加载 {len(compatible_state_dict)} 个兼容的权重")
            
            # 加载优化器状态（如果存在且兼容）
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    print(f"警告：无法加载优化器状态: {str(e)}")
            
            # 加载学习率调度器状态（如果存在且兼容）
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    print(f"警告：无法加载学习率调度器状态: {str(e)}")
            
            # 设置为评估模式
            self.eval()
            print(f"模型已从 {model_path} 加载")
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            raise 