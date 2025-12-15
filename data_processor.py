import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

HOT_EVENTS_BASE_DIR = "./event_新浪微博"

class DataProcessor:
    @staticmethod
    def clean_numeric_field(value):
        """清理数值字段，处理可能的字符串混合情况"""
        if pd.isna(value):
            return 0
        if isinstance(value, (int, float)):
            return int(value)
        # 如果是字符串，尝试提取数字
        try:
            # 移除可能的非数字字符（保留数字和小数点）
            cleaned = ''.join(c for c in str(value) if c.isdigit() or c == '.')
            return int(float(cleaned)) if cleaned else 0
        except:
            return 0
            
    @staticmethod
    def clean_dataframe(df):
        """清理整个数据框的数值字段"""
        df = df.copy()
        numeric_columns = ['转发数', '评论数', '点赞数']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(DataProcessor.clean_numeric_field)
        return df
    
    @staticmethod
    # def load_event_data(event_name):
    #     """加载单个事件的数据"""
    #     event_dir = os.path.join(HOT_EVENTS_BASE_DIR, event_name)
    #     if not os.path.isdir(event_dir):
    #         return None
            
    #     csv_files = [f for f in os.listdir(event_dir) if f.endswith('.csv')]
    #     if not csv_files:
    #         return None
            
    #     csv_file = next((f for f in csv_files if f.lower().startswith(event_name.lower())), 
    #                    csv_files[0])
    #     csv_path = os.path.join(event_dir, csv_file)
        
    #     try:
    #         event_df = pd.read_csv(csv_path)
    #         return DataProcessor.clean_dataframe(event_df)
    #     except Exception as e:
    #         print(f"加载事件 {event_name} 数据时出错: {str(e)}")
    #         return None
    def load_event_data(event_name):
        """加载单个事件的数据（事件目录名带序号，需去除）"""
        # 在事件基础目录下找到所有目录
        all_dirs = [d for d in os.listdir(HOT_EVENTS_BASE_DIR) if os.path.isdir(os.path.join(HOT_EVENTS_BASE_DIR, d))]
        # 匹配去除前缀（2位数字+下划线，或者2位+空格，或者空格+2位）再比对事件名
        matching_dir = None
        for dirname in all_dirs:
            # 去掉开头的2位数字和下划线/空格
            if len(dirname) > 2 and (dirname[2] == '_' or dirname[2] == ' '):
                pure_name = dirname[3:]
            # 有些情况可能是 01xx，中间没下划线/空格
            elif len(dirname) > 2 and dirname[:2].isdigit():
                pure_name = dirname[2:]
            else:
                pure_name = dirname

            # 有的事件会带 #，只按事件名包含匹配
            if event_name in pure_name or pure_name in event_name:
                matching_dir = dirname
                break
        if matching_dir is None:
            print(f"未找到对应的事件目录: {event_name}")
            return None

        event_dir = os.path.join(HOT_EVENTS_BASE_DIR, matching_dir)
        if not os.path.isdir(event_dir):
            return None

        csv_files = [f for f in os.listdir(event_dir) if f.endswith('.csv')]
        if not csv_files:
            return None

        # 试图优先选择包含事件名的csv，否则默认第一个
        csv_file = next((f for f in csv_files if event_name.lower() in f.lower()), csv_files[0])
        csv_path = os.path.join(event_dir, csv_file)

        try:
            event_df = pd.read_csv(csv_path)
            return DataProcessor.clean_dataframe(event_df)
        except Exception as e:
            print(f"加载事件 {event_name} 数据时出错: {str(e)}")
            return None
    
    @staticmethod
    def filter_events(stats_df, min_posts=10):
        """过滤事件，帖子数量达到最小要求的事件"""
        # 过滤帖子数量不足的事件
        filtered_stats = stats_df[stats_df['总帖子数'] >= min_posts]
        
        return filtered_stats

    def prepare_event_data(self, ground_truth_file, min_posts=10, std_multiplier=2):
        """
        统一处理事件数据的加载和过滤
        
        Args:
            ground_truth_file: 包含真实热度的CSV文件路径
            min_posts: 最小帖子数量
            std_multiplier: 标准差倍数（用于过滤异常值）
            
        Returns:
            tuple: (filtered_events_data, filtered_stats_df)
                - filtered_events_data: [(event_df, true_hot, event_name), ...]
                - filtered_stats_df: 过滤后的事件统计DataFrame
        """
        try:
            # 加载真实热度数据
            if not os.path.exists(ground_truth_file):
                raise FileNotFoundError(f"真实热度文件不存在: {ground_truth_file}")
            
            ground_truth_df = pd.read_csv(ground_truth_file)
            print(f"加载真实热度数据: {len(ground_truth_df)} 个事件")
            
            # 收集所有事件的原始数据
            event_stats = []
            for _, row in ground_truth_df.iterrows():
                event_name = row['关键词']
                true_hot = row['标准化热度']
                
                event_df = self.load_event_data(event_name)
                if event_df is None or len(event_df) == 0:
                    print(f"警告：事件 {event_name} 的数据为空或无法加载")
                    continue
                
                # 计算每个事件的统计信息
                event_stats.append({
                    '事件': event_name,
                    '总帖子数': len(event_df),
                    '真实热度': true_hot
                })
            
            # 转换为DataFrame进行分析
            stats_df = pd.DataFrame(event_stats)
            
            # 过滤事件
            filtered_stats = self.filter_events(stats_df, min_posts)
            
            # 准备过滤后的事件数据
            filtered_events_data = []

            for _, row in filtered_stats.iterrows():
                event = row['事件']
                true_hot = row['真实热度']
                
                event_df = self.load_event_data(event)
                if event_df is None or len(event_df) == 0:
                    continue
                
                # 添加事件名称列
                event_df['event_name'] = event
                
                # 直接使用全部帖子，不进行采样
                filtered_events_data.append((
                    event_df,
                    true_hot,
                    event
                ))
            
            if not filtered_events_data:
                raise ValueError("没有可用的事件数据")
            
            return filtered_events_data, filtered_stats
            
        except Exception as e:
            print(f"准备事件数据时出错: {str(e)}")
            raise 