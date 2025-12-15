import os
import torch

# 设备配置
def get_device():
    # 仅检查CUDA支持（适用于NVIDIA GPU），不再检测Apple Silicon MPS
    if torch.cuda.is_available():
        # 获取可用的GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"发现 {gpu_count} 个可用的CUDA GPU设备")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
        return "cuda"
    else:
        print("未找到可用的GPU，将使用CPU运行")
        return "cpu"

DEVICE = get_device()
print(f"当前使用的设备: {DEVICE}")

# 数据集选择
DATASET_TYPE = "WEIBO"  # 可选值: "CRISIS", "TWITTER100K", "WEIBO"

# Twitter100k数据集配置
# TEXT_FILE_PATH = "D:\\硕士学习资料\\研一\\硕士项目\\舆情分析\\twiiter100k\\filtered_texts_top20_hashtags.txt"
# IMAGE_DIR = "D:\\硕士学习资料\\研一\\硕士项目\\舆情分析\\twiiter100k\\Twitter100k-images-without-user-info\\"

# # Crisis数据集配置
# CRISIS_EVENTS = [
#     'california_wildfires',
#     'hurricane_maria',
#     'srilanka_floods',
#     'mexico_earthquake',
#     'iraq_iran_earthquake',
#     'hurricane_irma',
#     'hurricane_harvey'
# ]

# 微博数据集配置
WEIBO_BASE_DIR = "old_结果文件"
WEIBO_IMAGE_DIRNAME = "images"     # 图片目录的名称
WEIBO_REQUIRED_FIELDS = {
    "id": "id",
    "text": "微博正文"
}

# 文件路径配置
BASE_DIR = "舆情分析"
# CRISIS_DATA_DIR = os.path.join(BASE_DIR, "CRisis", "annotations")
# CRISIS_IMAGE_DIR = os.path.join(BASE_DIR, "CRisis")
OUTPUT_DIR = "./clustering_results"
RECLUSTER_OUTPUT_DIR = "./reclustering_results"

# 模型配置
CLIP_MODELS = {
    "CRISIS": "ViT-B/32",  # 英文数据集使用原始CLIP
    "TWITTER100K": "ViT-B/32",  # 英文数据集使用原始CLIP
    "WEIBO": "ViT-B-16"  # 中文数据集使用中文CLIP
}

CLIP_MODEL_SOURCES = {
    "CRISIS": "openai",  # 使用openai的原始CLIP
    "TWITTER100K": "openai",  # 使用openai的原始CLIP
    "WEIBO": "chinese"  # 使用chinese-clip
}

# NER模型配置
NER_MODELS = {
    "CRISIS": "dbmdz/bert-large-cased-finetuned-conll03-english",  # 英文NER
    "TWITTER100K": "dbmdz/bert-large-cased-finetuned-conll03-english",  # 英文NER
    "WEIBO": "hfl/chinese-roberta-wwm-ext-large"  # 中文NER
}

# 序列长度配置
MAX_SEQ_LENGTHS = {
    "CRISIS": 128,  # 英文文本
    "TWITTER100K": 128,  # 英文文本
    "WEIBO": 512  # 中文文本
}

# 特征提取配置
DESIRED_DIM = 512  # 特征维度
UMAP_COMPONENTS = 100

# 聚类配置
N_CLUSTERS = 10  # Crisis数据集对应7个灾难事件
# MIN_CLUSTERS = 2  # Twitter100k聚类配置
# MAX_CLUSTERS = 15 

DEEP_CLUSTER_CONFIG = {
    "hidden_dims": [512, 256, 128],
    "latent_dim": 64,
    "alpha": 1.0,  # KL散度权重
    "temperature": 0.1,  # InfoNCE loss温度参数
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-3
}

# 知识图谱配置
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
ENTITY_EMBEDDING_DIM = 100 

# 特征保存路径配置
FEATURES_DIR = "./extracted_features"
IMAGE_FEATURES_FILE = os.path.join(FEATURES_DIR, "image_features.npy")
TEXT_FEATURES_FILE = os.path.join(FEATURES_DIR, "text_features.npy")
TAGS_FILE = os.path.join(FEATURES_DIR, "tags.json")
NER_FEATURES_FILE = os.path.join(FEATURES_DIR, "ner_features.json")
TRUE_LABELS_FILE = os.path.join(FEATURES_DIR, "true_labels.npy")

# 用户认证权重配置
AUTH_WEIGHTS = {
    '金V': 1.5,
    '蓝V': 1.5,  # 蓝V与金V同等重要
    '红V': 1.3,
    '黄V': 1.1,
    '普通用户': 1.0
}

# 媒体内容奖励配置
MEDIA_BONUS = {
    'image': 0.5,
    'video': 1.0
}

# 事件过滤配置
MIN_POSTS = 1
STD_MULTIPLIER = 2

# 参数优化配置
NUM_ITERATIONS = 5
INITIAL_SEARCH_RADIUS = 0.2
COARSE_STEP = 0.2
FINE_STEP = 0.03

# 训练配置
TRAIN_RATIO = 0.8

# 文件路径配置
GROUND_TRUTH_FILE = 'popularity.csv'
