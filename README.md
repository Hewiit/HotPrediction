## 项目说明：微博事件热度预测

本项目实现的是基于多粒度事件特征的媒体事件热度评估算法，用于对微博热点事件进行热度预测与排序，综合考虑(hot_calculator.py)：

- **互动特征**：转发数、评论数、点赞数、用户认证类型  
- **文本特征**：情感强度、主题相关性、实体丰富度、信息量  

训练神经网络模型动态优化计算各热度因素权重（transformer_optimizer.py），加权计算事件热度分数

输入：事件级帖子，事件热度GT文件（训练需要）

输出：事件热度值（单事件预测）及热度影响因素权重，事件热度排行（事件热度批量预测）

---

## 环境与依赖

建议使用 Python 3.9+，并在虚拟环境中安装依赖。

```bash
pip install -r requirement.txt
```

> 说明：  
> - 本项目使用了语言模型`bert-base-chinese` ，会通过 `transformers` 自动下载，请确保可以访问 HuggingFace Hub（或提前离线缓存放在同级目录中）。  
> - 如需离线下载 `bert-base-chinese`，可使用此链接：[离线模型下载](https://drive.google.com/drive/folders/1Cfhq4qWLnLyKAHzYzt4Cs7zHOfK2DF2H?usp=drive_link)。

---

## 数据组织结构与格式

### 1. 真实热度标注文件 `popularity.csv`

放在项目根目录（与 `hot_prediction.py` 同级）：

```python
GROUND_TRUTH_FILE = 'popularity.csv'
```

必需字段：

- **`关键词`**：事件名称（将用作事件目录匹配与 `event_name`）
- **`热度`**：原始热度值（任意正数）

使用 `normalize_popularity.py` 会在该文件中新增一列：

- **`标准化热度`**：将 `热度` 归一化到 0–100 的区间，训练/评估时使用这一列。

---

### 2. 事件微博数据目录（批量训练/推理）

在 `data_processor.py` 中：

```python
HOT_EVENTS_BASE_DIR = "./event_新浪微博"
```

目录结构示例：

```text
project_root/
  hot_prediction.py
  data_processor.py
  ...
  popularity.csv
  event_新浪微博/
    01_事件A/
      事件A.csv
    02 事件B/
      事件B数据.csv
    03事件C/
      xxx.csv
```

**事件 CSV 文件必需字段（主要用于训练/推理）：**

- `转发数`
- `评论数`
- `点赞数`
- `认证类型`（如：`金V`、`蓝V` 等，缺失时默认为 `普通用户`）
- 文本字段：  
  - `全文内容`  
  - `原微博内容`  

---

## 训练流程

### 1. 准备与标准化真实热度

1. 在项目根目录创建/整理热度GroundTruth文件 `popularity.csv`，至少包含：
   - `关键词`
   - `热度`

现有的训练数据是初步采集的500+微博事件，[下载链接](https://bhpan.buaa.edu.cn/link/AA51654EE0CB5E442A86871BD66029C7A9)

对应的热度GroundTruth文件是本仓库中的normalized_hot_results.csv

2. 运行标准化脚本：

```bash
python normalize_popularity.py
```

执行后：

- 会在 `popularity.csv` 中添加 `标准化热度` 列，范围 0–100。

---

### 2. 准备事件微博数据

1. 在`data_processor.py`中设置事件目录参数
```python
HOT_EVENTS_BASE_DIR = "./event_新浪微博"
```
2. 每个子目录至少包含一个事件 CSV：  
   - 如：`武汉疫情.csv`、`北京暴雨数据.csv`
3. 确保 CSV 中有以下字段：
   - `转发数`、`评论数`、`点赞数`、`认证类型`、`全文内容`、`原微博内容`
4. config.py设置ground_truth文件路径：
```python
GROUND_TRUTH_FILE = 'popularity.csv'
```
---

### 3. 运行训练

项目主入口是 `hot_prediction.py`

在命令行中执行：

```bash
python hot_prediction.py
```
训练完成后会在根目录保存训练好的模型参数./models/best_model.pt，也可使用[模型参数文件](https://drive.google.com/drive/folders/1HXKNJHYnvCoE1EnAFfnTh2cXmkxEQcqF?usp=drive_link)直接下载

存在./models/best_model.pt之后，再执行：
```bash
python hot_prediction.py
```
会直接加载模型参数文件进行热度的评估，不再进行训练步骤。

---

## 加载模型（batch 与 single）结果

在 `hot_prediction.py` 的 `main()` 中，通过设置 `run_mode` 选择两种模式：

- `run_mode = "batch"`：**批量模式**（默认推荐）  
- `run_mode = "single"`：**单事件示例模式**

### 1. 批量模式（batch）

`run_mode = "batch"` 时：

- 调用 `init_predictor()`：  
  - 若存在 `models/best_model.pt`：直接加载  
  - 否则：先进行训练并保存该文件
- 调用 `FixedHotPredictor.predict_hot_scores()`，对 `popularity.csv` 中所有事件进行预测和排序。

主要输出文件（保存在 `./results/` 目录）：

- `event_rankings.csv`：  
  - `事件`：事件名称  
  - `预测分数`：归一化后的热度分数（0–100）  
  - `预测排名`：按预测分数从高到低的排名  
- `model_metrics.json`：  
  - `event_scores`：每个事件的原始热度得分  
  - `event_parameters`：每个事件对应的一组权重（互动权重、文本特征权重、整体权重，已在各自组内归一化）  
  - `metrics`：整体相关性指标（Spearman、Pearson 等）

### 2. 单事件模式（single）

将 `run_mode` 改为：

```python
run_mode = "single"
```

此时 `main()` 中会走示例代码路径：

- 在代码中指定：  
  - `event_name`：示例事件名称  
  - `csv_file_path`：该事件的 CSV 路径  
- 调用 `predict_single_event(...)`，只对这个单一事件进行热度预测。

主要输出：

- 返回值：`predict_single_event` 返回**结果 JSON 文件路径**  
- JSON 文件内容（保存在 `./results/某事件_prediction_result.json`）：  
  - `event_info`：事件名称、CSV 路径、预测时间等  
  - `hot_score`：  
    - `raw_score`：原始热度（所有微博总体得分之和）  
  - `event_statistics`：总微博数、总转发/评论/点赞数等基础统计  
  - `user_authentication_distribution`：各认证类型的微博数量分布  
  - `model_parameters`：该事件对应的一组权重（互动权重、文本特征权重、整体权重，已在各自组内归一化）  

- **batch 模式**：适合一次性评估所有事件，得到全局排序与整体指标。  
- **single 模式**：适合调试或分析某一个事件的详细热度构成与参数。
