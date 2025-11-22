import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import sys
from datetime import datetime
import config

# 导入自定义模块
from dataset_new import HateDataset, collate_fn, image_transform, tokenizer
from fusion_res import MultimodalHateSpeechDetector
from dataset_new import ProjectionHead
from dataloader import load_dataset

# 设置随机种子
def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="多模态仇恨言论检测")
    parser.add_argument('--device', type=str, default="cuda", help='指定使用的设备 (例如: cuda:0, cpu)')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=config.num_epochs, help='训练轮次')
    parser.add_argument('--lambda_ic', type=float, default=config.lambda_ic, help='交互约束损失权重')
    parser.add_argument('--learning_rate', type=float, default=config.learning_rate, help='学习率')
    parser.add_argument('--output_dir', type=str, default=config.output_dir, help='输出目录')
    parser.add_argument('--max_len', type=int, default=config.max_len, help='最大序列长度')
    parser.add_argument('--feature_dim', type=int, default=config.feature_dim, help='特征维度')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='类别数量')
    parser.add_argument('--weight_decay', type=float, default=config.weight_decay, help='权重衰减')
    # 新增参数
    parser.add_argument('--max_grad_norm', type=float, default=config.max_grad_norm, help='梯度裁剪的最大范数')
    parser.add_argument('--dropout', type=float, default=config.dropout, help='Dropout比例')
    # 添加参数解析中的数据集参数
    parser.add_argument("--dataset", type=str, default=config.dataset_name,
                        choices=["MultiOFF", "HatefulMemes"])

    return parser.parse_args()


# 训练函数
def train_epoch(model, dataloader, optimizer, device, max_grad_norm):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # 提取特征和标签
        text_fine = batch['text_feature_bert'].to(device)
        text_coarse = batch['text_features_clip'].to(device)
        image_fine = batch['image_features_resnet'].to(device)
        image_coarse = batch['image_features_clip'].to(device)
        labels = batch['label'].to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        logits, loss, loss_dict = model(text_fine, text_coarse, image_fine, image_coarse, labels)

        # 反向传播和优化
        loss.backward()

        # 添加梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # 优化步骤
        optimizer.step()

        # 累加损失
        total_loss += loss.item()

        # 收集预测和标签
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item()})

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')

    return total_loss / len(dataloader), accuracy, f1, precision, loss_dict


def evaluate(model, dataloader, device, desc="Evaluating"):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # 存储预测概率，用于计算AUCROC
    total_loss = 0  # 添加总损失变量

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            # 提取特征和标签
            text_fine = batch['text_feature_bert'].to(device)
            text_coarse = batch['text_features_clip'].to(device)
            image_fine = batch['image_features_resnet'].to(device)
            image_coarse = batch['image_features_clip'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            logits, loss, _ = model(text_fine, text_coarse, image_fine, image_coarse, labels)
            probs = torch.softmax(logits, dim=1)  # 获取预测概率
            
            # 累积损失
            if loss is not None:
                total_loss += loss.item()

            # 收集预测和标签
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            batch_probs = probs[:, 1].cpu().tolist()  # 假设正类是索引1
            all_probs.extend(batch_probs)  # 累加概率


    # 计算平均损失
    val_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    try:
        # 检查标签是否只有一个类别
        unique_labels = np.unique(all_labels)
        if len(unique_labels) < 2:
            print(f"警告：数据集中只有一个类别 {unique_labels}，无法计算AUC-ROC")
            auc_roc = 0.0
        elif len(all_labels) == 0 or len(all_probs) == 0:
            print("警告：标签或概率数组为空，无法计算AUC-ROC")
            auc_roc = 0.0
        else:
            auc_roc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        print(f"计算AUC-ROC时出错: {e}")
        auc_roc = 0.0

    # 返回时增加all_labels与all_probs
    return accuracy, f1, precision, recall, auc_roc, cm, val_loss, all_labels, all_probs


# 添加测试函数
def test_model(model, dataloader, device, output_dir=None):
    """
    在测试集上评估模型并保存结果

    参数:
        model: 已加载最佳模型的模型实例
        dataloader: 测试数据加载器
        device: 计算设备
        output_dir: 输出目录，如果提供则保存结果
    """
    print("\n开始在测试集上进行评估...")

    accuracy, f1, precision, recall, auc_roc, cm, _, all_labels, all_probs = evaluate(model, dataloader, device, desc="Testing")

    # 打印结果
    print("\n测试集评估结果:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"F1分数 (F1-Score): {f1:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")


    # 保存结果到文件
    if output_dir:
        results = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc_roc': auc_roc,
        }

        # 保存结果为JSON
        import json
        with open(f"{output_dir}/test_results.json", 'w') as f:
            json.dump(results, f, indent=4)


        print(f"测试结果已保存至 {output_dir}/test_results.json")

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'all_labels': all_labels,   # 新增
        'all_probs': all_probs      # 新增
    }


# 可视化训练过程
def plot_metrics(metrics, save_path):
    epochs = range(1, len(metrics['train_acc']) + 1)

    plt.figure(figsize=(20, 10))

    # 绘制损失 - 添加验证损失
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')  # 添加验证损失
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, metrics['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 绘制F1分数
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics['train_f1'], 'b-', label='Training F1')
    plt.plot(epochs, metrics['val_f1'], 'r-', label='Validation F1')
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    # 绘制精确率和召回率
    plt.subplot(2, 2, 4)
    plt.plot(epochs, metrics['val_precision'], 'r-', label='Validation Precision')
    plt.plot(epochs, metrics['val_recall'], 'm-', label='Validation Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"训练指标图已保存至 {save_path}")


# 添加绘制ROC曲线的函数
def plot_roc_curve(fpr, tpr, auc_roc, save_path, title="Receiver Operating Characteristic (ROC) Curve"):
    """
    绘制ROC曲线

    参数:
        fpr: 假阳性率 (False Positive Rate)
        tpr: 真阳性率 (True Positive Rate)
        auc_roc: 曲线下面积 (Area Under Curve - ROC)
        save_path: 保存图像的路径
        title: 图表标题
    """
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)


    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC曲线已保存至 {save_path}")

def save_experiment_info(output_dir, metrics):
    """
    保存实验设置和结果，直接从config模块导入超参数
    
    参数:
        output_dir: 输出目录
        metrics: 训练过程中记录的指标
        test_results: 测试结果
        train_df: 训练数据集DataFrame
        val_df: 验证数据集DataFrame
        test_df: 测试数据集DataFrame
    """
    
    # 收集实验信息
    experiment_info = {
        # 实验元数据
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version,
        
        # 直接从config读取超参数
        "hyperparameters": {
            "seed": config.seed,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "lambda_ic": config.lambda_ic,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "max_grad_norm": config.max_grad_norm,
        }
        
    }
    
    # 保存为JSON文件
    with open(f"{output_dir}/experiment_info.json", 'w') as f:
        json.dump(experiment_info, f, indent=4)
    
    print(f"实验信息已保存至 {output_dir}/experiment_info.json")


def main():
    args = parse_args()

    # 首先根据数据集名称创建子目录
    dataset_output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(dataset_output_dir, exist_ok=True)

    # 然后在数据集子目录下根据时间戳创建目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(dataset_output_dir, timestamp)
    os.makedirs(args.output_dir, exist_ok=True)
    
    set_seed()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 获取当前数据集配置和标签映射
    current_dataset_config = config.dataset_configs[args.dataset]
    current_label2id = current_dataset_config["label2id"]
    current_id2label = {v: k for k, v in current_label2id.items()}
    
    # 加载数据集
    train_loader, val_loader, test_loader = load_dataset(
        dataset_name=args.dataset,
        config=current_dataset_config,
        args=args
    )

    # 打印数据集大小
    print(f"训练集大小: {len(train_loader.dataset)}, 验证集大小: {len(val_loader.dataset)}, 测试集大小: {len(test_loader.dataset)}")

    # 初始化模型
    print("初始化模型...")
    model = MultimodalHateSpeechDetector(
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        lambda_ic=args.lambda_ic,
        dropout=args.dropout
    ).to(device)

    # 定义优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # 训练循环
    print("开始训练...")
    best_val_f1 = 0.0
    metrics = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_precision': [],
        'val_acc': [], 'val_f1': [], 'val_precision': [], 'val_recall': [], 'val_loss': []
    }

    # 记录开始时间
    start_time = time.time()

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # 训练
        train_loss, train_acc, train_f1, train_precision, loss_dict = train_epoch(
            model, train_loader, optimizer, device, args.max_grad_norm
        )

        # 评估
        val_acc, val_f1, val_precision, val_recall, val_auc_roc, val_cm, val_loss, _, _ = evaluate(
            model, val_loader, device, desc="Validating"
        )

        # 记录指标
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['train_f1'].append(train_f1)
        metrics['train_precision'].append(train_precision)
        metrics['val_loss'].append(val_loss)  # 同时记录训练和验证损失
        metrics['val_acc'].append(val_acc)
        metrics['val_f1'].append(val_f1)
        metrics['val_precision'].append(val_precision)
        metrics['val_recall'].append(val_recall)

        # 调整学习率
        scheduler.step(val_f1)

        # 打印当前训练结果
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(
            f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        print("损失明细:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f"{args.output_dir}/best_val_model.pth")
            print(f"保存验证集上的最佳模型，验证准确率: {val_f1:.4f}")

            # 保存混淆矩阵
            np.save(f"{args.output_dir}/best_val_model_cm.npy", val_cm)


    # 加载最佳模型用于最终评估
    best_model_path = f"{args.output_dir}/best_val_model.pth"
    model.load_state_dict(torch.load(best_model_path))
    print(f"已加载验证集上表现最佳的模型: {best_model_path}")

    # 在测试集上进行最终评估
    # 执行测试
    test_results = test_model(model, test_loader, device, args.output_dir)

    # 记录总训练时间
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n训练完成！总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")
    print(f"模型保存在: {args.output_dir}")

    # 绘制指标和ROC曲线
    print("生成训练过程可视化...")
    plot_metrics(metrics, f"{args.output_dir}/training_metrics.png")

    # 计算ROC曲线数据
    fpr, tpr, _ = roc_curve(test_results["all_labels"], test_results["all_probs"])
    plot_roc_curve(fpr, tpr, test_results['auc_roc'], f"{args.output_dir}/roc_curve.png")

    # 保存实验信息（超参数和结果）
    save_experiment_info(
        output_dir=args.output_dir,
        metrics=metrics
    )

    # 添加测试结果打印
    print(f"测试集F1分数: {test_results['f1']:.4f}")
    print(f"测试集准确率: {test_results['accuracy']:.4f}")
    print(f"测试集AUC-ROC: {test_results['auc_roc']:.4f}")
    print(f"测试集召回率: {test_results['recall']:.4f}")
    print(f"测试集精确率: {test_results['precision']:.4f}")


if __name__ == "__main__":
    main()

#python hsd.py --device cuda:1 --batch_size 64 --num_epochs 80 --lambda_ic 0.1 --learning_rate 0.001

