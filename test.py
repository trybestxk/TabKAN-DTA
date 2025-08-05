# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import dgl
from tqdm import tqdm
import pandas as pd

import Utils
from Loder import MultiModalDataLoader
from HGCN import HGCNDTA
from model import TabKANDTA


def evaluate_model(model, device, loader, loss_fn, dataset_name):
    """评估模型在指定数据集上的性能"""
    print(f"Evaluating on {dataset_name} dataset ({len(loader.graph_dataset)} samples)...")
    model.eval()
    total_loss = 0
    outputs = []
    targets = []

    with torch.no_grad():
        for batch_data in tqdm(loader, desc=f"Testing {dataset_name}"):
            # 解包数据
            g, protein_embeds, selfies_embeds, tabpfn_feats, labels = batch_data

            # 将图移动到设备
            g = g.to(device)

            # 准备节点特征
            node_feats = {
                'ligand_atom': g.nodes['ligand_atom'].data['feat'].to(device),
                'residue': g.nodes['residue'].data['feat'].to(device)
            }

            # 转换标签和特征到设备
            labels = labels.to(device).unsqueeze(1)
            protein_embeds = protein_embeds.to(device)
            selfies_embeds = selfies_embeds.to(device)
            tabpfn_feats = tabpfn_feats.to(device)

            # 前向传播
            out = model(g, node_feats, protein_embeds, selfies_embeds, tabpfn_feats)

            # 计算损失
            loss_value = loss_fn(out, labels)
            total_loss += loss_value.item()

            # 收集输出和目标
            outputs.append(out.cpu().numpy().reshape(-1))
            targets.append(labels.cpu().numpy().reshape(-1))

    # 合并所有批次的输出和目标
    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
    avg_loss = total_loss / len(loader)

    # 计算评估指标
    metrics = {
        'loss': avg_loss,
        'c_index': Utils.c_index(targets, outputs),
        'RMSE': Utils.RMSE(targets, outputs),
        'MAE': Utils.MAE(targets, outputs),
        'SD': Utils.SD(targets, outputs),
        'CORR': Utils.CORR(targets, outputs),
    }

    # 打印详细评估指标
    print(f"\n===== {dataset_name} 评估结果 =====")
    print(f"样本数: {len(targets)}")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"c-index: {metrics['c_index']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"SD: {metrics['SD']:.4f}")
    print(f"CORR: {metrics['CORR']:.4f}")
    print("=" * 40)

    return metrics, outputs, targets


def load_model_for_dataset(dataset_name, graph_model, device):
    """为指定数据集加载对应的模型权重"""
    # 从数据集名称中提取基本名称（去除.bin后缀）
    dataset_base_name = dataset_name.replace('.bin', '')

    # 构建模型权重路径
    model_weights_path = f'./best_result/{dataset_base_name}.bin/best_model.pt'

    # 检查权重文件是否存在
    if not os.path.exists(model_weights_path):
        print(f"警告: 模型权重文件不存在: {model_weights_path}")
        # 尝试使用corr版本的权重
        model_weights_path = f'./best_result/{dataset_base_name}.bin/best_model_corr.pt'
        if not os.path.exists(model_weights_path):
            print(f"警告: corr版本的模型权重文件也不存在: {model_weights_path}")
            return None

    # 初始化多模态模型
    model = TabKANDTA(
        tabpfn_dim=192,
        smiles_dim=384,
        esm_dim=1152,
        graph_model=graph_model,
        graph_feat_dim=432,
        fusion_dim=512,
        dropout=0.1
    )

    # 加载预训练的多模态模型权重
    print(f"加载模型权重: {model_weights_path}")
    checkpoint = torch.load(model_weights_path, map_location=device)

    # 如果是完整的checkpoint，需要提取模型状态字典
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"从checkpoint加载模型权重，来自epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # 如果直接是模型状态字典
        model.load_state_dict(checkpoint)
        print("直接加载模型状态字典")

    model.to(device)
    model.eval()
    return model


def main():
    """主函数，加载预训练模型并在所有数据集上进行评估"""
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据路径
    graph_data_path = './DataSet03/processed/'
    avg_embed_base = './avg_embeddings/'
    tabpfn_features_base = './tabpfn_learned_features/'

    # 批次大小
    BATCH_SIZE = 64

    # 创建结果目录
    results_dir = './model_evaluation_results'
    os.makedirs(results_dir, exist_ok=True)

    # 加载图数据集
    datasets = ['test2013.bin', 'test2016.bin', 'bind2020.bin', 'csar.bin']
    graph_datasets = {}
    for dataset_name in datasets:
        dataset_path = graph_data_path + dataset_name
        if os.path.exists(dataset_path):
            graph_datasets[dataset_name] = Utils.LoadedHeteroDataSet(dataset_path)
            print(f"加载图数据集: {dataset_name}, 样本数: {len(graph_datasets[dataset_name])}")
        else:
            print(f"警告: 图数据文件不存在: {dataset_path}")

    # 加载各个模态的特征
    multimodal_loaders = {}
    for dataset_name in datasets:
        if dataset_name not in graph_datasets:
            continue

        # 构建各个模态的路径
        dataset_prefix = dataset_name.replace('.bin', '')
        protein_embed_path = os.path.join(avg_embed_base, f"{dataset_prefix}_protein_avg_embeddings.pt")
        smiles_embed_path = os.path.join(avg_embed_base, f"{dataset_prefix}_smiles_avg_embeddings.pt")
        tabpfn_features_path = os.path.join(tabpfn_features_base, f"{dataset_prefix}.npy")

        # 检查文件是否存在
        if not os.path.exists(protein_embed_path):
            print(f"警告: 蛋白质平均嵌入文件不存在: {protein_embed_path}")
            continue

        if not os.path.exists(smiles_embed_path):
            print(f"警告: SMILES平均嵌入文件不存在: {smiles_embed_path}")
            continue

        # 加载TabPFN特征
        if not os.path.exists(tabpfn_features_path):
            print(f"警告: TabPFN特征文件不存在: {tabpfn_features_path}")
            tabpfn_features = np.zeros((len(graph_datasets[dataset_name]), 192))
            print(f"创建零填充TabPFN特征: {tabpfn_features.shape}")
        else:
            # 加载TabPFN特征 (npy格式)
            tabpfn_features = np.load(tabpfn_features_path)
            print(f"TabPFN特征已加载，形状: {tabpfn_features.shape}")

            # 检查特征维度，确保和训练时一致
            if tabpfn_features.shape[1] != 192:
                print(f"警告: TabPFN特征维度 ({tabpfn_features.shape[1]}) 与预期 (192) 不符")
                # 如果维度不对，可以进行填充或截断
                if tabpfn_features.shape[1] > 192:
                    tabpfn_features = tabpfn_features[:, :192]
                    print(f"已截断TabPFN特征到192维")
                else:
                    pad_width = ((0, 0), (0, 192 - tabpfn_features.shape[1]))
                    tabpfn_features = np.pad(tabpfn_features, pad_width, mode='constant')
                    print(f"已填充TabPFN特征到192维")

        # 创建多模态数据加载器
        multimodal_loaders[dataset_name] = MultiModalDataLoader(
            graph_dataset=graph_datasets[dataset_name],
            protein_embed_path=protein_embed_path,
            smiles_embed_path=smiles_embed_path,
            tabpfn_features=tabpfn_features,
            batch_size=BATCH_SIZE,
            shuffle=False  # 评估时不需要洗牌
        )
        print(f"创建数据加载器: {dataset_name}")

    # 加载预训练的图模型
    graph_model = HGCNDTA()
    graph_model.load_state_dict(torch.load('./HGCN_PTH/best_model.pt'))
    print("预训练图模型已加载")

    # 损失函数
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # 存储所有数据集的评估结果
    all_results = {}

    # 在所有数据集上评估模型
    for dataset_name in datasets:
        if dataset_name in multimodal_loaders:
            print(f"\n开始评估数据集: {dataset_name}")

            # 为当前数据集加载对应的模型权重
            model = load_model_for_dataset(dataset_name, graph_model, device)

            if model is None:
                print(f"跳过数据集 {dataset_name} 的评估，因为无法加载对应的模型权重")
                continue

            # 评估模型
            metrics, predictions, targets = evaluate_model(
                model=model,
                device=device,
                loader=multimodal_loaders[dataset_name],
                loss_fn=loss_fn,
                dataset_name=dataset_name
            )

            # 保存评估结果
            all_results[dataset_name] = metrics

            # 保存预测结果到CSV
            results_df = pd.DataFrame({
                'true': targets,
                'pred': predictions
            })
            results_df.to_csv(f'{results_dir}/{dataset_name}_predictions.csv', index=False)

            # 保存评估指标到文本文件
            with open(f'{results_dir}/{dataset_name}_metrics.txt', 'w') as f:
                f.write(f"数据集: {dataset_name}\n")
                f.write(f"样本数: {len(targets)}\n")
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.6f}\n")

    # 创建汇总结果表格
    summary_data = []
    for dataset_name, metrics in all_results.items():
        row = {'dataset': dataset_name}
        row.update({k: f"{v:.4f}" for k, v in metrics.items()})
        summary_data.append(row)

    # 保存汇总结果
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{results_dir}/all_datasets_summary.csv', index=False)

    # 打印汇总结果
    print("\n===== 所有数据集评估结果汇总 =====")
    print(summary_df.to_string())

    print(f"\n评估完成! 结果已保存到 {results_dir} 目录")


if __name__ == "__main__":
    # 设置 pandas 显示选项，使输出更易读
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    main()
