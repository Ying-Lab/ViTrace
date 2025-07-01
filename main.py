import os
import math
import argparse
import torch
from model2 import Transformer, VirusCNN
from init import matrix_from_fasta, setup_seed, filter_reads
from config import ModelConfig, RuntimeConfig
from typing import Tuple
import numpy as np

class FASTAProcessor:
    def __init__(self, config: RuntimeConfig, model_cfg: ModelConfig):
        self.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.model_cfg = model_cfg
        self.sigmoid = torch.nn.Sigmoid()
        setup_seed(1001)
        self._initialize_models()

    @staticmethod
    def batch_mean(inputs: torch.Tensor, outputs: torch.Tensor, seq_len: int) -> torch.Tensor:
        """修正的批次平均计算"""
        
        num_fragments = math.ceil(seq_len / 48)
        return outputs.view(-1, num_fragments).mean(dim=1)

    def _predict_reads(self, test_loader) -> Tuple[torch.Tensor, torch.Tensor]:
        """修复索引偏移的预测逻辑"""
        self.transformer.eval()
        self.cnn.eval()
        all_scores = []
        all_indices = []
        global_offset = 0  # 新增全局偏移计数器

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                batch_size, seq_len = batch.shape
                
                # 长序列处理逻辑
                if seq_len > 48:
                    fragments = []
                    for seq in batch:
                        total_len = seq.size(0)
                        starts = list(range(0, total_len, 48))
                        for start in starts:
                            end = start + 48
                            fragment = seq[-48:] if end > total_len else seq[start:end]
                            fragments.append(fragment)
                    
                    fragments = torch.stack(fragments).to(self.device)
                    
                    # CNN处理
                    c_inputs = torch.eye(4, device=self.device)[(fragments - 1).long()]
                    c_outputs = self.cnn(c_inputs.float().unsqueeze(1))
                    c_outputs = self.batch_mean(batch, c_outputs, seq_len)
                    
                    # Transformer处理
                    t_outputs = self.transformer(fragments)
                    t_outputs = self.batch_mean(batch, t_outputs, seq_len)
                    
                    outputs = torch.maximum(c_outputs, t_outputs)
                else:
                    # 短序列处理
                    c_inputs = torch.eye(4, device=self.device)[(batch - 1).long()]
                    c_outputs = self.cnn(c_inputs.float().unsqueeze(1)).squeeze(1)
                    t_outputs = self.transformer(batch).squeeze(1)
                    outputs = torch.maximum(c_outputs, t_outputs)

                # 概率计算
                probs = self.sigmoid(outputs).flatten()
                
                # 索引修正（添加全局偏移）
                batch_indices = torch.where(probs > self.config.threshold)[0]
                global_indices = batch_indices + global_offset
                
                all_scores.append(probs.cpu())
                all_indices.append(global_indices.cpu())
                
                # 更新偏移量
                global_offset += batch_size  # 关键修正点

        # 合并结果
        final_scores = torch.cat(all_scores) if all_scores else torch.tensor([])
        final_indices = torch.cat(all_indices) if all_indices else torch.tensor([])
        return final_scores, final_indices

    def _initialize_models(self):
        """模型初始化（添加维度验证）"""
        self.transformer = Transformer(
            src_vocab_size=5,
            src_pad_idx=0,
            device=self.device,
            max_length=48,
            out_size=self.model_cfg.out_size,
            embed_size=self.model_cfg.embed_size,
            num_layers=self.model_cfg.num_layers,
            forward_expansion=self.model_cfg.forward_expansion,
            heads=self.model_cfg.heads,
            dropout=self.model_cfg.dropout
        ).to(self.device)
        
        self.cnn = VirusCNN().to(self.device)
        
        # 加载权重（添加文件存在性检查）
        model_dir = self.config.model_path
        transformer_path = os.path.join(model_dir, "best_model_48_o.pkl")
        cnn_path = os.path.join(model_dir, "best_cnn_48_o.pkl")
        # transformer_path = os.path.join(model_dir, "best_model_48_mouse.pkl")
        # cnn_path = os.path.join(model_dir, "best_cnn_48_mouse.pkl")
        
        if not os.path.exists(transformer_path) or not os.path.exists(cnn_path):
            raise FileNotFoundError("模型权重文件缺失")
            
        self.transformer.load_state_dict(torch.load(transformer_path, map_location="cpu"))
        self.cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))

    def process_file(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """带维度验证的文件处理"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        print(f"正在处理: {os.path.basename(file_path)}")
        test_x = matrix_from_fasta(file_path)
        
        if test_x.ndim != 2:
            raise ValueError(f"输入维度错误: 应为二维矩阵，实际为 {test_x.shape}")
            
        test_tensor = torch.tensor(test_x, device=self.device)
        test_loader = torch.utils.data.DataLoader(
            test_tensor,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        return self._predict_reads(test_loader)

    def run_pipeline(self):
        """完整执行流程（添加路径验证）"""
        os.makedirs(self.config.output_folder, exist_ok=True)
        
        predictions = []
        for filename in sorted(os.listdir(self.config.input_folder)):
            if filename.endswith(".fasta"):
                file_path = os.path.join(self.config.input_folder, filename)
                try:
                    scores, indices = self.process_file(file_path)
                    if scores.numel() > 0:
                        filter_reads(file_path, indices.numpy(), self.config.output_folder)
                        predictions.append((
                            os.path.splitext(filename)[0], 
                            scores.tolist()
                        ))
                except Exception as e:
                    print(f"处理 {filename} 失败: {str(e)}")
                    continue
        
        # 保存结果（添加空结果保护）
        if predictions:
            with open(self.config.output_file, "w") as f:
                for name, scores in predictions:
                    f.write(f"{name}\t{scores}\n")
        else:
            print("警告: 未生成任何预测结果")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViTrace病毒序列分析工具")
    parser.add_argument("--in_folder", required=True, help="输入目录路径")
    parser.add_argument("--out_folder", required=True, help="输出目录路径")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=1024)
    
    args = parser.parse_args()
    
    # 配置类初始化（添加路径验证）
    try:
        runtime_cfg = RuntimeConfig(
            input_folder=os.path.abspath(args.in_folder),
            output_folder=os.path.abspath(args.out_folder),
            threshold=args.threshold,
            batch_size=args.batch_size,
            model_path=os.path.join(os.path.dirname(__file__), "params"),
            output_file=os.path.join(os.path.abspath(args.out_folder), "predictions.txt")
        )
        
        processor = FASTAProcessor(runtime_cfg, ModelConfig())
        processor.run_pipeline()
        print(f"处理完成！结果保存在: {args.out_folder}")
    except Exception as e:
        print(f"初始化失败: {str(e)}")