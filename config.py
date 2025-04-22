# ------------------------------
# 模块1：配置管理（config.py）
# ------------------------------
from dataclasses import dataclass

@dataclass
class ModelConfig:
    num_layers: int = 3
    embed_size: int = 512
    forward_expansion: int = 2
    heads: int = 8
    dropout: float = 0.1
    out_size: int = 256

@dataclass
class RuntimeConfig:
    batch_size: int = 1024
    threshold: float = 0.6
    input_folder: str = ""
    output_folder: str = ""
    model_path: str = "/workspace/data/ZF/VITrace/github/ViTrace/gihub_upload/params"
    output_file: str = "predictions.txt"  