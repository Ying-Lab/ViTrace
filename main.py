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
    def __init__(self, config: RuntimeConfig, model_cfg: ModelConfig, model_type: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.model_cfg = model_cfg
        self.model_type = model_type  # Store the model type for loading the correct weights
        self.sigmoid = torch.nn.Sigmoid()
        setup_seed(1001)
        self._initialize_models()

    @staticmethod
    def batch_mean(inputs: torch.Tensor, outputs: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Corrected batch averaging"""
        num_fragments = math.ceil(seq_len / 48)
        return outputs.view(-1, num_fragments).mean(dim=1)

    def _predict_reads(self, test_loader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prediction logic with index correction"""
        self.transformer.eval()
        self.cnn.eval()
        all_scores = []
        all_indices = []
        global_offset = 0  # Global offset counter

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                batch_size, seq_len = batch.shape
                
                # Long sequence processing
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
                    
                    # CNN processing
                    c_inputs = torch.eye(4, device=self.device)[(fragments - 1).long()]
                    c_outputs = self.cnn(c_inputs.float().unsqueeze(1))
                    c_outputs = self.batch_mean(batch, c_outputs, seq_len)
                    
                    # Transformer processing
                    t_outputs = self.transformer(fragments)
                    t_outputs = self.batch_mean(batch, t_outputs, seq_len)
                    
                    outputs = torch.maximum(c_outputs, t_outputs)
                else:
                    # Short sequence processing
                    c_inputs = torch.eye(4, device=self.device)[(batch - 1).long()]
                    c_outputs = self.cnn(c_inputs.float().unsqueeze(1)).squeeze(1)
                    t_outputs = self.transformer(batch).squeeze(1)
                    outputs = torch.maximum(c_outputs, t_outputs)

                # Probability calculation
                probs = self.sigmoid(outputs).flatten()
                
                # Index correction (with global offset)
                batch_indices = torch.where(probs > self.config.threshold)[0]
                global_indices = batch_indices + global_offset
                
                all_scores.append(probs.cpu())
                all_indices.append(global_indices.cpu())
                
                # Update offset
                global_offset += batch_size

        # Combine results
        final_scores = torch.cat(all_scores) if all_scores else torch.tensor([]) 
        final_indices = torch.cat(all_indices) if all_indices else torch.tensor([]) 
        return final_scores, final_indices

    def _initialize_models(self):
        """Model initialization (with dimension validation)"""
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
        
        # Load model weights according to model_type
        model_dir = self.config.model_path
        transformer_path = os.path.join(model_dir, f"best_model_48_{self.model_type}.pkl")
        cnn_path = os.path.join(model_dir, f"best_cnn_48_{self.model_type}.pkl")
        
        if not os.path.exists(transformer_path) or not os.path.exists(cnn_path):
            raise FileNotFoundError("Model weight files are missing")
            
        self.transformer.load_state_dict(torch.load(transformer_path, map_location="cpu"))
        self.cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))

    def process_file(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single FASTA file (with dimension validation)"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        print(f"Processing: {os.path.basename(file_path)}")
        test_x = matrix_from_fasta(file_path)
        
        if test_x.ndim != 2:
            raise ValueError(f"Invalid input dimensions: expected 2D matrix, got {test_x.shape}")
            
        test_tensor = torch.tensor(test_x, device=self.device)
        test_loader = torch.utils.data.DataLoader(
            test_tensor,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        return self._predict_reads(test_loader)

    def run_pipeline(self):
        """Run the complete pipeline (with path validation)"""
        os.makedirs(self.config.output_folder, exist_ok=True)
        
        predictions = []
        for filename in sorted(os.listdir(self.config.input_folder)):
            if filename.endswith(".fasta"):
                file_path = os.path.join(self.config.input_folder, filename)
                try:
                    scores, indices = self.process_file(file_path)
                    if scores.numel() > 0:
                        filter_reads(file_path, indices.numpy(), self.config.output_folder)
                        predictions.append((os.path.splitext(filename)[0], scores.tolist()))
                except Exception as e:
                    print(f"Failed to process {filename}: {str(e)}")
                    continue
        
        # Save results (with empty result check)
        if predictions:
            with open(self.config.output_file, "w") as f:
                for name, scores in predictions:
                    f.write(f"{name}\t{scores}\n")
        else:
            print("Warning: No predictions were generated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViTrace virus sequence analysis tool")
    parser.add_argument("--in_folder", required=True, help="Input folder path")
    parser.add_argument("--out_folder", required=True, help="Output folder path")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--model_type", choices=['human', 'microbe', 'mouse'], default='human', help="Select model type")

    args = parser.parse_args()

    # Initialize configuration (with path validation)
    try:
        runtime_cfg = RuntimeConfig(
            input_folder=os.path.abspath(args.in_folder),
            output_folder=os.path.abspath(args.out_folder),
            threshold=args.threshold,
            batch_size=args.batch_size,
            model_path=os.path.join(os.path.dirname(__file__), "params"),
            output_file=os.path.join(os.path.abspath(args.out_folder), "predictions.txt")
        )
        
        processor = FASTAProcessor(runtime_cfg, ModelConfig(), args.model_type)  # Use user-specified model_type
        processor.run_pipeline()
        print(f"Processing complete! Results saved in: {args.out_folder}")
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
