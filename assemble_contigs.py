#!/usr/bin/env python3
import os
import argparse
import ast
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# ===== 全局变量 =====
GLOBAL_READS_SEQ = None
GLOBAL_READS_SCORE = None
GLOBAL_KMER_INDEX = None
K_GLOBAL = None
EXT_THRESHOLD_GLOBAL = None

# -----------------------------
# 读取FASTA和分数
# -----------------------------
def read_fasta_and_scores_numpy(fasta_path, scores_path):
    # 读取分数
    with open(scores_path) as f:
        line = f.readline().strip().split("\t")
        try:
            scores = ast.literal_eval(line[1])  # 安全解析
        except Exception:
            raise ValueError(f"Scores file format error: {scores_path}")

    # 读取FASTA
    seqs = []
    with open(fasta_path) as f:
        seq_id, seq = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None:
                    seqs.append("".join(seq))
                seq_id = line[1:]
                seq = []
            else:
                seq.append(line)
        if seq_id is not None:
            seqs.append("".join(seq))

    if len(seqs) != len(scores):
        raise ValueError("FASTA and scores length mismatch!")

    seqs_np = np.array(seqs, dtype=object)
    scores_np = np.array(scores, dtype=np.float32)
    return seqs_np, scores_np

# -----------------------------
# 构建 k-mer 索引
# -----------------------------
def build_kmer_index_numpy(seqs_np, k):
    kmer_index = defaultdict(list)
    for idx, seq in enumerate(seqs_np):
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmer_index[kmer].append((idx, i))
    return kmer_index

# -----------------------------
# 初始化 worker
# -----------------------------
def init_worker(seqs_np, scores_np, kmer_index, k, ext_threshold):
    global GLOBAL_READS_SEQ, GLOBAL_READS_SCORE, GLOBAL_KMER_INDEX, K_GLOBAL, EXT_THRESHOLD_GLOBAL
    GLOBAL_READS_SEQ = seqs_np
    GLOBAL_READS_SCORE = scores_np
    GLOBAL_KMER_INDEX = kmer_index
    K_GLOBAL = k
    EXT_THRESHOLD_GLOBAL = ext_threshold

# -----------------------------
# 组装单个 contig
# -----------------------------
def assemble_single_contig_numpy(seed_idx):
    seqs_np = GLOBAL_READS_SEQ
    scores_np = GLOBAL_READS_SCORE
    kmer_index = GLOBAL_KMER_INDEX
    k = K_GLOBAL
    ext_threshold = EXT_THRESHOLD_GLOBAL

    seed_seq = seqs_np[seed_idx]
    seed_score = scores_np[seed_idx]

    used_reads = set([seed_idx])
    contig = seed_seq
    scores = [seed_score]

    def extend(direction):
        nonlocal contig, scores
        while True:
            kmer = contig[:k] if direction == "left" else contig[-k:]
            candidates = []
            for idx, pos in kmer_index.get(kmer, []):
                if idx in used_reads:
                    continue
                seq = seqs_np[idx]
                score = scores_np[idx]
                if direction == "left" and pos == len(seq) - k:
                    extension = seq[:-k]
                    if extension:
                        candidates.append((idx, extension, score))
                elif direction == "right" and pos == 0:
                    extension = seq[k:]
                    if extension:
                        candidates.append((idx, extension, score))
            if not candidates:
                break
            # 按分数和延伸长度排序
            idx, extension, score = max(candidates, key=lambda x: (x[2], len(x[1])))
            new_scores = scores + [score]
            if np.mean(new_scores) < ext_threshold:
                break
            contig = extension + contig if direction == "left" else contig + extension
            scores = new_scores
            used_reads.add(idx)

    extend("left")
    extend("right")
    return (contig, np.mean(scores))

# -----------------------------
# 主函数
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Assemble contigs from reads using k-mer overlap")
    parser.add_argument("--fasta", required=True, help="Path to input FASTA file")
    parser.add_argument("--scores", required=True, help="Path to prediction scores file")
    parser.add_argument("-k", type=int, default=24, help="k-mer length (default: 24)")
    parser.add_argument("--ext-threshold", type=float, default=0.5, help="Extension score threshold (default: 0.5)")
    parser.add_argument("--seed-threshold", type=float, default=0.7, help="Seed read score threshold (default: 0.7)")
    parser.add_argument("--max-proc", type=int, default=cpu_count(), help="Max number of processes (default: all CPUs)")
    parser.add_argument("-o", "--output", default="assembled_contigs.fasta", help="Output FASTA file path")
    parser.add_argument("--min-len", type=int, default=90, help="Minimum contig length to output")
    args = parser.parse_args()

    # 读取数据
    seqs_np, scores_np = read_fasta_and_scores_numpy(args.fasta, args.scores)
    print(f"[INFO] Total reads: {len(seqs_np)}")

    # 筛选种子
    seed_indices = np.where(scores_np > args.seed_threshold)[0]
    print(f"[INFO] Seed reads: {len(seed_indices)}")
    if len(seed_indices) == 0:
        print("[ERROR] No seed reads found. Exiting.")
        return

    # 构建 k-mer 索引
    print("[INFO] Building k-mer index...")
    kmer_index = build_kmer_index_numpy(seqs_np, args.k)
    print(f"[INFO] k-mer index built: {len(kmer_index)} unique k-mers.")

    # 多进程组装
    n_processes = min(cpu_count(), args.max_proc)
    with Pool(processes=n_processes,
              initializer=init_worker,
              initargs=(seqs_np, scores_np, kmer_index, args.k, args.ext_threshold)) as pool:
        results = list(tqdm(pool.imap(assemble_single_contig_numpy, seed_indices),
                            total=len(seed_indices),
                            desc="Assembling Contigs",
                            unit="contig"))

    # 保存结果
    kept = 0
    with open(args.output, "w") as f:
        for i, (contig, avg_score) in enumerate(results):
            if len(contig) >= args.min_len:
                f.write(f">contig_{i}_len{len(contig)}_avgscore{avg_score:.3f}\n{contig}\n")
                kept += 1
    print(f"[SUCCESS] Assembly completed: {kept} contigs saved to {args.output}")

if __name__ == "__main__":
    main()
