# ViTrace


![ViTrace](/assets/ViTrace.jpg)


## Intro
ViTrace is a state-of-the-art, dual-channel deep neural network designed to identify viral sequences within human transcriptomic data. It employs an integrated approach by combining Transformer and Convolutional Neural Network (CNN) architectures, maximizing their strengths to uncover viral signatures in human tumor sequencing data effectively. This tool is crucial for researchers focusing on the viral etiology of cancers.

## File Structure
```
ViTrace
├─ README.md
├─ assets
│  └─ ViTrace.jpg
├─ config.py
├─ demo
│  ├─ demo_assemble
│  │  ├─ AAACGGGAGAGCAATT-1.fasta
│  ├─ human_demo
│  │  ├─ AAACGGGAGAGCAATT-1.fasta
│  │  ├─ AAAGTAGCAAGTCATC-1.fasta
│  │  ├─ AACCGCGAGGCTAGCA-1.fasta
│  │  └─ AACCGCGTCCCTCAGT-1.fasta
│  └─ mouse_demo
│     ├─ AAGGTTCGTTGTCTTT-1.fasta
│     └─ AATCGGTAGTTGAGAT-1.fasta
├─ init.py
├─ main.py
├─ model2.py
├─ out
├─ params
│  ├─ best_cnn_48_microbe.pkl
│  ├─ best_cnn_48_mouse.pkl
│  ├─ best_cnn_48_human.pkl
│  ├─ best_model_48_microbe.pkl
│  ├─ best_model_48_mouse.pkl
│  └─ best_model_48_huamn.pkl
├─ ready
│  ├─ test_x.npy
│  ├─ test_y.npy
│  ├─ train_x.npy
│  └─ train_y.npy
└─ requirements.txt

```


---

## 🧠 Pretrained Models

ViTrace includes multiple pretrained models tailored to different scenarios. All models are located in the `params/` folder.

| Model Name                  | Application Scenario                                     | Negative Class                         |
|----------------------------|----------------------------------------------------------|----------------------------------------|
| `best_model_48_human.pkl`      | **Default model** for human tumor transcriptomes         | Human transcripts                      |
| `best_model_48_microbe.pkl`| Viral detection in metatranscriptomes                    | Microbial sequences (bacteria, fungi, etc.) |
| `best_model_48_mouse.pkl`  | Viral detection in mouse tumor transcriptomes            | Mouse transcripts                      |

> ✅ **Default production model:**  
> `best_model_48_human.pkl` is automatically used for inference unless changed.


**Note on Model Comparisons**  

To improve transparency, the GitHub repository now includes performance metrics for each model in its respective application scenario:

| Model Name                  | ACC     | Precision | Recall  | F1     |
|-----------------------------|---------|-----------|---------|--------|
| `best_model_48_human.pkl`       | 0.7810  | 0.8461    | 0.7486  | 0.7944 |
| `best_model_48_microbe.pkl` | 0.7646  | 0.7892    | 0.7703  | 0.7844 |
| `best_model_48_mouse.pkl`   | 0.8996  | 0.8342    | 0.9780  | 0.9104 |

End users can readily identify the most appropriate model for their data and interpret its expected performance accordingly.

---




## Dependencies
ViTrace is a Python package. To install it, run the following command in your terminal:
* clone repo, cd into it
```bash
git clone https://github.com/Ying-Lab/ViTrace && cd ViTrace
```
* create a conda environment, with Python 3.10+
```bash
conda create -n ViTrace python=3.10 && conda activate ViTrace
```
* install requirements
```bash
pip install -r requirements.txt
```


## Usage
To analyze human sequencing data, ensure that your unmapped sequencing FASTA read files are stored in the designated folder. Execute the following command:
```python
python main.py \
--in_folder <in_folder> \
--out_folder <out_folder> \
--threshold <threshold> \
--batch_size <batch_size>

In the output directory, you can find the filtered viral candidate contigs and the prediction score for each sequence.
```
### Demo
```python
python main.py \
--in_folder demo/human_demo \
--out_folder out \
--threshold 0.5 \
--batch_size 1024
```
### Parameter 
| Parameter       | Type   | Required | Default | Description          |
|:-----------------:|:--------:|:----------:|:---------:|:----------------------:|
| `--in_folder`   | String | Yes      | -       | Input directory path |
|`--out_folder` | String | Yes|       -| OutPut directory path|
| `--threshold`   | Float  | No       | 0.6     | Processing threshold |
| `--batch_size`  | Int    | No       | 1024    | Batch size           |
| `--model_type`  | String    | No       | human    | Specify model type (human, microbe, or mouse)         |
### Optional: Contig Assembly
If you wish to assemble viral contigs from the filtered reads, you can use the provided assemble_contigs.py script. Execute the following command:

```python
python assemble_contigs.py \
    --fasta reads.fasta \
    --scores predictions.txt \
    -k 24 \
    --ext-threshold 0.5 \
    --seed-threshold 0.7 \
    --max-proc 48 \
    -o assembled_contigs.fasta
```

### Demo

```python
python main.py \
--in_folder demo/demo_assemble \
--out_folder out \
--threshold 0.5 \
--batch_size 1024
```
```python
python assemble_contigs.py \
    --fasta reads.fasta \
    --scores out/predictions.txt \
    -k 24 \
    --ext-threshold 0.5 \
    --seed-threshold 0.7 \
    --max-proc 48 \
    -o assembled_contigs.fasta
```

### Parameters

| Flag             | Description                                                                 |
|------------------|------------------------------------------------------------------------------|
| `--fasta`        | Input FASTA file of reads (required).                                  |
| `--scores`       | File containing per-read viral-likelihood scores from the detection step (required). |
| `-k`             | K-mer size for graph construction (default: 24).                             |
| `--ext-threshold`| Minimum score for extending a contig (default: 0.5).                         |
| `--seed-threshold`| Minimum score for seeding a contig (default: 0.7).                          |
| `--max-proc`     | Maximum number of parallel processes (default: 8).                           |
| `-o`             | Output FASTA file for assembled contigs (required).                          |

> **Note**  
> [viRNATrap](https://github.com/AuslanderLab/virnatrap) provides an alternative assembly implementation in C, which offers certain speed advantages compared to our Python version.
