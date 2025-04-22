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
│  ├─ best_cnn_48_o.pkl
│  ├─ best_model_48_microbe.pkl
│  ├─ best_model_48_mouse.pkl
│  └─ best_model_48_o.pkl
└─ requirements.txt

```

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

## Data Availability Statement
All datasets used in this work, including training data, validation splits, and test benchmarks, are publicly available under open-access licenses via the Zenodo repository:
* Persistent URL: https://zenodo.org/records/15259039