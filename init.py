import random
import re
import numpy as np
import torch
import os

DEFAULT_NUC_ORDER = {y: x for x, y in enumerate(["A", "T", "C", "G"])}
NUCLEOTIDES = sorted([x for x in DEFAULT_NUC_ORDER.keys()])

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def random_base(seq=None):
    """
    Generate a random base.
    :return: Random base.
    """
    return random.choice(NUCLEOTIDES)


def handle_non_ATGC(sequence):
    """
    Handle non ATGCs.
    :param sequence: String input.
    :return: String output (only ATCGs), with randomly assigned bp to non-ATGCs.
    """
    ret = re.sub('[^ATCG]', random_base, sequence)
    assert len(ret) == len(sequence)
    return ret


def matrix_from_fasta(fasta_file):
    """
    Convert fasta file to matrix with A, C, G, T converted to 1, 3, 4, 2 respectively.
    :param fasta_file: String input.
    :return: Numpy array.
    """
    ret = []
    with open(fasta_file, 'r') as fasta:
        for line in fasta:
            line = line.strip()
            if line.startswith(">"):
                seq_value = ""
            else:
                line = handle_non_ATGC(line)
                seq_value += line.strip()
                ret.append(line)


    for i in range(len(ret)):
            ret[i] = list(ret[i])

    matrix = np.array(ret)

    matrix[matrix == 'A'] = 1
    matrix[matrix == 'C'] = 3
    matrix[matrix == 'G'] = 4
    matrix[matrix == 'T'] = 2
    matrix = matrix.astype(int)

    return matrix
    # matrix[matrix != 'A'] = 0

    # 将数据类型转换为整数
    # matrix = matrix.astype(np.int)



#############################################################################

def filter_reads(fasta_file, read_indices, output_path):
    """
    从 fasta 文件中过滤出指定的 reads

    :param fasta_file: fasta 文件路径
    :param read_indices: 包含要保留的 reads 的索引的列表
    :param output_path: 新的输出文件路径
    :return: 过滤后的 fasta 序列字符串
    """

    # 读取 fasta 文件
    with open(fasta_file) as f:
        fasta_dict = {}
        seq = ''
        header = ''
        i = 0
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    fasta_dict[header] = seq
                    seq = ''
                header = '>read{}'.format(i)  # 将索引转换为 read 名称
                i += 1
            else:
                seq += line
        fasta_dict[header] = seq

    # 过滤出指定的 reads
    filtered_dict = {}
    for header, seq in fasta_dict.items():
        read_index = int(header[5:])  # 从 read 名称中获取索引
        if read_index in read_indices:
            filtered_dict[header] = seq

    # 生成过滤后的 fasta 序列字符串
    filtered_fasta = ''
    for header, seq in filtered_dict.items():
        filtered_fasta += header + '\n' + seq + '\n'

    # 生成新的输出文件路径
    output_file = os.path.join(output_path, os.path.basename(fasta_file).replace('.fasta', 'virus_candiadate.fasta'))

    # 将过滤后的 fasta 序列写入新文件
    with open(output_file, 'w') as f:
        f.write(filtered_fasta)

    return filtered_fasta
