import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoModel
from itertools import combinations
from diffusers import StableDiffusionPipeline

from MDST import build_tree, get_ground_truth
from utils import _get_nodes, _get_layer_kinds, calc_ku


def test_SD():
    """ test MoTHer on the Stable Diffusion Model Tree"""
    nodes = _get_nodes(sd=True)
    ku = pd.Series(np.nan, index=[n[0] for n in nodes])
    SD_statistics = pd.read_csv('SD_layers_statistics.csv', index_col=0)

    for (i, model_path), (j, other_model_path) in tqdm(list(combinations(nodes, 2))):
        sd_pipe = StableDiffusionPipeline.from_pretrained(model_path)
        other_sd_pipe = StableDiffusionPipeline.from_pretrained(other_model_path)

        if ku.isna()[i]:
            ku[i] = calc_ku(sd_pipe.unet)

        if ku.isna()[j]:
            ku[j] = calc_ku(other_sd_pipe.unet)

    idx = [os.path.basename(n) for _, n in nodes]
    dist_minmax_square_max = {}
    files_df = pd.DataFrame(np.array([np.array(r) for r in pd.Series(os.listdir('SD_blocks/')).str.split('__')]),
                            columns=['node', 'layer'])

    for layer, slice_ in tqdm(files_df.groupby('layer')):
        layer = layer[:-3]
        dist_minmax_square_max[layer] = pd.DataFrame(np.nan, index=idx, columns=idx)

        for node_i, node_j in combinations(idx, 2):
            layer_data_i = torch.load(os.path.join(f'SD_blocks/{node_i}__{layer}.pt'))
            layer_data_j = torch.load(os.path.join(f'SD_blocks/{node_j}__{layer}.pt'))

            diff = layer_data_i - layer_data_j

            # minmax normalize diff:
            min_ = SD_statistics.loc[(SD_statistics['layer'] == layer), 'min'].values[0]
            max_ = SD_statistics.loc[(SD_statistics['layer'] == layer), 'max'].values[0]
            diff_minmax = (diff - min_) / (max_ - min_)

            if len(diff.shape) == 2 and diff.shape[0] == diff.shape[1]:
                dist_minmax_square_max[layer][node_i][node_j] = \
                    dist_minmax_square_max[layer][node_j][node_i] = diff_minmax.max().item()

    max_dist = pd.DataFrame(np.max(np.array([d.values for d in dist_minmax_square_max.values()
                                             if not np.isnan(d.values).all()]), axis=0), index=ku.index,
                            columns=ku.index).fillna(0)
    _, acc = build_tree(ku, max_dist, 0.3, get_ground_truth(sd=True), rev=True)
    print(f'{acc=}')


def test_llama():
    nodes = _get_nodes(llama=True)
    layer_kinds = _get_layer_kinds(llama=True)
    llama_statistics = pd.read_csv('llama_layers_statistics.csv', index_col=0)
    idx = [os.path.basename(n) for _, n in nodes]
    dist = {i: {k: pd.DataFrame(np.nan, index=idx, columns=idx) for k in layer_kinds} for i in range(32)}

    ku = pd.Series(np.nan, index=[i for i, _ in nodes])
    for i, node in tqdm(nodes):
        model = AutoModel.from_pretrained(node)
        ku[i] = calc_ku(model, 'self_attn.o_proj')

    for block_idx in tqdm(range(32)):
        for kind in layer_kinds:
            for node_i, node_j in combinations(idx, 2):
                layer_data_i = torch.load(os.path.join(f'llama_blocks/{block_idx}',
                                                       f'{node_i}-layers.{block_idx}.{kind}.weight.pt'))
                layer_data_j = torch.load(os.path.join(f'llama_blocks/{block_idx}',
                                                       f'{node_j}-layers.{block_idx}.{kind}.weight.pt'))
                diff = layer_data_i - layer_data_j

                # minmax normalize diff:
                min_ = llama_statistics.loc[(llama_statistics['block_idx'] == block_idx) & (
                        llama_statistics['layer_kind'] == kind), 'min'].values[0]
                max_ = llama_statistics.loc[(llama_statistics['block_idx'] == block_idx) & (
                        llama_statistics['layer_kind'] == kind), 'max'].values[0]
                diff_minmax = (diff - min_) / (max_ - min_)

                if len(diff.shape) == 2 and diff.shape[0] == diff.shape[1]:
                    dist[block_idx][kind][node_i][node_j] = \
                        dist[block_idx][kind][node_j][node_i] = diff_minmax.max().item()

    max_dist = pd.DataFrame(np.max(np.array([d.values for i in dist for d in dist[i].values() if
                                             not np.isnan(d.values).all()]), axis=0),
                            index=ku.index, columns=ku.index).fillna(0)
    _, acc = build_tree(ku, max_dist, 0.3, get_ground_truth(llama=True))
    print(f'{acc}')
