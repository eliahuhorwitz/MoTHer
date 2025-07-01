import os
import pickle
import warnings
import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import AutoModel

from MDST import build_tree, get_ground_truth
from utils import _get_layer_kinds, calc_ku
from model_graph import ModelGraphNode, ModelGraphNodeMetadata, ModelGraph

warnings.filterwarnings("ignore")


def calc_dist_and_ku(model_graph_, root_idx_, layer_kind='output.dense'):
    """ Calculates the distance and kurtosis between the models. """
    layer_kinds_ = _get_layer_kinds()
    idx = []
    root_ = model_graph_.get_roots()[root_idx_]
    root_str = f'{root_idx_}-X-X'
    idx.append(root_str)
    models = {root_str: AutoModel.from_pretrained(root_.metadata.model_path)}
    for i, child_node in enumerate(tqdm(root_.children, desc=f'loading models for root {root_idx_}', leave=False)):
        child_str = f'{root_idx_}-{i}-X'
        idx.append(child_str)
        models[child_str] = AutoModel.from_pretrained(child_node.metadata.model_path)
        for j, grandchild_node in enumerate(child_node.children):
            grandchild_str = f'{root_idx_}-{i}-{j}'
            idx.append(grandchild_str)
            models[grandchild_str] = AutoModel.from_pretrained(grandchild_node.metadata.model_path)

    ku_ = pd.Series(np.nan, index=idx)
    for i, model in tqdm(models.items()):
        ku_[i] = calc_ku(model, layer_kind)

    dist_ = {i_: {k_: pd.DataFrame(np.nan, index=idx, columns=idx) for k_ in layer_kinds_} for i_ in range(12)}

    for block_idx in tqdm(range(12)):
        for kind in layer_kinds_:
            for node_i, node_j in sorted(itertools.combinations(idx, 2)):
                layer_data_i = models[node_i].state_dict()[f'encoder.layer.{block_idx}.attention.{kind}.weight']
                layer_data_j = models[node_j].state_dict()[f'encoder.layer.{block_idx}.attention.{kind}.weight']

                diff = layer_data_i - layer_data_j
                l2_diff = diff.pow(2).mean().sqrt().item()

                if len(diff.shape) == 2 and diff.shape[0] == diff.shape[1]:
                    dist_[block_idx][kind][node_i][node_j] = dist_[block_idx][kind][node_j][node_i] = l2_diff

    mean_dist_ = pd.DataFrame(np.mean(np.array([d.values for i in dist_ for d in dist_[i].values() if not np.isnan(d.values).all()]), axis=0), index=idx, columns=idx).fillna(0)
    return ku_, mean_dist_


if __name__ == '__main__':
    with open('dataset/full_ft_model_graph.pkl', "rb") as f:
        model_graph = pickle.load(f)
        roots = model_graph.get_roots()
        for root_idx in range(len(roots)):
            ku, dist = calc_dist_and_ku(model_graph, root_idx)
            tree, acc = build_tree(ku, dist, 0.3, get_ground_truth(root_idx))
            print(f'[{root_idx}]: tree: {tree}')
            print(f'[{root_idx}]: accuracy: {acc:.2f}%')
            print('-' * 100)