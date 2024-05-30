import os
import pickle
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import AutoModel

from utils import calc_ku
from MDST import build_tree, get_ground_truth
from model_graph_lora import LoRAModelGraphNode, LoRAModelGraphNodeMetadata, LoRAModelGraph

warnings.filterwarnings("ignore")


def calc_dist_and_ku(nodes_: dict, layer_kind: str = 'value'):
    """ Calculates the distance and kurtosis between the models. """
    for i, child_node in enumerate(root.children):
        nodes_[f'{root_idx}-{i}-X'] = child_node
        for j, grandchild_node in enumerate(child_node.children):
            nodes_[f'{root_idx}-{i}-{j}'] = grandchild_node

    idx_ = sorted(list(nodes_.keys()))
    ku_ = pd.Series(np.nan, index=idx_)
    dist_ = pd.DataFrame(0, index=idx_, columns=idx_)

    for n, (i, node_) in enumerate(tqdm(nodes_.items())):
        model = AutoModel.from_pretrained(node_.metadata.model_path)

        # calculate kurtosis of the model:
        ku_[i] = calc_ku(model, layer_kind)

        # calculate the distance between the models:
        for j, other_node in list(nodes_.items())[n:]:
            if i == j:
                continue

            other_model = AutoModel.from_pretrained(other_node.metadata.model_path)
            model_dist = 0
            for (name, layer), (other_name, other_layer) in zip(model.state_dict().items(),
                                                                other_model.state_dict().items()):
                if layer_kind not in name:
                    continue
                if not (layer.shape == other_layer.shape):
                    continue
                if len(layer.shape) != 2 or layer.shape[0] != layer.shape[1]:
                    continue

                layer_dist = np.linalg.matrix_rank((layer - other_layer).numpy())
                model_dist += layer_dist

            dist_[i][j] = dist_[j][i] = model_dist
    return ku_, dist_


if __name__ == '__main__':
    for subset_name in ['lora_v', 'lora_f']:
        with open(os.path.join("dataset", f'{subset_name}_model_graph.pkl'), "rb") as f:
            model_graph = pickle.load(f)
            roots = model_graph.get_roots()
            for root_idx in range(len(roots)):
                ground_truth = get_ground_truth(root_idx)
                root = roots[root_idx]
                nodes = {f'{root_idx}-X-X': root}
                ku, dist = calc_dist_and_ku(nodes)
                _, acc = build_tree(ku, dist, 0.3, ground_truth)
                print(f'[{subset_name}] - ({root_idx}): accuracy: {acc:.2f}%')
                print('-' * 100)
