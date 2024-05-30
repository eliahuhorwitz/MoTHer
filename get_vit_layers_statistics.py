import torch
import pickle
import warnings
import pandas as pd

from tqdm import tqdm
from transformers import AutoModel
from itertools import combinations

from utils import _get_layer_kinds
from model_graph import ModelGraphNode, ModelGraphNodeMetadata, ModelGraph

warnings.filterwarnings("ignore")

model_graph_path = './dataset/full_ft_model_graph.pkl'

if __name__ == '__main__':
    layer_kinds = _get_layer_kinds(vit=True)
    with open(model_graph_path, "rb") as f:
        model_graph = pickle.load(f)

        for root_idx in range(len(model_graph.get_roots())):
            root = model_graph.get_roots()[root_idx]
            models = {f'{root_idx}-X-X': AutoModel.from_pretrained(root.metadata.model_path)}
            for i, child_node in enumerate(tqdm(root.children,
                                                desc=f'loading models for root {root_idx}', leave=False)):
                models[f'{root_idx}-{i}-X'] = AutoModel.from_pretrained(child_node.metadata.model_path)
                for j, grandchild_node in enumerate(child_node.children):
                    models[f'{root_idx}-{i}-{j}'] = AutoModel.from_pretrained(grandchild_node.metadata.model_path)

            res = pd.DataFrame(columns=['block_idx', 'layer_kind', 'mean', 'std', 'min', 'max'])

            for block_idx in tqdm(range(12)):
                for kind in layer_kinds:
                    layer_data = [model.state_dict()[f'encoder.layer.{block_idx}.attention.{kind}.weight']
                                  for node, model in models.items()]

                    dist_layer_data = torch.stack([(l1 - l2) for l1, l2 in combinations(layer_data, 2)])
                    res.loc[res.shape[0]] = {'block_idx': block_idx,
                                             'layer_kind': kind,
                                             'mean': dist_layer_data.mean().item(),
                                             'std': dist_layer_data.std().item(),
                                             'min': dist_layer_data.min().item(),
                                             'max': dist_layer_data.max().item()}

            res.to_csv(f'vit_[{root_idx}]_layers_statistics.csv')
