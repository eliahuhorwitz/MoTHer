import os
import torch
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import combinations
from diffusers import StableDiffusionPipeline

from utils import _get_nodes

warnings.filterwarnings("ignore")


def _save_SD_layers_locally():
    nodes_ = _get_nodes(sd=True)

    for i, (node_name, node) in enumerate(nodes_):
        sd_pipe = StableDiffusionPipeline.from_pretrained(node)
        model = sd_pipe.unet
        for name, layer_ in model.state_dict().items():
            splits = name.split('.weight')
            if len(splits) == 1:
                continue

            torch.save(layer_, f'SD_blocks/{os.path.basename(node)}__{name}.pt')


if __name__ == '__main__':
    nodes = _get_nodes(sd=True)

    res = pd.DataFrame(columns=['layer', 'mean', 'std', 'min', 'max'])
    files_df = pd.DataFrame(np.array([np.array(r) for r in pd.Series(os.listdir('SD_blocks/')).str.split('__')]),
                            columns=['node', 'layer'])

    for layer, slice_ in tqdm(files_df.groupby('layer')):
        layer_data = [torch.load(os.path.join('SD_blocks/', f'{node}__{file}')) for node, file in slice_.values]
        dist_layer_data = [(l1 - l2) for l1, l2 in combinations(layer_data, 2)]
        layer_data = torch.stack(layer_data).cpu()

        res.loc[res.shape[0]] = {'layer': layer[:-3],
                                 'mean': layer_data.mean().item(),
                                 'std': layer_data.std().item(),
                                 'min': layer_data.min().item(),
                                 'max': layer_data.max().item()}

    res.to_csv('SD_layers_statistics.csv')
