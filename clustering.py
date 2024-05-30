import pickle
import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import AutoModel
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

LORA = True


def calc_full_distance_matrix(model_graph_path: str, lora: bool = False):
    """ Calculate the distance between all models in the graph """
    with open(model_graph_path, "rb") as f:
        model_graph = pickle.load(f)
        nodes = {}
        roots = model_graph.get_roots()
        for root_idx in range(len(roots)):
            root = roots[root_idx]
            nodes[f'{root_idx}-X-X'] = root

            for i_, child_node in enumerate(root.children):
                nodes[f'{root_idx}-{i_}-X'] = child_node
                for j_, grandchild_node in enumerate(child_node.children):
                    nodes[f'{root_idx}-{i_}-{j_}'] = grandchild_node

        idx_ = sorted(list(nodes.keys()))
        dist_ = pd.DataFrame(0, index=idx_, columns=idx_)

        for i_, j_ in tqdm(list(itertools.combinations(nodes, 2))):
            node = nodes[i_]
            other_node = nodes[j_]

            model = AutoModel.from_pretrained(node.metadata.model_path)
            other_model = AutoModel.from_pretrained(other_node.metadata.model_path)

            model_dist = 0
            for (layer, other_layer) in zip(model.state_dict().values(), other_model.state_dict().values()):
                if not (layer.shape == other_layer.shape):
                    continue

                if len(layer.shape) != 2 or layer.shape[0] != layer.shape[1]:
                    continue

                if LORA:
                    layer_dist = np.linalg.matrix_rank((layer - other_layer).numpy())
                else:
                    layer_dist = (layer.flatten() - other_layer.flatten()).abs().mean().cpu().numpy()
                model_dist += layer_dist

            dist_[i_][j_] = dist_[j_][i_] = model_dist

        dist_.to_csv(f'{"lora_" if lora else ""}dist_checkpoint_full.csv')


if __name__ == '__main__':
    K = 5
    iters = 100

    model_tree_path = f'dataset/{"lora_v" if LORA else "full_ft"}_model_graph.pkl'
    calc_full_distance_matrix(model_tree_path, lora=LORA)

    dist = pd.read_csv(f'{"lora_" if LORA else ""}dist_checkpoint_full.csv', index_col=0)

    for node_percentage in np.linspace(1, 0.1, 10):
        node_percentage = round(node_percentage, 1)
        res = []
        for i in range(iters):
            num = int(len(dist.index) * node_percentage)
            idx = sorted(np.random.choice(dist.index, num, replace=False))
            tmp_dist = squareform(np.array(dist.loc[idx, idx]))
            Z = linkage(tmp_dist, method='ward')
            clusters = fcluster(Z, K, criterion='maxclust')
            best = 0
            for perm in itertools.permutations(range(K)):
                score = accuracy_score(pd.Series(idx).str[0].to_numpy().astype(int),
                                       np.array([perm[i] for i in clusters - 1]))
                if score > best:
                    best = score
            res.append(best)

        print(f'{node_percentage:.1f} of nodes: Accuracy of {np.mean(res):.3f} ± {np.std(res):.3f}')
