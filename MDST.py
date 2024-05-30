import numpy as np
import pandas as pd
import networkx as nx


def _find_min_weighted_directed_tree(kd: pd.DataFrame):
    """ Finds the minimum weighted directed tree in a graph."""
    G = nx.DiGraph()
    weights = [list(row) for name, row in kd.iterrows()]
    for i in range(len(weights)):
        for j in range(len(weights)):
            if weights[i][j] < float('inf'):
                G.add_edge(i, j, weight=weights[i][j])

    min_arborescence = nx.algorithms.tree.branchings.Edmonds(G).find_optimum(attr='weight', default=float('inf'),
                                                                             kind='min', style='arborescence',
                                                                             preserve_attrs=False)
    return min_arborescence


def build_tree(ku_, dist_, lam_: float, ground_truth_: set, rev: bool = False):
    """ Builds a tree from the given distance and kurtosis matrices. """
    res = []
    if not rev:
        T_ = pd.DataFrame(ku_.to_numpy().reshape(-1, 1) > ku_.to_numpy().reshape(1, -1),
                          index=ku_.index, columns=ku_.index).astype(int).T
    else:
        T_ = pd.DataFrame(ku_.to_numpy().reshape(-1, 1) < ku_.to_numpy().reshape(1, -1),
                          index=ku_.index, columns=ku_.index).astype(int).T

    diag_ = np.eye(len(T_))
    diag_[diag_ == 1] = float('inf')

    KD_ = dist_ + lam_ * dist_.mean().mean() * T_ + diag_
    KD_ = KD_.sort_index(ascending=False)
    KD_ = KD_[sorted(KD_, reverse=True)]

    tree_ = _find_min_weighted_directed_tree(KD_)

    correct = 0
    for edge in tree_.edges(data=True):
        formatted_edge = KD_.index[edge[0]] + ' -> ' + KD_.index[edge[1]]
        if formatted_edge in ground_truth_:
            correct += 1
        res.append(formatted_edge)
    acc_ = correct / len(ground_truth_) * 100
    return res, acc_


def get_ground_truth(r_i: int = None, llama=False, sd=False):
    """ Returns the ground truth for the given root index. """
    assert sum([r_i is not None, llama, sd]) == 1, \
        'Only one of r_i, llama, or sd should be provided.'

    if r_i is not None:
        return {f'{r_i}-X-X -> {r_i}-3-X', f'{r_i}-X-X -> {r_i}-2-X',
                f'{r_i}-X-X -> {r_i}-1-X', f'{r_i}-X-X -> {r_i}-0-X',
                f'{r_i}-3-X -> {r_i}-3-3', f'{r_i}-3-X -> {r_i}-3-2',
                f'{r_i}-3-X -> {r_i}-3-1', f'{r_i}-3-X -> {r_i}-3-0',
                f'{r_i}-2-X -> {r_i}-2-1', f'{r_i}-2-X -> {r_i}-2-0',
                f'{r_i}-2-X -> {r_i}-2-3', f'{r_i}-2-X -> {r_i}-2-2',
                f'{r_i}-1-X -> {r_i}-1-3', f'{r_i}-1-X -> {r_i}-1-2',
                f'{r_i}-1-X -> {r_i}-1-1', f'{r_i}-1-X -> {r_i}-1-0',
                f'{r_i}-0-X -> {r_i}-0-3', f'{r_i}-0-X -> {r_i}-0-2',
                f'{r_i}-0-X -> {r_i}-0-0', f'{r_i}-0-X -> {r_i}-0-1'}

    elif llama:
        return {
            f'0-X-X -> 0-2-X',
            f'0-X-X -> 0-1-X',
            f'0-X-X -> 0-0-X',
            f'0-0-X -> 0-0-0'
        }

    elif sd:
        return {
            '0-X-X -> 0-0-X',
            '0-0-X -> 0-0-0',
            '0-0-X -> 0-0-1',
            '0-0-1 -> 0-0-2'
        }
