from scipy.stats import kurtosis


def calc_ku(model, layer_kind=None):
    """ calculate kurtosis of a model """
    model_ku = 0
    for name, layer in model.state_dict().items():
        if len(layer.shape) != 2 or layer.shape[0] != layer.shape[1]:
            continue

        if layer_kind is not None:
            if layer_kind not in name:
                continue
        ku = kurtosis(layer.flatten())
        model_ku += ku
    return model_ku


def _get_layer_kinds():
    """ get layer kinds for a model """
    return ['attention.query', 'attention.key', 'attention.value', 'output.dense']



