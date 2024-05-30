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


def _get_layer_kinds(vit: bool = False, llama: bool = False):
    """ get layer kinds for a model """
    assert sum([vit, llama]) == 1, 'Exactly one of the flags should be set to True'
    if vit:
        return ['attention.query', 'attention.key', 'attention.value', 'output.dense']

    if llama:
        return ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'input_layernorm', 'post_attention_layernorm']


def _get_nodes(llama: bool = False, sd: bool = False):
    """ get nodes for a model """
    assert sum([llama, sd]) == 1, 'Exactly one of the flags should be set to True'
    if llama:
        return [
            ('0-X-X', 'meta-llama/Llama-2-7b-hf'),
            ('0-0-X', 'meta-llama/CodeLlama-7b-hf'),
            ('0-0-0', 'meta-llama/CodeLlama-7b-Instruct-hf'),
            ('0-1-X', 'meta-llama/CodeLlama-7b-Python-hf'),
            ('0-2-X', 'meta-llama/Llama-2-7b-chat-hf'),
        ]

    if sd:
        return [
            ('0-X-X', 'CompVis/stable-diffusion-v1-1'),
            ('0-0-X', 'CompVis/stable-diffusion-v1-2'),
            ('0-0-0', 'CompVis/stable-diffusion-v1-3'),
            ('0-0-1', 'CompVis/stable-diffusion-v1-4'),
            ('0-0-2', 'runwayml/stable-diffusion-v1-5'),
        ]
