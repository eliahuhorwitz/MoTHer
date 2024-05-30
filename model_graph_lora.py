from model_graph import ModelGraphNodeMetadata, ModelGraphNode, ModelGraph


class LoRAModelGraphNodeMetadata(ModelGraphNodeMetadata):
    """ Metadata for a node in the model graph of LoRA """
    def __init__(self, args, component_id, depth, dataset_name, num_classes, dataset_short_name, seed, lora_rank=-1,
                 model_path=None):
        super().__init__(args, component_id, depth, dataset_name, num_classes, dataset_short_name, seed, model_path)
        self.lora_rank = lora_rank

    def __str__(self):
        return 'LoRA' + super().__str__() + f", lora_rank={self.lora_rank}"

    def __repr__(self):
        return 'LoRA' + super().__repr__() + f", lora_rank={self.lora_rank}"


class LoRAModelGraphNode(ModelGraphNode):
    """ Node in the model graph of LoRA """
    def __str__(self):
        return f'LoRA{super().__str__()}'

    def __repr__(self):
        return f'LoRA{super().__repr__()}'


class LoRAModelGraph(ModelGraph):
    """ Model graph of LoRA """
    def __str__(self):
        return f'LoRA{super().__str__()}'

    def __repr__(self):
        return f'LoRA{super().__repr__()}'
