import shortuuid


class ModelGraphNodeMetadata:
    """ Metadata for a node in the model graph """
    def __init__(self, args, component_id, depth, dataset_name, num_classes, dataset_short_name, seed, model_path=None):
        self.component_id = component_id
        self.depth = depth
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.dataset_short_name = dataset_short_name
        self.seed = seed
        self.uuid = shortuuid.ShortUUID().random(length=8)
        if model_path is None:
            self.model_path = f"{args.output_root}/{dataset_short_name}__" \
                              f"component_{component_id}__depth_{depth}__uuid_{self.uuid}"
        else:
            self.model_path = model_path

    def __str__(self):
        return f"ModelGraphNodeMetadata(component_id={self.component_id}, depth={self.depth}, " \
               f"dataset_name={self.dataset_name}, num_classes={self.num_classes}, " \
               f"dataset_short_name={self.dataset_short_name}, seed={self.seed}, uuid={self.uuid}, " \
               f"model_path={self.model_path})"

    def __repr__(self):
        return f"ModelGraphNodeMetadata(component_id={self.component_id}, depth={self.depth}, " \
               f"dataset_name={self.dataset_name}, num_classes={self.num_classes}, " \
               f"dataset_short_name={self.dataset_short_name}, seed={self.seed}, uuid={self.uuid}, " \
               f"model_path={self.model_path})"


class ModelGraphNode:
    """ Node in the model graph """
    def __init__(self, parent, metadata, is_root=False, is_leaf=False):
        self.is_root = is_root
        self.is_leaf = is_leaf
        self.metadata = metadata
        self.children = []
        self.parent = parent

    def __str__(self):
        return f"ModelGraphNode(component={self.metadata.component_id}, depth={self.metadata.depth}, " \
               f"name={self.metadata.dataset_short_name}, model_path={self.metadata.model_path})"

    def __repr__(self):
        return f"ModelGraphNode(component={self.metadata.component_id}, depth={self.metadata.depth}, " \
               f"name={self.metadata.dataset_short_name}, model_path={self.metadata.model_path})"

    def __eq__(self, other):
        return self.metadata.uuid == other.metadata.uuid

    def __lt__(self, other):
        # check if metadata.component_id is the same, if so then check metadata.depth, otherwise check metadata.uuid
        if self.metadata.component_id == other.metadata.component_id:
            if self.metadata.depth == other.metadata.depth:
                return self.metadata.uuid < other.metadata.uuid
            return self.metadata.depth < other.metadata.depth
        return self.metadata.component_id < other.metadata.component_id


class ModelGraph:
    """ Model graph for a given dataset """
    def __init__(self, roots):
        self.roots = roots

    def get_roots(self):
        """ Returns all root nodes in the model graph """
        return self.roots

    def get_first_level_children(self):
        """ Returns all first level children in the model graph """
        first_level_children = []
        for root in self.roots:
            for child in root.children:
                assert child.metadata.depth == 1, f"Child {child} does not have depth 1"
                first_level_children.append(child)
        return sorted(first_level_children)

    def get_second_level_children(self):
        """ Returns all second level children in the model graph """
        second_level_children = []
        for root in self.roots:
            for first_level_child in root.children:
                for second_level_child in first_level_child.children:
                    assert second_level_child.metadata.depth == 2, f"Child {second_level_child} does not have depth 2"
                    second_level_children.append(second_level_child)
        return sorted(second_level_children)

    def get_all_nodes(self, sort=True):
        """ Returns all nodes in the model graph """
        all_nodes = []
        for root in self.roots:
            all_nodes.append(root)
            for first_level_child in root.children:
                all_nodes.append(first_level_child)
                for second_level_child in first_level_child.children:
                    all_nodes.append(second_level_child)
        if sort:
            return sorted(all_nodes)
        return all_nodes

    def get_node_count(self):
        """ Returns the number of nodes in the model graph """
        return len(self.get_all_nodes(sort=False))

    def __str__(self):
        return f"ModelGraph(roots={self.roots})"

    def __repr__(self):
        return f"ModelGraph(roots={self.roots})"
