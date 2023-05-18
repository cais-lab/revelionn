import networkx as nx
from semantic_loss_pytorch import SemanticLoss


class ConceptExtractor:
    """
    A class that provides concept extraction algorithms.

    Attributes
    ----------
    ontology : nxontology.NXOntology
        Ontology represented as a graph, where edge direction goes from superterm to subterm.
    trainer : MappingTrainer
        An instance of the MappingTrainer class that provides an interface for training mapping networks.

    Methods
    -------
    create_subgraph(graph, node)
        Returns a subgraph containing all child nodes for a given, including this one.
    order_concepts(target_concept, ontology)
        Performs topological sorting of a subgraph formed by a given parent node (target concept).
    exhaustive_search(concept, layer_names, mapping_neurons)
        Trains and evaluates mapping networks based on the activations of each of the specified layers of the network.
    linear_search(concept, top_layer_num, patience_layers, mapping_neurons)
        Trains and evaluates mapping networks based on the activations of each of the layers starting from the
        specified one, until the value of the quality metric deteriorates over several layers (the value of patience).
    heuristic_search(target_concept, top_layer_num, patience_layers, mapping_neurons)
        Due to the heuristic reduction of the set of specified layers, mapping networks are not trained for every
        combination of layer-concept. Uses linear search.
    simultaneous_extraction(target_concept, decoder_channels, num_shared_neurons, num_output_neurons,
                            sdd_path=None, vtree_path=None, sem_loss_weight=None, unlabeled_samples=None)
        Trains a mapping network that can simultaneously extract a set of relevant concepts from the entire set of
        layers of specified types (the types are set when initializing the MappingTrainer instance).
    """

    def __init__(self, mapping_trainer, nxonto):
        """
        Sets all the necessary attributes for the ConceptExtractor object.

        Parameters
        ----------
        mapping_trainer : MappingTrainer
            An instance of the MappingTrainer class that provides an interface for training mapping networks.
        nxonto : nxontology.NXOntology
            Ontology represented as a graph, where edge direction goes from superterm to subterm.
        """

        self.ontology = nxonto
        self.trainer = mapping_trainer

    @staticmethod
    def create_subgraph(graph, node):
        """
        Returns a subgraph containing all child nodes for a given node, including the given node.

        Parameters
        ----------
        graph : networkx.Graph
            The graph from which to extract the subgraph.
        node : str
            The node for which to create the subgraph.

        Returns
        -------
        networkx.Graph
            A subgraph of `graph` containing all child nodes of `node`, including `node`.
        """

        edges = nx.dfs_successors(graph, node)
        nodes = []
        for k, v in edges.items():
            nodes.extend([k])
            nodes.extend(v)
        return graph.subgraph(nodes)

    def order_concepts(self, target_concept, ontology):
        """
        Performs topological sorting of a subgraph formed by a given parent node (target concept).

        Parameters
        ----------
        target_concept : str
            The target concept node for which to perform topological sorting.
        ontology : nxontology.NXOntology
            The ontology graph.

        Returns
        -------
        list
            A list of concepts in topologically sorted order within the subgraph.
        """

        subgraph = self.create_subgraph(ontology.graph, target_concept)
        return list(nx.topological_sort(nx.line_graph(subgraph)))

    def exhaustive_search(self, concept, layer_names, mapping_neurons):
        """
        Trains and evaluates mapping networks based on the activations of each of the specified layers of the network.

        Parameters
        ----------
        concept : str
            The concept for which to perform the search.
        layer_names : list
            A list of layer names to consider for training and evaluation.
        mapping_neurons : list[int]
            The number of neurons in the mapping network.

        Returns
        -------
        dict
            A dict containing the best layer name and the corresponding evaluation value.
        """

        best_value = None
        best_layer = None

        for layer_name in layer_names:
            self.trainer.train_single_model(mapping_neurons, concept, [layer_name])
            cur_value = self.trainer.evaluate_model()

            if best_value is None or cur_value > best_value:
                best_value = cur_value
                best_layer = layer_name

        return best_layer, best_value

    def linear_search(self, concept, top_layer_num, patience_layers, mapping_neurons):
        """
        Trains and evaluates mapping networks based on the activations of each of the layers starting from the
        specified one, until the value of the quality metric deteriorates over several layers (the value of patience).

        Parameters
        ----------
        concept : str
            The concept for which to perform the search.
        top_layer_num : int
            The starting layer number for training and evaluation.
        patience_layers : int
            The number of layers to tolerate deterioration in the quality metric.
        mapping_neurons : list[int]
            The number of neurons in the mapping network.

        Returns
        -------
        tuple
            A tuple containing the best layer number and the corresponding evaluation value.
        """

        best_layer_num = None
        best_value = None

        cur_layer_num = top_layer_num
        while cur_layer_num >= 0:
            self.trainer.train_single_model(mapping_neurons, concept, [cur_layer_num])
            cur_value = self.trainer.evaluate_model()

            if best_value is None or cur_value > best_value:
                best_value = cur_value
                best_layer_num = cur_layer_num
            if best_layer_num - cur_layer_num > patience_layers:
                break
            cur_layer_num -= 1

        return best_layer_num, best_value

    def heuristic_search(self, target_concept, top_layer_num, patience_layers, mapping_neurons):
        """
        Due to the heuristic reduction of the set of specified layers, mapping networks are not trained for every
        combination of layer-concept. Uses linear search.

        Parameters
        ----------
        target_concept : str
            The target concept that should be obtained by ontological inference.
            Mapping networks are trained to extract concepts relevant to the target concept.
        top_layer_num : int
            The starting layer number for training and evaluation.
        patience_layers : int
            The number of layers to tolerate deterioration in the quality metric.
        mapping_neurons : list[int]
            The number of neurons in the mapping network.

        Returns
        -------
        dict
            A dictionary containing the best layer number and evaluation value for each concept in the subgraph.
        """

        ordered_concepts = self.order_concepts(target_concept, self.ontology)
        best_layers = {}

        for parent, child in ordered_concepts:
            if parent == target_concept:
                initial_layer = top_layer_num
            else:
                initial_layer = best_layers[parent][0]
            layer_num, auc = self.linear_search(child, initial_layer, patience_layers, mapping_neurons)

            if child not in best_layers.keys():
                best_layers[child] = [layer_num, auc]
            elif auc > best_layers[child][1]:
                best_layers[child] = [layer_num, auc]

        return best_layers

    def simultaneous_extraction(self, target_concept, decoder_channels, num_shared_neurons, num_output_neurons,
                                sdd_path=None, vtree_path=None, sem_loss_weight=None, unlabeled_samples=None):
        """

        Parameters
        ----------
        target_concept : str
            The target concept that should be obtained by ontological inference.
            Mapping networks are trained to extract concepts relevant to the target concept.
        decoder_channels : int
            The number of decoder channels. The output number of channels of the convolutional layer of the decoder or
            the output number of neurons of the decoder of the fully connected layer.
        num_shared_neurons : list[int]
            The number of neurons in consecutive fully connected layers of the common part of the network
            (internal representation of the simultaneous extraction network).
        num_output_neurons : list[int]
            The number of neurons in consecutive fully connected layers of each of the concept blocks.
        sdd_path : str
            The path to the .sdd file.
        vtree_path : str
            The path to the .vtree file.
        sem_loss_weight : float
            The contribution of semantic loss to the overall loss function.
        unlabeled_samples : int or float
            The number of unlabeled samples to include. If float, it represents the fraction of unlabeled samples.

        Returns
        -------
        concepts_auc : list[float]
            ROC AUC values for each of the concepts.
        all_auc : float
            ROC AUC value for all labels of a simultaneous mapping network.
        """

        concepts = self.create_subgraph(self.ontology.graph, target_concept)
        concepts.remove(target_concept)

        if sdd_path is None:
            self.trainer.train_simultaneous_model(concepts, decoder_channels, num_shared_neurons,
                                                  num_output_neurons)
        else:
            sl = SemanticLoss(sdd_path, vtree_path)
            self.trainer.train_simultaneous_model_semisupervised(concepts, decoder_channels, num_shared_neurons,
                                                                 num_output_neurons, sl, sem_loss_weight,
                                                                 unlabeled_samples)
        concepts_auc, all_auc = self.trainer.evaluate_model()
        return concepts_auc, all_auc
