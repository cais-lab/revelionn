Data
====

Dataset Representation
----------------------

RevelioNN uses image datasets that are binarily annotated by various attributes. The attributes represent ontology
concepts. Thus, if the value of some attribute equals 1, we can say that the corresponding ontology concept is
represented in the image. Otherwise, the value of the attribute is 0.

All images of the dataset should be in the same folder. The dataset is expected to be represented in a CSV file with the following fields (columns):

- column with the names of the images (indicating the image format);
- group of columns, each corresponding to a particular attribute (concept) that has a value of 1 or 0.

Ontology Representation
-----------------------

To form logical explanations through ontological inference the ontology must be represented in the OWL 2 language.
In addition, it is necessary to describe the so-called concepts map, which is a dictionary whose keys are the names of
the attributes of the dataset, and the values are the corresponding concepts of the ontology. For example, for the XTRAINS dataset ontology, the concept map would look like this:

.. code :: python

    concepts_map = {
        'TypeA': '(TypeA)',
        'TypeB': '(TypeB)',
        'TypeC': '(TypeC)',

        'LongWagon': '(has some LongWagon)',
        'PassengerCar': '(has some PassengerCar)',
        'FreightWagon': '(has some FreightWagon)',
        'EmptyWagon': '(has some EmptyWagon)',
        'LongTrain': '(LongTrain)',
        'WarTrain': '(WarTrain)',
        'PassengerTrain': '(PassengerTrain)',
        'FreightTrain': '(FreightTrain)',
        'LongFreightTrain': '(LongFreightTrain)',
        'EmptyTrain': '(EmptyTrain)',
        'MixedTrain': '(MixedTrain)',
        'ReinforcedCar': '(has some ReinforcedCar)',
        'RuralTrain': '(RuralTrain)',
    }

To use the ontology in concept extraction algorithms it is also necessary to represent the ontology as a
``networkx.DiGraph``, where edge direction goes from superterm to subterm. It is recommended to use the ``NXOntology`` class
of the nxontology library for this purpose, as shown in the following example:

.. code :: python

    from nxontology import NXOntology

    def create_xtrains_nxo() -> NXOntology[str]:

        nxo: NXOntology[str] = NXOntology()
        nxo.graph.graph["name"] = "XTRAINS"
        nxo.set_graph_attributes(node_name_attribute="{node}")
        edges = [
            ("TypeA", "WarTrain"),
            ("TypeA", "EmptyTrain"),
            ("WarTrain", "ReinforcedCar"),
            ("WarTrain", "PassengerCar"),
            ("EmptyTrain", "EmptyWagon"),

            ("TypeB", "PassengerTrain"),
            ("TypeB", "LongFreightTrain"),
            ("PassengerTrain", "PassengerCar"),
            ("PassengerTrain", "LongWagon"),
            ("LongFreightTrain", "LongTrain"),
            ("LongFreightTrain", "FreightTrain"),
            ("LongTrain", "LongWagon"),
            ("FreightTrain", "FreightWagon")
        ]
        nxo.graph.add_edges_from(edges)
        return nxo
