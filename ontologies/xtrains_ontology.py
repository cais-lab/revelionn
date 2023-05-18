from nxontology import NXOntology


def create_xtrains_nxo() -> NXOntology[str]:
    """
    Edges go from general to specific.
    """

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
