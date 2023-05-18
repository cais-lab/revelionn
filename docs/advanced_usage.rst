Advanced Usage
==============

Concept Extraction Algorithms
-----------------------------

+-------------------------+------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
| Extraction algorithm    | Type of mapping network      | What it does                                                                                                                           |
+=========================+==============================+========================================================================================================================================+
| Exhaustive search       | Single mapping network       | Trains and evaluates mapping networks based on the activations of each of the specified layers of the convolutional network            |
+-------------------------+------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
| Heuristic search        | Single mapping network       | Due to the heuristic reduction of the set of specified layers, mapping networks are not trained for every combination of layer-concept |
+-------------------------+------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
| Simultaneous extraction | Simultaneous mapping network | Trains a mapping network that can simultaneously extract a set of relevant concepts from the entire set of layers of specified types   |
+-------------------------+------------------------------+----------------------------------------------------------------------------------------------------------------------------------------+

Semi-Supervised Learning
------------------------

For a simultaneous mapping network, RevelioNN provides the ability to perform semi-supervised learning using semantic
loss. Semantic loss allows us to take into account the relationship between the concepts of ontology, which introduces additional regularization.

To use semantic loss during the training of a simultaneous mapping network, the following preparatory steps must be performed:

#. Following the sympy syntax, write logical constraints defined by the ontology on the output vector of probabilities of
   concepts. At the same time, it should be taken into account which of the outputs (concept block) of the mapping network
   corresponds to each of the concepts. Each of the constraints is expressed by a single string, and the strings are considered to be in an "and" relationship.

   For example, in the SCDB dataset, the concept `C1` is equivalent to the concepts `Hexagon` ⊓ `Star` or `Ellipse` ⊓ `Star` or `Triangle` ⊓ `Ellipse` ⊓ `Starmarker`.

   Suppose that the network will return a vector of probabilities of concepts arranged in the following order:

   ``['HexStar', 'EllStar', 'TEStarmarker', 'Hexagon', 'Star', 'Ellipse', 'Triangle', 'Starmarker']``

   In this case, the generated sympy file may look like this:

   .. code-block:: shell

        shape [8]

        Equivalent(X0, And(X3, X4))
        Equivalent(X1, And(X5, X4))
        Equivalent(X2, And(X6, X5, X7))

#. Compile the specified constraint to a ``vtree`` and an ``sdd`` file using the `semantic-loss-pytorch <https://github.com/lucadiliello/semantic-loss-pytorch>`_ library.

#. The resulting ``sdd`` and ``vtree`` files must be used to initialize an instance of the ``semantic_loss_pytorch.SemanticLoss`` class. After that, it can be used in the methods of the RevelioNN library that allow semi-supervised learning.
