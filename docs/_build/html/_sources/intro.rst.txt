Introduction
============

Installation
------------
The simplest way to install RevelioNN is using ``pip``:

.. code-block:: shell

  $ pip install revelionn
  $ pip install git+https://github.com/lucadiliello/semantic-loss-pytorch.git


RevelioNN supports versions of Python from 3.9 and up.

Another possible way to install RevelioNN is to copy the contents of the ``revelionn``
folder of the repository to your project. Install all the necessary missing dependencies for RevelioNN. In order to do so, you can run:

.. code-block:: shell

  $ pip install -r requirements.txt

To use ready-made scripts, you can clone the repository, create a
virtual environment and install dependencies:

.. code-block:: shell

  $ git clone https://github.com/cais-lab/revelionn.git
  $ cd revelionn
  $ python3 -m venv venv
  $ source venv/bin/activate
  $ pip install -r requirements.txt

It is also worth noting that `Java SE 8 <https://www.java.com/en/download/manual.jsp>`_ must be installed to form logical explanations.
This is due to the fact that explanations using ontology are implemented using the external `BUNDLE <https://ml.unife.it/bundle/>`_ library.

Quick Start
-----------

There are two ways to use the RevelioNN library. One of them is to use the API provided by the library. This method can
be useful for implementing your own library usage scenarios, as well as advanced use cases (see `Advanced Usage <advanced_usage.html>`_).
The second way is to use ready-made scripts via the command line.

Anyway, to use both methods, you need to do a few general steps:

#. Your network class must be described in a separate file in which the following variables must also be declared:

    - variable storing the number of channels of the image fed to the network;
    - variable storing the size of the image fed to the network;
    - the torchvision.transforms module object, which represents a transformation over images.

   Examples of network descriptions are given in the `main_net_classes <https://github.com/cais-lab/revelionn/tree/main/main_net_classes>`_ directory.

    .. note::

        The network class must be inherited from the ``nn.Module`` class, that is, your network must be implemented using PyTorch.


#. Prepare an image dataset according to the data format specification (see `Dataset Representation <data.html#dataset-representation>`_).

#. Prepare the dataset ontology according to the documentation instructions (see `Ontology Representation <data.html#ontology-representation>`_).
   Optional, but necessary to use concept extraction algorithms.

RevelioNN can interpret convolutional binary classification networks that have already been trained without using this
library. To do this, the specified model must be converted to RevelioNN format.

    .. note::

        To convert to the RevelioNN format, your model file should contain only ``state_dict``, which is typical for most cases.
