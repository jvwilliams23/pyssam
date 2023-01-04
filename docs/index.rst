.. pyssam documentation master file, created by
   sphinx-quickstart on Sun Dec 11 15:39:23 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyssam documentation
====================
Welcome to the documentation for **pyssam**! This package aims to provide a 
lightweight and low-dependency option for statistical modelling in Python. 

Main applications of this package include statistical shape and appearance modelling,
which can be used to parameterise biomedical structures and quantify changes in
these complex datasets across a population.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API documentation:

   modules

Installation
------------
This repository is currently not listed on PyPi. Therefore, to install one must do ::

   pip install numpy scikit_learn scikit-image networkx
   python setup.py install

Several jupyter notebooks also exist, which require::

   pip install matplotlib

To check the package is installed properly can be done by typing the following 
in a bash terminal ::

   python -c "import pyssam ; print(pyssam.SSAM.__doc__)"

Extending to new datasets
-------------------------
Landmarks are required to apply pyssam to a new dataset. These can be obtained using a 
variety of methods as reviewed in Section 4 of 
`Heimann and Meinzer (2009) <https://doi.org/10.1016/j.media.2009.05.004>`_.
This can be quite task specific, therefore we have not included specific methods for this.
On tree-like structures, we have previously used nodal coordinates as landmarks as done in 
``pyssam.datasets.Tree``.
For modelling lung lobes, we have also used an automatic algorithm detailed by
`Ferrarini et al. (2007) <https://doi.org/10.1016/j.media.2007.03.006>`_, but the code is not 
provided here as it has many task-specific parameters.

A script for manually landmarking an input shape is given in ``scripts/click_landmarking.py``.
This uses ``vedo`` to visualise an input mesh and create landmarks by left click and pressing "c".
The aim of this script is to provide a simple way to landmark meshes before progressing to more advanced 
automatic algorithms.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   tutorial/ssm_example.ipynb
   tutorial/sam_example.ipynb
   tutorial/ssam_example.ipynb



