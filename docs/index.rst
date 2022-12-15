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

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   tutorial/ssm_example.ipynb
   tutorial/sam_example.ipynb
   tutorial/ssam_example.ipynb



