---
title: '`pyssam` -- a Python library for statistical modelling of biomedical shape and appearance'
tags:
  - Python
  - Bioinformatics
  - Medical image processing
  - Statistics
authors:
  - name: Josh Williams
    corresponding: true
    affiliation: 1
  - name: Ali Ozel
    affiliation: 1
  - name: Uwe Wolfram
    affiliation: 1
affiliations:
 - name: School of Engineering and Physical Sciences, Heriot-Watt University, Edinburgh, UK
   index: 1
date: 10 January 2023
bibliography: paper.bib
---

# Summary

`pyssam` is a Python library for creating statistical shape and appearance models (SSAMs) for biological (and other) shapes such as bones, lungs or other organs.
A point cloud best describing the anatomical 'landmarks' of the organ are required from each sample in a small population as an input. 
Additional information such as landmark gray-value can be included to incorporate joint correlations of shape and 'appearance' into the model.
Our library performs alignment and scaling of the input data and creates a SSAM based on covariance across the population.
The output SSAM can be used to parameterise and quantify shape change across a population.
`pyssam` is a small and low dependency codebase with examples included as Jupyter notebooks for several common SSAM computations.
The given examples can easily be extended to alternative datasets, and also alternative tasks such as medical image segmentation by incorporating a SSAM as a constraint for segmented organs.

# Statement of need
Statistical shape (and appearance) models (SSAMs) have drawn significant interest in biomedical engineering and computer vision research due to their ability to automatically deduce a linear parameterisation of shape covariances across a small population of training data [@cootes1995active; @heimann2009statistical; @baka20112D3D; @vaananen2015generation].
The classic statistical shape model (SSM) approach uses a point cloud of landmarks which are in correspondence across several instances of a shape.
The covariances of how the shape changes across the training population are computed, and principal component analysis (PCA) is used to parameterise the different modes of shape variation [@cootes1995active].
This approach paved the way for automatic algorithms which could significantly aid medical image segmentation (similar to an atlas) [@irving2011segmentation], characterise how the organ shape varies over a population as a diagnostic tool [@osanlouy2020lung], or even reconstruct a full 3D structure from a sparser imaging modality such as planar X-ray images [@baka20112D3D; @vaananen2015generation].

We have found that available open-source toolkits such as Statismo and Scalismo [@luthi2012statismo] suffer from an exhaustive number of dependencies and are difficult to adapt to new tasks, datasets and I/O datatypes.
ShapeWorks [@cates2017shapeworks] is another strongly developed library for statistical shape modelling, but it uses an alternative method of establishing correspondence between samples (a so-called particle-based method) which is less broadly used and more complex than a landmark-based system (where landmarks can be defined in any desired way for different anatomical shapes).
Additionally, as the machine learning ecosystem has strong foundations in Python, building statistical models in C++, Scala or other languages reduces compatibility with the majority of modern machine learning developments [@bhalodia2018deepssm].
We therefore implemented a lightweight Python framework for SSAMs which is easily adaptable with few dependencies, making it suitable for integrating as part of a broader codebase, as well as installing and running on high-performance computing clusters where users do not have root access to install many dependencies. 
We provide Jupyter notebooks on [readthedocs](https://pyssam.readthedocs.io/en/latest/) and two example datasets that allow users new to coding or SSAMs to learn how these models work in an interactive way to ease access when learning a new research topic and library.

# Overview

The main modelling classes are built on the abstract base class `StatisticalModelBase`, which has several methods for pre-processing data and performing PCA (\autoref{fig:code}).
There are also several global variables that are inherited which are related to principal components, component variances and model parameters.
The classes for `SSM` and `SAM` pre-process the data (align to zero mean and standard deviation of one) and can compute the population mean shape/appearance.
Finally, the `SSAM` class for shape and appearance modelling inherits all of these, but also imports the `SSM` and `SAM` methods to pre-process shape and appearance features separately, before they are merged into one dataset for modelling.

![Schematic overview of the codebase. Each modelling class is abstracted from the `StatisticalModelBase` class and contains several inherited variables such as model weights and principal components. The `SSAM` class inherits from `StatisticalModelBase`, but also uses pre-processing pipelines from `SSM` and `SAM`.\label{fig:code}](figures/code-schematic.pdf){ width=100% }


\section{Examples}
Here we present two example applications of `pyssam`. 
The first example examines shape variations in a toy dataset created for this study, which has a tree structure.
Tree structures appear often in biology, including the lung airways and vascular system. 
Toy datasets such as these are a simple means to visualise and interpret the modelling and code framework.
We then provide a more complex example which considers the left lower lobe of human lungs obtained from CT data [@tang2019automatic].
This example considers shape and appearance, where the appearance is the gray-value at the landmark location on an X-ray projection (obtained with the `AppearanceFromXray` helper class).

\subsection{Statistical shape modelling toy dataset} \label{sec:tree}
To understand the shape modelling process, we have provided a dataset class called `Tree` which creates a number of tree shapes which are randomly computed based on global minimum and maximum values for angle and branch length ratio (between parent and child).
Tree parameters are shown in \autoref{fig:tree}a.
Tree nodes are converted to a numpy array and used to initialise `pyssam.SSM`.
At initialisation of the `SSM` class, the landmarks are aligned, scaled to unit standard deviation and stacked into a matrix of shape $(N_f, 3N_L)$ where $N_f$ is the number of features (samples in our training dataset) and $N_L$ is the number of landmarks (each with a $x,y,z$ coordinates).
All $y$ coordinates in this case are zero, meaning the data is actually 2D but we preserve a 3D coordinate system for simplicity in generalising the code to more common 3D applications.
The code below shows how we can simply obtain a SSM from a set of landmarks.

```python
from glob import glob
import numpy as np
import pyssam

tree_class = pyssam.datasets.Tree(num_extra_ends=1)
landmark_coordinates = np.array(
    [tree_class.make_tree_landmarks() for i in range(0, num_samples)]
)

ssm_obj = pyssam.SSM(landmark_coordinates)
ssm_obj.create_pca_model(ssm_obj.landmarks_scale)
mean_shape_columnvector = ssm_obj.compute_dataset_mean()
```

![Overview of tree dataset population. Panels show (a) a visualisation of 100 tree samples, and (b) cumulative variance versus the number of PCA components constructed by the statistical shape model. Inset of (a) shows a legend describing the morphological parameters varied to create the tree dataset. These parameters include the initial branch length, $L_1$, the branch length ratio $L_R = L_2/L_1$, and branching angle $\theta$.\label{fig:tree}](figures/figure2-tree-example.pdf){ width=100% }

\subsection{Shape and appearance modelling of lung shape and chest X-ray images}

In the following example, we show a real application where 3D landmark for the left lower lung lobe are projected onto digitally reconstructed X-rays [@vaananen2015generation] and the gray-value is used to obtain appearance.
Example landmark data was obtained using an automatic algorithm [@ferrarini2007games].
Appearance information is extracted from the X-ray images using `AppearanceFromXray` (part of `pyssam.utils`).
We use landmarks, X-ray images as well as origin and pixel spacing information for the X-ray images to extract appearance as follows

```python
appearance_xr = pyssam.AppearanceFromXray(
    IMAGE_DATASET, IMAGE_ORIGIN, IMAGE_SPACING
)
appearance_values = appearance_xr.all_landmark_density(
    landmarks_coordinates
)
```

The SSAM can then be trained in a similar way as the SSM in \autoref{sec:tree} with the following code snippet:

```python
ssam_obj = pyssam.SSAM(landmark_coordinates, appearance_values)
ssam_obj.create_pca_model(ssam_obj.shape_appearance_columns)
mean_shape_appearance_columnvector = ssam_obj.compute_dataset_mean()
```

The shape and appearance modes can then be computed based on the model parameters (`ssam.model_parameters`). 
The computed model parameters (eigenvectors and eigenvalues of the covariance matrix) can be used to morph the shape and appearance using `ssam.morph_model` (part of `StatisticalModelBase` in \autoref{fig:code}) by 

\begin{equation}\label{eq:ssm}
\boldsymbol{x} \approx \bar{\boldsymbol{x}} + \boldsymbol{\Phi} \cdot \boldsymbol{b}  
\end{equation}

where $\boldsymbol{x}$ is a new array containing shape and appearance, $\bar{\boldsymbol{x}}$ is the training dataset mean shape and appearance, $\boldsymbol{\Phi}$ is the model principal components (eigenvectors of the training data covariance matrix), $\boldsymbol{b}$ is the model parameters, which is an array of weights unique to each data sample.
The model parameter a mode $m$ should be within $[-3\sqrt{\boldsymbol{\sigma_m^2}}, 3\sqrt{\boldsymbol{\sigma_m^2}}]$, where $\sigma_m^2$ is the explained variance of $m$ ($m^{th}$ largest eigenvalue of the covariance matrix) [@cootes1995active]. 

Each mode of shape and appearance variation is visualised, as shown for a representative mode in \autoref{fig:lungSSAM}.
This shows how lung shape influences the gray-value of lung pixels on the X-ray image. 
In this case, the change in shape and appearance are mainly due to how the lung interacts with adjacent structures such as the heart, rib cage and diaphragm.

![First mode of SSAM variation for lung lobe dataset. Panels show shape and appearance morphed using `ssam.morph_model` method and varying the model parameters (`ssam.model_parameters`), from -2, 0 (mean shape) and 2.\label{fig:lungSSAM}](figures/figure3-300.png){ width=100% }

\section*{Acknowledgement}
JW was funded by a 2019 PhD Scholarship from the Carnegie-Trust for the Universities of Scotland. 

\bibliography{pyssam_refs}


# References

[def]: figures/code-schematic.pdf