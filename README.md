# Graph Network applied to Organic Chemistry

* Trained on Tensorflow 2.15.0

## Contents
* [Repository Files](#repository-files)
* [Model](#model)
* [Results](#results)

## Repository Files

* graph_model.py - This file contains the graph model. 
* train_and_test.py - This file contains the preprocessing functions and the training and testing scripts.
* graph_weights.h5 - This file contains the trained weights of the model.

## Model
This model is a graph neural network that models the Blood-Brain Barrier Penetration dataset. The model uses customized graph convolution layers and customized graph attention layers. A key difference between the two layers is that in the graph convolution layer, the sum of the features of neighboring atoms is taken while in the graph attention layer, a <u>weighted sum</u> of features of neighboring atoms is taken.

For the custom graph convolution layers, ideas for aggregating the features of neighboring atoms was taken from the following paper:

```Bibtex
@misc{hamilton2018inductiverepresentationlearninglarge,
      title={Inductive Representation Learning on Large Graphs}, 
      author={William L. Hamilton and Rex Ying and Jure Leskovec},
      year={2018},
      eprint={1706.02216},
      archivePrefix={arXiv},
      primaryClass={cs.SI},
      url={https://arxiv.org/abs/1706.02216}, 
}
```

For the custom graph attention layers, steps for normalizing the attention weights and aggregating the weighted features of neighboring atoms were based on the following Keras Tutorial:

```Bibtex
keras.io/examples/graph/gat_node_classification/
```

For preprocessing the data, ideas were taken from the following Keras tutorial:

```Bibtex
keras.io/examples/graph/mpnn-molecular-graphs/
```

Reference to the BBBP dataset (Deepchem):

```Bibtex
@book{Ramsundar-et-al-2019,
    title={Deep Learning for the Life Sciences},
    author={Bharath Ramsundar and Peter Eastman and Patrick Walters and Vijay Pande and Karl Leswing and Zhenqin Wu},
    publisher={O'Reilly Media},
    note={\url{https://www.amazon.com/Deep-Learning-Life-Sciences-Microscopy/dp/1492039837}},
    year={2019}
}
```

## Results

A test dataset was created by taking one percent of the total data. The model was able to achieve a 95% binary accuracy on this test dataset.
