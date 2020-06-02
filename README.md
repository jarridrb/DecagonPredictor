# DecagonPredictor

This repository maintains code which enables simple link prediction on top of pretrained
matrices from Decagon, along with an iterator for getting all training edges for a given
side effect.  These interfaces may be used for other experiments, as these pretrained
matrices already give relatively good performance for link prediction on a variety of 
drug-drug interaction side effects.

## Installation
We now cover installation.  Installing this repository is quite simple.  To do so, one must run the following:

```
$ conda create --name decagon-predictor python=3.6
$ conda activate decagon-predictor
$
$ ./install.sh
```

This will create a new conda environment, install all requirements (from `requirements.txt`), and download
and do some small preprocessing on the data.  After this is done, the code should be ready to run.

## Repository Descriptions
Before proceeding, it will be useful to recall the approach Decagon takes for link
prediction.  Decagon uses a tensor decomposition framework, wherein to predict whether
drugs i and j will have some side effect r, Decagon evaluates the following matrix
product and puts the result through a sigmoid:

z_i D_r R D_r z_j^T

Here, z_i, z_j \in R^d are the embeddings for drugs i and j, where d is the size of the
last layer in Decagon (in our case here, d is 32).  Meanwhile, D_r \in R^{d \times d} is
the diagonal feature importance matrix for side effect r.  Finally, the matrix R is the global
interaction matrix which is learned through the convolutional implication of the GNN.  All of 
these items in some way exposed to the user, as is detailed below.

There are two classes which may be used by the end user.  Specifically, these are
the `Predictor` and the `TrainingEdgeIterator`.  Both of these are defined within the
file [Predictor.py](src/Predictor/Predictor.py).  


### Predictor
The `Predictor` serves to allow link prediction for a given side effect, optionally using a user-defined feature importance
matrix for the prediction computation.  A `Predictor` object is defined by a `ModelType` and
a side effect ID, as it will predict for this specific tuple.  The `Predictor` has three public methods.

- `predict(importance_matrix=None)`: Computes the tensor decomposition given above for the `ModelType`
and side effect ID with which the `Predictor` was defined.  By default, this method predicts using
the importance matrix D_r obtained by the Decagon training procedure.  However, one may pass in a
non-null valued `importance_matrix` and this will be used as D_r in the prediction instead.  This
method returns a `PredictionsInformation` object, which is defined in 
[PredictionsInformation.py](src/Dtos/PredictionsInformation.py).

- `predict_as_dataframe(importance_matrix=None)`: This behaves almost identically to `predict`, but
returns data as a pandas DataFrame instead of a `PredictionsInformation` object.  This `DataFrame`
is a series with columns `FromEmbeddings`, `ToEmbeddings`, `Labels`, `Probabilities`, 
`GlobalInteractionMatrix`, and `ImportanceMatrix`.  As above, if `importance_matrix` is passed as
`None`, then the pretrained version of D_r is used.  Otherwise, a user-defined version of D_r is used.

### Accessing Graph Edges
To do any sort of manipulation on top of these pretrained matrices, one must have some way to
access edges within the drug-drug interaction graph.  To that end, we expose various 
[EdgeAccessor](src/Predictor/EdgeAccessor.py) objects to support such finding of edges.
An `EdgeAccessor` is relative to a specific `ModelType` and `SideEffectId`, and so each
of these must be passed in on initialization of the `EdgeAccessor`.  Each edge returned
by said `EdgeAccessor` will then be relative to the specified `ModelType` and `SideEffectId`.

For utility, we expose three usable EdgeAccessors.  First, the `AllEdgeAccessor`, which
provides access to all edges within the graph.  Second, the `TrainEdgeAccessor`, which
provides access to all edges within the graph which were used in training of the `ModelType`.
And finally, the `TestEdgeAccessor`, which provides access to all edges within the graph which
were held out for the test set for the model of `ModelType`.

All `EdgeAccessor` objects provide the following methods

- `get_edges()`: This returns an ndarray with training edges.  In particular, this is an ndarray
with three columns.  The first of these is the from node index, the second is the to node index, and
the third column is the label of the given edge.

- `get_edges_iterator()`: This returns an iterator over the ndarray that would be returned by `get_edges()`.

- `get_edges_as_embeddings()`: This returns a 4 columned ndarray.  The first column represents
the number of edges in the train set total.  The second index is the pretrained embeddings of the from 
node, while the third index is the pretrained embeddings of the to node.  Finally, the fourth column
is the label of the training edge.

- `get_edges_as_embeddings_iterator()`: This returns an iterator over the ndarray that would be
returned by `get_edges_as_embeddings()`.

Using an `EdgeAccessor` is quite simple.  For example, if we wanted to iterate over all edges within
the drug-drug interaction graph for side effect C0003126 (this corresponds to Anosmia) and apply some 
procedure for each available edge, we could simply do the below.

```
from src.Predictor.EdgeAccessor import AllEdgeAccessor
from src.Dtos.ModelType import ModelType

edge_accessor = AllEdgeAccessor(ModelType.TrainedOnAll, 'C0003126')

# edge will be an ndarray of shape (3,) wherein the first component is the from
# node index, the second component is the to node index, and the third component
# is the label of that edge
for edge in edge_accessor.get_edges_iterator():
   # Do my cool procedure using edge!
   
```

Note here that had we instead wanted to iterate over only the edges used in training, or similarly only those
for testing, we would only need to change the assignment to `edge_accessor` as

```
edge_accessor = TrainEdgeAccessor(ModelType.TrainedOnAll, 'C0003126')
```

or

```
edge_accessor = TestEdgeAccessor(ModelType.TrainedOnAll, 'C0003126')
```

respectively.

### An Example Predictor Script
This framework can be run quite easily.  For example, if we wanted to do link prediction across some of the
6 best performing side effects through training, we could simply write the below.

```
from src.Predictor.Predictor import Predictor
from src.Dtos.ModelType import ModelType

relations = [
    'C0003126',
    'C0020456',
    'C0027947',
    'C0026780',
    'C0009193',
    'C0038019'
]

for relation in relations:
    predictor = Predictor(ModelType.TrainedOnAll, relation)
    print('Relation: %s, AUC: %f' % (relation, predictor.predict().auroc))

```

This script yields the following:

```
Relation: C0003126, AUC: 0.888447
Relation: C0020456, AUC: 0.789948
Relation: C0027947, AUC: 0.802926
Relation: C0026780, AUC: 0.871485
Relation: C0009193, AUC: 0.836454
Relation: C0038019, AUC: 0.910391
```
