# DecagonPredictor

This repository maintains code which enables simple link prediction on top of pretrained
matrices from Decagon, along with an iterator for getting all training edges for a given
side effect.  These interfaces may be used for other experiments, as these pretrained
matrices already give relatively good performance for link prediction on a variety of 
drug-drug interaction side effects.

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

`predict(importance_matrix=None)`: Computes the tensor decomposition given above for the `ModelType`
and side effect ID with which the `Predictor` was defined.  By default, this method predicts using
the importance matrix D_r obtained by the Decagon training procedure.  However, one may pass in a
non-null valued `importance_matrix` and this will be used as D_r in the prediction instead.  This
method returns a `PredictionsInformation` object, which is defined in 
[PredictionsInformation.py](src/Dtos/PredictionsInformation.py).

`predict_as_dataframe(importance_matrix=None)`: This behaves almost identically to `predict`, but
returns data as a pandas DataFrame instead of a `PredictionsInformation` object.  This `DataFrame`
is a series with columns `FromEmbeddings`, `ToEmbeddings`, `Labels`, `Probabilities`, 
`GlobalInteractionMatrix`, and `ImportanceMatrix`.  As above, if `importance_matrix` is passed as
`None`, then the pretrained version of D_r is used.  Otherwise, a user-defined version of D_r is used.

### TrainingEdgeIterator

