from .PredictionsInfoHolder import PredictionsInfoHolder
from ..Dtos.PredictionsInformation import PredictionsInformation
from ..Dtos.ModelType import ModelType

from threading import Lock
import pandas as pd
import numpy as np
import sklearn.metrics

class Predictor:
    def __init__(self, modelType: ModelType, relationId: str) -> None:
        self.modelType: ModelType = modelType

        predsInfoHolder = PredictionsInfoHolder.getInstance()

        self.defaultImportanceMtx = \
            predsInfoHolder.modelInfos[self.modelType].importanceMatrices[relationId]

        baseTestEdges = predsInfoHolder.testEdgeDict[self.modelType][relationId]
        self.negTestEdges = baseTestEdges[baseTestEdges[:, 2] == 0]
        self.posTestEdges = baseTestEdges[baseTestEdges[:, 2] == 1]

    def predict(self, importance_matrix=None) -> PredictionsInformation:
        LABEL_IDX         = 2
        PROBABILITIES_IDX = 4

        ndarrayResults = self._predict(importance_matrix)

        labels = ndarrayResults[:, LABEL_IDX]
        probabilities = ndarrayResults[:, PROBABILITIES_IDX]

        auroc = sklearn.metrics.roc_auc_score(labels, probabilities)
        auprc = sklearn.metrics.average_precision_score(labels, probabilities)
        confusionMtx = sklearn.metrics.confusion_matrix(
            labels,
            np.round(probabilities)
        )

        return PredictionsInformation(
            probabilities=probabilities,
            labels=labels,
            auroc=auroc,
            auprc=auprc,
            confusionMatrix=confusionMtx,
        )

    def predict_as_dataframe(self, importance_matrix=None) -> pd.DataFrame:
        FROM_NODE_IDX     = 0
        TO_NODE_IDX       = 1
        LABEL_IDX         = 2
        PROBABILITIES_IDX = 3

        ndarrayResults = self._predict(importance_matrix)

        fromEmbeddings = self._getColEmbeddings(ndarrayResults[:, FROM_NODE_IDX].astype(np.int32))
        toEmbeddings = self._getColEmbeddings(ndarrayResults[:, TO_NODE_IDX].astype(np.int32))

        impMtx = importance_matrix if importance_matrix else self.defaultImportanceMtx

        predsInfoHolder = PredictionsInfoHolder.getInstance()
        interactionMtx = predsInfoHolder.modelInfos[self.modelType].globalInteractionMtx

        return pd.DataFrame().append({
            'FromEmbeddings': fromEmbeddings,
            'ToEmbeddings': toEmbeddings,
            'Labels': ndarrayResults[:, LABEL_IDX],
            'Probabilities': ndarrayResults[:, PROBABILITIES_IDX],
            'GlobalInteractionMatrix': interactionMtx,
            'ImportanceMatrix': impMtx,
        }, ignore_index=True)

    def _predict(self, importance_matrix=None) -> np.ndarray:
        importanceMtx = self.defaultImportanceMtx
        if importance_matrix is not None:
            importanceMtx = importance_mtx

        negEdgePreds = self._predictEdges(importanceMtx, self.negTestEdges, 0)
        posEdgePreds = self._predictEdges(importanceMtx, self.posTestEdges, 1)

        return np.vstack([negEdgePreds, posEdgePreds])

    def _predictEdges(self, importanceMtx, edges, label):
        FROM_EDGE_IDX = 0
        TO_EDGE_IDX   = 1
        COL_SHAPE_IDX = 1

        modelInfos = PredictionsInfoHolder.getInstance().modelInfos[self.modelType]

        globalInteractionMtx = modelInfos.globalInteractionMtx

        colEmbeddings = modelInfos.embeddings
        rowEmbeddings = modelInfos.embeddings.T

        rawPreds = colEmbeddings @ importanceMtx @ globalInteractionMtx @ importanceMtx @ rowEmbeddings
        probabilities = self._sigmoid(rawPreds)

        sampledProbabilities = self._getSampledPredictions(
            probabilities,
            edges,
            probabilities.shape[COL_SHAPE_IDX]
        )

        return np.hstack([edges, sampledProbabilities.reshape(-1, 1)])

    def _getSampledPredictions(
        self,
        predictions: np.ndarray,
        edgeSamples: np.ndarray,
        colShape: int
    ) -> np.ndarray:
        linearEdgeIdxs = (edgeSamples[:, 0] * colShape) + edgeSamples[:, 1]
        return np.take(predictions, linearEdgeIdxs)

    def _getRowEmbeddings(self, edgeIdxs) -> np.ndarray:
        predsInfoHolder = PredictionsInfoHolder.getInstance()
        return predsInfoHolder.modelInfos[self.modelType].embeddings[edgeIdxs].T

    def _getColEmbeddings(self, edgeIdxs) -> np.ndarray:
        predsInfoHolder = PredictionsInfoHolder.getInstance()
        return predsInfoHolder.modelInfos[self.modelType].embeddings[edgeIdxs]

    def _sigmoid(self, vals):
        return 1. / (1 + np.exp(-vals))

if __name__ == '__main__':
    predictor = Predictor(ModelType.TrainedOnAll, 'C0003126')
    x = predictor.predict_as_dataframe()
    print(x)

    edgeIter = TrainingEdgeIterator(ModelType.TrainedOnAll, 'C0003126')
    edgeIter.get_train_edges_as_embeddings()

