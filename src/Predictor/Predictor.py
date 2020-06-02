from ..Dtos.PredictionsInformation import PredictionsInformation
from ..Dtos.ModelType import ModelType
from ..Dtos.NodeLists import NodeLists
from ..Dtos.NodeIds import DrugId, SideEffectId
from ..Dtos.SavedModelInfo import SavedModelInfo
from ..Utils.ArgParser import ArgParser
from ..Utils.Config import Config

from typing import Type, Dict, Tuple
from threading import Lock
import pandas as pd
import numpy as np
import sklearn.metrics
import csv
import sys
import os

config = Config.getConfig()

# Will be set in the NpPredictor class
predsInfoHolder = None
predsInfoHolderLock = Lock()

# Shortcut for typing
NpzFile = np.lib.npyio.NpzFile

# Internal class
# This should only be instantiated once
class _PredictionsInfoHolder:
    def __init__(self):
        self.modelInfos: Dict[ModelType, SavedModelInfo] = self._getSavedModelInfos()
        self.testEdgeDict: Dict[ModelType, NpzFile] = self._buildTestEdgeDict()
        self.trainEdgeDict: Dict[ModelType, NpzFile] = self._buildTrainEdgeDict()

    def _getSavedModelInfos(self) -> Dict[ModelType, SavedModelInfo]:
        result = {}

        for modelType, saveDir in self._getModelTypeSaveDirs().items():
            result[modelType] = SavedModelInfo(
                embeddings=np.load(saveDir + 'embeddings.npy'),
                importanceMatrices=np.load(saveDir + 'EmbeddingImportance.npz'),
                globalInteractionMtx=np.load(saveDir + 'GlobalRelations.npy')
            )

        return result

    def _getModelTypeSaveDirs(self) -> Dict[ModelType, str]:
        global config

        trainedAllDirName = config.getSetting(
            'TrainedOnAllSavedNdarrayPath',
            default='data/ModelSpecificData/TrainedOnAll/'
        )

        trainedMaskedDirName = config.getSetting(
            'TrainedWithMaskSavedNdarrayPath',
            default='data/ModelSpecificData/TrainedWithMasks/'
        )

        return {
            ModelType.TrainedOnAll: trainedAllDirName,
            ModelType.TrainedWithMask: trainedMaskedDirName,
        }

    def _buildTestEdgeDict(self) -> Dict[ModelType, NpzFile]:
        return {
            modelType: np.load(saveDir + 'TestEdges.npz')
            for modelType, saveDir in self._getModelTypeSaveDirs().items()
        }

    def _buildTrainEdgeDict(self) -> Dict[ModelType, NpzFile]:
        return {
            modelType: np.load(saveDir + 'TrainEdges.npz')
            for modelType, saveDir in self._getModelTypeSaveDirs().items()
        }

class TrainingEdgeIterator:
    def __init__(self, modelType: ModelType, relationId: str) -> None:
        self._initGlobalInfosHolderIfNeeded()

        self.modelType: ModelType = modelType
        self.relationId: str = relationId

    def _initGlobalInfosHolderIfNeeded(self) -> None:
        global predsInfoHolder
        global predsInfoHolderLock
        if predsInfoHolder is None:
            predsInfoHolderLock.acquire()
            if predsInfoHolder is None:
                predsInfoHolder = _PredictionsInfoHolder()

            predsInfoHolderLock.release()

    # Returns 3-dim ndarray where the first column is the from node,
    # the second column is the to node, and the third column is the edge label
    def get_train_edges(self) -> np.ndarray:
        global predsInfoHolder
        return predsInfoHolder.trainEdgeDict[self.modelType][self.relationId]

    def get_train_edges_as_embeddings(self) -> np.ndarray:
        FROM_NODE_IDX = 0
        TO_NODE_IDX   = 1
        LABELS_IDX    = 2

        global predsInfoHolder
        raw = predsInfoHolder.trainEdgeDict[self.modelType][self.relationId].astype(np.int32)

        modelInfos = predsInfoHolder.modelInfos[self.modelType]
        fromEmbeddings = np.squeeze(modelInfos.embeddings[raw[:, FROM_NODE_IDX]])
        toEmbeddings = np.squeeze(modelInfos.embeddings[raw[:, TO_NODE_IDX]])

        result = np.empty((fromEmbeddings.shape[0], 32, 32, 1))
        result[:, 0, :, 0] = fromEmbeddings
        result[:, :, 0, 0] = toEmbeddings
        result[:, 0, 0, :] = raw[:, LABELS_IDX].reshape(-1, 1)

        return result

    def get_train_edges_as_embeddings_df(self) -> pd.DataFrame:
        FROM_NODE_IDX = 0
        TO_NODE_IDX   = 1
        LABELS_IDX    = 2

        global predsInfoHolder
        raw = predsInfoHolder.trainEdgeDict[self.modelType][self.relationId].astype(np.int32)

        modelInfos = predsInfoHolder.modelInfos[self.modelType]
        fromEmbeddings = np.squeeze(modelInfos.embeddings[raw[:, FROM_NODE_IDX]])
        toEmbeddings = np.squeeze(modelInfos.embeddings[raw[:, TO_NODE_IDX]])

        return pd.DataFrame().append({
            'FromEmbeddings': fromEmbeddings,
            'ToEmbeddings': toEmbeddings,
            'Labels': raw[:, LABELS_IDX],
            'GlobalInteractionMatrix': modelInfos.globalInteractionMtx,
        }, ignore_index=True)

class Predictor:
    def __init__(self, modelType: ModelType, relationId: str) -> None:
        self._initGlobalInfosHolderIfNeeded()
        self.modelType: ModelType = modelType

        global predsInfoHolder

        self.defaultImportanceMtx = \
            predsInfoHolder.modelInfos[self.modelType].importanceMatrices[relationId]

        baseTestEdges = predsInfoHolder.testEdgeDict[self.modelType][relationId]
        self.negTestEdges = baseTestEdges[baseTestEdges[:, 2] == 0]
        self.posTestEdges = baseTestEdges[baseTestEdges[:, 2] == 1]

    def _initGlobalInfosHolderIfNeeded(self) -> None:
        global predsInfoHolder
        global predsInfoHolderLock
        if predsInfoHolder is None:
            predsInfoHolderLock.acquire()
            if predsInfoHolder is None:
                predsInfoHolder = _PredictionsInfoHolder()

            predsInfoHolderLock.release()

    def predict(self, importance_matrix=None) -> PredictionsInformation:
        LABEL_IDX         = 2
        PROBABILITIES_IDX = 3

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
        global predsInfoHolder
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

        global predsInfoHolder
        modelInfos = predsInfoHolder.modelInfos[self.modelType]

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
        global predsInfoHolder
        return predsInfoHolder.modelInfos[self.modelType].embeddings[edgeIdxs].T

    def _getColEmbeddings(self, edgeIdxs) -> np.ndarray:
        global predsInfoHolder
        return predsInfoHolder.modelInfos[self.modelType].embeddings[edgeIdxs]

    def _sigmoid(self, vals):
        return 1. / (1 + np.exp(-vals))

if __name__ == '__main__':
    predictor = Predictor(ModelType.TrainedOnAll, 'C0003126')
    x = predictor.predict_as_dataframe()
    print(x)

    edgeIter = TrainingEdgeIterator(ModelType.TrainedOnAll, 'C0003126')
    edgeIter.get_train_edges_as_embeddings()

