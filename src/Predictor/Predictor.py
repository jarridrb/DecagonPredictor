from ..DataSetParsers.DecagonPublicDataAdjacencyMatricesBuilder import DecagonPublicDataAdjacencyMatricesBuilder
from ..DataSetParsers.DecagonPublicDataNodeListsBuilder import DecagonPublicDataNodeListsBuilder
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

# Internal class
# This should only be instantiated once
class _PredictionsInfoHolder:
    def __init__(self):
        #self.modelInfos: Dict[ModelType, SavedModelInfo] = self._getSavedModelInfos()
        self.nodeLists: NodeLists = self._getNodeLists()
        self.drugIdToIdx: Dict[str, int] = {
            DrugId.toDecagonFormat(drugId): idx
            for idx, drugId in enumerate(self.nodeLists.drugNodeList)
        }

        self.testEdgeDict: Dict = self._buildTestEdgeDict()
        self.trainEdgeDict: Dict = self._buildTrainEdgeDict()

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

    def _getNodeLists(self) -> NodeLists:
        listBuilder = DecagonPublicDataNodeListsBuilder(config)
        return listBuilder.build()

    def _buildTestEdgeDict(self) -> Dict:
        result = {}

        testEdgeReader = self._getTestEdgeReader()
        for row in testEdgeReader:
            if not self._isRowValid(row):
                continue

            fromNodeIdx = self.drugIdToIdx[row['FromNode']]
            toNodeIdx = self.drugIdToIdx[row['ToNode']]

            newArr = np.array([fromNodeIdx, toNodeIdx, int(row['Label'])])

            relId = row['RelationId']
            if relId not in result:
                result[relId] = newArr.reshape((1, -1))
            else:
                result[relId] = np.vstack([result[relId], newArr])

        return result

    def _isRowValid(self, row):
        def _isDrugNode(strVal: str) -> bool:
            return strVal[:3] == 'CID'

        return _isDrugNode(row['FromNode']) and _isDrugNode(row['ToNode'])

    def _buildTrainEdgeDict(self) -> None:
        result = {}

        # Define indices here to not redefine it a bunch
        indices = None
        for relId, mtx in self._getDrugDrugMtxs().items():
            if indices is None:
                indices = self._getIndices(mtx.shape)

            relId = SideEffectId.toDecagonFormat(relId)
            trainEdgeIdxs = self._getTrainEdgeIdxs(indices, relId, mtx.shape)
            trainEdgeLabels = self._getTrainEdgeLabels(mtx, trainEdgeIdxs)

            intermed = np.hstack([trainEdgeIdxs, trainEdgeLabels])
            if relId in self.testEdgeDict:
                result[relId] = np.hstack([trainEdgeIdxs, trainEdgeLabels])
            else:
                self._assignNewEdges(result, relId, intermed)

        return result

    def _getIndices(self, shape) -> np.ndarray:
        xx, yy = np.indices(shape)
        return np.dstack([xx, yy]).reshape((-1, 2))

    def _getTrainEdgeIdxs(
        self,
        indices: np.ndarray,
        relId: str,
        mtxShape: Tuple[int, int]
    ) -> np.ndarray:
        # Get test edges and remove that labels from indices (slice of :2)
        if not relId in self.testEdgeDict:
            return indices

        testEdges = self.testEdgeDict[relId][:, :2] if relId in self.testEdgeDict else np.empty((0,2))

        indicesLinear   = (indices[:, 0] * mtxShape[1]) + indices[:, 1]
        testEdgesLinear = (testEdges[:, 0] * mtxShape[1]) + testEdges[:, 1]

        trainEdges = np.setdiff1d(indicesLinear, testEdgesLinear)

        return np.dstack(np.unravel_index(trainEdges, mtxShape)).reshape(-1, 2)

    def _getTrainEdgeLabels(self, mtx, edgeIdxs: np.ndarray) -> np.ndarray:
        idxsLinear = (edgeIdxs[:, 0] * mtx.shape[1]) + edgeIdxs[:, 1]
        return np.take(mtx.todense(), idxsLinear).T

    def _getDrugDrugMtxs(self):
        adjMtxBuilder = DecagonPublicDataAdjacencyMatricesBuilder(
            self.nodeLists,
            config,
        )

        return adjMtxBuilder.build().drugDrugRelationMtxs

    def _getTestEdgeReader(self) -> csv.DictReader:
        testEdgeFname = config.getSetting('TestEdgeFilename')
        return csv.DictReader(open(testEdgeFname))

    def _assignNewEdges(self, trainEdgeDict, relationId, allPossibleEdges) -> None:
        testEdgeIdxs = self._getNewTestEdgeIdxs(allPossibleEdges)

        allEdgeIdxs = np.arange(allPossibleEdges.shape[0])
        trainEdgeIdxs = np.setdiff1d(allEdgeIdxs, testEdgeIdxs)

        self.testEdgeDict[relationId] = allPossibleEdges[testEdgeIdxs]
        trainEdgeDict[relationId] = allPossibleEdges[trainEdgeIdxs]

    def _getNewTestEdgeIdxs(self, allPossibleEdges) -> np.ndarray:
        testEdgeProportion = 0.1

        negativeEdgeIdxs = np.where(allPossibleEdges[:, 2] == 0)[0]
        positiveEdgeIdxs = np.where(allPossibleEdges[:, 2] == 1)[0]

        numTestEdgesPerLabel = int(np.ceil(positiveEdgeIdxs.shape[0] * testEdgeProportion))

        negTestIdxs = negativeEdgeIdxs[np.random.choice(negativeEdgeIdxs.shape[0], numTestEdgesPerLabel)]
        posTestIdxs = positiveEdgeIdxs[np.random.choice(positiveEdgeIdxs.shape[0], numTestEdgesPerLabel)]

        return np.hstack([negTestIdxs, posTestIdxs])

class TrainingEdgeIterator:
    def __init__(self, relationId: str) -> None:
        self._initGlobalInfosHolderIfNeeded()

        self.relationId = relationId

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
        return predsInfoHolder.trainEdgeDict[self.relationId]

    def get_train_edges_as_embeddings(self) -> np.ndarray:
        FROM_NODE_IDX = 0
        TO_NODE_IDX   = 1
        LABELS_IDX    = 2

        global predsInfoHolder
        raw = predsInfoHolder.trainEdgeDict[self.relationId].astype(np.int32)

        modelInfos = predsInfoHolder.modelInfos[self.modelType]
        fromEmbeddings = np.squeeze(modelInfos.embeddings[raw[:, FROM_NODE_IDX]])
        toEmbeddings = np.squeeze(modelInfos.embeddings[raw[:, TO_NODE_IDX]])

        result = np.empty((fromEmbeddings.shape[0], 32, 32, 1))
        result[:, 0, :, 0] = fromEmbeddings
        result[:, :, 0, 0] = toEmbeddings
        result[:, 0, 0, :] = raw[:, LABELS_IDX]

        return result

    def get_train_edges_as_embeddings_df(self) -> pd.DataFrame:
        FROM_NODE_IDX = 0
        TO_NODE_IDX   = 1
        LABELS_IDX    = 2

        global predsInfoHolder
        raw = predsInfoHolder.trainEdgeDict[self.relationId].astype(np.int32)

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
    def __init__(self, relationId: str) -> None:
        self._initGlobalInfosHolderIfNeeded()

        global predsInfoHolder

        self.defaultImportanceMtx = \
            predsInfoHolder.modelInfos[self.modelType].importanceMatrices[relationId]

        baseTestEdges = predsInfoHolder.testEdgeDict[relationId]
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
        interactionMtx = predsInfoHolder.modelInfos[self.modelType].globalInteraction

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

def _write_as_parquet(relations_to_write):
    for relationId in tqdm(relations_to_write):
        edgeIterator = TrainingEdgeIterator(relationId)

        df = edgeIterator.get_train_edges_as_embeddings_df()

        fname = os.getcwd() + '/train-edges-%s.pkl.gzip' % relationId
        df.to_pickle(fname, compression='gzip')

if __name__ == '__main__':
    infoHolder = _PredictionsInfoHolder()

    import pdb; pdb.set_trace()
    np.savez_compressed('TrainEdges', **infoHolder.trainEdgeDict)
    np.savez_compressed('TestEdges', **infoHolder.testEdgeDict)

