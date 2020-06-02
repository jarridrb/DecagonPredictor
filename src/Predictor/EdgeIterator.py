from .PredictionsInfoHolder import PredictionsInfoHolder
from ..Dtos.ModelType import ModelType

from typing import Iterator
from abc import ABCMeta, abstractmethod
import numpy as np

class EdgeAccessor(metaclass=ABCMeta):
    def __init__(self, modelType: ModelType, relationId: str) -> None:
        # The entire ndarray for edges
        self.edges: np.ndarray = self._getEdges(modelType, relationId)
        self.modelInfos = PredictionsInfoHolder.getInstance().modelInfos[modelType]

    @abstractmethod
    def _getEdges(self, modelType: ModelType, relationId: str) -> np.ndarray:
        pass

    def get_edges(self) -> np.ndarray:
        '''
        Resulting ndarray has its first column as the from node index,
        its second column as the to node index, and the third column as its
        labels
        '''
        return self.edges

    def get_edges_iterator(self) -> Iterator[np.ndarray]:
        '''
        Resulting ndarrays have their first column as the from node index,
        their second columns as the to node index, and their third columns
        as its labels
        '''
        for i in range(self.edges.shape[0]):
            yield self.edges[i]

    def get_edges_as_embeddings(self) -> np.ndarray:
        '''
        Resulting ndarray has its first column as the from embeddings,
        its second column as the to embeddings, and the third column as its
        labels
        '''
        FROM_NODE_IDX = 0
        TO_NODE_IDX   = 1
        LABELS_IDX    = 2

        raw = self.get_edges().astype(np.int32)

        fromEmbeddings = np.squeeze(self.modelInfos.embeddings[raw[:, FROM_NODE_IDX]])
        toEmbeddings = np.squeeze(self.modelInfos.embeddings[raw[:, TO_NODE_IDX]])

        result = np.empty((fromEmbeddings.shape[0], 32, 32, 1))
        result[:, 0, :, 0] = fromEmbeddings
        result[:, :, 0, 0] = toEmbeddings
        result[:, 0, 0, :] = raw[:, LABELS_IDX].reshape(-1, 1)

        return result

    def get_edges_as_embeddings_iterator(self) -> Iterator[np.ndarray]:
        '''
        Resulting ndarrays have as their first column the from embedding,
        their second column the to embedding, and the third column their
        label
        '''
        allEdges = self.get_edges_as_embeddings()
        for i in range(allEdges.shape[0]):
            yield allEdges[i]

class AllEdgeAccessor(EdgeAccessor):
    def _getEdges(self, modelType: ModelType, relationId: str) -> np.ndarray:
        predsInfoHolder = PredictionsInfoHolder.getInstance()

        trainEdges = predsInfoHolder.trainEdgeDict[modelType][relationId]
        testEdges  = predsInfoHolder.testEdgeDict[modelType][relationId]

        return np.vstack([trainEdges, testEdges])

class TrainEdgeAccessor(EdgeAccessor):
    def _getEdges(self, modelType: ModelType, relationId: str) -> np.ndarray:
        predsInfoHolder = PredictionsInfoHolder.getInstance()
        return predsInfoHolder.trainEdgeDict[modelType][relationId]

class TestEdgeAccessor(EdgeAccessor):
    def _getEdges(self, modelType: ModelType, relationId: str) -> np.ndarray:
        predsInfoHolder = PredictionsInfoHolder.getInstance()
        return predsInfoHolder.testEdgeDict[modelType][relationId]

