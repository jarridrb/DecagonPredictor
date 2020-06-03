from ..Dtos.ModelType import ModelType
from ..Dtos.SavedModelInfo import SavedModelInfo
from ..Utils.Config import Config

from typing import Type, Dict
from threading import Lock
import numpy as np

_instance: Type['_PredictionsInfoHolder'] = None
_instanceLock = Lock()

# Shortcut for typing
NpzFile = np.lib.npyio.NpzFile

# Internal class
# This should only be instantiated once
class PredictionsInfoHolder:
    @staticmethod
    def getInstance() -> Type['PredictionsInfoHolder']:
        global _instance
        global _instanceLock
        if _instance is None:
            _instanceLock.acquire()
            if _instance is None:
                _instance = PredictionsInfoHolder()

            _instanceLock.release()

        return _instance

    def __init__(self):
        self._config = Config.getConfig()
        self.modelInfos: Dict[ModelType, SavedModelInfo] = self._getSavedModelInfos()

        self.testEdgeDict: Dict[ModelType, NpzFile] = self._buildEdgeDict(isTrain=False)
        self.trainEdgeDict: Dict[ModelType, NpzFile] = self._buildEdgeDict(isTrain=True)

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
        trainedAllDirName = self._config.getSetting(
            'TrainedOnAllSavedNdarrayPath',
            default='data/ModelSpecificData/TrainedOnAll/'
        )

        trainedMaskedDirName = self._config.getSetting(
            'TrainedWithMaskSavedNdarrayPath',
            default='data/ModelSpecificData/TrainedWithMasks/'
        )

        return {
            ModelType.TrainedOnAll: trainedAllDirName,
            ModelType.TrainedWithMask: trainedMaskedDirName,
        }

    def _buildEdgeDict(self, isTrain: bool) -> Dict[ModelType, NpzFile]:
        loadFname = ''
        indicatorFxn = None
        if isTrain:
            loadFname = 'TrainEdges.npz'
            indicatorFxn = np.ones
        else:
            loadFname = 'TestEdges.npz'
            indicatorFxn = np.zeros

        result = {}
        for modelType, saveDir in self._getModelTypeSaveDirs().items():
            result[modelType] = {}
            for relationId, edges in np.load(saveDir + loadFname).items():
                result[modelType][relationId] = \
                    np.hstack([edges, indicatorFxn((edges.shape[0], 1))]).astype(np.int32)

        return result

