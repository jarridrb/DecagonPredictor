from src.Predictor.Predictor import Predictor
from src.Dtos.ModelType import ModelType

modelTypes = [ModelType.TrainedOnAll, ModelType.TrainedWithMask]

3126, 20456, 27947, 26780, 9193, 38019
relations = [
    'C0003126',
    'C0020456',
    'C0027947',
    'C0026780',
    'C0009193',
    'C0038019'
]

for modelType in modelTypes:
    for relation in relations:
        predictor = Predictor(modelType, relation)
        print('Model type: %s, Relation: %s, AUC: %f' % (modelType, relation, predictor.predict().auroc))

