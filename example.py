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
    print(
        'Model type: %s, Relation: %s, AUC: %f' % (
            ModelType.TrainedOnAll,
            relation,
            predictor.predict().auroc
        )
    )

