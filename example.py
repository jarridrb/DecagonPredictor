from src.Predictor.Predictor import Predictor
from src.Dtos.ModelType import ModelType
from src.Predictor.EdgeAccessor import *
import sklearn

relations = [
    'C0003126',
    'C0020456',
    'C0027947',
    'C0026780',
    'C0009193',
    'C0038019'
]

# for relation in relations:
#     predictor = Predictor(ModelType.TrainedOnAll, relation)
#     print(
#         'Model type: %s, Relation: %s, AUC: %f' % (
#             ModelType.TrainedOnAll,
#             relation,
#             predictor.predict().auroc
#         )
#     )

# for relation in relations:
#     ea = AllEdgeAccessor(ModelType.TrainedOnAll, relation)
#     df = ea.get_edges_dataframe()
#
#     Z = df['node_embedding_matrix'].to_numpy()[0]
#     D = df['importance_matrix'].to_numpy()[0]
#     R = df['global_interaction_matrix'].to_numpy()[0]
#
#     labels = df["label"].to_numpy()[0]
#     reindex = np.argsort(df["from_node_index"].to_numpy()[0] * Z.shape[0] + df["to_node_index"].to_numpy()[0])
#     labels = labels[reindex]
#     is_train = np.asarray(df["is_train"].to_numpy()[0][reindex], dtype=np.bool)
#
#     np.savez(relation + ".npz", Z=Z, D=D, R=R, labels=labels, is_train=is_train)
    #df.to_pickle(relation + ".pkl")


df = np.load(relations[0] + ".npz")


print(df["Z"])

# make predictions
# print(Z.shape, D.shape, R.shape)
# preds = (Z @ D @ R @ D @ Z.T).reshape(-1)
# test_labels = labels[~is_train]
# test_preds = preds[~is_train]
# print("AUC", sklearn.metrics.roc_auc_score(test_labels, test_preds))

#print("num train/num_test", is_train.sum(), (~is_train).sum())
#print("train/test positive edges", labels[is_train].sum(), labels[~is_train].sum())


# from torch.utils.data import dataset, dataloader
#
# print(dataset)
#
# class EdgeEmbeddingDataset:
#
#     def __init__(self):
#         self.embeddings = np.random.normal(size=[1000,10])
#
#     def __getitem__(self, index):
#         return self.embeddings[index]
#
#     def __len__(self):
#         return self.embeddings.shape[0]
#
# data = EdgeEmbeddingDataset()



# DataLoader:
    # give_index()
    # sample(label_idx)
    # take, Zi, Zj
    # [Zi, Zj, label)

# AUC 0.898
# todo: train_loader, test_loader