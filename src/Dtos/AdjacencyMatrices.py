from ..Dtos.TypeShortcuts import RelationIDToSparseMtx
import scipy.sparse as sp

class AdjacencyMatrices:
    def __init__(
        self,
        drugDrugRelationMtxs: RelationIDToSparseMtx,
        drugProteinRelationMtx: sp.csr_matrix,
        proteinProteinRelationMtx: sp.spmatrix
    ) -> None:
        self.drugDrugRelationMtxs: RelationIDToSparseMtx = drugDrugRelationMtxs
        self.drugProteinRelationMtx: sp.csr_matrix       = drugProteinRelationMtx
        self.proteinProteinRelationMtx: sp.spmatrix      = proteinProteinRelationMtx

