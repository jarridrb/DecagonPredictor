from .BaseAdjacencyMatricesBuilder import BaseAdjacencyMatricesBuilder
from ...Dtos.AdjacencyMatrices import AdjacencyMatrices
from ...Dtos.Enums.DataSetType import DataSetType
from ...Dtos.NodeIds import DrugId, ProteinId, BaseNodeId
from ...Dtos.NodeLists import NodeLists
from ...Dtos.TypeShortcuts import EdgeList, RelationIDToEdgeList, RelationIDToGraph, RelationIDToSparseMtx
from ...Utils import Config
from ...Utils.Sparse import RelationCsrMatrix
from typing import Type, Tuple
from collections import defaultdict
import networkx as nx
import numpy as np
import scipy.sparse as sp

Edge = Tuple[str, str, int]

class DecagonPublicDataAdjacencyMatricesBuilder(
    BaseAdjacencyMatricesBuilder,
    functionalityType = DataSetType.DecagonPublicData
):
    def __init__(self, nodeLists: NodeLists, config: Config) -> None:
        self.drugDrugRelationGraph: nx.MultiGraph = nx.read_edgelist(
            config.getSetting('DecagonDrugDrugRelationsFilename'),
            delimiter=',',
            create_using=nx.MultiGraph(),
            nodetype=DrugId,
            data=(('relationType', str),)
        )

        self.drugProteinRelationGraph: nx.Graph = nx.read_edgelist(
            config.getSetting('DecagonDrugProteinRelationsFilename'),
            delimiter=','
        )

        self.ppiGraph: nx.Graph = nx.read_edgelist(
            config.getSetting('DecagonProteinProteinRelationsFilename'),
            nodetype=ProteinId,
            delimiter=','
        )

        # When building adjacency matrices with networkx, we can
        # guarantee ordering of the built matrices.  Thus, we precompute
        # them here so they can be used later in the building of the matrices.
        self.drugNodeList: EdgeList = nodeLists.drugNodeList
        self.proteinNodeList: EdgeList = nodeLists.proteinNodeList

    def build(self) -> AdjacencyMatrices:
        return AdjacencyMatrices(
            drugDrugRelationMtxs=self._buildDrugDrugRelationMtxs(),
            drugProteinRelationMtx=self._buildDrugProteinRelationMtx(),
            proteinProteinRelationMtx=self._buildPpiMtx(),
        )

    def _buildDrugDrugRelationMtxs(self) -> RelationIDToSparseMtx:
        validEdgeSets = self._getValidEdgeSets()
        graphs = self._getDrugDrugRelationGraphs(validEdgeSets)

        adjMtxs = {
            relID: RelationCsrMatrix(
                nx.adjacency_matrix(graph, nodelist=self.drugNodeList)
            ) for relID, graph in graphs.items()
        }

        return adjMtxs

    def _getDrugDrugRelationGraphs(
        self,
        validEdgeSets: RelationIDToEdgeList
    ) -> RelationIDToGraph:
        graphs = {}
        for relID, validEdgeSet in validEdgeSets.items():
            graph = nx.Graph()

            graph.add_nodes_from(self.drugNodeList)
            graph.add_edges_from(validEdgeSet)

            graphs[relID] = graph

        return graphs

    def _getValidEdgeSets(self) -> RelationIDToEdgeList:
        allEdgeSets = self._buildAllEdgeSets()
        return self._filterEdgeSets(allEdgeSets)

    def _buildAllEdgeSets(self) -> RelationIDToEdgeList:
        result = defaultdict(list)
        for edge in self.drugDrugRelationGraph.edges:
            relationId = self._extractRelationId(edge)

            # Remove edge type from edge
            truncatedEdge = edge[:2]
            result[relationId].append(truncatedEdge)

        return result

    def _extractRelationId(self, edge: Edge) -> str:
        FIRST_EDGE_NODE_IDX = 0
        SCND_EDGE_NODE_IDX  = 1
        RELATION_TYPE_IDX   = 2

        thisEdgeDict = self.drugDrugRelationGraph.get_edge_data(
            edge[FIRST_EDGE_NODE_IDX],
            edge[SCND_EDGE_NODE_IDX]
        )

        rawRelationId = thisEdgeDict[edge[RELATION_TYPE_IDX]]['relationType']

        # First element of the relation ID str is a 'C'.  Return an int which
        # excludes this 'C'.
        return int(rawRelationId[1:])

    def _filterEdgeSets(
        self,
        allEdgeSets: RelationIDToEdgeList
    ) -> RelationIDToEdgeList:
        result = {}
        # Filter out edge types that don't have 500
        for edgeType, edgeList in allEdgeSets.items():
            if self._isEdgeListValid(edgeList):
                result[edgeType] = edgeList

        return result

    def _isEdgeListValid(self, edgeList: EdgeList) -> bool:
        return len(edgeList) >= 500

    def _buildDrugProteinRelationMtx(self) -> Type[sp.csr_matrix]:
        drugToIdx = {drug: idx for idx, drug in enumerate(self.drugNodeList)}
        proteinToIdx = {protein: idx for idx, protein in enumerate(self.proteinNodeList)}

        drugProteinMtx = np.zeros((len(proteinToIdx), len(drugToIdx)))
        for edge in self.drugProteinRelationGraph.edges:
            drug, protein = self._extractDrugProtein(edge)
            drugProteinMtx[proteinToIdx[protein], drugToIdx[drug]] = 1

        return RelationCsrMatrix(drugProteinMtx)

    def _extractDrugProtein(self, edge: tuple) -> tuple:
        drugIdx = 0 if edge[0][:3] == 'CID' else 1
        proteinIdx = 1 - drugIdx

        return (
            DrugId.fromDecagonFormat(edge[drugIdx]),
            ProteinId.fromDecagonFormat(edge[proteinIdx])
        )

    def _buildPpiMtx(self) -> Type[sp.spmatrix]:
        self.ppiGraph.add_nodes_from(self.proteinNodeList)

        return RelationCsrMatrix(
            nx.adjacency_matrix(self.ppiGraph, nodelist=self.proteinNodeList)
        )

