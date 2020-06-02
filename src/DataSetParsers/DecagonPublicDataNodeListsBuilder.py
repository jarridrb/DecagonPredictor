from .BaseNodeListsBuilder import BaseNodeListsBuilder
from ...Dtos.Enums.DataSetType import DataSetType
from ...Dtos.NodeLists import NodeLists
from ...Dtos.NodeIds import DrugId, SideEffectId, ProteinId
from ...Dtos.TypeShortcuts import EdgeList, RelationIDToEdgeList, RelationIDToGraph, RelationIDToSparseMtx
from ...Utils import Config
from collections import defaultdict
from typing import Iterator
import networkx as nx
import numpy as np
import scipy.sparse as sp

class DecagonPublicDataNodeListsBuilder(
    BaseNodeListsBuilder,
    functionalityType = DataSetType.DecagonPublicData
):
    def __init__(self, config: Config) -> None:
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

    def build(self) -> NodeLists:
        proteinNodeList = self._getOrderedProteinNodeList()
        drugNodeList    = self._getOrderedDrugNodeList()

        return NodeLists(proteinNodeList, drugNodeList)

    def _getOrderedDrugNodeList(self) -> EdgeList:
        allDrugs = set(
            self.drugDrugRelationGraph.nodes
        ).union(set(self._getDrugProteinGraphDrugs()))

        return sorted(list(allDrugs))

    def _getDrugProteinGraphDrugs(self) -> Iterator[tuple]:
        # In preprocessed dataset, all drug identifiers are prefixed with 'CID'
        # while protein identifiers are not
        drugNodes = filter(
            lambda x: x[:3] == 'CID',
            self.drugProteinRelationGraph.nodes
        )

        # Convert all drug id strs to DrugIds
        return map(DrugId, drugNodes)

    def _getOrderedProteinNodeList(self) -> EdgeList:
        allProteins = set(
            self.ppiGraph.nodes
        ).union(set(self._getDrugProteinGraphProteins()))

        return sorted(list(allProteins))

    def _getDrugProteinGraphProteins(self) -> Iterator[tuple]:
        # In preprocessed dataset, all drug identifiers are prefixed with 'CID'
        # while protein identifiers are not
        proteinNodes = filter(
            lambda x: x[:3] != 'CID',
            self.drugProteinRelationGraph.nodes
        )


        return map(ProteinId, proteinNodes)

