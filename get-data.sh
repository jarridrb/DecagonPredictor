#!/bin/bash

cwd=$(echo $PWD)
git clone https://github.com/jrectorb/DecagonPredictorDataSets.git data

cd data/CommonData/NetworkEdgeLists

tar -xvf DrugDrugInteractionNetworkEdgeList.tar.gz
rm DrugDrugInteractionNetworkEdgeList.tar.gz

cd ${cwd}
