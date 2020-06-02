#!/bin/bash

cwd=$(echo $PWD)
git clone https://github.com/jrectorb/DecagonPredictorDataSets.git data

cd data/CommonData/NetworkEdgeLists

tar -xvf DrugDrugInteractionNetworkEdgeList.tar.gz
rm DrugDrugInteractionNetworkEdgeList.tar.gz

cd ${cwd}
cd data/ModelSpecificData/TrainedOnAll
bzip2 -d TrainEdges.npz.bz2

cd ../TrainedWithMasks
bzip2 -d TrainEdges.npz.bz2

cd ${cwd}
