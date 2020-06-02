#!/bin/bash

echo "Starting to download data for use in running.  This may take a few minutes..."
echo ""

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

echo ""
echo "Finished downloading data"
