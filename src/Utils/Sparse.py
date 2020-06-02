from typing import Type
import scipy.sparse as sp
import os

class RelationCooMatrix(sp.coo_matrix):
    _numMtxsCreated = 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.isTranspose: bool = False

        # A reference to the transposed matrix
        self.transposedMtxLink: RelationCooMatrix = None

        self.id = "RelationCooMatrix|%d|%d" % (os.getpid(), RelationCooMatrix._numMtxsCreated)
        RelationCooMatrix._numMtxsCreated += 1

    def transpose(self, axes=None, copy=False, setId=False) -> Type['RelationCooMatrix']:
        newMtx = RelationCooMatrix(super().transpose(axes, copy))

        if setId:
            self.isTranspose = True
            self.tranposedMtxLink = newMtx

            newMtx.isTranspose = True
            newMtx.transposedMtxLink = self

        return newMtx

    def tocsr(self, copy=False) -> Type['RelationCsrMatrix']:
        newMtx = RelationCsrMatrix(super().tocsr(copy))

        newMtx.id = self.id
        newMtx.isTranspose = self.isTranspose
        newMtx.transposedMtxLink = self.transposedMtxLink

        return newMtx

class RelationCsrMatrix(sp.csr_matrix):
    _numMtxsCreated = 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.isTranspose: bool = False

        # A reference to the transposed matrix
        self.transposedMtxLink: RelationCsrMatrix = None

        self.id = "RelationCsrMatrix|%d|%d" % (os.getpid(), RelationCsrMatrix._numMtxsCreated)
        RelationCsrMatrix._numMtxsCreated += 1

    def transpose(self, axes=None, copy=False, setId=False) -> Type['RelationCsrMatrix']:
        newMtx = RelationCsrMatrix(super().transpose(axes, copy))

        if setId:
            self.isTranspose = True
            self.transposedMtxLink = newMtx

            newMtx.isTranspose = True
            newMtx.transposedMtxLink = self

        return newMtx

    def tocoo(self, copy=False) -> Type['RelationCooMatrix']:
        newMtx = RelationCooMatrix(super().tocoo(copy))

        newMtx.id = self.id
        newMtx.isTranspose = self.isTranspose
        newMtx.transposedMtxLink = self.transposedMtxLink

        return newMtx

