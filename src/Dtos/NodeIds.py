from typing import Type

class BaseNodeId(int):
    '''
    Not meant to be instantiated, but rather used as a utility for
    other Id classes (namely, to convert from decagon format)
    '''
    def __new__(cls, val: object) -> Type['BaseNodeId']:
        if isinstance(val, str):
            val = BaseNodeId._formatStr(val)

        return int.__new__(cls, val)

    @classmethod
    def toDecagonFormat(cls, val=None):
        toConvToStr = cls
        if not isinstance(cls, BaseNodeId):
            if val is None:
                raise ValueError('If cls is not a BaseNodeId, val must not be None')

            toConvToStr = cls(val)

        return str(toConvToStr)

    @classmethod
    def fromDecagonFormat(cls, val: object) -> Type['BaseNodeId']:
        return cls(BaseNodeId._formatStr(val))

    @staticmethod
    def _formatStr(preStr: str) -> str:
        '''
        Removes letters and prefixed 0s from a string  E.g.,

        CID0012314 -> 12314
        C00512341  -> 512341
        SID123     -> 123

        '''
        # If the protein is just 0, then return 0
        if preStr == '0' or preStr[-1] == '0':
            return '0'

        # Remove leters from string
        preStr = ''.join(filter(str.isdigit, preStr))

        # Remove prefixed 0s
        preStr = preStr.lstrip('0')

        return preStr

class DrugId(BaseNodeId):
    def toDecagonFormat(self):
        preStr = str(self)

        # All STITCH drug IDs are of the format CID<nums> where <nums> is a
        # sequence of 9 numbers.  If a preStr, as above, has less than 9
        # digits, the STITCH ID has 0s prepended to those preStr digits.
        numPrecedingZeros = 9 - len(preStr)
        zerosStr = '0' * numPrecedingZeros

        return 'CID' + zerosStr + preStr

class ProteinId(BaseNodeId):
    pass

class SideEffectId(BaseNodeId):
    def toDecagonFormat(self):
        preStr = str(self)

        # All STITCH side effect IDs are of the format C<nums> where <nums> is a
        # sequence of 7 numbers.  If a preStr, as above, has less than 7
        # digits, the STITCH ID has 0s prepended to those preStr digits.
        numPrecedingZeros = 7 - len(preStr)
        zerosStr = '0' * numPrecedingZeros

        return 'C' + zerosStr + preStr

