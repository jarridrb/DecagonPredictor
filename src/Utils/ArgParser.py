from typing import ClassVar, Type
from argparse import ArgumentParser, Namespace

class ArgParser:
    DESCRIPTION_STR: ClassVar[str] = '''
        Trains the main (currently decagon) model. Code is modular, and provides
        a series of configuration options for modules to use (e.g., learning policy,
        optimizer, graph builder).
    '''

    def __init__(self) -> None:
        self.parser: ArgumentParser = ArgumentParser(
            description=ArgParser.DESCRIPTION_STR
        )

        self.args: Namespace = None
        self._addArgs()

    def getKey(self, key: str) -> Type[object]:
        if hasattr(self.args, key):
            return getattr(self.args, key)
        else:
            raise AttributeError(self._getErrStr(key))

    def _getErrStr(self, key: str) -> str:
        errStr = ''
        if self.args is None:
            errStr = 'Parse has not yet been called on ArgParser object'
        elif not hasattr(self.args, key):
            errStr = 'Did not have arg %s in parsed args' % key
        else:
            errStr = 'Arg %s was not specified in process invocation' % key

        return errStr

    def _addArgs(self) -> None:
        # TODO: Must add all arguments!
        self.parser.add_argument('--config')
        self.parser.add_argument('--fxn-call-type')

    def parse(self) -> None:
        self.args = self.parser.parse_args()

