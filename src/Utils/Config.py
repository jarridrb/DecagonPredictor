from .ArgParser import ArgParser
import json

class Config:
    CONF_FILE_PATH = 'configuration.json'

    @staticmethod
    def getConfig():
        argParser = ArgParser()
        argParser.parse()

        return Config(argParser)

    def __init__(self, argParserObj: ArgParser) -> None:
        if argParserObj is None or not isinstance(argParserObj, ArgParser):
            raise TypeError('argParserObj was passed to config uninitialized')

        self.argParserObj: ArgParser = argParserObj

        with open(self._getConfFilename()) as f:
            self.confJson: Dict[str, str] = json.load(f)

    def _getConfFilename(self) -> str:
        fname = None
        if hasattr(self.argParserObj.args, 'config'):
            fname = self.argParserObj.getKey('config')

        if fname is None:
            fname = Config.CONF_FILE_PATH

        return fname

    def getSetting(self, settingName: str):
        if hasattr(self.argParserObj, settingName):
            return self.argParserObj.getKey(settingName)
        elif settingName in self.confJson:
            return self.confJson[settingName]
        else:
            raise KeyError(
                'Setting %s not in args or configuration file' % settingName
            )

