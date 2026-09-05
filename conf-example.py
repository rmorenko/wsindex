'''Example of a config singleton idea
'''
import tomllib


class Config:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, config_file='wsindex.toml'):
        # TODO: check if config_file exists, if not, create it with default values, if necessary
        try:
            open(config_file, 'rb')
        except FileNotFoundError:
            print('warning')
            self._config = Config.DEFAULT_CONFIG
        if not hasattr(self, '_config'):
            self._config = tomllib.load(open(config_file, 'rb'))

    @property
    def base_url(self):
        return self._config['tensorus']['base_url']


c = Config()
print(c.config['tensorus']['base_url'])


c2 = Config()