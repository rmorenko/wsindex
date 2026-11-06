import functools
import random
import time


def benchmark(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        '''wrapper!'''
        start = time.perf_counter()
        result = f(*args, **kwargs)
        end = time.perf_counter()
        print(f'{f.__name__} took {end - start:.6f} s')
        return result
    return wrapper


@benchmark
def square(x):
    '''xxx'''
    return x * x

@benchmark
def add(x, y):
    '''yyy'''
    return x + y

square(3)



def retry(n, exceptions=(Exception,)):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(n):
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    print(f'Attempt {i + 1} failed: {e}')
                    last_exception = e
            raise last_exception
        return wrapper
    return decorator




@retry(3)
def connect_to_db():
    time.sleep(1)
    if random.random() < 0.9:
        raise ConnectionError('Failed to connect to db')
    return 'Connected to db'

@retry(1)
def connect_to_remote_display():
    time.sleep(1)
    if random.random() < 0.5:
        raise ConnectionError('Failed to connect to display')
    return 'Connected to display'


def save_to_db(f):
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if result.isinstance(...):
            # save_to_database(result)
            ...
        return result
    return wrapper

@save_to_db
def create_user(name):
    user = {'name': name}



def version(ver):
    def decorator(cls):
        cls.version = ver
        return cls
    return decorator

def dataclass(cls):
    def __repr__(self):
        return f'{cls.__name__}({self.__dict__})'

    cls.__repr__ = __repr__
    return cls

def collection(cls):
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        cls._collection.append(self)

    def delete(self):
        self._collection.remove(self)

    cls._collection = []
    original_init = cls.__init__
    cls.__init__ = __init__
    cls.delete = delete
    return cls


@collection
@dataclass
@version(ver='2.0.0')
class User:
    def __init__(self):
        self.x = 0

