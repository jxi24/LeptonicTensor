import subprocess
import os
import importlib
import pkgutil

import Models


def iter_namespace(namespace):
    return pkgutil.iter_modules(namespace.__path__, namespace.__name__ + '.')


def get_model(name):
    try:
        model = importlib.import_module(name)
    except ImportError:
        print('Converting Model file {} to python3 format!'.format(name))
        path = os.path.join(Models.__path__[0], name.split('.')[-1])
        subprocess.run(['2to3', '-w', path],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)
        model = importlib.import_module(name)
    return model


def discover_models():
    return {
        name: get_model(name)
        for finder, name, ispkg
        in iter_namespace(Models)
    }


def main():
    print(discover_models())


if __name__ == '__main__':
    main()
