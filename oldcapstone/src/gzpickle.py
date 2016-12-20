"""Generic object pickler and compressor

This module saves and reloads compressed representations of generic Python
objects to and from the disk. Added Protocol field.

Module code obtained from https://wiki.python.org/moin/Asking%20for%20Help/How%20do%20I%20use%20gzip%20module%20with%20pickle%3F
"""

__author__ = "Bill McNeill <billmcn@speakeasy.net>"
__version__ = "1.1"

import cPickle as pickle
import gzip


def save(object, filename, protocol = 2):
    """Saves a compressed object to disk
    """
    file = gzip.GzipFile(filename, 'wb')
    file.write(pickle.dumps(object, protocol))
    file.close()

def load(filename):
    """Loads a compressed object from disk
        """
    file = gzip.GzipFile(filename, 'rb')
    buffer = ""
    while True:
        data = file.read()
        if data == "":
            break
        buffer += data
    object = pickle.loads(buffer)
    file.close()
    return object
