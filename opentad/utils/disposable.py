from wrapt import ObjectProxy

class Disposable(ObjectProxy):
    """
    A wrapper for objects that need to be disposed of.
    """

    def dispose(self):
        """
        Dispose of the wrapped object.
        """
        self.__wrapped__ = None