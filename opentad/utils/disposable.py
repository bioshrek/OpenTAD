from wrapt import ObjectProxy

class Disposable(ObjectProxy):
    """
    A wrapper for objects that need to be disposed of.
    """

    def _dispose(self):
        """
        Dispose of the wrapped object.
        """
        self.__wrapped__ = None

    def _unwrap(self):
        """
        Unwrap the object and return it.
        """
        return self.__wrapped__
    
    @classmethod
    def dispose(cls, obj):
        """
        Dispose of the object if it is a Disposable.
        """
        if isinstance(obj, cls):
            obj._dispose()
    
    @classmethod
    def unwrap(cls, obj):
        """
        Unwrap the object and return it.
        """
        if isinstance(obj, cls):
            return obj._unwrap()
        return obj