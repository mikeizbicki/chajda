from collections import OrderedDict
 
class LRUCache:
    '''
    A dictionary that automatically removes the least recently used (LRU) items.
    This is useful for preventing infrequently used items from taking up memory.

    FIXME:
    Currently, it's only possible to set a limit on the number of items,
    but in practice it's probably more useful to set a limit on the amount of RAM used.
    This can be done with the rememberme library.
    '''

    def __init__(self, maxitems=None):
        self.cache = OrderedDict()
        self.maxitems = maxitems

    def set_maxitems(self, maxitems):
        self.maxitems = maxitems
        self._trim()

    def _trim(self):
        if self.maxitems:
            while len(self.cache) > self.maxitems:
                self.cache.popitem(last = False)
 
    def __getitem__(self, key):
        self.cache.move_to_end(key)
        return self.cache[key]
 
    def __setitem__(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        self._trim()

    def __len__(self):
        return len(self.cache)

    def __cmp__(self, dict_):
        return self.__cmp__(self.cache, dict_)

    def __contains__(self, item):
        return item in self.cache

    def __iter__(self):
        return iter(self.cache)

    def keys(self):
        return self.cache.keys()

    def values(self):
        return self.cache.values()

    def items(self):
        return self.cache.items()
