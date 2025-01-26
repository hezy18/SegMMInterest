import inspect
import sys

def decorate_all_functions(func_name, decorator):
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj):
            setattr(sys.modules[__name__], name, decorator(obj))