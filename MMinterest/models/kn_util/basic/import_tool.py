import importlib
import os

def import_modules(dir, namespace):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + name)
