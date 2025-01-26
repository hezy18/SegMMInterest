import json
import yaml


def yapf_pformat(obj):
    from pprint import pformat
    from yapf.yapflib.yapf_api import FormatCode
    """pretty print object using yapf
    https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries

    Args:
        obj : object to be printed
    """

    string = pformat(obj)
    format_code, _ = FormatCode(string)

    return format_code

def black_pformat(obj):
    import black
    from pprint import pformat
    string = pformat(obj)
    format_code = black.format_str(string, mode=black.FileMode(line_length=119))

    return format_code


def json2yaml(json_str, is_file=False):
    if not is_file:
        json_dict = json.loads(json_str)
    else:
        json_dict = json.load(open(json_str, "r"))  # json_str is a file path
    return yaml.dump(json_dict, default_flow_style=False)
