import sys


def get_image_path(default_path):
    if len(sys.argv) > 1:
        return sys.argv[1]

    return default_path