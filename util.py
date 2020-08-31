import os


def make_directory(*dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    