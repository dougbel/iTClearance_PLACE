import os

def get_file_names_with_extension_in(directory, extension):
    return [file_name for file_name in os.listdir(directory) if file_name.endswith(extension)]