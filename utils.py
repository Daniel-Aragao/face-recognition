import os

constants = {
    'samples_folder' : 'samples',
    'samples_map_file' : 'samples/samples-map.txt',
    'trained_folder' : 'trained/trainner.yml'
}

def import_file(path, show_progress=False):
    lines = []
    with open(path, 'r') as file_obj:
        for i, line in enumerate(file_obj):
            lines.append((i, line))
        
    return lines

def printToFile(string, filename, append=True):
    path = filename

    with open(path, 'a+' if append else 'w+') as file_obj:
        # file_obj.write('\n')
        file_obj.write(string)
