import os

def search(dirname):
    filenames = os.listdir(dirname)
    fileList = []
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        fileList.append(full_filename)
    return fileList

fileList = search("../evaluate_result/")