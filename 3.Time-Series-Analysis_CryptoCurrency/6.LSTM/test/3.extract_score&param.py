import pandas as pd
import os

def search(dirname):
    filenames = os.listdir(dirname)
    fileList = []
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        fileList.append(full_filename)
    return fileList

fileList = search("../evaluate_result/")
temp_list = []
for file in fileList:
    temp_list.append(pd.read_pickle(file))

for i in range(len(temp_list)):
    max_idx = 0
    max_score = 0
    # print(temp_list[i])
    # print()
    print(temp_list[i][list(temp_list[i].keys())[0]])
    print()
    for idx, item in enumerate(temp_list[i][list(temp_list[i].keys())[0]]['Score']):
        if max_score < item:
            max_idx = idx
            max_score = item
        if max_score == max(temp_list[i][list(temp_list[i].keys())[0]]['Score']):
            break

    print(max_score)
    print(max_idx)
    print()
    # filename_element_1 = temp_list[i][list(temp_list[i].keys())[0]]['Cryptocurrency']
    # print(filename_element_1)
    best_param = temp_list[i][list(temp_list[i].keys())[0]]['Params'][max_idx]
    print(best_param)
    print()
    print()