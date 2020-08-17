#!/usr/bin/env python
# coding: utf-8

# In[3]:


def load_data_file(filename):
    file = open(filename, "r")
    whole_FASTA = file.read()
    whole_FASTA_list = whole_FASTA.split('>')
    
    return whole_FASTA_list

def split_data_file(whole_FASTA_list):
    FASTA_data = {}
    keys = []
    for idx in range(1,len(whole_FASTA_list)):
        FASTA_tmp = whole_FASTA_list[idx].split('\n')
        key = FASTA_tmp[0]
        keys.append(key)
        value = FASTA_tmp[1]
        FASTA_data[key] = value
        
    return keys, FASTA_data

def compute_GC_content(keys, FASTA_data):
    num_of_GC_content = {}
    result_list = []
    for key in keys:
        cnt = 0
        DNA_seq = FASTA_data[key]
        for idx, item in enumerate(DNA_seq):
            if item == 'G' or item == 'C':
                cnt += 1
        num_of_GC_content[key] = (cnt / len(DNA_seq))*100
#         result_list.append(num_of_GC_content[key])
        print("{}, {:2.6f}".format(key, num_of_GC_content[key]))
        
    return num_of_GC_content


# In[4]:


if __name__ == "__main__":
    filename = "database/rosalind_gc.txt"
    keys, FASTA_data = split_data_file(load_data_file(filename))
    compute_GC_content(keys, FASTA_data)


# In[65]:


#filename = "rosalind_gc.txt"
#file = open(filename, "r")
#whole_FASTA = file.read()
#whole_FASTA_list = whole_FASTA.split('\n')

# print(len(whole_FASTA_list))
# print(type(whole_FASTA_list))
# print(whole_FASTA_list[-1])


#whole_FASTA_list = whole_FASTA_list[:-1]
# print(whole_FASTA_list[-1][0])

# print(whole_FASTA_list)

# seq_list = []
# sequence = ""

# for i in range(len(whole_FASTA_list)-1):
#     if whole_FASTA_list[i][0] != '>':
#         for nucleic_base in whole_FASTA_list[i]:
#             sequence += nucleic_base
# #             print("debug_point_1: {}".format(nucleic_base))
# #             print("debug_point_2: {}".format(sequence))
# #         print(type(whole_FASTA_list[i]))
# #         print(whole_FASTA_list[i])
        
#     elif whole_FASTA_list[i][0] == '>':
# #         continue
#         seq_list.append('-')
#     seq_list.append(sequence)
    

# seq_list = ''.join(seq_list)

# seq_list = seq_list.split('-')
# print(seq_list)
# print(len(seq_list))
# print(seq_list[0])


# In[66]:


#filename = "rosalind_gc.txt"
#file = open(filename, "r")
#whole_FASTA = file.read()
#whole_FASTA_list = whole_FASTA.split('\n')

# print(len(whole_FASTA_list))
# print(type(whole_FASTA_list))
# print(whole_FASTA_list[-1])


#whole_FASTA_list = whole_FASTA_list[:-1]


# In[68]:


#whole_FASTA_list

#FASTA_dict = {}
#FASTA_dict_key = ''
#for i in range(len(whole_FASTA_list)):
#    if whole_FASTA_list[i][0] == ">":
#        FASTA_dict_keys = whole_FASTA_list[i][1:]
#        FASTA_dict[FASTA_dict_keys] = ''
#    elif whole_FASTA_list[i][0] != ">" and whole_FASTA_list[i][0] != '':
#        FASTA_dict[FASTA_dict_keys] += whole_FASTA_list[i]
#        FASTA_dict_keys = ''
#        continue
#FASTA_dict


# In[56]:


#seq_list


# In[109]:


#tmp = str(whole_FASTA_list).split("\n")
#print(tmp)


# In[123]:


#len(tmp)


# In[ ]:




