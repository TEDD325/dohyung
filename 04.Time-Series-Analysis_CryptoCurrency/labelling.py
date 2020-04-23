
# coding: utf-8

# In[9]:


import csv
coin = csv.reader(open('coin_BTC_cleanup.csv'))
price_list = []
for row in coin:
    price_list.append(row)
print(len(price_list))
print(price_list[0])


# In[26]:


label = []
for i in range(1,len(price_list)):
    if price_list[i][3] < price_list[i-1][3]: # closing price of Nth < closing price of (N-1)th
#         label.append([price_list[i][0], 0])
        label.append(0)
    else:
#         label.append([price_list[i][0], 1])
        label.append(1)
label = [1] + label
print(len(label))
print(label)


# In[35]:


import csv

# res = ['x', 'y', 'z']
csvfile = "label.csv"

#Assuming res is a flat list
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in label:
        writer.writerow([val])    

# #Assuming res is a list of lists
# with open(csvfile, "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     writer.writerows(label)

