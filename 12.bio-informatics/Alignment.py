#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random
import numpy as np
import sys


# In[2]:


# Intro
print ('''
=================================================================

 The program that score coincidence of between sequence A and B.

==============The first program made by Jonghun_Woo==============
        Seq_Alignment               Ver. 1.00 / April 24, 2020.
=================================================================


''')


# In[6]:


def gen_random_seq():
    gnum_a = input("Enter a base sequence or length of A: ") # 'gene number'
    gnum_b = input("Enter a base sequence or length of B: ") # 'gene number'
    # User input the number of base.
    gcode = ["A", "T", "G", "C"] # 'genetic code' .
    sqc_a = [''] # 첫 element는 일부러 빈 문자열을 넣어준다.
    sqc_b = ['']
    n = 1
    
    while True :
        if gnum_a or gnum_b == 'done' : # Quit the program when typing 'done'
            print ("Exit the program")
            sys.exit()
        else :
            try :
                gn_a = int(gnum_a) # 'gene number'
                gn_b = int(gnum_b)

                while n <= gn_a :
                    sqc1.append(random.choice(gcode))
                    n = n + 1
                n = 1
                while n < gn_b:
                    sqc2.append(random.choice(gcode))
                    n = n + 1
                break
            except :
                gnum_a = gnum_a.upper()
                gnum_b = gnum_b.upper()
                gn_a = len(gnum)
                gn_b = len(gnum)
                gnum = list(gnum)
                gnum.insert(0, "")
                sqc1 = gnum
                sqc2 = gnum
                break


# In[7]:


gen_random_seq()


# In[3]:


# User input the number of base.
# Sequence A
while True :
    gnum = input("Enter a base sequence or length of A : ") # 'gene number'
    if gnum == 'done' : # Quit the program when typing 'done'
        print ("Exit the program")
        sys.exit()
    else :
        try :
            gn1 = int(gnum) # 'gene number'
            gcode = ["A", "T", "G", "C"] # 'genetic code' .
            sqc1 = [""] # 첫 element는 일부러 빈 문자열을 넣어준다.
            n = 1
            while n <= gn1 :
                sqc1.append(random.choice(gcode))
                n = n + 1
            break
        except :
            gnum = gnum.upper()
            gn1 = len(gnum)
            gnum = list(gnum)
            gnum.insert(0, "")
            sqc1 = gnum
            break


# In[4]:


while True :
    gnum = input("Enter a base sequence or length of B : ") # 'gene number'
    if gnum == 'done' : # Quit the program when typing 'done'
        print ("Exit the program")
        sys.exit()
    else :
        try :
            gn2 = int(gnum) # 'gene number'
            gcode = ["A", "T", "G", "C"] # 'genetic code' .
            sqc2 = [""] # 첫 element는 일부러 빈 문자열을 넣어준다.
            n = 1
            while n <= gn2 :
                sqc2.append(random.choice(gcode))
                n = n + 1
            break
        except :
            gnum = gnum.upper()
            gn2 = len(gnum)
            gnum = list(gnum)
            gnum.insert(0, "")
            sqc2 = gnum
            break


# In[ ]:





# In[5]:


print ("A is", gn1, "sequences")
print (sqc1)
print ("B is", gn2, "sequences")
print (sqc2)


# In[9]:


# Making frame of dot matrix.
gdata = pd.DataFrame(index = sqc1, columns = sqc2) # 'gene data'

# Inputting the first row & column.
# Sequence A
gdata.iloc[0, 0] = 0
count = 0
n = 0
while n <= gn1 :
    gdata.iloc[n, 0] = count
    count = count - 0 # Gap penalty (If negetive value is available.)
    n = n + 1
# Sequence B
count = 0
n = 0
while n <= gn2 :
    gdata.iloc[0, n] = count
    count = count - 0 # Gap penalty (If negetive value is available.)
    n = n + 1


# In[10]:


print (gdata)


# In[11]:


# Scoring Match, Mismatch, Gap in the direction of reading a book.
v = 1
cprn = list() # 'comparing number'
while v <= gn1 :
    h = 1
    while h <= gn2 :
        del cprn[:]
        if sqc1[v] == sqc2[h] :
            cprn.append(gdata.iloc[v-1, h-1] + 8) # Match point
        else :
            cprn.append(max(gdata.iloc[v-1, h-1] - 5, 0)) # Mismatch penalty
        cprn.append(max(gdata.iloc[v-1, h] - 3, 0)) # Gap penalty
        cprn.append(max(gdata.iloc[v, h-1] - 3, 0)) # Gap penalty
        gdata.iloc[v, h] = max(cprn) # Input Maximum value.
        h = h + 1
    v = v + 1
print (gdata)


# In[17]:


##### 리스트 생성 함수 ######
def a_list_gen() :
    gen_num = 1
    while True : 
        yield ("a_sqc" + str(gen_num))
        gen_num += 1
               
def b_list_gen() :
    gen_num = 1
    while True : 
        yield ("b_sqc" + str(gen_num))
        gen_num += 1

a_sqc_gen = iter(a_list_gen()) # 'a_sequence_list'
b_sqc_gen = iter(b_list_gen()) # 'b_sequence_list'


# In[21]:


# Searching the max values.
a = gdata.max(axis=1).argmax()
b = gdata.max(axis=0).argmax()
a_sqc0 = [gdata.max(axis=1).idxmax()] # 'a_sequence 0'
b_sqc0 = [gdata.max(axis=0).idxmax()] # 'b_sequence 0'
a_sqc_sqc = [a_sqc0] # total a_sequence_list
b_sqc_sqc = [b_sqc0] # total b_sequence_list


# In[27]:


# Finding routes for the max.
while a > 0 & b > 0 :
    if gdata.iloc[a, b] - 8 == gdata.iloc[a-1, b-1] or gdata.iloc[a, b] + 5 == gdata.iloc[a-1, b-1] : 
        for i in a_sqc_sqc :
            a_sqc_sqc.insert(0, sqc1[a-1])
            b_sqc_sqc.insert(0, sqc2[b-1])
        a = a - 1
        b = b - 1
    elif gdata.iloc[a-1, b] != gdata.iloc[a, b-1] and gdata.iloc[a, b] + 3 == gdata.iloc[a-1, b] :
        for i in a_sqc_sqc :
            a_sqc_sqc.insert(0, sqc1[a-1])
            b_sqc_sqc.insert(0, "-")
        a = a - 1
    elif gdata.iloc[a-1, b] != gdata.iloc[a, b-1] and gdata.iloc[a, b] + 3 == gdata.iloc[a, b-1] :
        a_sqc_sqc.insert(0, "-")
        b_sqc_sqc.insert(0, sqc2[b-1])
        b = b - 1
        


# In[29]:


print (a, b)
print (a_sqc0, b_sqc0)
print (a_sqc_sqc)
print (b_sqc_sqc)


# 동시에 성립하면 공통부분까지 확정해두고, 두갈래로 리스트 생성하여 추후비교 방식으로 구현.

# In[2]:


# print ("A : ", a_sqc)
# print ("B : ", b_sqc)
# input()

