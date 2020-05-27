#!/usr/bin/env python
# coding: utf-8

# ### 작성자: 권도형
# ### 작성일자: 200515
# ### 최종수정일자: 200518

# In[1]:


var_1 = "Hello world"


# In[2]:


print(var_1)


# In[3]:


var_1.upper()
# var_1은 str 타입의 object이므로, str object가 갖는 모든 property와 method를 사용할 수 있게 된다.
# str.upper()


# In[8]:


var_1.count('e')


# In[11]:


type(var_1.count('e'))


# In[ ]:





# In[9]:


dna_1 = "ATGCCGCATTC"


# In[19]:


print("number of A: ", dna_1.count("A"))
print("number of T: ", dna_1.count("T"))
print("number of G: ", dna_1.count("G"))
print("number of C: ", dna_1.count("C"))


# In[24]:


(dna_1.count('A') + dna_1.count('G')) / len(dna_1) * 100


# In[ ]:





# In[49]:


def count_seq_of_DNA(DNA_seq, seq_user):
    print("number of A: ", DNA_seq.count("A"))
    print("number of T: ", DNA_seq.count("T"))
    print("number of G: ", DNA_seq.count("G"))
    print("number of C: ", DNA_seq.count("C"))
    
    total_ = []
    for item in seq_user:
        total_.append(DNA_seq.count(item))
#         print("debug point_1", total_)
        print(item, end="")
#     print("debug point_2", sum(total_))
    print("은(는) DNA 서열 중", int(per), "% 입니다.")
    return sum(total_) / len(DNA_seq) * 100
'''
DNA 뉴클레오타이드 이외의 문자열이 들어오지 않도록 처리 필요
seq_user의 개수가 네 개가 넘어가지 않도록 처리 필요
대문자, 소문자 모두 가능하도록 수정할 필요
'''


# In[51]:


per = count_seq_of_DNA(dna_1, ['A', 'G', 'C'])
'''
count_seq_of_DNA(DNA 서열, 찾고자 하는 뉴클레오타이드-list 타입)
'''


# In[ ]:




