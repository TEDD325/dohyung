#!/usr/bin/env python
# coding: utf-8

# # Sequence Alignment python 구현

# In[116]:


# 각 case에 대한 점수를 전역변수로 만듭니다.
score_rule_dict = {}
score_rule_dict["match"] = 8
score_rule_dict["mismatch"] = -5
score_rule_dict["gap"] = -3
print(score_rule_dict)


# In[117]:


# 예제에서 배웠던 sequence를 그대로 사용했으며, 비어있는 칸은 문자 X로 치환했습니다.
seq_a = ["X", "C", "A", "A", "T", "T", "G", "A"]
seq_b = ["X", "G", "A", "A", "T", "C", "T", "G", "C"]
print("seq_a: ", seq_a)
print("seq_b: ", seq_b)


# In[96]:


'''
염기 정보 한 글자를 받아서 상호 비교하는 함수입니다.
추후, 결과에 대한 분석과 디버깅을 위해 print문을 넣었으며, 
print문이 반복되기에 이 역시 함수로 만들어 호출했습니다.
추후, 더 긴 서열에 대한 처리가 가능하도록 코드를 개선시킬 필요가 있습니다.
'''

def debug_print(base_from_seq_a, 
                base_from_seq_b,
                score_rule):
    print("base_from_seq_a: ",base_from_seq_a)
    print("base_from_seq_b: ",base_from_seq_b)
    print("state: ", score_rule)
    print("score: ", score_rule_dict[score_rule])
    
def scoring(base_from_seq_a, base_from_seq_b):
    if base_from_seq_a==base_from_seq_b and (base_from_seq_a!="X" and base_from_seq_b!="X"):
#         debug_print(base_from_seq_a, base_from_seq_b, score_rule="match")
        return score_rule_dict["match"]
    elif base_from_seq_a!=base_from_seq_b:
        if base_from_seq_a=="X" or base_from_seq_b=="X":
#             debug_print(base_from_seq_a, base_from_seq_b, score_rule="gap")
            return score_rule_dict["gap"]
        else:
#             debug_print(base_from_seq_a, base_from_seq_b, score_rule="mismatch")
            return score_rule_dict["mismatch"]
    else:
        return 0


# In[97]:


# test를 한 번 해봤습니다. 
test = scoring("X", "C")
print(test)


# In[98]:


# score를 저장할 table을 만들어 주기 위한 함수입니다.
# 처음은 모두 값을 0으로 초기화합니다.
def init_score_table(score_table):
#     score_table = []
    row = []
    for num_i in range(len(seq_b)):
        for num_j in range(len(seq_a)):
            row.append(0)
        score_table.append(row)
        row = []
    return score_table


# In[99]:


# 초기화 수행 후 출력
score_table = []
score_table = init_score_table(score_table)
score_table


# In[32]:


# 1열에 대한 값들을 사전에 넣어주었습니다.
# local alignment인 경우 수행하지 않습니다.
val = 0
for i in range(len(seq_b)):
    score_table[i][0] = val
    val = val-3
score_table


# In[33]:


# 1행에 대한 값들도 마찬가지로 넣어주었습니다.
# local alignment인 경우 수행하지 않습니다.
val = 0
for j in range(len(seq_a)):
    score_table[0][j] = val
    val = val-3
    
score_table


# In[100]:


'''
PPT에 나온 수식 그대로 score table에 있는 값과 현재의 값을 더한 
세 개의 값 중 최대의 값을 구합니다.
print 부분은 테이블 상 어느 좌표인지를 살펴보기 위해 넣었으며, 최대값도 찍도록 했습니다.

'''

for j in range(1, len(seq_a)):
    for i in range(1, len(seq_b)):
        candidate_list = []
        candidate_list.append(score_table[i-1][j] + scoring(seq_a[j], "X"))
        candidate_list.append(score_table[i][j-1] + scoring("X", seq_b[i]))
        candidate_list.append(score_table[i-1][j-1] + scoring(seq_a[j], seq_b[i]))
        candidate_list.append(0) #local alignment
        score_table[i][j] = max(candidate_list)
        print("index of b: ", i)
        print("index of a: ", j)
        print("max val: ", max(candidate_list), end="\n\n")


# In[101]:


score_table


# In[102]:


import numpy as np 
score_table_np = np.array(score_table)
score_table_np_flatten = score_table_np.flatten() 


# In[103]:


score_table_np_flatten


# In[104]:


max(score_table_np_flatten)


# In[105]:


[(ix,iy) for ix, row in enumerate(list(score_table_np)) for iy, i in enumerate(row) if i == max(score_table_np_flatten)]


# In[106]:


score_table_np[7][6]


# In[107]:


i, j = [(ix,iy) for ix, row in enumerate(list(score_table_np)) for iy, i in enumerate(row) if i == max(score_table_np_flatten)][0]
print(i)
print(j)


# In[108]:


score_table = score_table[:i+1][:j+1]
score_table


# In[ ]:





# In[109]:


import numpy as np 
score_table_np = np.array(score_table[:i+1][:j+1])
score_table_np_flatten = score_table_np.flatten() 
i, j = [(ix,iy) for ix, row in enumerate(list(score_table_np)) for iy, i in enumerate(row) if i == max(score_table_np_flatten)][0]
print(i)
print(j)
print(score_table_np[i][j])
candidate_list = [score_table[i-1][j], score_table[i][j-1], score_table[i-1][j-1], 0]
score_table = score_table[:i+1][:j+1]
score_table


# In[114]:


import numpy as np 
score_table_np = np.array(score_table[:i+1][:j+1])
score_table_np_flatten = score_table_np.flatten() 
i, j = [(ix,iy) for ix, row in enumerate(list(score_table_np)) for iy, i in enumerate(row) if i == max(score_table_np_flatten)][0]
print(i)
print(j)
print(score_table_np[i][j])
candidate_list = [score_table[i-1][j], score_table[i][j-1], score_table[i-1][j-1], 0]
score_table = score_table[:i+1][:j+1]
score_table


# In[115]:


import numpy as np 
score_table_np = np.array(score_table[:i+1][:j+1])
score_table_np_flatten = score_table_np.flatten() 
i, j = [(ix,iy) for ix, row in enumerate(list(score_table_np)) for iy, i in enumerate(row) if i == max(score_table_np_flatten)][0]
print(i)
print(j)
print(score_table_np[i][j])
candidate_list = [score_table[i-1][j], score_table[i][j-1], score_table[i-1][j-1], 0]
score_table = score_table[:i+1][:j+1]
score_table


# In[ ]:





# In[ ]:


candidate_list = [score_table[i-1][j], score_table[i][j-1], score_table[i-1][j-1], 0]
max_val.append(max(candidate_list))


# In[ ]:





# <img src="../../../documents/__etc/dot_matrix.png">
# <!-- <img src="https://previews.dropbox.com/p/thumb/AAzTT1YNiln2JdRsYqxL9w7cucORreZTsCtFeQVe5qKar0rhTl2PSHjm4OCZBo-ZLFnJHMkIYgLUnSFVPH46j0IocA32CIbZbQAEgRUFlrbWrmm3-2BC797IDYh5CPKgo752wS9J0BrfZYe-SgCYsMjdOWPfrZLVOlT_cWspArHk602y4LXGysHOlqUSq5BXJ385oecnj1c0BirGJ6awrcfdZeTQT9_xcnE61PfHIobTQCv2dEFQo61VawSJiuT99XHu8HikcIF-0CRRYK2uHesrqQKlJHZnUk0S7e0R7Hp1O82bh1s4_MrO8dXU-ZF8DMFX4uFvp1DfPmQioalNEFHOVF930vd594cYR7OiSBa5wf-tGw2XrQ7JhMMO_Jpy-uJmrNVqKQvTD_6MhzJvnrbpTiiXWQH5CFvqQXGVTIfn1Q/p.png?fv_content=true&size_mode=5"> -->

# 결과가 PPT 상의 결과와 다른 부분들이 몇 군데 보여서 제가 잘못 구현한 부분이 있는 것일 수도 있겠다는 생각이 들었습니다만, 분자생물학 공부하느라 시간이 없어서 세세하게 살펴보지는 못했습니다. 급하게 구현한만큼 부족한 부분이 많을 것 같습니다. 피드백을 주신다면 감사드리겠습니다.
