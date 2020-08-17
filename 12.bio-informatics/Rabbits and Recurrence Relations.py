#!/usr/bin/env python
# coding: utf-8

# In[15]:


def rabiit_fibo(n, k):
    f1 = 1
    f2 = 1
    rabbit_fibo = [f1, f2]
    
    for n_idx in range(1, n):
        if n_idx == f1 or n_idx == f2:
            continue
        else:
            rabbit_fibo.append(rabbit_fibo[n_idx-2]*k+rabbit_fibo[n_idx-1])
            
    return rabbit_fibo


# In[23]:


if __name__ == "__main__":
    n = 30
    k = 5
    print("{}".format(rabiit_fibo(n, k)[-1]))

