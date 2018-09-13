
# coding: utf-8

# In[1]:


import consulate


# In[2]:


print("..")
consul = consulate.Consul()


# In[4]:


consul.agent.service.register(name="iot-example", address="192.168.255.2", port=18546, httpcheck="http://192.168.255.2:18546/health", interval='10s')

