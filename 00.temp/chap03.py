
# coding: utf-8

# #### BeautifulSoup 1: use file

# In[1]:


from bs4 import BeautifulSoup
# html 코드를 Python이 이해하는 객체 구조로 변환하는 Parsing을 맡고 있다.


# In[2]:


page = open("../data/03. test_first.html", 'r').read()
# open()메소드는 defualt로 제공되는 메소드


# In[5]:


type(page)


# In[6]:


soup = BeautifulSoup(page, 'html.parser')

# html의 내용을 BeautifulSoup의 첫번째 인자값으로 전달하고, 
# 두번째 인자값으로 파서를 전달하여 html로 해석되게 한다.

# https://www.crummy.com/software/BeautifulSoup/bs4/doc/
    
# 사용가능한 parser의 종류가 네 가지가 있다;html.parser, lxml, lxml-xml, html5lib

# 파서란 "구문 분석기" 라 한다. 

# 문서를 읽어서 단위 구문으로 나누는 컴파일러.
# 언어학에서는 구문(문장의 구성)분석이라고 함.

# 위계 관계를 분석하여 문장의 구조를 결정함. --> 일종의 트리가 나옴


# In[7]:


type(soup)


# In[4]:


print(soup.prettify())


# In[22]:


list(soup.children)


# In[23]:


list(soup.children)[0]


# In[24]:


list(soup.children)[1]


# In[25]:


list(soup.children)[2]


# In[26]:


len(list(soup.children))


# In[27]:


html = list(soup.children)[2]
html


# In[29]:


list(html.children)


# In[30]:


len(list(html.children))


# In[31]:


body = list(html.children)[3]
body


# In[32]:


body.parent


# In[33]:


html.parent


# - https://namu.wiki/w/HTML/%ED%83%9C%EA%B7%B8

# In[34]:


soup.find_all('p')


# In[35]:


soup.find_all('p', class_="outer-text")


# In[36]:


# 곧바로 태그에 접근 가능
soup.head


# In[37]:


soup.p


# In[38]:


soup.body


# In[39]:


soup.html


# In[40]:


soup.head.next_sibling


# In[43]:


list(soup.head.next_siblings)


# In[44]:


soup.head.next_element


# In[46]:


list(soup.head.next_elements)


# In[47]:


soup.head.next_sibling.next_sibling


# In[48]:


# 태그는 제외하고 텍스트만 추출하려면 get_text() 메소드
for each_tag in soup.find_all('p'):
    print(each_tag.get_text())


# In[50]:


for each_tag in soup.find_all("body"):
    print(each_tag.get_text())


# In[49]:


for each_tag in soup.body:
    print(each_tag.get_text())


# In[51]:


type(soup.find_all("body"))


# In[52]:


type(soup.body)


# - from bs4 import BeautifulSoup
# 
# - beautifulsoup라는 라이브러리를 이용하면 html파일 속의 정보에 접근할 수 있겠다.
# - BeautifulSoup(open(".html", 'r').read(), 'html.parser')
# - BeautifulSoup(open(".html", 'r').read(), 'html.parser').find('tag')
# - BeautifulSoup(open(".html", 'r').read(), 'html.parser').find_all('tag')
# - BeautifulSoup(open(".html", 'r').read(), 'html.parser').find_all('tag', class_="class name")
# - BeautifulSoup(open(".html", 'r').read(), 'html.parser').find_all('tag', id="id name")
# - BeautifulSoup(open(".html", 'r').read(), 'html.parser').find_all(class_="class name")
# - BeautifulSoup(open(".html", 'r').read(), 'html.parser').find_all(id="id name")
# - BeautifulSoup(open(".html", 'r').read(), 'html.parser').find_all(id="id name")[index].get_text()
# - BeautifulSoup(open(".html", 'r').read(), 'html.parser').tag
# - BeautifulSoup(open(".html", 'r').read(), 'html.parser').tag.next_sibling
# - list(BeautifulSoup(open(".html", 'r').read(), 'html.parser').tag.next_siblings)
# - BeautifulSoup(open(".html", 'r').read(), 'html.parser').tag.next_element
# - list(BeautifulSoup(open(".html", 'r').read(), 'html.parser').tag.next_elements)
# - BeautifulSoup(open(".html", 'r').read(), 'html.parser').prettify()
# - list(BeautifulSoup(open(".html", 'r').read(), 'html.parser').children)
# - list(BeautifulSoup(open(".html", 'r').read(), 'html.parser').parent)

# #### BeautifulSoup 2: use url

# In[1]:


from urllib.request import urlopen


# In[4]:


url = "http://info.finance.naver.com/marketindex/"
# span 태그의 class name은 value


# In[5]:


page = urlopen(url)


# In[6]:


type(page)


# In[8]:


from bs4 import BeautifulSoup
soup = BeautifulSoup(page, "html.parser")


# In[11]:


soup.find_all('span', class_="value")


# In[12]:


soup.find_all('span', class_="value")[0]


# In[16]:


soup.find_all('span', class_="value")[0].get_text()


# - from urllib.request import urlopen
# - BeautifulSoup(urlopen("url"), 'html.parser')

# ##### example

# http://www.chicagomag.com/Chicago-Magazine/November-2012/Best-Sandwiches-Chicago/

# In[1]:


from bs4 import BeautifulSoup


# In[2]:


from urllib.request import urlopen


# In[5]:


chicago = BeautifulSoup(urlopen("https://goo.gl/wAtv1s"), "html.parser").find_all("div", class_="sammy")


# In[6]:


len(chicago)


# In[7]:


type(chicago)


# In[8]:


chicago[0]


# In[11]:


chicago[0].find_all("div", "sammyListing")[0].get_text()


# In[12]:


import re


# In[16]:


re.split("\r\n|\n", chicago[0].find_all("div", "sammyListing")[0].get_text())


# In[17]:


type(re.split("\r\n|\n", chicago[0].find_all("div", "sammyListing")[0].get_text()))


# In[19]:


re.split("\r\n|\n", chicago[0].find_all("div", "sammyListing")[0].get_text())[:len(re.split("\r\n|\n", chicago[0].find_all("div", "sammyListing")[0].get_text()))-1]


# In[25]:


chicago[0].find('a')['href']


# In[28]:


type(chicago[0].find('a'))
# bs4.element.Tag['attr']


# In[31]:


chicago[0].find('a')['href']


# In[32]:


for i in range(len(chicago)):
    print(chicago[i].find('a')['href'])


# In[39]:


from urllib.request import urljoin
type(urljoin("http://chicagomag.com", chicago[0].find('a')['href']))


# In[41]:


type(chicago[0].find('a')['href'])


# In[42]:


"http://chicagomag.com" + chicago[0].find('a')['href']


# In[52]:


chicago[0]


# In[66]:


rank = []
main_menu = []
cafe_name = []
url_add = []

chicago = BeautifulSoup(urlopen("http://goo.gl/wAtv1s"), "html.parser").find_all("div", class_="sammy")
for i in range(len(chicago)):
    rank.append(chicago[i].find("div", class_="sammyRank").get_text())
    main_menu.append(re.split("\r\n|\n", chicago[i].find("div", class_="sammyListing").get_text())[0])
    cafe_name.append(re.split("\r\n|\n", chicago[i].find("div", class_="sammyListing").get_text())[1])
    url_add.append(urljoin("http://chicagomag.com",chicago[i].find("a")["href"]))


# In[67]:


main_menu


# In[68]:


cafe_name


# In[69]:


url_add


# - 상대경로를 절대경로로, 절대경로를 절대경로로 변경시켜주는 urljoin()
#   - from urllib.request import urljoin
#   - urljoin("http://...", 상대경로 또는 절대경로)
#   - urljoin("http://www.chicagomag.com", BeautifulSoup(urlopen("http://goo.gl/wAtv1s"), "html.parser").find_all("div", class_="sammy")[0].find("a")["href"])
#   
# - 정규표현식(regular expression)
#   - import re
#   - re.split("정규표현", str)
#   - re.split("\r\n|\n", (BeautifulSoup(urlopen("http://goo.gl/wAtv1s"), "html.parser").find_all("div", class_="sammy")[0].find("div", class_="sammyListing").get_text()))[0] == "BLT"

# #### pandas화

# In[72]:


df = {"Rank":rank, "Menu":main_menu, "Cafe":cafe_name, 'URL':url_add}


# In[73]:


import pandas as pd


# In[74]:


df = pd.DataFrame(df)


# In[75]:


df


# In[77]:


df = df[["Rank", "Cafe", "Menu", "URL"]]


# In[78]:


df


# In[79]:


df.to_csv("./chicago.csv")
'''
Signature: df.to_csv(path_or_buf=None, sep=',', na_rep='', float_format=None, columns=None, header=True, 
index=True, index_label=None, mode='w', encoding=None, compression=None, quoting=None, quotechar='"', 
line_terminator='\n', chunksize=None, tupleize_cols=False, date_format=None, doublequote=True, escapechar=None, 
decimal='.')
Docstring:
Write DataFrame to a comma-separated values (csv) file

Parameters
----------
path_or_buf : string or file handle, default None
    File path or object, if None is provided the result is returned as
    a string.
sep : character, default ','
    Field delimiter for the output file.
na_rep : string, default ''
    Missing data representation
float_format : string, default None
    Format string for floating point numbers
columns : sequence, optional
    Columns to write
header : boolean or list of string, default True
    Write out column names. If a list of string is given it is assumed
    to be aliases for the column names
index : boolean, default True
    Write row names (index)
index_label : string or sequence, or False, default None
    Column label for index column(s) if desired. If None is given, and
    `header` and `index` are True, then the index names are used. A
    sequence should be given if the DataFrame uses MultiIndex.  If
    False do not print fields for index names. Use index_label=False
    for easier importing in R
mode : str
    Python write mode, default 'w'
encoding : string, optional
    A string representing the encoding to use in the output file,
    defaults to 'ascii' on Python 2 and 'utf-8' on Python 3.
compression : string, optional
    a string representing the compression to use in the output file,
    allowed values are 'gzip', 'bz2', 'xz',
    only used when the first argument is a filename
line_terminator : string, default ``'\n'``
    The newline character or character sequence to use in the output
    file
quoting : optional constant from csv module
    defaults to csv.QUOTE_MINIMAL. If you have set a `float_format`
    then floats are converted to strings and thus csv.QUOTE_NONNUMERIC
    will treat them as non-numeric
quotechar : string (length 1), default '\"'
    character used to quote fields
doublequote : boolean, default True
    Control quoting of `quotechar` inside a field
escapechar : string (length 1), default None
    character used to escape `sep` and `quotechar` when appropriate
chunksize : int or None
    rows to write at a time
tupleize_cols : boolean, default False
    write multi_index columns as a list of tuples (if True)
    or new (expanded format) if False)
date_format : string, default None
    Format string for datetime objects
decimal: string, default '.'
    Character recognized as decimal separator. E.g. use ',' for
    European data

    .. versionadded:: 0.16.0
File:      ~/anaconda3/lib/python3.6/site-packages/pandas/core/frame.py
Type:      method
'''


# In[80]:


get_ipython().run_line_magic('ls', '')


# - pandas의 DataFrame으로 만들고싶다? dict화 시키고 DataFrame화 시키면 끝이다.
#   - import pandas as pd
#   - df = {}
#   - df["Rank"] = rank; df["Cafe"] = cafe_name; ...
#   - pd.DataFrame(df)
#   
# 
# - pandas의 DataFrame을 csv로 저장하고 싶다? to_csv() csv를 읽고싶다? read_csv()
#   - pd.to_csv("path and file name", encoding="UTF-8")
#   - pd.read_csv("./chicago.csv")

# In[84]:


pd.read_csv("./chicago.csv", index_col=0)


# In[91]:


df["URL"][0]


# 가격, 주소 부분을 가져오고 싶다는 목적이 있다. 페이지를 보니 p, addy 부분이다.

# In[105]:


((BeautifulSoup(urlopen(df["URL"][0]), "html.parser").find_all("p", class_="addy")[0].get_text()).split()[0])[:-1]


# In[104]:


(' '.join((BeautifulSoup(urlopen(df["URL"][0]), "html.parser").find_all("p", class_="addy")[0].get_text()).split()[1:-2]))[:-1]


# In[108]:


from tqdm import tqdm_notebook


# In[109]:


df.index


# In[112]:


for i in (df.index):
    print(i)


# In[118]:


price = []
address = []

for n in tqdm_notebook(df.index):
    price.append(((BeautifulSoup(urlopen(df["URL"][n]), "lxml").find_all("p", class_="addy")[0].get_text()).split()[0])[:-1])
    address.append((' '.join((BeautifulSoup(urlopen(df["URL"][n]), "lxml").find_all("p", class_="addy")[0].get_text()).split()[1:-2]))[:-1])


# In[119]:


price


# In[120]:


address


# In[122]:


df["price"] = price
df["address"] = address


# In[125]:


df


# In[126]:


df[["Rank", "Cafe", "Menu", "price", "address", "URL"]]


# In[127]:


df.set_index("Rank", inplace = True)


# In[128]:


df


# In[129]:


df.to_csv("./chicago.csv")


# - 반복문에서 걸리는 수행 시간을 상태바로 나타내주는 모듈; tqdm
#   - 설치: conda install -c conda-force tqdm
#   - from tqdm import tqdm_notebook
#   - for n in tqdm_notebook(df.index):...
#     - df.index 그 자체가 숫자
# 
