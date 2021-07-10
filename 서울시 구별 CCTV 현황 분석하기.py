#!/usr/bin/env python
# coding: utf-8

# # 서울시 CCTV 현황

# In[20]:


import pandas as pd


# In[21]:


import os
os.getcwd()

CCTV_Seoul = pd.read_csv('C:/Users/kccistc/Desktop/파이썬 실습/참고 자료/CCTV_in_Seoul.csv', encoding = 'utf-8')
CCTV_Seoul.head()


# In[22]:


CCTV_Seoul.columns


# In[23]:


CCTV_Seoul.columns[0]


# In[24]:


CCTV_Seoul.rename(columns = {CCTV_Seoul.columns[0] : '구별'}, inplace = True)
CCTV_Seoul.head()


# # 엑셀파일 읽기 - 서울시 인구현황

# In[25]:


#pop_Seoul = pd.read_excel('C:/Users/kccistc/Desktop/파이썬 실습/참고 자료/01. population_in_Seoul.xls', encoding = 'utf-8')
pop_Seoul = pd.read_excel('C:/Users/kccistc/Desktop/파이썬 실습/참고 자료/01. population_in_Seoul.xls')
pop_Seoul.head()


# In[26]:


# pop_Seoul = pd.read_excel('C:/Users/kccistc/Desktop/파이썬 실습/참고 자료/01. population_in_Seoul.xls',
#                             header = 2,
#                             usecols = 'B, D, G, J, N',
#                             encoding = 'utf-8')
pop_Seoul = pd.read_excel('C:/Users/kccistc/Desktop/파이썬 실습/참고 자료/01. population_in_Seoul.xls',
                           header = 2,
                           usecols = 'B, D, G, J, N')
pop_Seoul.head()


# In[27]:


pop_Seoul.rename(columns = {pop_Seoul.columns[0] : '구별',
                           pop_Seoul.columns[1] : '인구수',
                           pop_Seoul.columns[2] : '한국인',
                           pop_Seoul.columns[3] : '외국인',
                           pop_Seoul.columns[4] : '고령자'}, inplace = True)
pop_Seoul.head()


# # Pandas기초

# In[28]:


import pandas as pd
import numpy as np


# In[29]:


s = pd.Series([1,3,5,np.nan,6,8])
s


# In[30]:


dates = pd.date_range('20130101', periods = 6)
dates


# In[31]:


df = pd.DataFrame(np.random.randn(6,4), index = dates, columns = ['A','B','C','D'])
df


# In[32]:


df.head()


# In[33]:


df.head(3)


# In[34]:


df.index


# In[35]:


df.columns


# In[36]:


df.values


# In[37]:


df.info()


# In[38]:


df.describe()


# In[39]:


df.sort_values(by = 'B', ascending = False)


# In[40]:


df


# In[41]:


df['A']


# In[42]:


df[0:3]


# In[43]:


df['20130102':'20130104']


# In[44]:


df.loc[dates[0]]


# In[45]:


df.loc[:,['A','B']]


# In[46]:


df.loc['20130102':'20130104',['A','B']]


# In[47]:


df.loc['20130102', ['A','B']]


# In[48]:


df.loc[dates[0],'A']


# In[49]:


df.iloc[3]


# In[50]:


df.iloc[3:5,0:2]


# In[51]:


df.iloc[[1,2,4],[0,2]]


# In[52]:


df.iloc[:,1:3]


# In[53]:


df


# In[54]:


df[df.A > 0]


# In[55]:


df[df > 0]


# In[56]:


df2 = df.copy()


# In[57]:


df2['E'] = ['one','one','two','three','four','three']
df2


# In[58]:


df2['E'].isin(['two','four'])


# In[59]:


df2[df2['E'].isin(['two','four'])]


# In[60]:


df


# In[61]:


df.apply(np.cumsum)


# In[62]:


df.apply(lambda x: x.max() - x.min())


# # CCTV 데이터 파악하기

# In[63]:


CCTV_Seoul.head()


# In[64]:


CCTV_Seoul.sort_values(by = '소계',ascending = True).head(5)


# In[65]:


CCTV_Seoul.sort_values(by = '소계',ascending = False).head(5)


# In[66]:


CCTV_Seoul['최근증가율'] = (CCTV_Seoul['2016년'] + CCTV_Seoul['2015년'] +                        CCTV_Seoul['2014년']) / CCTV_Seoul['2013년도 이전'] * 100
CCTV_Seoul.sort_values(by = '최근증가율', ascending = False).head(5)


# # 서울시 인구 데이터 파악하기

# In[67]:


pop_Seoul.head()


# In[68]:


pop_Seoul.drop([0], inplace = True)
pop_Seoul.head()


# In[69]:


pop_Seoul['구별'].unique()


# In[70]:


pop_Seoul[pop_Seoul['구별'].isnull()]


# In[71]:


pop_Seoul.drop([26], inplace = True)
pop_Seoul.head()


# In[73]:


pop_Seoul['외국인비율'] = pop_Seoul['외국인'] / pop_Seoul['인구수'] * 100
pop_Seoul['고령자비율'] = pop_Seoul['고령자'] / pop_Seoul['인구수'] * 100
pop_Seoul.head()


# In[74]:


pop_Seoul.sort_values(by = '인구수', ascending = False).head(5)


# In[75]:


pop_Seoul.sort_values(by = '외국인', ascending = False).head(5)


# In[76]:


pop_Seoul.sort_values(by = '외국인비율', ascending = False).head(5)


# In[77]:


pop_Seoul.sort_values(by = '고령자', ascending = False).head(5)


# In[78]:


pop_Seoul.sort_values(by = '고령자비율', ascending = False).head(5)


# # Pandas 고급 두 DataFrame 병합하기

# In[84]:


df1 = pd.DataFrame({'A' : ['A0', 'A1', 'A2', 'A3'],
                   'B' : ['B0', 'B1', 'B2', 'B3'],
                   'C' : ['C0', 'C1', 'C2', 'C3'],
                   'D' : ['D0', 'D1', 'D2', 'D3']},
                  index = [0, 1, 2, 3])
df2 = pd.DataFrame({'A' : ['A4', 'A5', 'A6', 'A7'],
                   'B' : ['B4', 'B5', 'B6', 'B7'],
                   'C' : ['C4', 'C5', 'C6', 'C7'],
                   'D' : ['D4', 'D5', 'D6', 'D7']},
                  index = [4, 5, 6, 7])
df3 = pd.DataFrame({'A' : ['A8', 'A9', 'A10', 'A11'],
                   'B' : ['B8', 'B9', 'B10', 'B11'],
                   'C' : ['C8', 'C9', 'C10', 'C11'],
                   'D' : ['D8', 'D9', 'D10', 'D11']},
                  index = [8, 9, 10, 11])


# In[85]:


df1


# In[82]:


df2


# In[86]:


df3


# In[87]:


result = pd.concat([df1, df2, df3])
result


# In[88]:


result = pd.concat([df1, df2, df3], keys = ['x', 'y', 'z'])
result


# In[89]:


result.index


# In[90]:


result.index.get_level_values(0)


# In[91]:


result.index.get_level_values(1)


# In[92]:


result


# In[93]:


df4 = pd.DataFrame({'B' : ['B2', 'B3', 'B6', 'B7'],
                   'D' : ['D2', 'D3', 'D6', 'D7'],
                   'F' : ['F2', 'F3', 'F6', 'F7']},
                  index = [2, 3, 6, 7])
result = pd.concat([df1, df4], axis = 1)


# In[94]:


df1


# In[95]:


df4


# In[96]:


result


# In[97]:


result = pd.concat([df1, df4], axis = 1, join = 'inner')
result


# In[98]:


result = pd.concat([df1, df4], ignore_index = True)
result


# In[107]:


left = pd.DataFrame({'key' : ['K0', 'K4', 'K2', 'K3'],
                    'A' : ['A0', 'A4', 'A2', 'A3'],
                    'B' : ['B0', 'B4', 'B2', 'B3']})
right = pd.DataFrame({'C' : ['C0', 'C1', 'C2', 'C3'],
                    'D' : ['D0', 'D1', 'D2', 'D3'],
                    'key' : ['K0', 'K1', 'K2', 'K3']})
                    


# In[100]:


left


# In[112]:


right


# In[113]:


pd.merge(left, right, on = 'key')


# In[110]:


pd.merge(left, right, how = 'right', on = 'key')


# In[109]:


pd.merge(left, right, how = 'left', on = 'key')


# In[114]:


pd.merge(left, right, how = 'outer', on = 'key')


# In[115]:


pd.merge(left, right, how = 'inner', on = 'key')


# # CCTV 데이터와 인구 데이터 합치고 분석하기

# In[116]:


data_result = pd.merge(CCTV_Seoul, pop_Seoul, on = '구별')
data_result.head()


# In[117]:


del data_result['2013년도 이전']
del data_result['2014년']
del data_result['2015년']
del data_result['2016년']
data_result.head()


# In[118]:


data_result.set_index('구별', inplace = True)
data_result.head()


# In[119]:


np.corrcoef(data_result['고령자비율'],data_result['소계'])


# In[120]:


np.corrcoef(data_result['외국인비율'],data_result['소계'])


# In[121]:


np.corrcoef(data_result['인구수'],data_result['소계'])


# In[122]:


data_result.sort_values(by = '소계', ascending = False).head(5)


# In[123]:


data_result.sort_values(by = '인구수', ascending = False).head(5)


# # 그래프 그리기 기초 - matplotlib

# In[124]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[125]:


plt.figure()
plt.plot([1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,0])
plt.show()


# In[127]:


import numpy as np
t = np.arange(0,12,0.01)
y = np.sin(t)


# In[128]:


plt.figure(figsize = (10,6))
plt.plot(t,y)
plt.show()


# In[130]:


plt.figure(figsize = (10,6))
plt.plot(t,y)
plt.grid()
plt.show()


# In[131]:


plt.figure(figsize = (10,6))
plt.plot(t,y)
plt.grid()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.show()


# In[134]:


plt.figure(figsize = (10,6))
plt.plot(t,y)
plt.grid()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.show()


# In[135]:


plt.figure(figsize = (10,6))
plt.plot(t, np.sin(t))
plt.plot(t, np.cos(t))
plt.grid()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.show()


# In[137]:


plt.figure(figsize = (10,6))
plt.plot(t, np.sin(t), label = 'sin')
plt.plot(t, np.cos(t), label = 'cos')
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.show()


# In[138]:


plt.figure(figsize = (10,6))
plt.plot(t, np.sin(t), lw = 3, label = 'sin')
plt.plot(t, np.cos(t), 'r', label = 'cos')
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.show()


# In[139]:


plt.figure(figsize = (10,6))
plt.plot(t, np.sin(t), lw = 3, label = 'sin')
plt.plot(t, np.cos(t), 'r', label = 'cos')
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.ylim(-1.2, 1.2)
plt.xlim(0,np.pi)
plt.show()


# In[140]:


t = np.arange(0, 5, 0.5)

plt.figure(figsize = (10,6))
plt.plot(t, t, 'r--')
plt.plot(t, t**2, 'bs')
plt.plot(t, t**3, 'g^')
plt.show()


# In[141]:


t = np.arange(0, 5, 0.5)

plt.figure(figsize = (10,6))
pl1 = plt.plot(t, t**2, 'bs')

plt.figure(figsize = (10,6))
plt.plot(t, t**3, 'g^')

plt.show()


# In[142]:


t = [0,1,2,3,4,5,6]
y = [1,4,5,8,9,5,3]

plt.figure(figsize = (10,6))
plt.plot(t, y, color = 'green')
plt.show()


# In[146]:


plt.figure(figsize = (10,6))
plt.plot(t, y, color = 'green', linestyle = 'dashed')
plt.show()


# In[147]:


plt.figure(figsize = (10,6))
plt.plot(t, y, color = 'green', linestyle = 'dashed', marker = 'o')
plt.show()


# In[148]:


plt.figure(figsize = (10,6))
plt.plot(t, y, color = 'green', linestyle = 'dashed', marker = 'o',
        markerfacecolor = 'blue')
plt.show()


# In[149]:


plt.figure(figsize = (10,6))
plt.plot(t, y, color = 'green', linestyle = 'dashed', marker = 'o',
        markerfacecolor = 'blue', markersize = 12)

plt.xlim([-0.5, 6.5])
plt.ylim([0.5, 9.5])
plt.show()


# In[151]:


t = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([9,8,7,9,8,3,2,4,3,4])


# In[152]:


plt.figure(figsize = (10,6))
plt.scatter(t,y)
plt.show()


# In[153]:


plt.figure(figsize = (10,6))
plt.scatter(t,y, marker = '>')
plt.show()


# In[154]:


colormap = t

plt.figure(figsize = (10,6))
plt.scatter(t, y, s = 50, c = colormap, marker = '>')
plt.show()


# In[155]:


colormap = t

plt.figure(figsize = (10,6))
plt.scatter(t, y, s = 50, c = colormap, marker = '>')
plt.colorbar()
plt.show()


# In[156]:


s1 = np.random.normal(loc = 0, scale = 1, size = 1000)
s2 = np.random.normal(loc = 5, scale = 0.5, size = 1000)
s3 = np.random.normal(loc = 10, scale = 2, size = 1000)


# In[157]:


plt.figure(figsize = (10,6))
plt.plot(s1, label = 's1')
plt.plot(s2, label = 's2')
plt.plot(s3, label = 's3')
plt.legend()
plt.show()


# In[158]:


plt.figure(figsize = (10,6))
plt.boxplot((s1, s2, s3))
plt.grid()
plt.show()


# In[159]:


plt.figure(figsize = (10, 6))

plt.subplot(221)
plt.subplot(222)
plt.subplot(212)

plt.show()


# In[160]:


plt.figure(figsize = (10,6))

plt.subplot(411)
plt.subplot(423)
plt.subplot(424)
plt.subplot(413)
plt.subplot(414)

plt.show()


# In[161]:


t = np.arange(0,5,0.01)

plt.figure(figsize = (10,12))

plt.subplot(411)
plt.plot(t, np.sqrt(t))
plt.grid()

plt.subplot(423)
plt.plot(t, t**2)
plt.grid()

plt.subplot(424)
plt.plot(t, t**3)
plt.grid()

plt.subplot(413)
plt.plot(t, np.sin(t))
plt.grid()

plt.subplot(414)
plt.plot(t, np.cos(t))
plt.grid()

plt.show()


# # CCTV와 인구현황 그래프로 분석하기

# In[162]:


import platform

from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':
    rc('font', family = 'AppleGothic')
elif platform.system() == 'Windows':
    path = "C:/Users/kccistc/Desktop/파이썬 실습/참고 자료/malgun.ttf"
    font_name = font_manager.FontProperties(fname = path).get_name()
    rc('font', family = font_name)
else:
    print('Unknown system... sorry~~~~')


# In[163]:


data_result.head()


# In[165]:


plt.figure()
data_result['소계'].plot(kind = 'barh', grid = True, figsize = (10,10))
plt.show()


# In[167]:


data_result['소계'].sort_values().plot(kind = 'barh', grid = True, figsize = (10,10))
plt.show()


# In[168]:


data_result['CCTV비율'] = data_result['소계'] / data_result['인구수'] * 100

data_result['CCTV비율'].sort_values().plot(kind = 'barh', grid = True, figsize = (10,10))
plt.show()


# In[169]:


plt.figure(figsize = (6,6))
plt.scatter(data_result['인구수'], data_result['소계'], s = 50)
plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()


# In[170]:


fp1 = np.polyfit(data_result['인구수'], data_result['소계'], 1)
fp1


# In[172]:


f1 = np.poly1d(fp1)
fx = np.linspace(100000, 700000, 100)


# In[173]:


plt.figure(figsize = (10,10))
plt.scatter(data_result['인구수'], data_result['소계'], s = 50)
plt.plot(fx, f1(fx), ls = 'dashed', lw = 3, color = 'g')
plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()


# # 조금 더 설득력 있는 자료 만들기

# In[174]:


fp1 = np.polyfit(data_result['인구수'], data_result['소계'], 1)

f1 = np.poly1d(fp1)
fx = np.linspace(100000, 700000, 100)

data_result['오차'] = np.abs(data_result['소계'] - f1(data_result['인구수']))

df_sort = data_result.sort_values(by = '오차', ascending = False)
df_sort.head()


# In[175]:


plt.figure(figsize = (14,10))
plt.scatter(data_result['인구수'], data_result['소계'],
           c = data_result['오차'], s = 50)
plt.plot(fx, f1(fx), ls = 'dashed', lw = 3, color = 'g')

for n in range(10):
    plt.text(df_sort['인구수'][n] * 1.02, df_sort['소계'][n] * 0.98,
            df_sort.index[n], fontsize = 15)
    
plt.xlabel('인구수')
plt.ylabel('인구당비율')
plt.colorbar()
plt.grid()
plt.show()


# In[ ]:




