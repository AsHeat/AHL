#우리나라 인구 분석

import csv
data = csv.reader(open('age.csv','r'), delimiter = ",")
temp = [ d for d in data]                                   #리스트로 만듬

'''
temp=[]
for d in data:
    temp.append(d)
'''
temp.remove(temp[0])

woman = {'age': 0, 'loc': ""}

for row in temp:
    row[2] = float(row[2])
    if woman['age'] < row[2]:
        woman['age'] = row[2]
        woman['loc'] = row[0]

man = {'age': 50, 'loc': ""}

for row in temp:
    row[1] = float(row[1])
    if man['age'] > row[1]:
        man['age'] = row[1]
        man['loc'] = row[0]


print('여성 평균 연령이 가장 높은 지역은', woman['loc'], '이고 평균 연령은', woman['age'], '입니다.')
print('남성 평균 연령이 가장 낮은 지역은', man['loc'], '이고 평균 연령은', man['age'], '입니다.')