import numpy as np
import pandas as pd
import re

a1 = " 2.094(3) &      380.88(1) &    2.97(3) &     8.63(1) &   2.13(2) &    10.00(1) &       2.76(2) &  20.27(1) &  65.06(1) &  6.95(1) "
b1 = a1.replace('&', '+')
res1 = re.findall(r'\(.*?\)', b1)

converted_list1 = []

for element in res1:
    converted_list1.append(element.strip("(,)"))

print(converted_list1)

for i in range(0, len(converted_list1)):
    converted_list1[i] = float(converted_list1[i])

new_res1 = sum(converted_list1)
avg1 = new_res1 / 10
print(avg1)
print('\n')
print('\n')




"""
a2 = "  100(4) & 100(4) & 100(4.5) & 100(3) & 100(5) & 100(4.5) & 100(5) & 100(4) & 100(3.5)  "
b2 = a2.replace('&', '+')
res2 = re.findall(r'\(.*?\)', b2)

converted_list2 = []

for element in res2:
    converted_list2.append(element.strip("(,)"))

print(converted_list2)

for i in range(0, len(converted_list2)):
    converted_list2[i] = float(converted_list2[i])

new_res2 = sum(converted_list2)
avg2 = new_res2 / 9
print(avg2)
print('\n')
print('\n')


a3 = " 60(2.5) & 60(3.5) & 60(4.5) & 60(3.5) & 60(4.5) & 60(5) & 60(4.5) & 62(2) & 62(1)   "
b3 = a3.replace('&', '+')
res3 = re.findall(r'\(.*?\)', b3)

converted_list3 = []

for element in res3:
    converted_list3.append(element.strip("(,)"))

print(converted_list3)

for i in range(0, len(converted_list3)):
    converted_list3[i] = float(converted_list3[i])

new_res3 = sum(converted_list3)
avg3 = new_res3 / 9
print(avg3)
print('\n')
print('\n')


final_average_rank = (avg1 + avg2) / 2

print(final_average_rank)

"""
