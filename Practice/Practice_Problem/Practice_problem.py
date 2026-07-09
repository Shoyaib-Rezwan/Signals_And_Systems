import numpy as np
import matplotlib.pyplot as plt

##################################### Problem-1  ##########################################
# n=int(input())
# events=[]
# for i in range(n):
#     rawInput= input().split()
#     events.append((rawInput[0], int(rawInput[1]), int(rawInput[2])))

# dictionary={}

# for e in events:
#     if e[0] not in dictionary:
#         dictionary[e[0]]={'total_dur':0, 'total_energy':0}
#     dictionary[e[0]]['total_dur']+=e[2]
#     dictionary[e[0]]['total_energy']+=e[1]*e[1]*e[2]

# dictionary=dict(sorted(dictionary.items()))

# for d in dictionary.items():
#     print(d[0], d[1]['total_dur'], d[1]['total_energy'])


# max_val=max(dictionary.items(), key= lambda item:item[1]['total_energy'])

# print(f"TOP {max_val[0]} {max_val[1]['total_energy']}")

##################################### Problem-2  ##########################################

"""input:

    5 4
    70 80 90 100
    85 88 72 98
    58 67 79 90
    82 54 77 12
    11 13 19 27
    120 120 120 120
"""

# import numpy as np
# s,m=map(int, input().split())

# mat=np.zeros((s,m), dtype=np.int8)

# for i in range(s):
#     mat[i]=list(map(int,input().split()))

# print(mat)

# max_marks=np.zeros(s, dtype=np.int8)

# max_marks= list(map(int,input().split()))

# p= np.round(mat/max_marks * 100,2)

# print(f"Percentage Marks:\n{p}")

# stu_summary=np.hstack((np.reshape(np.mean(mat, axis=1), (-1, 1)),np.reshape(np.argmax(mat, axis=1), (-1, 1))))

# sub_summary=np.hstack((np.reshape(np.mean(mat, axis=0), (-1, 1)),np.reshape(np.argmax(mat, axis=0), (-1, 1))))

# print(f"Student Summary:\n{stu_summary}")

# print(f"Subject Summary:\n{sub_summary}")

# stu_ids=np.argsort(np.mean(p, axis=1))[::-1]

# print(f"Student ID based on highest mean percentage:\n{stu_ids}")

# grades=np.select(condlist=[p>=80, p>=60, p>=40], choicelist=['A','B','C'],default='D')

# print(f"Grades:\n{grades}")

##################################### Problem-3  ##########################################

"""input:

    5 3
    A
    100
    70 80 90
    B
    101
    60 70 90
    C
    102
    70 70 80
    D
    103
    60 60 100
    E
    104
    100 20 10
"""

# class Student:
#     uni_name=""
#     def __init__(self):
#         self.name=""
#         self.stuID=-1
#         self.grades=[]
#     def add_grade(self, grade):
#         self.grades.append(grade)
#     def get_average(self):
#         return np.mean(self.grades)
#     def get_letter_grade(self):
#         avg= float(self.get_average())
#         return'A' if avg>=80 else ('B' if avg>=60 else ('C' if avg>=40 else 'D'))
#     def __str__(self):
#         return f"Name: {self.name}\nStuID: {self.stuID}\nGarde: {self.get_average()}: {self.get_letter_grade()}"

# s,m=map(int, input().split())

# students=np.empty(s,dtype=object)

# for i in range (s):
#     name=input()
#     stuId= int(input())
#     grades=list(map(int,input().split()))
#     students[i]=Student()
#     students[i].name=name
#     students[i].stuID=stuId
#     for grade in grades:
#         students[i].add_grade(grade)

# for student in students:
#     print(student)

##################################### Problem-4  ##########################################

# Monthly data 
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] 
# Sales data (in thousands) 
product_a = np.array([23, 25, 28, 30, 35, 38, 42, 45, 48, 50, 53, 55]) 
product_b = np.array([30, 32, 31, 33, 35, 37, 39, 41, 43, 45, 47, 50]) 
product_c = np.array([15, 18, 22, 25, 28, 32, 35, 38, 42, 45, 48, 52])

fig, axs= plt.subplots(2,2, figsize=(14,10), layout='constrained', label='Practice Plot')

#first plot
axs[0,0].plot(months, product_a, color='blue',label='product_a')
axs[0,0].plot(months, product_b, color='green', linestyle='--',label='product_b')
axs[0,0].plot(months, product_c, color='red', linestyle='--',label='product_c')
axs[0,0].set_title( "Monthly Sales Trends 2024" )
axs[0,0].set_xlabel("Month")
axs[0,0].set_ylabel("Sales (in thousands)")
axs[0,0].legend()
axs[0,0].grid(True, alpha=0.2)

#second plot
container=axs[0,1].bar(['Product A', 'Product B', 'Product C'],[product_a[11], product_b[11], product_c[11]],color=['red','green', 'blue'])
axs[0,1].set_title( "December Sales Comparison" )
axs[0,1].set_xlabel("Month")
axs[0,1].set_ylabel("Sales (in thousands)")
axs[0,1].bar_label(container)
axs[0,1].set_ylim(0, max([product_a[11], product_b[11], product_c[11]])+10)
axs[0,1].grid(True, axis='y', alpha=0.2)

#third plot
axs[1,0].scatter(months, product_a,color='blue',s=50)
axs[1,0].set_title("Product A Sales Distribution")
axs[1,0].set_xlabel("Month")
axs[1,0].set_ylabel("Sales (in thousands)")
axs[1,0].grid(True, alpha=0.2)

#fourth plot
x=['Q1','Q2','Q3','Q4']
quarters_A=np.sum(np.reshape(product_a,(4,3)),axis=1)
quarters_B=np.sum(np.reshape(product_b,(4,3)),axis=1)
quarters_C=np.sum(np.reshape(product_c,(4,3)),axis=1)
axs[1,1].bar(x,quarters_A,color='blue',label="Product A")
axs[1,1].bar(x,quarters_B,bottom=quarters_A,color='green',label="Product B")
axs[1,1].bar(x,quarters_C,bottom=quarters_A+quarters_B,color='red',label="Product C")
axs[1,1].set_title("Quarterly Sales (Stacked)" )
axs[1,1].set_xlabel( "Quarter" )
axs[1,1].set_ylabel("Sales (in thousands)")
axs[1,1].legend()
axs[1,1].grid(True, axis='y', alpha=0.2)

# plt.bar
fig.suptitle("2024 Product Sales Analysis" )
# plt.tight_layout()
plt.show()