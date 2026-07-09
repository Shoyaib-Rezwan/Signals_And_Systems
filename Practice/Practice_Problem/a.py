n=int(input())
lst=[]
summary={}
for i in range(n):
    raw=input().split()
    lst.append((raw[0], int(raw[1]), int(raw[2])))
    if raw[0] not in summary:
        summary[raw[0]]={'total_duration':0, 'total_energy':0}
    summary[raw[0]]['total_duration']+=int(raw[2])
    summary[raw[0]]['total_energy']+=int(raw[1])*int(raw[1])*int(raw[2])

print(lst)
print(summary)
