import numpy as np

expert_list=list()
expert_list.append({'p': 'dd'})
expert_list.append({'p': 'cc'})
expert_list.append({'p': 'cc'})
expert_list.append({'p': 'cc'})
expert_list.append({'p': 'cc'})
expert_list.append({'p': 'cc'})
expert_list.append({'p': 'cc'})
expert_list.append({'p': 'cc'})
expert_list.append({'p': 'cc'})
expert = np.array(expert_list)

for i in range(len(expert)):
    if(i+1)%3==0:
        print('yeah!')
    print(i)
