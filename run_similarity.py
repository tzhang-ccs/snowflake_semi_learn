import os
import numpy as np

beg = np.arange(0,500,50)
end = beg + 50
idx = list(zip(beg,end))

for i, curr_id in enumerate(idx):
    print(i, curr_id)
    os.system(f'python similarity.py -b {curr_id[0]} -e {curr_id[1]} -l {i+1} &')
