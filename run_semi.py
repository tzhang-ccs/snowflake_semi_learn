import os
import numpy as np

for j in [200,100,50,25]:
    for i in range(10):
        out_name = f'out.seed_{i}.num_{j}'
        print(out_name)
        os.system(f'python snowflake_semi.py -p train -s {i} -n {j} | tee {out_name}')
