import numpy as np
from tqdm import tqdm
import time

# for i in tqdm(range(100)):
#     time.sleep(0.1)
#     # print(f"Progress: {i+1}/100", end="\r")
#     tqdm.write('Progress: %d/100' % (i+1), end='\r')


with tqdm(total=100) as pbar:
    for i in range(100):
        time.sleep(0.1)
        pbar.set_description('Processing item %d' % (i+1))
        pbar.update(1)
