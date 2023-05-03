import matplotlib.pyplot as plt
import os
import numpy as np
import shutil
if not os.path.exists('vis/'):
    os.makedirs('vis/')
else:
    shutil.rmtree('vis/')
    os.makedirs('vis/')
function_pool = []
for i, functions in enumerate(os.listdir('transfer_function')):
    data = np.load('transfer_function/' + functions)
    response = data['response']
    variance = data['variance']
    function_pool.append(np.expand_dims(np.column_stack([response, variance]), axis=0))
    plt.plot(response)
    plt.savefig('vis/' + str(i) + '.png')
function_pool = np.row_stack(function_pool).astype(np.float32)
np.save('function_pool.npy', function_pool)
print('get function pool of', function_pool.shape[0])
