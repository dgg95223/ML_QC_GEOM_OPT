import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

with open('./debug_copy.txt', 'r') as file:
    results = file.readlines()

data = []
_data = []
for ii, i in enumerate(results):
    if i != '\n':
        _data_ = i.split()[0:2]
        _data.append(_data_)   
    elif i == '\n':
        data.append(np.array(_data, dtype=np.float64))
        _data = []

fig,axs = plt.subplots(figsize=(16,9))
for ii, i in enumerate(data):
    axs.plot(list(range(1, len(i)+1)),i[:,0], label='QC '+str(ii+1)) # QC
    axs.plot(list(range(1, len(i)+1)),i[:,1], label='ML '+str(ii+1)) # ML

# plt.rc('xtick', labelsize=18)
# plt.rc('ytick', labelsize=18)
axs.set_xlabel('Cycle', fontsize=20)
axs.set_ylabel('Total Energy (eV)', fontsize=20)
axs.set_xlim(0, 53)
axs.set_ylim(-4137, -4130)
axs.set_title('QC energy VS ML energy', y=1.05, fontsize=24)
axs.legend(fontsize=11)
    

plt.tight_layout()
plt.savefig('comparsion.pdf')