#%%
import numpy as np
import matplotlib.pyplot as plt
# %%
fn = "/mnt/sda/avirmani/libmsi/npy_files/patch_15_nmf_20.npy"
data = np.load(fn, allow_pickle=True)[()]
#%%
dmin = data['patches'].min()
dmax = 0.12 # 2*data['patches'].max()/3
# %%
fig, ax = plt.subplots(5, 2, figsize=(15,15))
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
for i in range(5):
    ax[i,0].imshow(data['patches'][i].reshape(15,15,20)[:,:, 2], vmin=dmin, vmax=dmax)
    im = ax[i,1].imshow(data['patches'][7999-i].reshape(15,15,20)[:,:, 2], vmin=dmin, vmax=dmax)
fig.colorbar(im, cax=cbar_ax)
plt.savefig('patches.svg')
# %%
## plotting the nmf and pca components relative to h&e components
pca_fn = 'pca_dmsi0002_neg_tics'