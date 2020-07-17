import argparse

import torch
import torchvision.utils as vutils
import numpy as np
# import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', required=True, help='Checkpoint to load path from')
args = parser.parse_args()

from models.dsprites_model import Generator

# Load the checkpoint file
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator().to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['netG'])
print(netG)

c = np.linspace(-2, 2, 10).reshape(1, -1)
c = np.repeat(c, 3, 0).reshape(-1, 1)
c = torch.from_numpy(c).float().to(device)
c = c.view(-1, 1, 1, 1)

zeros = torch.zeros(30, 1, 1, 1, device=device)

# Continuous latent code.
# c2 = torch.cat((c, zeros), dim=1)
# c3 = torch.cat((zeros, c), dim=1)
num_con_c = 4
num_z = 64
cs = []
for i in range(num_con_c):
	tmp = []
	for j in range(num_con_c):
		if j == i:
			tmp.append(c)
		else:
			tmp.append(zeros)
	cs.append(torch.cat(tmp, dim=1))

idx = np.arange(3).repeat(10)
dis_c = torch.zeros(30, 3, 1, 1, device=device)
dis_c[torch.arange(0, 30), idx] = 1.0
# Discrete latent code.
c1 = dis_c.view(30, -1, 1, 1)

z = torch.randn(1, num_z, 1, 1, device=device).repeat(30, 1, 1 ,1)

# # To see variation along c2 (Horizontally) and c1 (Vertically)
# noise1 = torch.cat((z, c1, c2), dim=1)
# # To see variation along c3 (Horizontally) and c1 (Vertically)
# noise2 = torch.cat((z, c1, c3), dim=1)

for i in range(num_con_c):
	# Generate image.
	noise = torch.cat((z, c1, cs[i]), dim=1)
	with torch.no_grad():
	    generated_img = netG(noise).detach().cpu()
	vutils.save_image(vutils.make_grid(generated_img, nrow=10, padding=2, normalize=True),
	        'sample_dsprites/c0_c%d.jpg' % i)


# To see variation along z (horizontally) and c1 (Vertically)
z = torch.randn(10, num_z, 1, 1, device=device).repeat(3, 1, 1, 1)
tmp = [z, c1]
for i in range(num_con_c):
	tmp.append(zeros)
noise = torch.cat(tmp, dim=1)
with torch.no_grad():
    generated_img = netG(noise).detach().cpu()
vutils.save_image(vutils.make_grid(generated_img, nrow=10, padding=2, normalize=True),
        'sample_dsprites/c0_z.jpg')
