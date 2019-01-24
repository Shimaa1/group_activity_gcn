import torch
import torch.nn as nn
from torch.autograd import Variable
import video_dataset_processing as vdpro

class Encoder(nn.Module):
	def __init__(self, x_dim, h_dim, z_dim):
		super(Encoder, self).__init__()
		self.x_dim = x_dim
		self.h_dim = h_dim
		self.z_dim = z_dim

		self.relu = nn.LeakyReLU()
		#self.norm = nn.BatchNorm1d(x_dim)
		self.fc1 = nn.Linear(x_dim, h_dim)
		self.fc21 = nn.Linear(h_dim, z_dim)
		self.fc22 = nn.Linear(h_dim, z_dim)

	def reparameterize(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mul(std).add_(mu)
		else:
			return mu

	def forward(self, x):
		h = self.relu(self.fc1(x))
		mu = self.fc21(h)
		logvar = self.fc22(h)
		z = self.reparameterize(mu, logvar)
		return z


class Decoder1(nn.Module): # for generating full video features
	def __init__(self, in_dim, h_dim, out_dim):
		super(Decoder1, self).__init__()
		self.in_dim = in_dim
		self.h_dim = h_dim
		self.out_dim = out_dim

		self.relu = nn.LeakyReLU()
		self.fc1 = nn.Linear(in_dim+10, h_dim)
		self.fc2 = nn.Linear(h_dim, out_dim)
		self.sigmoid = nn.Tanh()

	def forward(self, in_data, l):
		in_data = vdpro.CombineSample(in_data, l, 10)
		h = self.relu(self.fc1(in_data))
		return self.sigmoid(self.fc2(h))


class Decoder2(nn.Module):  # for generating partial video features
	def __init__(self, in_dim, h_dim, out_dim):
		super(Decoder2, self).__init__()
		self.in_dim = in_dim
		self.h_dim = h_dim
		self.out_dim = out_dim

		self.relu = nn.LeakyReLU()
		self.fc1 = nn.Linear(in_dim+10, h_dim)
		self.fc2 = nn.Linear(h_dim, out_dim)
		self.sigmoid = nn.Tanh()

	def forward(self, in_data, l):
		in_data = vdpro.CombineSample(in_data, l, 10)
		h = self.relu(self.fc1(in_data))
		return self.sigmoid(self.fc2(h))


class Discriminator(nn.Module):
	def __init__(self, x_dim, h_dim, num_class):
		super(Discriminator, self).__init__()
		self.dim = x_dim
		self.x_dim = x_dim
		self.h_dim = h_dim
		self.num_class = num_class

		self.fc1 = nn.Linear(x_dim+10, num_class+1)
		#self.fc2 = nn.Linear(h_dim, num_class+1)
		#self.relu = nn.LeakyReLU()
		#self.norm = nn.BatchNorm1d(x_dim)
	#def forward(self, x):
	#	h = self.relu(self.fc1(x))
	#	return self.fc2(h)

	def forward(self, x, l):
		#x = self.norm(x)
		x = vdpro.CombineSample(x, l, 10)
		#h = self.relu(self.fc1(x))
		return self.fc1(x)

