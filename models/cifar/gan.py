import torch
import torch.nn as nn
from torch.autograd import Variable
import video_dataset_processing as vdpro
from mypath import Path

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
		self._initialize_weights()

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

	def _initialize_weights(self):
    
		for m in self.modules():
			if isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

class Decoder1(nn.Module): # for generating full video features
	def __init__(self, in_dim, h_dim, out_dim):
		super(Decoder1, self).__init__()
		self.in_dim = in_dim
		self.h_dim = h_dim
		self.out_dim = out_dim

		self.relu = nn.LeakyReLU()
		self.fc1 = nn.Linear(in_dim+10, h_dim)
		self.fc2 = nn.Linear(h_dim, out_dim)
		self._initialize_weights()
		#self.sigmoid = nn.Tanh()

	def forward(self, in_data, l):
		in_data = vdpro.CombineSample(in_data, l, 10)
		h = self.relu(self.fc1(in_data))
		return self.fc2(h)

	def _initialize_weights(self):
    
		for m in self.modules():
			if isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

class Decoder2(nn.Module):  # for generating partial video features
	def __init__(self, in_dim, h_dim, out_dim):
		super(Decoder2, self).__init__()
		self.in_dim = in_dim
		self.h_dim = h_dim
		self.out_dim = out_dim

		self.relu = nn.LeakyReLU()
		self.fc1 = nn.Linear(in_dim+10, h_dim)
		self.fc2 = nn.Linear(h_dim, out_dim)
		self._initialize_weights()
		#self.sigmoid = nn.Tanh()

	def forward(self, in_data, l):
		in_data = vdpro.CombineSample(in_data, l, 10)
		h = self.relu(self.fc1(in_data))
		return self.fc2(h)

	def _initialize_weights(self):
    
		for m in self.modules():
			if isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

class Discriminator1(nn.Module):
	def __init__(self, x_dim, h_dim, num_class, model_dir):
		super(Discriminator1, self).__init__()
		#self.dim = x_dim
		self.x_dim = x_dim
		self.h_dim = h_dim
		self.num_class = num_class

		self.fc1 = nn.Linear(x_dim, num_class)
		self._load_pretrained_weights(model_dir)

	def forward(self, x):
		#x = vdpro.CombineSample(x, l, 10)
		return self.fc1(x)

	def _load_pretrained_weights(self, model_dir):
		corresp_name = {
				"gclassifier.weight": "fc1.weight",
				"gclassifier.bias": "fc1.bias",                               
				}
		p_dict = torch.load(model_dir)['state_dict']
		#p_dict = torch.load(Path.group_dir('vgg19'))['state_dict']
		s_dict = self.state_dict()
		for item in s_dict:
			print('sdict', item)
		for name in p_dict:
			print('pdict', name)
		#assert 1==0
		self.load_state_dict({corresp_name[k]:v for k,v in p_dict.items() if k in corresp_name})

class Discriminator2(nn.Module):
	def __init__(self, x_dim, h_dim):
		super(Discriminator2, self).__init__()
		self.dim = x_dim
		self.x_dim = x_dim
		self.h_dim = h_dim

		self.fc1 = nn.Linear(x_dim+10, 1)
		self._initialize_weights()
		#self._load_pretrained_weights()
		#self.fc2 = nn.Linear(h_dim, num_class+1)
		#self.relu = nn.LeakyReLU()
		#self.norm = nn.BatchNorm1d(x_dim)
	#def forward(self, x):
	#	h = self.relu(self.fc1(x))
	#	return self.fc2(h)

	def forward(self, x, l):
		x = vdpro.CombineSample(x, l, 10)
		return self.fc1(x)


	def _initialize_weights(self):
    
		for m in self.modules():
			if isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()
