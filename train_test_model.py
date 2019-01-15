from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import video_dataset_processing as vdpro
import time
import util

def train(epoch, device, train_data_loader, model, E_model, E_solver, G1_model, G1_solver, G2_model, G2_solver, D_model,
		  D_solver, train_file, num_class):
	ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	#ratio_list = [0.1]
	#for i_batch, sample_batched in enumerate(train_data_loader):
	for i_batch, (inputs, targets, dists) in enumerate(train_data_loader):
		start = time.time()
		inputs = inputs.to(device)
		label_batched = targets.to(device)
		dists = dists.to(device)

		with torch.no_grad():
			_, data_batched = model(inputs, dists)

		length_full=9
		#data_batched, label_batched, length_full = sample_batched
		data_batched = data_batched.permute(1,0,2)  # B*L*D -> L*B*D
		#data_batched = Variable(data_batched).cuda()
		#label_batched = Variable(label_batched).cuda()

		# labels for GAN
		ones_label = Variable(torch.ones(data_batched.size(1)) + (torch.rand(data_batched.size(1)) - 0.5) * 0.2).to(device)  # full videos
		zeros_label = Variable(torch.zeros(data_batched.size(1)) + torch.rand(data_batched.size(1)) * 0.3).to(device)  # partial videos

		# batch normalization
		max_len = 9
		#max_len = train_data_loader.dataset.max_len
		X_full = data_batched.permute(2, 1, 0) #L*B*D -> D*B*L
		X_full = F.avg_pool1d(X_full, max_len, stride=1)
		X_full = torch.squeeze(X_full, dim=2).permute(1,0)
		X_full = util.norm_data(X_full)

		# sample partial data
		for ratio in ratio_list:	
			X_partial, length_partial = vdpro.sample_data(data_batched, length_full, ratio)
			max_len_partial = length_partial
			#max_len_partial = int(max(length_partial))

			# temporal pooling
			X_partial = X_partial.permute(2, 1, 0)
			X_partial = F.avg_pool1d(X_partial, max_len_partial, stride=1)
			X_partial = torch.squeeze(X_partial, dim=2).permute(1,0)
			X_partial = util.norm_data(X_partial)
			# BEGIN optimize E and G1
			z_sample = E_model(X_partial)
			progress_label = vdpro.GetProgressLabel(ratio)
			X_gen_partial = G1_model(z_sample, progress_label)
			G1_loss = F.mse_loss(X_gen_partial, X_partial)
			G1_loss.backward()
			E_solver.step()
			G1_solver.step()
			E_solver.zero_grad()
			G1_solver.zero_grad()
			# END optimizing E, G1

			# BEGIN optimize D -- true or fake full videos
			for i in range(5):
				z_sample = E_model(X_partial)
				progress_label = vdpro.GetProgressLabel(1.0)
				X_gen_full = G2_model(z_sample, progress_label)  # generate fake full videos
				D_fake_score = D_model(X_gen_full, progress_label)
				D_real_score = D_model(X_full, progress_label)

				# compute score
				D_loss_action = F.cross_entropy(D_real_score[:, :num_class], label_batched)
				D_loss_real = F.binary_cross_entropy_with_logits(D_real_score[:, num_class], ones_label)
				D_loss_fake = F.binary_cross_entropy_with_logits(D_fake_score[:, num_class], zeros_label)
				D_loss = D_loss_action + D_loss_real + D_loss_fake

				D_loss.backward()
				D_solver.step()  # update parameters in D1_solver

				# reset gradient
				D_solver.zero_grad()
			# END optimizing D

			# BEGIN optimize E, G2 -- generator
			z_sample = E_model(X_partial)
			progress_label = vdpro.GetProgressLabel(1.0)
			X_gen_full = G2_model(z_sample, progress_label)
			D_fake_score = D_model(X_gen_full, progress_label)

			G2_loss = 0.001*F.mse_loss(X_gen_full, X_full)
			G_loss_action = F.cross_entropy(D_fake_score[:, :num_class], label_batched)
			G_loss_fake = F.binary_cross_entropy_with_logits(D_fake_score[:, num_class], ones_label)
			G_loss = G2_loss + G_loss_action + G_loss_fake

			G_loss.backward()
			E_solver.step()
			G2_solver.step()

			# reset gradient
			E_solver.zero_grad()
			G2_solver.zero_grad()
			# END optimizing E, G2

			# nn.utils.clip_grad_norm(model.parameters(), clip)
			end = time.time()

			print("Training epoch %d, ratio %f, G1 loss %f, G2 loss %f, D loss %f, G loss %f, time %f"
				  % (epoch, ratio, G1_loss.data[0], G2_loss.data[0], D_loss.data[0], G_loss.data[0], end - start),
				  file=train_file)
			print("Training epoch %d, ratio %f, G1 loss %f, G2 loss %f, D loss %f, G loss %f, time %f"
				  % (epoch, ratio, G1_loss.data[0], G2_loss.data[0], D_loss.data[0], G_loss.data[0], end - start))
			train_file.flush()

def test(epoch, device, test_data_loader, model, E_model, G1_model, G2_model, D_model, test_file, num_class):
	ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	#ratio_list = [0.1]
	num_total_correct = torch.zeros(10)
	num_total = torch.zeros(10)

	for i_batch, (inputs, targets, dists) in enumerate(test_data_loader):
		start = time.time()
		inputs = inputs.to(device)
		label_batched = targets.to(device)
		dists = dists.to(device)

		with torch.no_grad():
			_, data_batched = model(inputs, dists)

		length_full=9
		#data_batched, label_batched, length_full = sample_batched
		data_batched = data_batched.permute(1,0,2)  # B*L*D -> L*B*D


		for ratio in ratio_list:
			X_partial, length_partial = vdpro.sample_data(data_batched, length_full, ratio)
			max_len_partial = length_partial
			#max_len_partial = int(max(length_partial))

			# temporal pooling
			X_partial = X_partial.permute(2, 1, 0)
			X_partial = F.avg_pool1d(X_partial, max_len_partial, stride=1)
			X_partial = torch.squeeze(X_partial, dim=2).permute(1,0)
			X_partial = util.norm_data(X_partial)
			
			# forward
			z_sample = E_model(X_partial)
			progress_label = vdpro.GetProgressLabel(1.0)
			X_gen_full = G2_model(z_sample, progress_label)  # generate fake full videos
			D_real_score1 = D_model(X_gen_full, progress_label)
			output1 = D_real_score1[:, :num_class]

			# forward again
			D_real_score2 = D_model(X_partial, progress_label)
			output2 = D_real_score2[:, :num_class]
			output = (output1 + output2) / 2 #torch.mean(output1, output2)

			index = int(ratio * 10) - 1
			# compute predicted labels
			pred_label = util.get_pred_label(output)
			accuracy, num_correct = util.compute_acc(pred_label, label_batched)
			print(accuracy)
			print(num_correct)
			num_total_correct[index] = num_total_correct[index] + num_correct
			num_total[index] = num_total[index] + pred_label.size(0)
			print("Testing epoch %d, ratio %f, acc. %f" % (epoch, ratio, accuracy), file=test_file)
			print("Testing epoch %d, ratio %f, acc. %f" % (epoch, ratio, accuracy))
			test_file.flush()

	return num_total_correct / num_total



	for i_batch, sample_batched in enumerate(test_data_loader):
		data_batched, label_batched, length_full = sample_batched
		data_batched = data_batched.transpose(0, 1)  # B*L*D -> L*B*D

		# prepare data
		if cuda:
			data_batched = data_batched.cuda()
			label_batched = label_batched.cuda()

		data_batched = Variable(data_batched)

		for ratio in ratio_list:
			X_partial, length_partial = vdpro.sample_data(data_batched, length_full, ratio)
			max_len_partial = int(max(length_partial))

			# temporal pooling
			X_partial = X_partial.transpose(0, 2)
			X_partial = F.avg_pool1d(X_partial, max_len_partial, stride=1)
			X_partial = X_partial.transpose(0, 2)
			X_partial = torch.squeeze(X_partial)

			# batch normalization
			X_partial = util.norm_data(X_partial)

			# forward
			z_sample = E_model(X_partial)
			progress_label = vdpro.GetProgressLabel(1.0)
			X_gen_full = G2_model(z_sample, progress_label)  # generate fake full videos
			D_real_score1 = D_model(X_gen_full, progress_label)
			output1 = D_real_score1[:, :num_class]

			# forward again
			D_real_score2 = D_model(X_partial, progress_label)
			output2 = D_real_score2[:, :num_class]
			output = (output1 + output2) / 2 #torch.mean(output1, output2)

			index = int(ratio * 10) - 1
			# compute predicted labels
			pred_label = util.get_pred_label(output)
			accuracy, num_correct = util.compute_acc(pred_label, label_batched)
			num_total_correct[index] = num_total_correct[index] + num_correct
			num_total[index] = num_total[index] + pred_label.size(0)

			print("Testing epoch %d, ratio %f, acc. %f" % (epoch, ratio, accuracy), file=test_file)
			print("Testing epoch %d, ratio %f, acc. %f" % (epoch, ratio, accuracy))
			test_file.flush()

	return num_total_correct / num_total
