from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import video_dataset_processing as vdpro
import time
import util
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from losses import TripletLoss


#def train(epoch, device, train_data_loader, model, E_model, E_solver, G1_model, G1_solver, G2_model, G2_solver, D_model,
#          D_solver, train_file, num_class):
def train(epoch, device, train_data_loader, base_model, model_solver, E_model, E_solver, G1_model, G1_solver, G2_model, G2_solver, D1_model, D1_solver, D2_model,
          D2_solver, train_file, num_class):
    
    #model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    #T_losses = AverageMeter()
    G1_losses = AverageMeter()
    G2_losses = AverageMeter()
    G_action_losses = AverageMeter()
    D_action_losses = AverageMeter()
    D_real_losses = AverageMeter()
    D_fake_losses = AverageMeter()
    end = time.time()

    #L1_criterion = nn.CosineSimilarity(dim=1)
    #L1_criterion = nn.L2Loss()
    #L1_criterion = F.mse_loss()

    bar = Bar('Processing', max=len(train_data_loader)) 

    #ratio_list = [0.2, 0.5, 0.8]
    ratio_list = [0.5]

    D_interval = 20
    for i_batch, (inputs, targets, dists) in enumerate(train_data_loader):
    #for i_batch, (inputs, targets, dists, pos_inputs, pos_dists, neg_inputs, neg_dists) in enumerate(train_data_loader):
        start = time.time()
        inputs = inputs.to(device)
        label_batched = targets.to(device)
        dists = dists.to(device)
        #inputs = inputs.cuda()
        #label_batched = targets.cuda()
        #dists = dists.cuda()
        #pos_inputs = pos_inputs.to(device)
        #neg_inputs = neg_inputs.to(device)
        #pos_dists = pos_dists.to(device)
        #neg_dists = neg_dists.to(device)

        data_time.update(time.time() - end)

        with torch.no_grad():
            _, data_batched = base_model(inputs, dists)
        #_, pos_data_batched = model(pos_inputs, pos_dists)
        #_, neg_data_batched = model(neg_inputs, neg_dists)

        length_full = data_batched.size(1)

        # labels for GAN
        #ones_label = Variable(torch.ones(data_batched.size(0))).to(device)  # full videos
        #ones_label = Variable(torch.ones(data_batched.size(0)) + (torch.rand(data_batched.size(0)) - 0.5) * 0.2).cuda()  # full videos
        ones_label = Variable(torch.ones(data_batched.size(0)) + (torch.rand(data_batched.size(0)) - 0.5) * 0.2).to(device)  # full videos
        #zeros_label = Variable(torch.zeros(data_batched.size(0))).to(device)  # partial videos
        #zeros_label = Variable(torch.zeros(data_batched.size(0)) + torch.rand(data_batched.size(0)) * 0.3).cuda()  # partial videos
        zeros_label = Variable(torch.zeros(data_batched.size(0)) + torch.rand(data_batched.size(0)) * 0.3).to(device)  # partial videos

        # batch normalization
        max_len = length_full
        X_full = data_batched[:,-1,:]
        #pos_partial = pos_data_batched[:,-1,:]
        #neg_partial = neg_data_batched[:,-1,:]

        # sample partial data
        for ratio in ratio_list:    
            X_partial, length_partial = vdpro.sample_data(data_batched, length_full, ratio)
            #pos_partial, length_partial = vdpro.sample_data(pos_data_batched, length_full, ratio)
            #neg_partial, length_partial = vdpro.sample_data(neg_data_batched, length_full, ratio)
            max_len_partial = length_partial

            # temporal pooling
            # BEGIN optimize E and G1
            z_sample = E_model(X_partial)
            #pos_z_sample = E_model(pos_partial)
            #neg_z_sample = E_model(neg_partial)
            #margin = 1.
            #T_criterion = TripletLoss(margin)
            #T_loss = T_criterion(z_sample, pos_z_sample, neg_z_sample)
            #T_loss.backward(retain_graph=True)
            #E_solver.step()
            #E_solver.zero_grad()

            progress_label = vdpro.GetProgressLabel(ratio)
            X_gen_partial = G1_model(z_sample, progress_label)
            G1_loss = F.mse_loss(X_gen_partial, X_partial)
            G1_loss.backward(retain_graph=True)
            E_solver.step()
            G1_solver.step()
            E_solver.zero_grad()
            G1_solver.zero_grad()
            #G1_loss = L1_criterion(X_gen_partial, X_partial)
            #G1_loss.backward(retain_graph=True)
            #E_solver.step()
            #G1_solver.step()
            #E_solver.zero_grad()
            #G1_solver.zero_grad()
            # END optimizing E, G1

            progress_label = vdpro.GetProgressLabel(1.0)
            #noise = torch.FloatTensor(np.random.normal(loc=0.0, scale=0.3, size=(X_gen_partial.shape))) 
            X_gen_full = G2_model(z_sample, progress_label)
            #print(X_gen_full)

            G2_loss = F.mse_loss(X_gen_full, X_full)
            #G2_loss = L1_criterion(X_gen_full, X_full)
            D2_fake_score = D2_model(X_gen_full, progress_label)
            D2_real_score = D2_model(X_full, progress_label)
           
            D1_cls_fake_score = D1_model(X_gen_full)
            D1_cls_real_score = D1_model(X_full)

            D_loss_action = F.cross_entropy(D1_cls_fake_score, label_batched)
            #D_loss_action = F.cross_entropy(D1_cls_real_score, label_batched)
            D_loss_real = F.binary_cross_entropy_with_logits(D2_real_score[:,0], ones_label)
            D_loss_fake = F.binary_cross_entropy_with_logits(D2_fake_score[:,0], zeros_label)
            G_loss_fake = F.binary_cross_entropy_with_logits(D2_fake_score[:,0], ones_label)
            G_loss = D_loss_action + G2_loss + G_loss_fake

            G_loss.backward(retain_graph=True)
            E_solver.step()
            G2_solver.step() 
            model_solver.step() 
            E_solver.zero_grad()
            G2_solver.zero_grad()
            model_solver.zero_grad()

            if(i_batch%D_interval==0):
            # BEGIN optimize D -- true or fake full videos
            #for i in range(1):
                #z_sample = E_model(X_partial)
                #progress_label = vdpro.GetProgressLabel(1.0)
                #X_gen_full = G2_model(z_sample, progress_label)  # generate fake full videos
                #D2_fake_score = D2_model(X_gen_full, progress_label)
                #D2_real_score = D2_model(X_full, progress_label)
                #D1_cls_fake_score = D1_model(X_gen_full)
                #D1_cls_real_score = D1_model(X_full)
                # compute score

                #D_loss_action = F.cross_entropy(D_real_score[:, :num_class], label_batched)
                #D_loss_real = F.binary_cross_entropy_with_logits(D_real_score[:, num_class], ones_label)
                #D_loss_fake = F.binary_cross_entropy_with_logits(D_fake_score[:, num_class], zeros_label)
                #D_loss_action = F.cross_entropy(D1_cls_real_score, label_batched)
                #D1_loss.backward(retain_graph=True)
                #D1_solver.step()  # update parameters in D1_solver
                ## reset gradient
                #D1_solver.zero_grad()

                #D_loss_real = F.binary_cross_entropy_with_logits(D2_real_score[:,0], ones_label)
                #D_loss_fake = F.binary_cross_entropy_with_logits(D2_fake_score[:,0], zeros_label)
                #D_loss = D_loss_real + D_loss_fake

                #D_loss.backward(retain_graph=True)
                D_loss = G2_loss + D_loss_fake + D_loss_real
                D_loss.backward()
                #D1_solver.step()
                #D1_solver.step()
                D2_solver.step()  # update parameters in D1_solver
                # reset gradient
                #D1_solver.zero_grad()
                D2_solver.zero_grad()
                #D1_solver.zero_grad()
            # END optimizing D

            # BEGIN optimize E, G2 -- generator
            #z_sample = E_model(X_partial)
            #progress_label = vdpro.GetProgressLabel(1.0)
            #X_gen_full = G2_model(z_sample, progress_label)
            #D2_fake_score = D2_model(X_gen_full, progress_label)
            #D1_cls_fake_score = D1_model(X_gen_full)

            #G2_loss = L1_criterion(X_gen_full, X_full)
            #G_loss_action = F.cross_entropy(D1_cls_fake_score, label_batched)
            #G_loss_fake = F.binary_cross_entropy_with_logits(D_fake_score[:, num_class], ones_label)
            #G_loss = G2_loss + G_loss_action
            #G_loss = G2_loss + G_loss_action + G_loss_fake

            #G_loss.backward(retain_graph=True)
            #E_solver.step()
            #G2_solver.step()

            ## reset gradient
            #E_solver.zero_grad()
            #G2_solver.zero_grad()
            # END optimizing E, G2
       
            #T_losses.update(T_loss.data[0], inputs.size(0))
            G1_losses.update(G1_loss.data[0], inputs.size(0))
            G2_losses.update(G2_loss.data[0], inputs.size(0))
            #G_action_losses.update(G_loss_action.data[0], inputs.size(0))
            D_action_losses.update(D_loss_action.data[0], inputs.size(0))
            D_real_losses.update(D_loss_real.data[0], inputs.size(0))
            D_fake_losses.update(D_loss_fake.data[0], inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | G1_loss: {G1_loss: .4f}| G2_loss: {G2_loss: .4f} | D_action_loss: {D_action_loss: .4f} | D_real_loss: {D_real_loss: .4f} | D_fake_loss: {D_fake_loss: .4f}'.format(
                    batch=i_batch + 1,
                    size=len(train_data_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    #T_loss=T_losses.avg,
                    G1_loss=G1_losses.avg,
                    G2_loss=G2_losses.avg,
                    #G_action_loss=G_action_losses.avg,
                    D_action_loss=D_action_losses.avg,
                    D_real_loss=D_real_losses.avg,
                    D_fake_loss=D_fake_losses.avg,
                    )
        bar.next()
    bar.finish()
    #        print("G1_loss %.4f D_loss_action %.4f D_loss_real %.4f D_loss_fake %.4f G2_loss %.4f G_loss_action %.4f" \
    #                    % G1_loss, D_loss_action, D_loss_real, D_loss_fake, G2_loss, G_loss_action)
        
    return D_action_losses.avg

def test(epoch, device, test_data_loader, model, E_model, G1_model, G2_model, D1_model, D2_model, test_file, num_class):

    criterion = nn.CrossEntropyLoss()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    OR_top1 = AverageMeter()
    OR_top5 = AverageMeter()

    model.eval()
    end = time.time()
    bar = Bar('Processing', max=len(test_data_loader))
    #ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ratio_list = [0.5]

    num_total_correct = torch.zeros(10)
    num_total = torch.zeros(10)

    for i_batch, (inputs, targets, dists) in enumerate(test_data_loader):

        data_time.update(time.time() - end)
        #start = time.time()
        #inputs = inputs.cuda()
        #label_batched = targets.cuda()
        #dists = dists.cuda()

        inputs = inputs.to(device)
        label_batched = targets.to(device)
        dists = dists.to(device)
        with torch.no_grad():
            _, data_batched = model(inputs, dists)

            length_full = data_batched.size(1)
            #data_batched = data_batched.permute(1,0,2)  # B*L*D -> L*B*D

            for ratio in ratio_list:
                X_partial, length_partial = vdpro.sample_data(data_batched, length_full, ratio)
                max_len_partial = length_partial

                # temporal pooling
                #X_partial = X_partial.permute(0, 2, 1)
                #X_partial = F.avg_pool1d(X_partial, max_len_partial, stride=1)
                #X_partial = torch.squeeze(X_partial, dim=2)
                #X_partial = util.norm_data(X_partial)
            
                # forward
                z_sample = E_model(X_partial)
                progress_label = vdpro.GetProgressLabel(1.0)
                X_gen_full = G2_model(z_sample, progress_label)  # generate fake full videos
                D_real_score1 = D1_model(X_gen_full)
                output1 = D_real_score1

                # forward again
                D_real_score2 = D1_model(X_partial)
                output2 = D_real_score2
                OR_outputs = output2 #torch.mean(output1, output2)
                outputs = (output1 + output2) / 2 #torch.mean(output1, output2)
 
                loss = criterion(outputs, label_batched)
                max_value, max_index = torch.max(outputs.data, 1)
                prec1, prec5 = accuracy(outputs.data, label_batched.data, topk=(1, 5))
                OR_prec1, OR_prec5 = accuracy(OR_outputs.data, label_batched.data, topk=(1, 5))

                losses.update(loss.data[0], inputs.size(0))
                top1.update(prec1[0], inputs.size(0))
                top5.update(prec5[0], inputs.size(0))
                OR_top1.update(OR_prec1[0], inputs.size(0))
                OR_top5.update(OR_prec5[0], inputs.size(0))
                # measure elapsed time

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | OR_top1: {OR_top1: .4f} | OR_top5: {OR_top5: .4f}'.format(
                    batch=i_batch + 1,
                    size=len(test_data_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    OR_top1=OR_top1.avg,
                    OR_top5=OR_top5.avg,
                    )
        bar.next()
    bar.finish()
    #model.train()
    return (losses.avg, top1.avg)
    #return num_total_correct / num_total
