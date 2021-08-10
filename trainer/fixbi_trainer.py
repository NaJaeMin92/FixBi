import time
import torch
import torch.nn as nn
import src.utils as utils


def train_fixbi(args, loaders, optimizers, models_sd, models_td, sp_params, losses, epoch):
    print("Epoch: [{}/{}]".format(epoch, args.epochs))
    start = time.time()
    src_train_loader, tgt_train_loader = loaders[0], loaders[1]
    optimizer_sd, optimizer_td = optimizers[0], optimizers[1]
    sp_param_sd, sp_param_td = sp_params[0], sp_params[1]
    ce, mse = losses[0], losses[1]

    utils.set_model_mode('train', models=models_sd)
    utils.set_model_mode('train', models=models_td)

    models_sd = nn.Sequential(*models_sd)
    models_td = nn.Sequential(*models_td)

    for step, (src_data, tgt_data) in enumerate(zip(src_train_loader, tgt_train_loader)):
        src_imgs, src_labels = src_data
        tgt_imgs, tgt_labels = tgt_data
        src_imgs, src_labels = src_imgs.cuda(non_blocking=True), src_labels.cuda(non_blocking=True)
        tgt_imgs, tgt_labels = tgt_imgs.cuda(non_blocking=True), tgt_labels.cuda(non_blocking=True)

        x_sd, x_td = models_sd(tgt_imgs), models_td(tgt_imgs)

        pseudo_sd, top_prob_sd, threshold_sd = utils.get_target_preds(args, x_sd)
        fixmix_sd_loss = utils.get_fixmix_loss(models_sd, src_imgs, tgt_imgs, src_labels, pseudo_sd, args.lam_sd)

        pseudo_td, top_prob_td, threshold_td = utils.get_target_preds(args, x_td)
        fixmix_td_loss = utils.get_fixmix_loss(models_td, src_imgs, tgt_imgs, src_labels, pseudo_td, args.lam_td)

        total_loss = fixmix_sd_loss + fixmix_td_loss

        if step == 0:
            print('Fixed MixUp Loss (SDM): {:.4f}'.format(fixmix_sd_loss.item()))
            print('Fixed MixUp Loss (TDM): {:.4f}'.format(fixmix_td_loss.item()))

        # Bidirectional Matching
        if epoch > args.bim_start:
            bim_mask_sd = torch.ge(top_prob_sd, threshold_sd)
            bim_mask_sd = torch.nonzero(bim_mask_sd).squeeze()

            bim_mask_td = torch.ge(top_prob_td, threshold_td)
            bim_mask_td = torch.nonzero(bim_mask_td).squeeze()

            if bim_mask_sd.dim() > 0 and bim_mask_td.dim() > 0:
                if bim_mask_sd.numel() > 0 and bim_mask_td.numel() > 0:
                    bim_mask = min(bim_mask_sd.size(0), bim_mask_td.size(0))
                    bim_sd_loss = ce(x_sd[bim_mask_td[:bim_mask]], pseudo_td[bim_mask_td[:bim_mask]].cuda().detach())
                    bim_td_loss = ce(x_td[bim_mask_sd[:bim_mask]], pseudo_sd[bim_mask_sd[:bim_mask]].cuda().detach())

                    total_loss += bim_sd_loss
                    total_loss += bim_td_loss

                    if step == 0:
                        print('Bidirectional Loss (SDM): {:.4f}'.format(bim_sd_loss.item()))
                        print('Bidirectional Loss (TDM): {:.4f}'.format(bim_td_loss.item()))

        # Self-penalization
        if epoch <= args.sp_start:
            sp_mask_sd = torch.lt(top_prob_sd, threshold_sd)
            sp_mask_sd = torch.nonzero(sp_mask_sd).squeeze()

            sp_mask_td = torch.lt(top_prob_sd, threshold_td)
            sp_mask_td = torch.nonzero(sp_mask_td).squeeze()

            if sp_mask_sd.dim() > 0 and sp_mask_td.dim() > 0:
                if sp_mask_sd.numel() > 0 and sp_mask_td.numel() > 0:
                    sp_mask = min(sp_mask_sd.size(0), sp_mask_td.size(0))
                    sp_sd_loss = utils.get_sp_loss(x_sd[sp_mask_sd[:sp_mask]], pseudo_sd[sp_mask_sd[:sp_mask]], sp_param_sd)
                    sp_td_loss = utils.get_sp_loss(x_td[sp_mask_td[:sp_mask]], pseudo_td[sp_mask_td[:sp_mask]], sp_param_td)

                    total_loss += sp_sd_loss
                    total_loss += sp_td_loss

                    if step == 0:
                        print('Penalization Loss (SDM): {:.4f}', sp_sd_loss.item())
                        print('Penalization Loss (TDM): {:.4f}', sp_td_loss.item())

        # Consistency Regularization
        if epoch > args.cr_start:
            mixed_cr = 0.5 * src_imgs + 0.5 * tgt_imgs
            out_sd, out_td = models_sd(mixed_cr), models_td(mixed_cr)
            cr_loss = mse(out_sd, out_td)
            total_loss += cr_loss
            if step == 0:
                print('Consistency Loss: {:.4f}', cr_loss.item())

        optimizer_sd.zero_grad()
        optimizer_td.zero_grad()
        total_loss.backward()
        optimizer_sd.step()
        optimizer_td.step()

    print("Train time: {:.2f}".format(time.time() - start))
