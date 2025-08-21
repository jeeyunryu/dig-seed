# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable, Optional
import os

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from utils import utils
from loss import *
import evaluation_metric

import logging
import string
from loss.embeddingRegressionLoss import EmbeddingRegressionLoss

import wandb

voc = list(string.printable[:-6])

# def decode_indices(indices, eos_token=95, padding_token=94):
#     """정수 인덱스 리스트를 문자열로 디코딩"""
#     chars = []
#     for idx in indices:
#         if idx == eos_token:  # EOS token
#             break
#         if idx == padding_token:  # padding은 무시
#             continue
#         if 0 <= idx < len(voc):
#             chars.append(voc[idx])
#         else:
#             chars.append('?')  # 범위 벗어남 → ?
#     return ''.join(chars)


def train_class_batch(model, samples, target, tgt_lens, criterion, criterion_aux=None, args=None, teacher_model=None, metric_logger=None):
    
    # if args.use_seq_cls_token:
    #     outputs = model(samples)
    # else:
    #     outputs = model((samples, target, tgt_lens))
    outputs = model((samples, target, tgt_lens))
    # embed_crit = EmbeddingRegressionLoss(loss_func='cosin')

    

    if isinstance(outputs, tuple):
        if teacher_model is not None:
            outputs, s_feat = outputs
            t_feat = teacher_model.module.encoder(samples)
            t_feat = F.layer_norm(t_feat, (t_feat.size(-1),))
            loss_distill = F.smooth_l1_loss(s_feat, t_feat, beta=2)
            loss_rec = criterion(outputs, target, tgt_lens)
            loss = loss_rec + args.loss_weight_feat_distill * loss_distill
            if metric_logger is not None:
                metric_logger.update(loss_distill=loss_distill)
                metric_logger.update(loss_rec=loss_rec)
        else:
            outputs, sem_feat, sem_feat_attn_maps, dec_attn_maps = outputs
            loss = criterion(outputs, target, tgt_lens)
            return loss, outputs, None
            # if tgt_embeds is None:
            #     outputs, sem_feat, sem_feat_attn_maps, dec_attn_maps = outputs
            #     loss = criterion(outputs, target, tgt_lens)
            #     return loss, outputs, None
            # else:
            #     outputs, sem_feat, sem_feat_attn_maps, dec_attn_maps, embedding_vectors = outputs
            #     loss = criterion(outputs, target, tgt_lens)
            #     # tgt_embeds = tgt_embeds.cuda()
            #     # loss_embed = embed_crit(embedding_vectors, tgt_embeds)
            #     # total_loss = loss + 5 * loss_embed
            #     # loss_dict = {
            #     #     "total_loss": total_loss,
            #     #     "rec_loss": loss,
            #     #     "embed_loss": loss_embed,
            #     # }
            #     # loss_dict = {
            #     #     "total_loss": 10,
            #     #     "rec_loss": loss,
            #     #     "embed_loss": 10,
            #     # }
                # return loss, outputs, None


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,
                    args=None, data_loader_val=None, max_accuracy=0., criterion_aux=None,
                    teacher_model=None): # used for evaluation
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    
    
    # for data_iter_step, (samples, targets, tgt_lens) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # samples, targets, tgt_lens, img_key = data
        samples = data['image']
        targets = data['label']
        tgt_lens = data['length']
        img_key = data['imgkey']
        # if args.dig_mode == 'dig':
        #     samples, targets, tgt_lens, img_key = data
        #     tgt_embeds = None
        # else: # dig-seed
        #     samples, targets, tgt_lens, tgt_embeds, img_key = data

        # samples, targets, tgt_lens, tgt_embeds = data

        # # samples = data[:][0]
        # samples = torch.stack([x[0] for x in data])
        # targets = torch.stack([x[1] for x in data])
        # # targets = data[:][1]
        # tgt_lens = torch.stack([x[2] for x in data])
        # # tgt_lens = data[:][2]
        # tgt_embeds = [x[2] for x in data]
        # # tgt_embeds = data[:][3]
        
        # samples, targets, tgt_lens, tgt_embeds = data
        
        # if args.w2v_path is None:
        #     samples, targets, tgt_lens, tgt_embeds = data
            
        # else:
        #     samples, targets, tgt_lens, sem_vecs = data
        # During training, evaluation is conducted, therefore, the training flag should be reset again.
        model.train(True)
        
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            # loss_dict, output, f_measure = train_class_batch(
            #     model, samples, targets, tgt_lens, criterion, tgt_embeds, criterion_aux, args, teacher_model, metric_logger)
            loss, output, f_measure = train_class_batch(
                model, samples, targets, tgt_lens, criterion, criterion_aux, args, teacher_model, metric_logger)
        else:
            with torch.cuda.amp.autocast():
                # loss_dict, output, f_measure = train_class_batch(
                #     model, samples, targets, tgt_lens, criterion, tgt_embeds, criterion_aux, args, teacher_model, metric_logger)
                loss, output, f_measure = train_class_batch(
                    model, samples, targets, tgt_lens, criterion, criterion_aux, args, teacher_model, metric_logger)
   

        # if args.dig_mode == 'dig':
        #     loss = loss_value = loss_dict
        # else:
        #     loss = loss_value = loss_dict['total_loss']
        #     rec_loss = loss_dict['rec_loss']
        #     embed_loss = loss_dict['embed_loss']

        loss_value = loss.item()
        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            if output is None:
                class_acc = 0.
            else:
                output = F.softmax(output, dim=-1)
                _, pred_ids = output.max(-1)
                class_acc, _, _ = evaluation_metric.factory()['accuracy'](pred_ids, targets, data_loader.dataset)
                # class_acc = evaluation_metric.factory()['ctc_accuracy'](pred_ids, targets, data_loader.dataset)
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        # if args.dig_mode == 'dig-seed':

        #     metric_logger.update(rec_loss=rec_loss)
        #     metric_logger.update(embed_loss=embed_loss)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        if f_measure is not None:
            metric_logger.update(f_measure=f_measure)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()
        
        
        wandb.log({"step_loss": loss, "step_acc": class_acc}) # dig

        # # evaluation during training
        # if step >= 1 and step % args.eval_freq == 0:
        #     if data_loader_val is not None:
        #         test_stats = evaluate(data_loader_val, model, device, args=args)
        #         print(f"Accuracy of the network on the {len(data_loader_val.dataset)} test images: {test_stats['acc']:.4f}%")
        #         if max_accuracy < test_stats["acc"]:
        #             max_accuracy = test_stats["acc"]
        #             if args.output_dir and args.save_ckpt:
        #                 utils.save_model(
        #                     args=args, model=model, model_without_ddp=model.module, optimizer=optimizer,
        #                     loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

        # if step >= 1 and step % (args.eval_freq * 10) == 0:
        #     utils.save_model(
        #         args=args, model=model, model_without_ddp=model.module, optimizer=optimizer,
        #         loss_scaler=loss_scaler, epoch="{0}_{1}".format(epoch, step), model_ema=model_ema)

        # # flush the screen info to disk_file.
        # # if utils.is_main_process():
        sys.stdout.flush()


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    wandb.log({"acc": metric_logger.meters['class_acc'].global_avg, "loss": metric_logger.meters['loss'].global_avg})
    train_stats.update({'max_accuracy': max_accuracy})
    return train_stats


@torch.no_grad()
def evaluate(data_loader, model, device, args=None):    
    criterion = SeqCrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    save_dir = args.output_dir if args and hasattr(args, 'output_dir') else './logs'
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'eval_result.txt')
    log_lines = []

    # summary_log_path = os.path.join(save_dir, 'eval_summary.txt')
    detail_log_path = os.path.join(save_dir, 'eval_predictions.txt')
    
    # summary_log = []
    detailed_log = []

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):

        images = batch['image']
        target = batch['label']
        lens = batch['length']
        img_key = batch['imgkey']

        # images = batch[0]
        # target = batch[1]
        # lens = batch[2]
        # img_key = batch[3]
        # if args.dig_mode == 'dig':
        #     img_key = batch[3]
        # else: # dig-seed
        #     # embed_ved = batch[3]
        #     img_key = batch[4]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # output = model((images, target, lens, img_key))
            output = model((images, target, lens))
            if isinstance(output, tuple):
                if len(output) == 3:
                    output, ctc_rec_score, _ = output
                    cls_logit = None
                elif len(output) == 2:
                    output, _ = output
                    cls_logit = None
                else:
                    output, cls_logit, _, _ = output
                    cls_logit = None
            else:
                cls_logit = None
            # TODO:
            if output is not None:
                if args.beam_width > 0:
                    loss = torch.Tensor([0.])
                else:
                    loss = criterion(output, target, lens)
                    # tgt_embeds = tgt_embeds.cuda()
                    # loss_embed = embed_crit(embedding_vectors, tgt_embeds)
                    # total_loss = loss + loss_embed

        # evaluation metrics.
        if output is not None:
            if args.beam_width > 0:
                pred_ids = output
            else:
                _, pred_ids = output.max(-1)
            acc, pred_list, targ_list = evaluation_metric.factory()['accuracy'](pred_ids, target, data_loader.dataset)
            # acc, pred_list, targ_list = evaluation_metric.factory()['ctc_accuracy'](pred_ids, target, data_loader.dataset)
            recognition_fmeasure = evaluation_metric.factory()['recognition_fmeasure'](pred_ids, target, data_loader.dataset)

            # if hasattr(data_loader.dataset, 'label_decode'):
            #     pred_strs = data_loader.dataset.label_decode(pred_ids)
            #     gt_strs = data_loader.dataset.label_decode(target)
            # else:
            #     pred_strs = [str(p.tolist()) for p in pred_ids]
            #     gt_strs = [str(g.tolist()) for g in target]

            for img_key, pred, tgt in zip(img_key, pred_list, targ_list):
                # filename = os.path.basename(path)
                detailed_log.append(f'{img_key} | GT: {tgt} | Pred: {pred}')
            
            # 상세 로그에 이미지 파일 이름 포함
            # for img_key, pred, gt in zip(img_key, pred_ids, target):
            #     # filename = os.path.basename(path)
            #     detailed_log.append(f'{img_key} | GT: {decode_indices(gt)} | Pred: {decode_indices(pred)}')
        else:
            acc = 0.
            recognition_fmeasure = 0.
        if cls_logit is not None:
            target_aux = F.one_hot(target, cls_logit.size(-1))
            target_aux = target_aux.sum(1)
            target_aux = (target_aux >= 1).float()
            f_measure = evaluation_metric.factory()['multi_label_fmeasure'](cls_logit, target_aux)
            metric_logger.meters['fmeasure'].update(f_measure, n=images.shape[0])
            if output is None:
                loss = F.binary_cross_entropy_with_logits(cls_logit, target_aux)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(acc, n=batch_size)
        # metric_logger.meters['recognition_fmeasure'].update(recognition_fmeasure, n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('* {eval_data.root}: {acc.count} images, Acc {acc.global_avg:.4f} loss {losses.global_avg:.4f} Rec_fmeasure {rec_f.global_avg:.4f}'
    #       .format(eval_data=data_loader.dataset, acc=metric_logger.acc, losses=metric_logger.loss, rec_f=metric_logger.recognition_fmeasure))
    
    # result_str = '* {}: {} images, Acc {:.4f} loss {:.4f} Rec_fmeasure {:.4f}'.format(
    #     data_loader.dataset.root,
    #     metric_logger.acc.count,
    #     metric_logger.acc.global_avg,
    #     metric_logger.loss.global_avg,
    #     metric_logger.recognition_fmeasure.global_avg
    # )
    result_str = '* {}: {} images, Acc {:.4f} loss {:.4f}'.format(
        data_loader.dataset.root,
        metric_logger.acc.count,
        metric_logger.acc.global_avg,
        metric_logger.loss.global_avg,
    )

    # wandb.log({"test_acc": metric_logger.acc.global_avg, "test_loss": metric_logger.loss.global_avg, "test_rec_fmeasure": metric_logger.recognition_fmeasure.global_avg})
    wandb.log({"test_acc": metric_logger.acc.global_avg, "test_loss": metric_logger.loss.global_avg})

    print(result_str)



    if cls_logit is not None:
        print('F_measure: {f.global_avg:.4f}'.format(f=metric_logger.fmeasure))
    # the window size of smoothedvalue is set to 20, therefore there may be imprecise.
    if len(metric_logger.meters['acc'].deque) == metric_logger.meters['acc'].window_size:
        print('there are too many batches, therefore this accuracy may be not accurate.')
    

    with open(detail_log_path, 'w') as f:
        for line in detailed_log:
            f.write(line + '\n')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
