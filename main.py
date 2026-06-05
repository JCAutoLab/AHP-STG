import math
import time
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import numpy as np
import configparser
from tqdm import tqdm

from models.model import PatchSTG
from lib.utils import log_string, loadSpatialManagedIndices, _compute_loss, metric
from lib.data_loader import data_provider

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        log_string(log, '\n------------ Loading Data -------------')
        self.train_data, self.train_loader = self._get_data(flag='train')
        _, self.vali_loader = self._get_data(flag='val')
        _, self.test_loader = self._get_data(flag='test')

        # ori_parts_idx indicates the original indices of input points
        # reo_parts_idx indicates the patched indices of input points
        # reo_all_idx indicates the patched indices of input and padded points
        self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx = loadSpatialManagedIndices(self.train_data, self.meta_file, self.adj_file,
                                                                                                self.recur_times, self.spa_patchsize, log)
        log_string(log, '------------ End -------------\n')

        self.best_epoch = 0
        self.history = {
            "train_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "val_mape": [],
            "merge_rate": [],
        }

        self.device = torch.device(f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")
        self.build_model()
    
    def build_model(self):
        spatial_attn_bias = self._build_spatial_attn_bias()
        self.model = PatchSTG(self.tem_patchsize, self.tem_patchnum,
                            self.node_num, self.spa_patchsize, self.spa_patchnum,
                            self.tod, self.dow,
                            self.layers, self.factors,
                            self.input_dims, self.node_dims, self.tod_dims, self.dow_dims,
                            self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx,
                            self.merge_tau, self.output_len, self.temporal_mixer,
                            self.temporal_kernel, spatial_attn_bias,
                            self.graph_bias_scale, self.merge_temperature,
                            self.merge_target).to(self.device)

        if getattr(self, 'pretrained_model', None):
            state_dict = torch.load(self.pretrained_model, map_location=self.device)
            load_msg, skipped_keys = self._load_pretrained_state(state_dict)
            log_string(log, f'Loaded pretrained model from {self.pretrained_model}')
            if not self.load_strict:
                log_string(log, f'Missing keys: {len(load_msg.missing_keys)}, Unexpected keys: {len(load_msg.unexpected_keys)}')
                if skipped_keys:
                    preview = ', '.join(skipped_keys[:8])
                    log_string(log, f'Skipped shape-mismatched keys: {len(skipped_keys)} ({preview})')

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=self.learning_rate,weight_decay=self.weight_decay)

        self.lr_scheduler = self._build_scheduler()

    def _load_pretrained_state(self, state_dict):
        if self.load_strict:
            return self.model.load_state_dict(state_dict, strict=True), []

        model_state = self.model.state_dict()
        compatible_state = {}
        skipped_keys = []
        skip_patterns = []
        if getattr(self, 'reset_refiner_res_scale', False):
            skip_patterns.append('adaptive_refiner.res_scale')
        extra_skip = str(getattr(self, 'skip_pretrained_keys', '') or '')
        skip_patterns.extend([x.strip() for x in extra_skip.split(',') if x.strip()])
        for key, value in state_dict.items():
            if any(pattern in key for pattern in skip_patterns):
                skipped_keys.append(key)
                continue
            if key not in model_state:
                compatible_state[key] = value
            elif model_state[key].shape == value.shape:
                compatible_state[key] = value
            else:
                skipped_keys.append(key)
        return self.model.load_state_dict(compatible_state, strict=False), skipped_keys

    def _build_spatial_attn_bias(self):
        if not getattr(self, 'graph_bias', False):
            return None
        if self.spa_patchnum % self.factors != 0:
            raise ValueError('spa_patchnum must be divisible by factors when graph_bias is enabled')

        adj = np.load(self.adj_file).astype(np.float32)
        reordered_idx = np.asarray(self.reo_all_idx, dtype=np.int64)
        local_window_count = self.spa_patchnum // self.factors
        local_window_size = self.spa_patchsize * self.factors
        expected_tokens = local_window_count * local_window_size
        if reordered_idx.shape[0] != expected_tokens:
            raise ValueError(f'graph_bias expects {expected_tokens} reordered tokens, got {reordered_idx.shape[0]}')

        idx = reordered_idx.reshape(local_window_count, local_window_size)
        local_adj = adj[idx[:, :, None], idx[:, None, :]]
        local_adj = np.log1p(np.maximum(local_adj, 0.0))
        mean = local_adj.mean(axis=(1, 2), keepdims=True)
        std = local_adj.std(axis=(1, 2), keepdims=True)
        local_adj = (local_adj - mean) / np.maximum(std, 1e-6)
        local_adj = np.nan_to_num(local_adj, copy=False).astype(np.float32)
        log_string(log, f'Graph attention bias enabled: {local_adj.shape}, scale {self.graph_bias_scale}')
        return torch.from_numpy(local_adj)

    def _build_scheduler(self):
        scheduler = str(getattr(self, 'scheduler', 'multistep')).lower()
        if scheduler in ('none', 'off'):
            return None
        if scheduler in ('cosine', 'warmup_cosine'):
            warmup_epochs = int(getattr(self, 'warmup_epochs', 0))
            min_lr = float(getattr(self, 'min_lr', 0.0))
            base_lr = float(self.learning_rate)
            min_factor = min_lr / base_lr if base_lr > 0 else 0.0

            def lr_lambda(epoch):
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return float(epoch + 1) / float(warmup_epochs)
                total = max(1, int(self.max_epoch) - warmup_epochs)
                progress = min(1.0, max(0.0, float(epoch - warmup_epochs) / float(total)))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_factor + (1.0 - min_factor) * cosine

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        milestones = getattr(self, 'lr_milestones', '1,35,40')
        if isinstance(milestones, str):
            milestones = [int(x) for x in milestones.split(',') if x.strip()]
        return torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=milestones,
            gamma=0.5,
        )
    
    def _get_data(self, flag):
        # tod and dow means the time of the day and the day of the week
        data_set, data_loader = data_provider(self.num_workers, self.batch_size, self.traffic_file,
                                                self.train_ratio, self.test_ratio,
                                                self.input_len, self.output_len,
                                                self.tod, self.dow, flag, log)
        return data_set, data_loader

    def vali(self):
        self.model.eval()
        pred = []
        label = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_te, batch_y_te) in enumerate(self.vali_loader):
                if isinstance(self.model, torch.nn.Module):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_te = batch_x_te.to(self.device)
                    batch_y_te = batch_y_te.to(self.device)

                    y_hat = self.model(batch_x, batch_x_te, batch_y_te)

                    pred.append(self.train_data.inverse_transform(y_hat).cpu().numpy())
                    label.append(batch_y.cpu().numpy())
        
        pred = np.concatenate(pred, axis = 0)
        label = np.concatenate(label, axis = 0)

        maes = []
        rmses = []
        mapes = []

        for i in range(pred.shape[1]):
            mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)

    def train(self):
        log_string(log, "======================TRAIN MODE======================")
        min_loss = 10000000.0

        for epoch in tqdm(range(1,self.max_epoch+1)):
            self.model.train()
            train_l_sum, batch_count, start = 0.0, 0, time.time()
            
            num_batch = math.ceil(len(self.train_data) / self.batch_size)
            with tqdm(total=num_batch) as pbar:
                for i, (batch_x, batch_y, batch_x_te, batch_y_te) in enumerate(self.train_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_te = batch_x_te.to(self.device)
                    batch_y_te = batch_y_te.to(self.device)
                    
                    self.optimizer.zero_grad()

                    y_hat = self.model(batch_x, batch_x_te, batch_y_te)

                    loss = _compute_loss(batch_y, self.train_data.inverse_transform(y_hat))
                    if self.merge_reg_weight > 0:
                        loss = loss + self.merge_reg_weight * self.model.merge_regularization()
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    
                    train_l_sum += loss.cpu().item()

                    batch_count += 1
                    pbar.update(1)

            log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
                % (epoch, self.optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
            self.history["train_loss"].append(float(train_l_sum / batch_count))
            mae, rmse, mape = self.vali()
            self.history["val_mae"].append(float(mae[-1]))
            self.history["val_rmse"].append(float(rmse[-1]))
            self.history["val_mape"].append(float(mape[-1]))
            if getattr(self.model, "latest_merge_rates", None):
                avg_merge = sum(self.model.latest_merge_rates) / len(self.model.latest_merge_rates)
                self.history["merge_rate"].append(float(avg_merge))
                log_string(log, f'epoch {epoch}, avg merge rate {avg_merge:.4f}')
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if mae[-1] < min_loss:
                self.best_epoch = epoch
                min_loss = mae[-1]
                torch.save(self.model.state_dict(), self.model_file)
        
        log_string(log, f'Best epoch is: {self.best_epoch}')

    def test(self):
        log_string(log, "======================TEST MODE======================")
        load_msg = self.model.load_state_dict(torch.load(self.model_file, map_location=self.device), strict=self.load_strict)
        if not self.load_strict:
            log_string(log, f'Test load missing keys: {len(load_msg.missing_keys)}, unexpected keys: {len(load_msg.unexpected_keys)}')
        self.model.eval()
        pred = []
        label = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_te, batch_y_te) in enumerate(self.test_loader):
                if isinstance(self.model, torch.nn.Module):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_te = batch_x_te.to(self.device)
                    batch_y_te = batch_y_te.to(self.device)

                    y_hat = self.model(batch_x, batch_x_te, batch_y_te)

                    pred.append(self.train_data.inverse_transform(y_hat).cpu().numpy())
                    label.append(batch_y.cpu().numpy())
        
        pred = np.concatenate(pred, axis = 0)
        label = np.concatenate(label, axis = 0)

        maes = []
        rmses = []
        mapes = []

        for i in range(pred.shape[1]):
            mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0), pred, label

    def save_experiment_result(self, maes, rmses, mapes, pred=None, label=None):
        result_dir = os.path.dirname(self.result_file)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        payload = {
            "config": self.config_path,
            "best_epoch": int(self.best_epoch),
            "merge_tau": float(self.merge_tau),
            "graph_bias": bool(self.graph_bias),
            "graph_bias_scale": float(self.graph_bias_scale),
            "merge_temperature": float(self.merge_temperature),
            "merge_target": float(self.merge_target),
            "merge_reg_weight": float(self.merge_reg_weight),
            "mae": [float(x) for x in maes.tolist()],
            "rmse": [float(x) for x in rmses.tolist()],
            "mape": [float(x) for x in mapes.tolist()],
            "history": self.history,
        }
        if getattr(self.model, "latest_merge_rates", None):
            payload["merge_rates"] = [float(x) for x in self.model.latest_merge_rates]
        with open(self.result_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log_string(log, f'Results saved to {self.result_file}')

        if self.prediction_file and pred is not None and label is not None:
            pred_dir = os.path.dirname(self.prediction_file)
            if pred_dir:
                os.makedirs(pred_dir, exist_ok=True)
            sample_count = min(pred.shape[0], self.plot_samples)
            np.savez_compressed(
                self.prediction_file,
                pred=pred[:sample_count],
                label=label[:sample_count],
            )
            log_string(log, f'Predictions saved to {self.prediction_file}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='configuration file')
    args, _ = parser.parse_known_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    parser.add_argument('--num_workers', type = int, default = 32)
    parser.add_argument('--cuda', type=str, default=config['train']['cuda'])
    parser.add_argument('--seed', type = int, default = config['train']['seed'])
    parser.add_argument('--batch_size', type = int, default = config['train']['batch_size'])
    parser.add_argument('--max_epoch', type = int, default = config['train']['max_epoch'])
    parser.add_argument('--learning_rate', type=float, default = config['train']['learning_rate'])
    parser.add_argument('--weight_decay', type=float, default = config['train']['weight_decay'])
    parser.add_argument('--scheduler', type=str, default=config['train'].get('scheduler', fallback='multistep'))
    parser.add_argument('--warmup_epochs', type=int, default=config['train'].get('warmup_epochs', fallback=0))
    parser.add_argument('--min_lr', type=float, default=config['train'].get('min_lr', fallback=0.0))
    parser.add_argument('--lr_milestones', type=str, default=config['train'].get('lr_milestones', fallback='1,35,40'))

    parser.add_argument('--input_len', type = int, default = config['data']['input_len'])
    parser.add_argument('--output_len', type = int, default = config['data']['output_len'])
    parser.add_argument('--train_ratio', type = float, default = config['data']['train_ratio'])
    parser.add_argument('--val_ratio', type = float, default = config['data']['val_ratio'])
    parser.add_argument('--test_ratio', type = float, default = config['data']['test_ratio'])

    parser.add_argument('--layers', type=int, default = config['param']['layers'])
    parser.add_argument('--tem_patchsize', type = int, default = config['param']['tps'])
    parser.add_argument('--tem_patchnum', type = int, default = config['param']['tpn'])
    parser.add_argument('--factors', type=int, default = config['param']['factors'])
    parser.add_argument('--recur_times', type = int, default = config['param']['recur'])
    parser.add_argument('--spa_patchsize', type = int, default = config['param']['sps'])
    parser.add_argument('--spa_patchnum', type = int, default = config['param']['spn'])
    parser.add_argument('--node_num', type = int, default = config['param']['nodes'])
    parser.add_argument('--tod', type=int, default = config['param']['tod'])
    parser.add_argument('--dow', type=int, default = config['param']['dow'])
    parser.add_argument('--input_dims', type=int, default = config['param']['id'])
    parser.add_argument('--node_dims', type=int, default = config['param']['nd'])
    parser.add_argument('--tod_dims', type=int, default = config['param']['td'])
    parser.add_argument('--dow_dims', type=int, default = config['param']['dd'])
    parser.add_argument('--merge_tau', type=float, default = config['param'].getfloat('merge_tau', fallback=0.5))
    parser.add_argument('--temporal_mixer', type=lambda x: str(x).lower() == 'true', default=config['param'].getboolean('temporal_mixer', fallback=False))
    parser.add_argument('--temporal_kernel', type=int, default=config['param'].getint('temporal_kernel', fallback=3))
    parser.add_argument('--graph_bias', type=lambda x: str(x).lower() == 'true', default=config['param'].getboolean('graph_bias', fallback=False))
    parser.add_argument('--graph_bias_scale', type=float, default=config['param'].getfloat('graph_bias_scale', fallback=0.0))
    parser.add_argument('--merge_temperature', type=float, default=config['param'].getfloat('merge_temperature', fallback=1.0))
    parser.add_argument('--merge_target', type=float, default=config['param'].getfloat('merge_target', fallback=-1.0))
    parser.add_argument('--merge_reg_weight', type=float, default=config['param'].getfloat('merge_reg_weight', fallback=0.0))
    parser.add_argument('--reset_refiner_res_scale', type=lambda x: str(x).lower() == 'true', default=config['param'].getboolean('reset_refiner_res_scale', fallback=False))
    parser.add_argument('--skip_pretrained_keys', type=str, default=config['param'].get('skip_pretrained_keys', fallback=''))

    parser.add_argument('--traffic_file', default = config['file']['traffic'])
    parser.add_argument('--meta_file', default = config['file']['meta'])
    parser.add_argument('--adj_file', default = config['file']['adj'])
    parser.add_argument('--model_file', default = config['file']['model'])
    parser.add_argument('--log_file', default = config['file']['log'])
    parser.add_argument('--result_file', default = config['file'].get('result', fallback='./results/experiment_result.json'))
    parser.add_argument('--prediction_file', default = config['file'].get('prediction', fallback='./results/prediction_sample.npz'))
    parser.add_argument('--config_path', default = args.config)
    parser.add_argument('--pretrained_model', default = config['file'].get('pretrained', fallback=''))
    parser.add_argument('--load_strict', type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument('--plot_samples', type=int, default=288)
    parser.add_argument('--mode', type=str, default='train', choices=['train','test'])
    
    args = parser.parse_args()

    log = open(args.log_file, 'w')

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    log_string(log, '------------ Options -------------')
    for k, v in vars(args).items():
        log_string(log, '%s: %s' % (str(k), str(v)))
    log_string(log, '-------------- End ----------------')

    solver = Solver(vars(args))

    if args.mode == 'train':
        solver.train()
    maes, rmses, mapes, pred, label = solver.test()
    solver.save_experiment_result(maes, rmses, mapes, pred, label)
    
