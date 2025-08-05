from argparse import ArgumentParser
import torch
import torch.nn as nn
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

import os
from sys import stdout


# snn
from spikingjelly.clock_driven import functional
from syops import get_model_complexity_info

# model
from timm import create_model
from timm.optim.optim_factory import create_optimizer_v2

from config import set_random_seed, parse_arguments, Config
from utils import get_pred, get_scheduler, get_class_weights, get_energy_consumption, tsne_visual, plot_eval
from dataloader import create_loader
import multi_model
from multi_model import TemporalBlock

def load_model(args):
    saved_model_path = os.path.join(args.save_result_path, "model_state", f"{args.best_fold}_{args.saved_epoch[-1]:03d}+model.pt")
    # model_checkpoint = torch.load(saved_model_path, map_location=args.device)
    
    spikformer = create_model(
        'spikformer',
        pretrained=False,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
        checkpoint_path=saved_model_path,
        drop_rate=0.,
        drop_path_rate=0.,
        drop_block_rate=None,
        gating=args.gating,
        train_mode='training',
        window_size=args.window_size,
        data_patch_size=args.data_patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        qkv_bias=False, 
        mlp_ratios=args.mlp_emb,
        depths=args.num_layers, 
        sr_ratios=1,
        time_num_layers=args.time_num_layers,
        T=args.time_steps, 
        lif_bias=args.bias, 
        data_patching_stride=args.stride,
        num_channels=args.num_channels,
        padding_patches=None,
        tau=args.tau,
        spk_encoding=args.spk_encoding,
        attn=args.attn, 
    )

    spikformer = spikformer.to(args.device)
    functional.reset_net(spikformer)

    
    print(f"Model was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
    
    return spikformer


class PathEval(nn.Module):
    def __init__(self, args, train_loader, test_loader, class_num_samples, model=None, path=['time', 'data']):
        super().__init__()
        
        self.time_steps = args.time_steps
        self.num_classes = args.num_classes
        self.bias = args.bias
        self.device = args.device
        self.weight_decay = args.weight_decay
        self.spk_encoding = args.spk_encoding
        
        self.scheduler = "reduce"
        self.save_log_path = args.save_log_path
        self.path = path
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.total_epoch = 50
        
        if model is None:
            self.model = load_model(args)
        
        self.cal_weights = get_class_weights(class_num_samples=class_num_samples)
        
    def path_eval(self):
        
        self.model.head = nn.Linear(self.time_steps, self.num_classes, bias=self.bias).to(self.device)
        
        self.set_requires_grad()

        criterion = nn.CrossEntropyLoss(weight=self.cal_weights).to(self.device)
        optimizer = create_optimizer_v2(self.model.parameters(), opt='adamw', lr=1e-2)
        scheduler = get_scheduler(self.scheduler, optimizer)

        self.embeddings = []
        self.actual = []
        
        print(f"{self.path} path evaluation start.")
        time_acc = self._path_evaluation(criterion, optimizer, scheduler)
        
        self.log_csv(time_acc)
        
        savefig_path = self.set_figsave_path()
        tsne_visual(feature=self.embeddings, actual_label=self.actual, metric='euclidean', save_path=savefig_path, num_classes=self.num_classes) 
        
        print(f"Final tsne result saved to `{savefig_path}`")
        
    def set_log_path(self):
        return self.save_log_path + f'/only+{self.path}+result.csv'
    
    def set_figsave_path(self):
        return self.save_log_path + f'/tsne_result_{self.path}_only_emb.png'
    
    def set_requires_grad(self):
        # freeze
        for name, params in self.model.named_parameters():
            params.requires_grad = False if not name == 'head.weight' else True
        
        for params in self.model.head.parameters():
            params.requires_grad = True
            
    def log_csv(self, time_acc):
        
        log_path = self.set_log_path()
        
        if not isinstance(time_acc, float):
            time_acc = time_acc.mean()
        
        with open(log_path, 'w', encoding='utf-8') as log_csv:
            print("final_epoch", "overall_acc", sep=", ", end="\n", file=log_csv)
            print(f"{self.total_epoch}", f"{time_acc:.6f}", sep=", ", end="\n", file=log_csv)
            
    def get_forward_path(self):
        
        if self.path == 'time' :
            return getattr(self.model, '_time_stream')
        elif self.path == 'data' :
            return getattr(self.model, '_fusing')
            
    def _forward_path(self, data):
        
        forward_path = self.get_forward_path()
        out = forward_path(data) # [T B 1 D]
        if isinstance(out, tuple):
            out = out[0]
        # feature = out.mean(2).mean(-1).transpose(0, 1).contiguous()
        feature = out.mean(2).mean(-1).transpose(0, 1).contiguous()
        out = self.model.head(feature)
        
        return out, feature
    
    def _path_evaluation(self, criterion, optimizer, scheduler=None):
        
        tot_acc = MulticlassAccuracy(num_classes=self.num_classes, average=None).to(self.device)

        for epoch in range(self.total_epoch):
            epoch_loss = 0
            
            self.model.train()
            self.model.train_mode = 'testing'
            # train
            for data, label in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                
                x = getattr(self.model, '_input_encoding')(data)
                out, _ = self._forward_path(x)
                
                loss = criterion(out, label)
                epoch_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                functional.reset_net(self.model)
                
            epoch_loss /= len(self.train_loader)
                
            # test
            epoch_test_loss = 0
            self.model.eval()
            with torch.no_grad():
                for data, label in self.test_loader:
                    data = data.to(self.device)
                    label = label.to(self.device)
                    
                    x = getattr(self.model, '_input_encoding')(data)
                    out, feature = self._forward_path(x)

                    loss = criterion(out, label)
                    epoch_test_loss += loss.item()

                    functional.reset_net(self.model)

                    pred = get_pred(out)
                    tot_acc.update(pred, label)
                    
                    if epoch == (self.total_epoch - 1): 
                        self.embeddings += feature.cpu().numpy().tolist()
                        self.actual += label.cpu().numpy().tolist()
                        
                    
            epoch_test_loss /= len(self.test_loader)
            
            print(f"{'epoch':15s}{epoch:>10d}")
            print(f"{'epoch_test_loss':15s}{epoch_test_loss:>10.5f}")
            print(f"{'overall_acc':15s}{tot_acc.compute().mean().clone():>10.5f}")

            if self.scheduler == 'reduce':
                scheduler.step(epoch_test_loss)
            elif scheduler is None:
                pass
            else:
                scheduler.step()

        return tot_acc.compute()
    

def evaluation(args:Config, criterion, model=None):
    
    loader, _ = create_loader(train=False, batch_size=args.batch_size, num_classes=args.num_classes, data_root=args.test_data_root, data_window_size=args.window_size, segment_len=args.segment_len, num_workers=args.num_workers)


    overall_acc = MulticlassAccuracy(num_classes=args.num_classes).to(args.device)
    acc = MulticlassAccuracy(num_classes=args.num_classes, average=None).to(args.device)
    f1 = MulticlassF1Score(num_classes=args.num_classes, average='macro').to(args.device)
    pre = MulticlassPrecision(num_classes=args.num_classes, average='macro').to(args.device)
    re = MulticlassRecall(num_classes=args.num_classes, average='macro').to(args.device)
    
    if model is None:
        spikformer = load_model(args)
        
    else:
        spikformer = model

    spikformer.eval()
    spikformer.train_mode = 'testing'

    if hasattr(spikformer, 'weak_decoder'): delattr(spikformer, 'weak_decoder')
    if hasattr(spikformer, 'replay') : delattr(spikformer, 'replay')
 
    
    with torch.no_grad():
        epoch_loss = 0
        sim = 0
        for data, label in loader:
            data = data.to(args.device)
            label = label.to(args.device)
                
            output = spikformer(data)
            if isinstance(output, tuple):
                
                output, org_x, rec_x = output

            loss = criterion(output, label)
            epoch_loss += loss.item()
            
            functional.reset_net(spikformer)

            pred = get_pred(output)
            
            overall_acc.update(pred, label)
            acc.update(pred, label)
            f1.update(pred, label)
            pre.update(pred, label)
            re.update(pred, label)
            
        input_res = data[0].shape
        
        model_info_per_layer_path = os.path.join(args.save_log_path, 'model+info+per+layer.txt') 
        file_out = open(model_info_per_layer_path, 'w', encoding='utf-8')
       
        spikformer.train_mode = 'testing'
        ops, params, fr = get_model_complexity_info(
                                    model=spikformer,
                                    input_res=(input_res,), 
                                    dataloader=loader,
                                    as_strings=False,
                                    print_per_layer_stat=True,
                                    # custom_modules_hooks=modules,
                                    # ignore_modules=ignore_modules,
                                    verbose=False,
                                    ost=file_out,
                                )
        
        file_out.close()
        
    test_result = {
        'loss' : epoch_loss/len(loader),
        'overall_acc' : overall_acc.compute(),
        'acc' : acc.compute(),
        'f1' : f1.compute(),
        'pre' : pre.compute(),
        're' : re.compute(),
        'sim' : sim/len(loader)
    }
    
    print("Test was successfully done")

    with open(args.save_log_path + '/final+result.csv', 'a', encoding='utf-8') as log_csv:
        acc_col = ', '.join(f"acc{i}" for i in range(args.num_classes))
        print("loss", "overall_acc", "average_acc", "f1", "precision", "sensitivity", "total_op", "ACop", "MACop", "capacity", "firing_rate", "energy", acc_col,
              sep=", ", end="\n", file=log_csv)
        
        acc_value = ', '.join(f"{test_result['acc'][i]:.6f}" for i in range(args.num_classes))
            
        print(f"{test_result['loss']:.6f}", 
              f"{test_result['overall_acc']:.6f}",
              f"{test_result['acc'].clone().mean():.6f}", 
              f"{test_result['f1']:.6f}", 
              f"{test_result['pre']:.6f}", 
              f"{test_result['re']:.6f}",
            #   f"{test_result['sim']:.6f}",
              f"{ops[0] / 1e6:.2f} M Ops",
              f"{ops[1] / 1e6:.2f} M Ops",
              f"{ops[2] / 1e6:.2f} M Ops",
              f"{params / 1e6:.4f} M",
              f"{fr:.4f} %",
              f"{get_energy_consumption(O_ac=ops[1], O_mac=ops[2], unit='u'):.2f} uJ",
              acc_value,
              sep=", ", end="\n", file=log_csv)
        
    print(f"Final result saved to `{args.save_log_path}`")
    
    # savefig_path = args.save_log_path
    
    # spikformer.eval()
    # spikformer.train_mode = 'visual'
    # spikformer.head = nn.Identity()
    
    # plot_eval(model=spikformer, loader=loader, num_classes=args.num_classes, save_path=savefig_path, device=args.device)
    
    # print(f"Final tsne result saved to `{savefig_path}`")

def test(args, test, only_path_test):

    # TODO: data loader
    test_loader, class_num_samples = create_loader(train=False, num_classes=args.num_classes, batch_size=args.batch_size, data_root=args.test_data_root, data_window_size=args.window_size, num_workers=args.num_workers) 
    
    # TODO: criterion
    cal_weights = get_class_weights(class_num_samples=class_num_samples)
    criterion = nn.CrossEntropyLoss(weight=cal_weights).to(args.device)

    if test:
        evaluation(args=args, loader=test_loader, criterion=criterion)
    
    if only_path_test:
        loader, class_num_samples = create_loader(train=True, batch_size=args.batch_size, num_classes=args.num_classes, data_root=args.train_data_root, train_val_ratio=args.train_val_ratio, data_window_size=args.window_size, sampling=args.sampling, num_workers=args.num_workers)
        
        if isinstance(loader, tuple):
            train_loader, test_loader = loader[0], loader[1]
        else:
            train_loader = loader
            
        data_path = PathEval(args, train_loader, test_loader, class_num_samples, path='data')
        time_path = PathEval(args, train_loader, test_loader, class_num_samples, path='time')
        
        data_path.path_eval()
        time_path.path_eval()

    
        
if __name__ == '__main__':
    
    
    config = parse_arguments()
    args = Config()
    
    if config.config:
        config_path = config.config
        args.load_args(config_path, config)
    
    else:
        args.set_args(config)
        
    set_random_seed(args.seed)
    args.print_info()

    test(args, config.test, config.only_path_test)
    
    