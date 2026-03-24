import os
import glob
import torch
import importlib
import torch.nn as nn
from thop import profile, clever_format
from config import system_configs
from models.py_utils.data_parallel import DataParallel

torch.manual_seed(317)

class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()

        self.model = model
        self.loss  = loss

    def forward(self, iteration, save, viz_split,
                xs, ys, **kwargs):

        preds, weights = self.model(*xs, targets=ys, **kwargs)

        loss  = self.loss(iteration,
                          save,
                          viz_split,
                          preds,
                          ys,
                          **kwargs)
        return loss

# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)

class NetworkFactory(object):
    def __init__(self, flag=False):
        super(NetworkFactory, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        module_file = "models.{}".format(system_configs.snapshot_name)
        # print("module_file: {}".format(module_file)) # models.CornerNet
        nnet_module = importlib.import_module(module_file)

        self.model   = DummyModule(nnet_module.model(flag=flag))
        self.loss    = nnet_module.loss()
        self.network = Network(self.model, self.loss)
        self.network = DataParallel(self.network, chunk_sizes=system_configs.chunk_sizes)
        self.flag    = flag
        self.network.to(self.device)

        # Count total parameters
        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print("Total parameters: {}".format(total_params))
        print("Device: {}".format(self.device))

        # Count MACs when input is 360 x 640 x 3. Skip on CPU to keep local smoke usable.
        if self.device.type == "cuda":
            input_test = torch.randn(1, 3, 360, 640, device=self.device)
            input_mask = torch.randn(1, 3, 360, 640, device=self.device)
            macs, params, = profile(self.model, inputs=(input_test, input_mask), verbose=False)
            macs, _ = clever_format([macs, params], "%.3f")
            print('MACs: {}'.format(macs))
        else:
            print('MACs: skipped on CPU')


        if system_configs.opt_algo == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )
        elif system_configs.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate, 
                momentum=0.9, weight_decay=0.0001
            )
        elif system_configs.opt_algo == 'adamW':
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate,
                weight_decay=1e-4
            )
        else:
            raise ValueError("unknown optimizer")

    def cuda(self):
        self.network.to(self.device)

    def _move_batch(self, tensors):
        non_blocking = self.device.type == "cuda"
        return [tensor.to(self.device, non_blocking=non_blocking) for tensor in tensors]

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def train(self,
              iteration,
              save,
              viz_split,
              xs,
              ys,
              **kwargs):
        xs = self._move_batch(xs)
        ys = self._move_batch(ys)

        self.optimizer.zero_grad()
        forward_kwargs = dict(kwargs)
        loss_kp = self.network(iteration,
                               save,
                               viz_split,
                               xs,
                               ys,
                               **forward_kwargs)

        loss      = loss_kp[0]
        loss_dict = loss_kp[1:]
        loss      = loss.mean()

        loss.backward()
        self.optimizer.step()

        return loss, loss_dict

    def validate(self,
                 iteration,
                 save,
                 viz_split,
                 xs,
                 ys,
                 **kwargs):

        with torch.no_grad():
            xs = self._move_batch(xs)
            ys = self._move_batch(ys)
            forward_kwargs = dict(kwargs)
            loss_kp = self.network(iteration,
                                   save,
                                   viz_split,
                                   xs,
                                   ys,
                                   **forward_kwargs)
            loss      = loss_kp[0]
            loss_dict = loss_kp[1:]
            loss      = loss.mean()

            return loss, loss_dict

    def test(self, xs, **kwargs):
        with torch.no_grad():
            # xs = [x.cuda(non_blocking=True) for x in xs]
            return self.model(*xs, **kwargs)

    def set_lr(self, lr):
        print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def load_pretrained_params(self, pretrained_model):
        print("loading from {}".format(pretrained_model))
        with open(pretrained_model, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

    def load_params(self, iteration, is_bbox_only=False):
        cache_file = system_configs.snapshot_file.format(iteration)

        if not os.path.exists(cache_file):
            snapshot_dir = system_configs.snapshot_dir
            snapshot_name = system_configs.snapshot_name
            # Gizli karakter/sonek farklarında tolerans: <name>_<iter>*.pkl*
            pattern = os.path.join(snapshot_dir, f"{snapshot_name}_{int(iteration)}*.pkl*")
            candidates = sorted(glob.glob(pattern))

            if candidates:
                cache_file = candidates[0]
                print(
                    "[NetworkFactory] exact checkpoint bulunamadı; eşleşen dosya kullanılıyor: {}"
                    .format(cache_file)
                )
            else:
                nearby = sorted(glob.glob(os.path.join(snapshot_dir, f"{snapshot_name}_*.pkl*")))
                nearby_tail = nearby[-8:]
                raise FileNotFoundError(
                    "Checkpoint bulunamadı.\n"
                    "  expected: {}\n"
                    "  absolute: {}\n"
                    "  cwd: {}\n"
                    "  snapshot_dir: {}\n"
                    "  nearby: {}".format(
                        system_configs.snapshot_file.format(iteration),
                        os.path.abspath(system_configs.snapshot_file.format(iteration)),
                        os.getcwd(),
                        snapshot_dir,
                        nearby_tail,
                    )
                )

        with open(cache_file, "rb") as f:
            params = torch.load(f)
            model_dict = self.model.state_dict()
            if len(params) != len(model_dict):
                pretrained_dict = {k: v for k, v in params.items() if k in model_dict}
            else:
                pretrained_dict = params
            model_dict.update(pretrained_dict)

            self.model.load_state_dict(model_dict)


    def save_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = self.model.state_dict()
            torch.save(params, f)
