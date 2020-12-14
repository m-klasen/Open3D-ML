import torch
import logging
from tqdm import tqdm
import numpy as np
import re

from datetime import datetime

from os.path import exists, join
from torch.utils.data import DataLoader
from pathlib import Path

from .base_pipeline import BasePipeline
from ..dataloaders import TorchDataloader
from torch.utils.tensorboard import SummaryWriter
from ..utils import latest_torch_ckpt
from ...utils import make_dir, PIPELINE, LogRecord, get_runid, code2md
import numpy as np

from ...metrics.mAP import mAP, convert_data_eval

logging.setLogRecordFactory(LogRecord)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class ObjectDetection(BasePipeline):
    """
    Pipeline for object detection. 
    """

    def __init__(self,
                 model,
                 dataset=None,
                 name='ObjectDetection',
                 main_log_dir='./logs/',
                 device='gpu',
                 split='train',
                 **kwargs):
        super().__init__(model=model,
                         dataset=dataset,
                         name=name,
                         main_log_dir=main_log_dir,
                         device=device,
                         split=split,
                         **kwargs)

    def run_inference(self, data):
        """
        Run inference on given data.

        Args:
            data: A raw data.
        Returns:
            Returns the inference results.
        """
        model = self.model
        device = self.device

        model.to(device)
        model.device = device
        model.eval()

        with torch.no_grad():
            if not isinstance(data['point'], torch.Tensor):
                inputs = torch.tensor([data['point']],
                                    dtype=torch.float32,
                                    device=self.device)
            else:
                inputs = data['point'].unsqueeze(0)
            results = model(inputs)
            boxes = model.inference_end(results, data)

        return boxes

    def run_test(self):
        """
        Run test with test data split, computes mean average precision of the prediction results.
        """
        model = self.model
        dataset = self.dataset
        device = self.device
        cfg = self.cfg
        model.device = device
        model.to(device)

        model.eval()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log.info("DEVICE : {}".format(device))
        log_file_path = join(cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        test_split = TorchDataloader(dataset=dataset.get_split('test'),
                                     preprocess=model.preprocess,
                                     transform=None,
                                     use_cache=False,
                                     shuffle=False)

        self.load_ckpt(model.cfg.ckpt_path)

        log.info("Started testing")
        self.test_ious = []

        pred = []
        gt = []
        with torch.no_grad():
            for i in tqdm(range(len(test_split)), desc='testing'):
                results = self.run_inference(test_split[i]['data'])

                pred.append(convert_data_eval(results[0], [40, 25]))
                gt.append(convert_data_eval(test_split[i]['data']['bboxes']))

        if cfg.get('test_compute_metric', True):
            ap = mAP(pred, gt, [0, 1, 2], [0, 1, 2], [0.5, 0.5, 0.7], similar_classes={0:4, 2:3})
            log.info("mAP BEV:")
            log.info("Pedestrian: {} (easy) {} (medium) {} (hard)".format(*ap[0,:,0]))
            log.info("Bicycle: {} (easy) {} (medium) {} (hard)".format(*ap[1,:,0]))
            log.info("Car: {} (easy) {} (medium) {} (hard)".format(*ap[2,:,0]))

            ap = mAP(pred, gt, [0, 1, 2], [0, 1, 2], [0.5, 0.5, 0.7], bev=False, similar_classes={0:4, 2:3})
            log.info("")
            log.info("mAP 3D:")
            log.info("Pedestrian: {} (easy) {} (medium) {} (hard)".format(*ap[0,:,0]))
            log.info("Bicycle: {} (easy) {} (medium) {} (hard)".format(*ap[1,:,0]))
            log.info("Car: {} (easy) {} (medium) {} (hard)".format(*ap[2,:,0]))
        #dataset.save_test_result(results, attr)

    def run_valid(self):
        """
        Run validation with validation data split, computes mean average precision and the loss of the prediction results.
        """
        model = self.model
        dataset = self.dataset
        device = self.device
        cfg = self.cfg
        model.device = device
        model.to(device)

        model.eval()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log.info("DEVICE : {}".format(device))
        log_file_path = join(cfg.logs_dir, 'log_valid_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        valid_dataset = dataset.get_split('validation')
        valid_loader = TorchDataloader(dataset=valid_dataset,
                                      preprocess=model.preprocess,
                                      transform=None,
                                      use_cache=dataset.cfg.use_cache,
                                      shuffle=False)

        log.info("Started validation")

        self.valid_losses = {}
        self.valid_mAP = {}

        pred = []
        gt = []
        with torch.no_grad():
            for i in tqdm(range(len(valid_loader)), desc='validation'):
                inputs = model.transform(valid_loader[i]['data'], None)
                results = model(inputs['point'])
                loss = model.loss(results, inputs)
                for l, v in loss.items():
                    if not l in self.valid_losses:
                        self.valid_losses[l] = []
                    self.valid_losses[l].append(v.cpu().item())

                # convert to bboxes for mAP evaluation
                boxes = model.inference_end(results, inputs)
                pred.append(convert_data_eval(boxes[0], [40, 25]))
                gt.append(convert_data_eval(valid_loader[i]['data']['bboxes']))
        
        sum_loss = 0
        desc = "validation - "
        for l, v in self.valid_losses.items():
            desc += " %s: %.03f" % (l, np.mean(v))
            sum_loss += np.mean(v)
        desc += " > loss: %.03f" % sum_loss

        log.info(desc)

        ap = mAP(pred, gt, [0, 1, 2], [0, 1, 2], [0.5, 0.5, 0.7], similar_classes={0:4, 2:3})
        log.info("mAP BEV:")
        log.info("Pedestrian: {} (easy) {} (medium) {} (hard)".format(*ap[0,:,0]))
        log.info("Bicycle: {} (easy) {} (medium) {} (hard)".format(*ap[1,:,0]))
        log.info("Car: {} (easy) {} (medium) {} (hard)".format(*ap[2,:,0]))

        ap = mAP(pred, gt, [0, 1, 2], [0, 1, 2], [0.5, 0.5, 0.7], bev=False, similar_classes={0:4, 2:3})
        log.info("")
        log.info("mAP 3D:")
        log.info("Pedestrian: {} (easy) {} (medium) {} (hard)".format(*ap[0,:,0]))
        log.info("Bicycle: {} (easy) {} (medium) {} (hard)".format(*ap[1,:,0]))
        log.info("Car: {} (easy) {} (medium) {} (hard)".format(*ap[2,:,0]))


    def run_train(self):
        """
        Run training with train data split.
        """
        model = self.model
        device = self.device
        model.device = device
        dataset = self.dataset

        cfg = self.cfg
        model.to(device)

        log.info("DEVICE : {}".format(device))
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log_file_path = join(cfg.logs_dir, 'log_train_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        train_dataset = dataset.get_split('training')
        train_loader = TorchDataloader(dataset=train_dataset,
                                      preprocess=model.preprocess,
                                      transform=model.transform,
                                      use_cache=dataset.cfg.use_cache,
                                      steps_per_epoch=dataset.cfg.get(
                                          'steps_per_epoch_train', None))

        self.optimizer, self.scheduler = model.get_optimizer(cfg.optimizer)

        is_resume = model.cfg.get('is_resume', True)
        start_ep = self.load_ckpt(model.cfg.ckpt_path, is_resume=is_resume)

        dataset_name = dataset.name if dataset is not None else ''
        tensorboard_dir = join(
            self.cfg.train_sum_dir,
            model.__class__.__name__ + '_' + dataset_name + '_torch')
        runid = get_runid(tensorboard_dir)
        self.tensorboard_dir = join(self.cfg.train_sum_dir,
                                    runid + '_' + Path(tensorboard_dir).name)

        writer = SummaryWriter(self.tensorboard_dir)
        self.save_config(writer)
        log.info("Writing summary in {}.".format(self.tensorboard_dir))

        log.info("Started training")
        for epoch in range(start_ep, cfg.max_epoch + 1):
            log.info(f'=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===')
            model.train()

            self.losses = {}
            process_bar = tqdm(range(len(train_loader)), desc='training')        
            for i in process_bar:
                inputs = train_loader[i]['data']

                results = model(inputs['point'])
                loss = model.loss(results, inputs)
                loss_sum = sum(loss.values())

                self.optimizer.zero_grad()
                loss_sum.backward()
                if model.cfg.get('grad_clip_norm', -1) > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(),
                                                    model.cfg.grad_clip_norm)
                self.optimizer.step()
                desc = "training - "
                for l, v in loss.items():
                    if not l in self.losses:
                        self.losses[l] = []
                    self.losses[l].append(v.cpu().item())
                    desc += " %s: %.03f" % (l, v.cpu().item())
                desc += " > loss: %.03f" % loss_sum.cpu().item()
                process_bar.set_description(desc)
                process_bar.refresh()

            #self.scheduler.step()

            # --------------------- validation
            self.run_valid()

            self.save_logs(writer, epoch)

            if epoch % cfg.save_ckpt_freq == 0:
                self.save_ckpt(epoch)

    def save_logs(self, writer, epoch):
        for key, val in self.losses.items():
            writer.add_scalar("train/" + key, np.mean(val), epoch)

        for key, val in self.valid_losses.items():
            writer.add_scalar("valid/" + key, np.mean(val), epoch)

    def load_ckpt(self, ckpt_path=None, is_resume=True):
        train_ckpt_dir = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(train_ckpt_dir)

        epoch = 0
        if ckpt_path is None:
            ckpt_path = latest_torch_ckpt(train_ckpt_dir)
            if ckpt_path is not None and is_resume:
                log.info('ckpt_path not given. Restore from the latest ckpt')
                epoch = int(re.findall(r'\d+', ckpt_path)[-1]) + 1
            else:
                log.info('Initializing from scratch.')
                return epoch

        if not exists(ckpt_path):
            raise FileNotFoundError(f' ckpt {ckpt_path} not found')

        log.info(f'Loading checkpoint {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt and hasattr(self, 'optimizer'):
            log.info(f'Loading checkpoint optimizer_state_dict')
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt and hasattr(self, 'scheduler'):
            log.info(f'Loading checkpoint scheduler_state_dict')
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        return epoch

    def save_ckpt(self, epoch):
        path_ckpt = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(path_ckpt)
        torch.save(
            dict(epoch=epoch,
                 model_state_dict=self.model.state_dict(),
                 optimizer_state_dict=self.optimizer.state_dict()),
            #scheduler_state_dict=self.scheduler.state_dict()),
            join(path_ckpt, f'ckpt_{epoch:05d}.pth'))
        log.info(f'Epoch {epoch:3d}: save ckpt to {path_ckpt:s}')

    def save_config(self, writer):
        '''
        Save experiment configuration with tensorboard summary
        '''
        writer.add_text("Description/Open3D-ML", self.cfg_tb['readme'], 0)
        writer.add_text("Description/Command line", self.cfg_tb['cmd_line'], 0)
        writer.add_text('Configuration/Dataset',
                        code2md(self.cfg_tb['dataset'], language='json'), 0)
        writer.add_text('Configuration/Model',
                        code2md(self.cfg_tb['model'], language='json'), 0)
        writer.add_text('Configuration/Pipeline',
                        code2md(self.cfg_tb['pipeline'], language='json'), 0)


PIPELINE._register_module(ObjectDetection, "torch")
