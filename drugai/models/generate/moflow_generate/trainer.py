# -*- encoding: utf-8 -*-
'''
Filename         :trainer.py
Description      :
Time             :2022/08/02 00:28:29
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from __future__ import annotations, print_function
import logging, os
from functools import partial
from typing import List, Optional, Text, Dict, Any

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from rdkit import Chem

from drugai.models.dataset import moflow_collate_fn
from drugai.models.generate.gen_component import GenerateComponent
from drugai.models.generate.moflow_generate.model import MoFlow
from drugai.shared.importers.training_data_importer import TrainingDataImporter
from drugai.shared.preprocess.moflow_preprocessor import MoFlowPreprocessor
from drugai.shared.preprocess.utils import construct_mol, correct_mol, to_numpy_array, valid_mol, valid_mol_can_with_seg

try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

logger = logging.getLogger(__name__)


class MoFlowGenerate(GenerateComponent):
    defaults = {
        # preprocess
        "max_atoms": None, # max number of atoms in a molecule, ie. qm9:9 zinc250k:38
        "add_Hs":False, # add hydrogens to molecules
        "kekulize":True, # kekulize molecule
        "pre_transform":None, # pre-transform function
        "type":"all", # adj normalization,  default: all, other: view
        "usecols": ["SMILES1"], # columns to use

        # training 
        "epochs": 5000,
        "batch_size": 512,
        "learning_rate": 0.001,
        "warmup_steps":10,
        "gamma": 0.5,
        "max_grad_norm": 1.0,

        ## model
        "b_n_type": 4, # number of bond types
        "b_n_flow": 10, # number of flows
        "b_n_block": 1, # number of blocks
        "b_n_squeeze": 3, # number of squeezes 3 or 2 
        "b_hidden_ch": [128,128], # hidden channels
        "b_affine": True, # affine layer
        "b_conv_lu": 1, # convolutional layer
        "a_n_node": 9, # number of nodes in the graph
        "a_n_type": 5, # number of atom types ie qm9:[6, 7, 8, 9, 0] zinc250k:[6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        "a_hidden_gnn": [64], # hidden channels in the graph neural network
        "a_hidden_lin": [128, 64], # hidden channels in the linear neural network
        "a_n_flow": 27, # number of flows
        "a_n_block": 1, # number of blocks
        "a_affine": True, # affine layer
        "mask_row_size_list": [1], # row size of the mask
        "mask_row_stride_list": [1], # row stride of the mask
        "learn_dist":1, # learn distance
        "noise_scale":0.6, # noise scale

        ## sample hyperparameters
        "temp": 1.0, # temperature
        "correct_validity": True, # correct validity
        "largest_connected_comp":True, # largest connected component
        "n_sample": 10000, # number of samples
        


    }

    def __init__(self, 
                 component_config: Optional[Dict[Text, Any]] = None,
                 model=None,
                 **kwargs):
        super().__init__(component_config=component_config, **kwargs)

        self.model = model

    def config_optimizer(self, 
                         *args, 
                         **kwargs) -> Any:
        optimizer = optim.Adam(self.model.parameters(), lr=self.component_config["learning_rate"])
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              self.component_config["warmup_steps"],
                                              self.component_config["gamma"])
        return optimizer, scheduler

    def config_criterion(self, *args, **kwargs) -> Any:
        pass

    def train(self, 
              file_importer: TrainingDataImporter,
              **kwargs) -> Any:
        
        preprocessor = MoFlowPreprocessor(max_atoms=self.component_config["max_atoms"],
                                          add_Hs=self.component_config["add_Hs"],
                                          kekulize=self.component_config["kekulize"])
        
        training_data = file_importer.get_data(preprocessor = preprocessor,
                                               num_workers=kwargs.get("num_workers", None) \
                                                   if kwargs.get("num_workers", None) else 0,
                                                   usecols=self.component_config["usecols"])
        ## 校验参数
        atomic_num_list = training_data.get_atom_list
        if len(atomic_num_list)!=self.component_config["a_n_type"]:
            raise ValueError("a_n_type is not equal to the number of atomic numbers")
        self.component_config["atomic_num_list"] = atomic_num_list

        if self.component_config["max_atoms"]!=self.component_config["a_n_node"]:
            raise ValueError("max_atoms is not equal to a_n_node")
        logger.info("Model Initialization....." )        
        self.model = MoFlow(b_n_type=self.component_config["b_n_type"],
                            a_n_node= self.component_config["a_n_node"],
                            a_n_type=self.component_config["a_n_type"],
                            noise_scale=self.component_config["noise_scale"],
                            learn_dist=self.component_config["learn_dist"],
                            b_n_flow=self.component_config["b_n_flow"],
                            b_n_block=self.component_config["b_n_block"],
                            b_n_squeeze=self.component_config["b_n_squeeze"],
                            b_hidden_ch=self.component_config["b_hidden_ch"],
                            b_affine=self.component_config["b_affine"],
                            b_conv_lu=self.component_config["b_conv_lu"],
                            a_hidden_gnn=self.component_config["a_hidden_gnn"],
                            a_hidden_lin=self.component_config["a_hidden_lin"],
                            a_n_flow=self.component_config["a_n_flow"],
                            a_n_block=self.component_config["a_n_block"],
                            mask_row_size_list=self.component_config["mask_row_size_list"],
                            mask_row_stride_list=self.component_config["mask_row_stride_list"],
                            a_affine=self.component_config["a_affine"])
        self.model.to(self.device)

        
        train_dataloader = training_data.dataloader(batch_size=self.component_config["batch_size"],
                                                    collate_fn=partial(moflow_collate_fn, len(atomic_num_list), self.component_config["max_atoms"]),
                                                    shuffle=True,
                                                    mode="train")

        eval_dataloader = training_data.dataloader(batch_size=self.component_config["batch_size"],
                                                   collate_fn=partial(moflow_collate_fn, len(atomic_num_list), self.component_config["max_atoms"]),
                                                   shuffle=False,
                                                   mode="eval")
        self.optimizer, scheduler = self.config_optimizer()
        self.compute_metric = None

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.zero_grad()

        for epoch in range(self.component_config["epochs"]):
            logger.info("Train: Current epoch {} Start...".format(epoch))
            scheduler.step()  # Update learning rate schedule
            self.logs = {"loss": 0.0, "eval_loss": 0.0}
            self.epoch_data = tqdm(train_dataloader, desc='Training (epoch #{})'.format(epoch))
            self.model.train()
            self.train_epoch()
            self.logs["learning_rate"] = scheduler.get_lr()[0]
            logger.info("Train: Current epoch {} End...".format(epoch))
            if training_data.eval_data is not None:
                logger.info("Evaluate: Current epoch {} Start...".format(epoch))
                self.evaluate(eval_dataloader=eval_dataloader)
                logger.info("Evaluate: Current epoch {} End...".format(epoch))
            for key, value in self.logs.items():
                self.tb_writer.add_scalar(key, value, epoch)

    def train_epoch(self, *args, **kwargs) -> Any:
        for step, batch_data in enumerate(self.epoch_data):
            self.train_step(batch_data, step)
    
    def train_step(self, batch_data, step, *args, **kwargs) -> Any:
        x, adj = batch_data
        x = torch.from_numpy(x).to(self.device)
        adj = torch.from_numpy(adj).to(self.device)
        adj_normalized = self.rescale_adj(adj).to(self.device)
        batch = {
            "x": x,
            "adj": adj,
            "adj_normalized": adj_normalized}
        # Forward, backward and optimize
        z, sum_log_det_jacs = self(**batch)
        if self.n_gpu > 1:
            nll = self.model.module.log_prob(z, sum_log_det_jacs)
        else:
            nll = self.model.log_prob(z, sum_log_det_jacs)
        loss = nll[0] + nll[1]

        self.optimizer.zero_grad()

        if self.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()
        self.logs["loss"] = loss.item()
        self.logs["nll_x"] = nll[0].item()
        self.logs["nll_adj"] = nll[1].item()
        self.epoch_data.set_postfix({**{"loss":self.logs["loss"], "nll_x":self.logs["nll_x"], "nll_adj":self.logs["nll_adj"]},
                                     **{"step": step + 1}})
        self.global_step = 1

        if self.fp16:
            torch.nn.utils.clip_grad_norm(amp.master_params(self.optimizer), self.component_config["max_grad_norm"])
        else:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.component_config["max_grad_norm"])
        return loss.item()
    
    def evaluate(self, eval_dataloader):
        self.eval_data = tqdm(eval_dataloader, desc='Evaluation')
        self.evaluate_epoch()

    def evaluate_epoch(self, *args, **kwargs):
        self.model.eval()

        for step, batch_data in enumerate(self.eval_data):
            self.evaluate_step(batch_data, step)

    @torch.no_grad()
    def evaluate_step(self, batch_data, step, **kwargs):
        x, adj = batch_data
        x = torch.from_numpy(x).to(self.device)
        adj = torch.from_numpy(adj).to(self.device)
        adj_normalized = self.rescale_adj(adj).to(self.device)
        batch = {"x": x,
                 "adj": adj,
                 "adj_normalized": adj_normalized}
        z, sum_log_det_jacs = self(**batch)
        if self.n_gpu > 1:
            nll = self.model.module.log_prob(z, sum_log_det_jacs)
        else:
            nll = self.model.log_prob(z, sum_log_det_jacs)
        loss = nll[0] + nll[1]
        self.logs["eval_loss"] = loss.item()
        self.logs["eval_nll_x"] = nll[0].item()
        self.logs["eval_nll_adj"] = nll[1].item()
        self.epoch_data.set_postfix({**{"eval_loss":self.logs["eval_loss"], 
                                        "eval_nll_x":self.logs["eval_nll_x"], 
                                        "eval_nll_adj":self.logs["eval_nll_adj"]},
                                     **{"eval_step": step + 1}})
        return loss.item()
        

    def get_predict_dataloader(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def predict(self, 
                batch_size: int, 
                z_mu:Any = None,
                true_adj:Any = None,
                **kwargs) -> List[str]:
        z_dim = self.model.b_size + self.model.a_size  # 324 + 45 = 369   9*9*4 + 9 * 5
        mu = np.zeros(z_dim)  # (369,) default , dtype=np.float64
        sigma_diag = np.ones(z_dim)  # (369,)

        if self.component_config["learn_dist"]:
            if len(self.model.ln_var) == 1:
                sigma_diag = np.sqrt(np.exp(self.model.ln_var.item())) * sigma_diag
            elif len(self.model.ln_var) == 2:
                sigma_diag[:self.model.b_size] = np.sqrt(np.exp(self.model.ln_var[0].item())) * sigma_diag[:self.model.b_size]
                sigma_diag[self.model.b_size+1:] = np.sqrt(np.exp(self.model.ln_var[1].item())) * sigma_diag[self.model.b_size+1:]

            # sigma_diag = xp.exp(xp.hstack((model.ln_var_x.data, model.ln_var_adj.data)))

        sigma = self.component_config["temp"] * sigma_diag

        if z_mu is not None:
            mu = z_mu
            sigma = 0.01 * np.eye(z_dim)
        # mu: (369,), sigma: (369,), batch_size: 100, z_dim: 369
        z = np.random.normal(mu, sigma, (batch_size, z_dim))  # .astype(np.float32) [100,369]
        z = torch.from_numpy(z).float().to(self.device)
        adj, x = self.model.reverse(z, true_adj=true_adj) #  (bs, 4, 9, 9), (bs, 9, 5)

        return self.atom_adj_array_to_smiles(adj=adj, 
                                   x=x, 
                                   atomic_num_list=self.component_config["atomic_num_list"])

    def process(self, *args, **kwargs) -> Dict:
        n_sample = self.component_config["n_sample"]
        batch_size = self.component_config["batch_size"]
        self.model.to(self.device)

        samples = []
        n = n_sample
        with tqdm(n, desc="Generating sample") as T:
            while n_sample > 0:
                current_sample = self.predict(min(n, batch_size))
                samples.extend(current_sample)
                n_sample -= len(current_sample)
                T.update(len(current_sample))
        return {"SMILES": samples}

    def persist(self, model_dir: Text
                ) -> Optional[Dict[Text, Any]]:
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        ## 保存模型
        torch.save(model_to_save.state_dict(), os.path.join(model_dir, self.name + "_model.pt"))
        ## 保存参数
        torch.save(self.component_config, os.path.join(model_dir, self.name + "_component_config.pt"))
        return {"model_file": os.path.join(model_dir, self.name + "_model.pt"),
                "component_config": os.path.join(model_dir, self.name + "_component_config.pt")}
    
    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             **kwargs: Any
             ) -> "Component":
        model = MoFlow(b_n_type=meta["b_n_type"],
                       a_n_node=meta["a_n_node"],
                       a_n_type=meta["a_n_type"],
                       noise_scale=meta["noise_scale"],
                       learn_dist=meta["learn_dist"],
                       b_n_flow=meta["b_n_flow"],
                       b_n_block=meta["b_n_block"],
                       b_n_squeeze=meta["b_n_squeeze"],
                       b_hidden_ch=meta["b_hidden_ch"],
                       b_affine=meta["b_affine"],
                       b_conv_lu=meta["b_conv_lu"],
                       a_hidden_gnn=meta["a_hidden_gnn"],
                       a_hidden_lin=meta["a_hidden_lin"],
                       a_n_flow=meta["a_n_flow"],
                       a_n_block=meta["a_n_block"],
                       mask_row_size_list=meta["mask_row_size_list"],
                       mask_row_stride_list=meta["mask_row_stride_list"],
                       a_affine=meta["a_affine"])
        model_state_dict = torch.load(os.path.join(model_dir, meta["name"] + "_model.pt"))
        model.load_state_dict(model_state_dict)
        return cls(component_config=meta,
                   model=model,
                   **kwargs)

    @staticmethod
    def rescale_adj(adj, type='all'):
        # Previous paper didn't use rescale_adj.以前的论文没有使用 rescale_adj
        # In their implementation, the normalization sum is: num_neighbors = F.sum(adj, axis=(1, 2)) 在他们的实现中，归一化和是： num_neighbors = F.sum(adj, axis=(1, 2))
        # In this implementation, the normaliztion term is different 在这个实现中，归一化项是不同的
        # raise NotImplementedError
        # (256,4,9, 9):
        # 4: single, double, triple, and bond between disconnected atoms (negative mask of sum of previous) 4：断开原子之间的单、双、三和键（前一个总和的负掩码）
        # 1-adj[i,:3,:,:].sum(dim=0) == adj[i,4,:,:]
        # usually first 3 matrices have no diagnal, the last has.
        # A_prime = self.A + sp.eye(self.A.shape[0])
        if type == 'view':
            out_degree = adj.sum(dim=-1)
            out_degree_sqrt_inv = out_degree.pow(-1)
            out_degree_sqrt_inv[out_degree_sqrt_inv == float('inf')] = 0
            adj_prime = out_degree_sqrt_inv.unsqueeze(-1) * adj  # (256,4,9,1) * (256, 4, 9, 9) = (256, 4, 9, 9)
        else:  # default type all
            num_neighbors = adj.sum(dim=(1, 2)).float()
            num_neighbors_inv = num_neighbors.pow(-1)
            num_neighbors_inv[num_neighbors_inv == float('inf')] = 0
            adj_prime = num_neighbors_inv[:, None, None, :] * adj
        return adj_prime
    

    def atom_adj_array_to_smiles(self,
                                 adj, 
                                 x, 
                                 atomic_num_list):
        """
        :param adj:  (100,4,9,9)
        :param x: (100.9,5)
        :param atomic_num_list: [6,7,8,9,0]
        :return:
        """
        adj = to_numpy_array(adj)  # , gpu)  (1000,4,9,9)
        x = to_numpy_array(x)  # , gpu)  (1000,9,5)
        if self.component_config["correct_validity"]:
            # valid = [valid_mol_can_with_seg(construct_mol_with_validation(x_elem, adj_elem, atomic_num_list)) # valid_mol_can_with_seg
            #          for x_elem, adj_elem in zip(x, adj)]
            valid = []
            for x_elem, adj_elem in zip(x, adj):
                mol = construct_mol(x_elem, adj_elem, atomic_num_list)
                # Chem.Kekulize(mol, clearAromaticFlags=True)
                cmol = correct_mol(mol)
                vcmol = valid_mol_can_with_seg(cmol, largest_connected_comp=self.component_config["largest_connected_comp"])   #  valid_mol_can_with_seg(cmol)  # valid_mol(cmol)  # valid_mol_can_with_seg
                # Chem.Kekulize(vcmol, clearAromaticFlags=True)
                valid.append(vcmol)
        else:
            valid = [valid_mol(construct_mol(x_elem, adj_elem, atomic_num_list))
                for x_elem, adj_elem in zip(x, adj)]   #len()=1000

        valid_smiles = [Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else None for mol in valid]
        return valid_smiles


