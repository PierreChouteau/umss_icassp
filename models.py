import torch
import torch.nn as nn
from torch.nn import functional as F

import network_components as nc
from ddsp import spectral_ops, synths, core
from model_utls import _Model

import matplotlib.pyplot as plt

import librosa as lb
from scipy.ndimage import filters
import numpy as np

from preprocessing_multif0_cuesta_BCBQ import f0_assignement

import data


# -------- F0 extraction model based on Cuesta's model -------------------------------------------------------------------------------------------
# Correspond to Multi-F0 Estimator
class F0Extractor(_Model):
    def __init__(self, audio_transform=spectral_ops.hcqt_torch, trained_cuesta=False, use_cuda=True, in_channel=5, k_filter=32, k_width=5, k_height=5):
        super(F0Extractor, self).__init__()

        self.audio_transform = audio_transform
        self.use_cuda = use_cuda
        self.trained_cuesta = trained_cuesta
        
        self.in_channel = in_channel
        self.k_filter = k_filter
        self.k_width = k_width
        self.k_height = k_height
        
        self.base_model1 = nn.Sequential(
            nn.BatchNorm2d(in_channel, eps=0.001, momentum=0.99),
            
            # conv1
            nn.Conv2d(
                self.in_channel,
                self.k_filter // 2,
                (self.k_width, self.k_height),
                padding=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.k_filter // 2, eps=0.001, momentum=0.99),
            
            # conv2
            nn.Conv2d(
                self.k_filter // 2,
                self.k_filter,
                (self.k_width, self.k_height),
                padding=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.k_filter, eps=0.001, momentum=0.99),
            
            # conv2
            nn.Conv2d(
                self.k_filter,
                self.k_filter,
                (self.k_width, self.k_height),
                padding=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.k_filter, eps=0.001, momentum=0.99),
            
            # conv2
            nn.Conv2d(
                self.k_filter,
                self.k_filter,
                (self.k_width, self.k_height),
                padding=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(k_filter, eps=0.001, momentum=0.99),
            
            # conv2
            nn.ZeroPad2d(((3-1)//2, (3-1) - (3-1)//2, (70-1)//2, (70-1) - (70-1)//2)),
            nn.Conv2d(self.k_filter, self.k_filter, (70, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(self.k_filter, eps=0.001, momentum=0.99),
            
            # conv2
            nn.ZeroPad2d(((3-1)//2, (3-1) - (3-1)//2, (70-1)//2, (70-1) - (70-1)//2)),
            nn.Conv2d(self.k_filter, self.k_filter, (70, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(self.k_filter, eps=0.001, momentum=0.99),
        )
        
        self.base_model2 = nn.Sequential(
            nn.BatchNorm2d(self.in_channel, eps=0.001, momentum=0.99),
            
            # conv1
            nn.Conv2d(
                self.in_channel,
                self.k_filter // 2,
                (self.k_width, self.k_height),
                padding=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.k_filter // 2, eps=0.001, momentum=0.99),
            
            # conv2
            nn.Conv2d(
                self.k_filter // 2,
                self.k_filter,
                (self.k_width, self.k_height),
                padding=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.k_filter, eps=0.001, momentum=0.99),
            
            # conv2
            nn.Conv2d(
                self.k_filter,
                self.k_filter,
                (self.k_width, self.k_height),
                padding=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.k_filter, eps=0.001, momentum=0.99),
            
            # conv2
            nn.Conv2d(
                self.k_filter,
                self.k_filter,
                (self.k_width, self.k_height),
                padding=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.k_filter, eps=0.001, momentum=0.99),
            
            # conv2
            nn.ZeroPad2d(((3-1)//2, (3-1) - (3-1)//2, (70-1)//2, (70-1) - (70-1)//2)),
            nn.Conv2d(self.k_filter, self.k_filter, (70, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(self.k_filter, eps=0.001, momentum=0.99),
            
            # conv2
            nn.ZeroPad2d(((3-1)//2, (3-1) - (3-1)//2, (70-1)//2, (70-1) - (70-1)//2)),
            nn.Conv2d(self.k_filter, self.k_filter, (70, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(self.k_filter, eps=0.001, momentum=0.99),
        )
        
        self.cuesta = nn.Sequential(
            # conv7 layer
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.99),
            
            # conv8 layer
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.99),
            
            # conv9 layer
            nn.ZeroPad2d((0, 0, (360-1)//2, (360-1) - (360-1)//2)),
            nn.Conv2d(64, 8, kernel_size=(360, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(8, eps=0.001, momentum=0.99),
            
            # output layer
            nn.Conv2d(8, 1, (1, 1), padding=0),
            nn.Sigmoid(),
        )
        
        if self.trained_cuesta == True:
            self.initialize_model(self.base_model1, './pre-trained_models/multi-f0_estimator/base_model1.npy')
            self.initialize_model(self.base_model2, './pre-trained_models/multi-f0_estimator/base_model2.npy')
            self.initialize_model(self.cuesta, './pre-trained_models/multi-f0_estimator/cuesta.npy')
        
    @classmethod
    def from_config(cls, config: dict):
        return cls()
    
    def initialize_model(self, model, np_file):
        
        weights = np.load(np_file, allow_pickle=True)
        
        n_layer = 0
        for i, layer in enumerate(model):
            if isinstance(layer, nn.Conv2d):
                layer.weight = torch.nn.Parameter(torch.from_numpy(weights[n_layer][0]).permute(3, 2, 0, 1))
                layer.bias = torch.nn.Parameter(torch.from_numpy(weights[n_layer][1]))
                n_layer += 1
                
            if isinstance(layer, nn.BatchNorm2d):
                # in tf weight is gamma and bias is beta
                layer.weight = torch.nn.Parameter(torch.from_numpy(weights[n_layer][0]))
                layer.bias = torch.nn.Parameter(torch.from_numpy(weights[n_layer][1]))
                layer.running_mean = torch.from_numpy(weights[n_layer][2])
                layer.running_var = torch.from_numpy(weights[n_layer][3])
                n_layer += 1
                

    # def forward(self, mags, dphases):
    def forward(self, x):
                    
        mags, dphases = self.audio_transform(x, x.device)
        
        mags = mags.to(x.device)
        dphases = dphases.to(x.device)
        
        y6a = self.base_model1(mags)
        y6b = self.base_model2(dphases)
        
        # concatenate features
        y6c = torch.cat((y6a, y6b), dim=1)
                
        y10 = self.cuesta(y6c)
        predictions = y10

        return predictions


# Correspond to Multi-F0 Assigner
class F0Assigner(nn.Module):
    def __init__(self, trained_VA=False):
        super(F0Assigner, self).__init__()
        
        self.F0Assigner = nn.Sequential(
            nn.BatchNorm2d(1, eps=0.001, momentum=0.99),
            
            # conv1
            nn.Conv2d(1, 32, (3, 3), padding=1,),
            nn.ReLU(),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.99),
            
            # conv2
            nn.Conv2d(32, 32, (3, 3), padding=1,),
            nn.ReLU(),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.99),
            
            # conv3
            nn.ZeroPad2d(((3-1)//2, (3-1) - (3-1)//2, (70-1)//2, (70-1) - (70-1)//2)),
            nn.Conv2d(32, 16, (70, 3),),
            nn.ReLU(),
            nn.BatchNorm2d(16, eps=0.001, momentum=0.99),
            
            # conv4
            nn.ZeroPad2d(((3-1)//2, (3-1) - (3-1)//2, (70-1)//2, (70-1) - (70-1)//2)),
            nn.Conv2d(16, 16, (70, 3),),
            nn.ReLU(),
            nn.BatchNorm2d(16, eps=0.001, momentum=0.99),
        )
            
            
        self.branch1 = nn.Sequential(   
            ## branch 1
            nn.Conv2d(16, 16, (3, 3), padding=1,),
            nn.ReLU(),
            nn.BatchNorm2d(16, eps=0.001, momentum=0.99),
            
            nn.Conv2d(16, 16, (3, 3), padding=1,),
            nn.ReLU(),
        )
        
        self.branch2 = nn.Sequential(   
            ## branch 2
            nn.Conv2d(16, 16, (3, 3), padding=1,),
            nn.ReLU(),
            nn.BatchNorm2d(16, eps=0.001, momentum=0.99),
            
            nn.Conv2d(16, 16, (3, 3), padding=1,),
            nn.ReLU(),
        )
        
        self.branch3 = nn.Sequential(   
            ## branch 3
            nn.Conv2d(16, 16, (3, 3), padding=1,),
            nn.ReLU(),
            nn.BatchNorm2d(16, eps=0.001, momentum=0.99),
            
            nn.Conv2d(16,16,(3, 3),padding=1,),
            nn.ReLU(),
        )
        
        self.branch4 = nn.Sequential(   
            ## branch 4
            nn.Conv2d(16, 16, (3, 3), padding=1,),
            nn.ReLU(),
            nn.BatchNorm2d(16, eps=0.001, momentum=0.99),
            
            nn.Conv2d(16, 16, (3, 3), padding=1,),
            nn.ReLU(),
        )
        
        self.post_process1 = nn.Sequential(
            nn.Conv2d(16, 1, 1, padding=0),
            nn.Sigmoid(),
        )
        
        self.post_process2 = nn.Sequential(
            nn.Conv2d(16, 1, 1, padding=0),
            nn.Sigmoid(),
        )
        
        self.post_process3 = nn.Sequential(
            nn.Conv2d(16, 1, 1, padding=0),
            nn.Sigmoid(),
        )
        
        self.post_process4 = nn.Sequential(
            nn.Conv2d(16, 1, 1, padding=0),
            nn.Sigmoid(),
        )
        
        if trained_VA:
            self.initialize_model(self.F0Assigner, "./pre-trained_models/multi-f0_assigner/voas_cnn_f0_assigner.npy")
            
            self.initialize_model(self.branch1, "./pre-trained_models/multi-f0_assigner/voas_cnn_branch1.npy")
            self.initialize_model(self.branch2, "./pre-trained_models/multi-f0_assigner/voas_cnn_branch2.npy")
            self.initialize_model(self.branch3, "./pre-trained_models/multi-f0_assigner/voas_cnn_branch3.npy") 
            self.initialize_model(self.branch4, "./pre-trained_models/multi-f0_assigner/voas_cnn_branch4.npy") 
            
            self.initialize_model(self.post_process1, "./pre-trained_models/multi-f0_assigner/voas_cnn_postprocess1_weights.npy", "./pre-trained_models/multi-f0_assigner/voas_cnn_postprocess1_bias.npy")
            self.initialize_model(self.post_process2, "./pre-trained_models/multi-f0_assigner/voas_cnn_postprocess2_weights.npy", "./pre-trained_models/multi-f0_assigner/voas_cnn_postprocess2_bias.npy")
            self.initialize_model(self.post_process3, "./pre-trained_models/multi-f0_assigner/voas_cnn_postprocess3_weights.npy", "./pre-trained_models/multi-f0_assigner/voas_cnn_postprocess3_bias.npy") 
            self.initialize_model(self.post_process4, "./pre-trained_models/multi-f0_assigner/voas_cnn_postprocess4_weights.npy", "./pre-trained_models/multi-f0_assigner/voas_cnn_postprocess4_bias.npy")
    
        

    def initialize_model(self, model, npy_file, npy_bias_file=None):
        
        weights = np.load(npy_file, allow_pickle=True)
        
        if npy_bias_file is None:
            
            n_layer = 0
            for i, layer in enumerate(model):
                if isinstance(layer, nn.Conv2d):
                    layer.weight = torch.nn.Parameter(torch.from_numpy(weights[n_layer][0]).permute(3, 2, 0, 1))
                    layer.bias = torch.nn.Parameter(torch.from_numpy(weights[n_layer][1]))
                    n_layer += 1
                    
                if isinstance(layer, nn.BatchNorm2d):
                    # in tf weight is gamma and bias is beta
                    layer.weight = torch.nn.Parameter(torch.from_numpy(weights[n_layer][0]))
                    layer.bias = torch.nn.Parameter(torch.from_numpy(weights[n_layer][1]))
                    layer.running_mean = torch.from_numpy(weights[n_layer][2])
                    layer.running_var = torch.from_numpy(weights[n_layer][3])
                    n_layer += 1
                    
        else:
            bias = np.load(npy_bias_file, allow_pickle=True)
            n_layer = 0
            for i, layer in enumerate(model):
                if isinstance(layer, nn.Conv2d):
                    layer.weight = torch.nn.Parameter(torch.from_numpy(weights[n_layer]).permute(3, 2, 0, 1))
                    layer.bias = torch.nn.Parameter(torch.from_numpy(bias[n_layer]))
                    n_layer += 1
                   


    def forward(self, x):
        x = self.F0Assigner(x)
        
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        
        y1 = self.post_process1(x1)
        y2 = self.post_process2(x2)
        y3 = self.post_process3(x3)
        y4 = self.post_process4(x4)
        
        out = torch.cat((y1, y2, y3, y4), dim=1)
        
        return out


class mf0Extract_from_salience():
    def __init__(self, 
                method='sigmoid', 
                sigmoid_factor=1000,
                thresh=[0.23, 0.17, 0.15, 0.17],
                ):
        super(mf0Extract_from_salience, self).__init__()
        
        self.method = method
        self.sigmoid_factor = sigmoid_factor
        self.thresh = thresh

    def sigmoid(self, assigned_saliences):
        
        # Extraction des amplitudes max à partir de saliences assignées
        sop = nn.MaxPool2d(kernel_size=(360, 1))(assigned_saliences[:, 0, :, :])               
        alto = nn.MaxPool2d(kernel_size=(360, 1))(assigned_saliences[:, 1, :, :])                
        tenor = nn.MaxPool2d(kernel_size=(360, 1))(assigned_saliences[:, 2, :, :])                
        bass = nn.MaxPool2d(kernel_size=(360, 1))(assigned_saliences[:, 3, :, :])
        
        # Sigmoid pour obtenir des amplitudes binarisées
        sop = sop - self.thresh[0]
        sop = nn.Sigmoid()(self.sigmoid_factor * sop)
        
        alto = alto - self.thresh[1]
        alto = nn.Sigmoid()(self.sigmoid_factor * alto)
        
        tenor = tenor - self.thresh[2]
        tenor = nn.Sigmoid()(self.sigmoid_factor * tenor )
        
        bass = bass - self.thresh[3]
        bass = nn.Sigmoid()(self.sigmoid_factor * bass)
        
        # Extraction des f0s
        mf0 = data.mf0_predict_batch(assigned_saliences.detach().cpu().numpy())
        
        # Obtention des f0s: 1- binarisation puis multiplication par les fréquences exactes
        f0_sop = sop[:, 0, :] * torch.from_numpy(mf0[:, :, 1]).to(assigned_saliences.device)
        f0_alto = alto[:, 0, :] * torch.from_numpy(mf0[:, :, 2]).to(assigned_saliences.device)
        f0_tenor = tenor[:, 0, :] * torch.from_numpy(mf0[:, :, 3]).to(assigned_saliences.device)
        f0_bass = bass[:, 0, :] * torch.from_numpy(mf0[:, :, 4]).to(assigned_saliences.device)
        
        f0s = torch.stack((f0_sop, f0_alto, f0_tenor, f0_bass), dim=1) # [batch_size, n_sources, n_frames]
        
        return f0s, None
        
        
    def ste(self, assigned_saliences):
        
        # Extraction des amplitudes max à partir de saliences assignées
        sop = nn.MaxPool2d(kernel_size=(360, 1))(assigned_saliences[:, 0, :, :])               
        alto = nn.MaxPool2d(kernel_size=(360, 1))(assigned_saliences[:, 1, :, :])                
        tenor = nn.MaxPool2d(kernel_size=(360, 1))(assigned_saliences[:, 2, :, :])                
        bass = nn.MaxPool2d(kernel_size=(360, 1))(assigned_saliences[:, 3, :, :])

        #------ 1- Ste puis multiplication --------------# 
        # STE pour obtenir les amplitude binarisées à la place du soft threshold
        sop = StraightThroughEstimator(thresh=self.thresh[0])(sop)
        alto = StraightThroughEstimator(thresh=self.thresh[1])(alto)
        tenor = StraightThroughEstimator(thresh=self.thresh[2])(tenor)
        bass = StraightThroughEstimator(thresh=self.thresh[3])(bass)
        
        # Extraction des f0s
        mf0 = data.mf0_predict_batch(assigned_saliences.detach().cpu().numpy())
        
        # Obtention des f0s: 1- binarisation puis multiplication par les fréquences exactes
        f0_sop = sop[:, 0, :] * torch.from_numpy(mf0[:, :, 1]).to(assigned_saliences.device)
        f0_alto = alto[:, 0, :] * torch.from_numpy(mf0[:, :, 2]).to(assigned_saliences.device)
        f0_tenor = tenor[:, 0, :] * torch.from_numpy(mf0[:, :, 3]).to(assigned_saliences.device)
        f0_bass = bass[:, 0, :] * torch.from_numpy(mf0[:, :, 4]).to(assigned_saliences.device)
                
        #------ 2- Juste ste (car dans ste, on englobe binarisation et multiplication --------------# 
        # mf0 = torch.from_numpy(mf0).to(assigned_saliences.device)
                
        # f0_sop = StraightThroughEstimator(thresh=self.thresh[0], mf0=mf0[:, :, 1])(sop[:, 0, :])
        # f0_alto = StraightThroughEstimator(thresh=self.thresh[1], mf0=mf0[:, :, 2])(alto[:, 0, :])
        # f0_tenor = StraightThroughEstimator(thresh=self.thresh[2], mf0=mf0[:, :, 3])(tenor[:, 0, :])
        # f0_bass = StraightThroughEstimator(thresh=self.thresh[3], mf0=mf0[:, :, 4])(bass[:, 0, :])
        
        f0s = torch.stack((f0_sop, f0_alto, f0_tenor, f0_bass), dim=1) # [batch_size, n_sources, n_frames]
        
        
        # Reconstruction des saliences assignées
        assigned_salience_sop_rec, f0_bins_salience = data.mf0_assigned_to_salience_map_batch(mf0[:, :, 1], assigned_saliences.shape)
        assigned_salience_alto_rec, _ = data.mf0_assigned_to_salience_map_batch(mf0[:, :, 2], assigned_saliences.shape)
        assigned_salience_tenor_rec, _ = data.mf0_assigned_to_salience_map_batch(mf0[:, :, 3], assigned_saliences.shape)
        assigned_salience_bass_rec, _ = data.mf0_assigned_to_salience_map_batch(mf0[:, :, 4], assigned_saliences.shape)
                        
        assigned_salience_rec = torch.stack((torch.from_numpy(assigned_salience_sop_rec.astype(np.float32)), 
                                             torch.from_numpy(assigned_salience_alto_rec.astype(np.float32)), 
                                             torch.from_numpy(assigned_salience_tenor_rec.astype(np.float32)), 
                                             torch.from_numpy(assigned_salience_bass_rec.astype(np.float32))
                                             ), 
                                            dim=1).to(assigned_saliences.device)
        
        return f0s, assigned_salience_rec
        
        
    def amplitude(self, assigned_saliences):
        
        # Extraction des amplitudes max à partir de saliences assignées
        sop = nn.MaxPool2d(kernel_size=(360, 1))(assigned_saliences[:, 0, :, :])               
        alto = nn.MaxPool2d(kernel_size=(360, 1))(assigned_saliences[:, 1, :, :])                
        tenor = nn.MaxPool2d(kernel_size=(360, 1))(assigned_saliences[:, 2, :, :])                
        bass = nn.MaxPool2d(kernel_size=(360, 1))(assigned_saliences[:, 3, :, :])
        
        # Extraction des f0s
        mf0 = data.mf0_predict_batch(assigned_saliences.detach().cpu().numpy())
        
        # Test de la division par l'amplitude pour obtenir les f0s (Gaël test)
        f0_sop = sop[:, 0, :] / sop[:, 0, :] * torch.from_numpy(mf0[:, :, 1]).to(assigned_saliences.device)
        f0_alto = alto[:, 0, :] / alto[:, 0, :] * torch.from_numpy(mf0[:, :, 2]).to(assigned_saliences.device)
        f0_tenor = tenor[:, 0, :] / tenor[:, 0, :] * torch.from_numpy(mf0[:, :, 3]).to(assigned_saliences.device)
        f0_bass = bass[:, 0, :] / bass[:, 0, :] * torch.from_numpy(mf0[:, :, 4]).to(assigned_saliences.device)
        
        f0s = torch.stack((f0_sop, f0_alto, f0_tenor, f0_bass), dim=1) # [batch_size, n_sources, n_frames]
        
        return f0s, None
        
        
    def reconstruction(self, assigned_saliences):
        # Extraction des f0s
        mf0 = data.mf0_predict_batch(assigned_saliences.detach().cpu().numpy())
        
        assigned_salience_sop_rec, f0_bins_salience = data.mf0_assigned_to_salience_map_batch(mf0[:, :, 1], assigned_saliences.shape)
        assigned_salience_alto_rec, _ = data.mf0_assigned_to_salience_map_batch(mf0[:, :, 2], assigned_saliences.shape)
        assigned_salience_tenor_rec, _ = data.mf0_assigned_to_salience_map_batch(mf0[:, :, 3], assigned_saliences.shape)
        assigned_salience_bass_rec, _ = data.mf0_assigned_to_salience_map_batch(mf0[:, :, 4], assigned_saliences.shape)
                        
        assigned_salience_rec = torch.stack((torch.from_numpy(assigned_salience_sop_rec.astype(np.float32)), 
                                             torch.from_numpy(assigned_salience_alto_rec.astype(np.float32)), 
                                             torch.from_numpy(assigned_salience_tenor_rec.astype(np.float32)), 
                                             torch.from_numpy(assigned_salience_bass_rec.astype(np.float32))
                                             ), 
                                            dim=1).to(assigned_saliences.device)
        
        f0_bins_batch = torch.stack([torch.from_numpy(f0_bins_salience) for _ in range(assigned_salience_rec.size(1))], dim=1).to(assigned_saliences.device)

        # Multiplication de la salience entière par l'axe des fréquences
        assigned_salience_rec_f0 = assigned_salience_rec * f0_bins_batch
        
        # Copy du gradient - eq à l'opération de Straight Through Estimator
        assign_salience = assigned_saliences + (assigned_salience_rec_f0 - assigned_saliences).detach()
        

        # Somme sur l'axe des fréquences pour trouver la bonne fréquence
        f0s = assign_salience.sum(dim=2)
        
        return f0s, assigned_salience_rec
    
    
    def softmax(self, assigned_saliences):
                
        freq_grid = data.get_freq_grid()       
        freq_grid_frame = freq_grid[:, None].repeat(assigned_saliences.size(3), axis=1)
        freq_grid_frame = freq_grid_frame[None, :,:].repeat(assigned_saliences.size(1), axis=0)
        freq_grid_batch = freq_grid_frame[None,:,:,:].repeat(assigned_saliences.size(0), axis=0)
        
        assign_saliences = assigned_saliences * torch.from_numpy(freq_grid_batch).to(assigned_saliences.device)
        f0s = assign_saliences.sum(dim=2)
                    
        return f0s, None
        
        
    def threshold(self, assigned_saliences):
        
        # Extraction des amplitudes max à partir de saliences assignées
        sop = nn.MaxPool2d(kernel_size=(360, 1))(assigned_saliences[:, 0, :, :])   
        alto = nn.MaxPool2d(kernel_size=(360, 1))(assigned_saliences[:, 1, :, :])                
        tenor = nn.MaxPool2d(kernel_size=(360, 1))(assigned_saliences[:, 2, :, :])                
        bass = nn.MaxPool2d(kernel_size=(360, 1))(assigned_saliences[:, 3, :, :])
        
        # Threshold operation
        sop[sop < self.thresh[0]] = 0.
        alto[alto < self.thresh[1]] = 0.
        tenor[tenor < self.thresh[2]] = 0.
        bass[bass < self.thresh[3]] = 0.
        
        # Extraction des f0s
        mf0 = data.mf0_predict_batch(assigned_saliences.detach().cpu().numpy())
        
        # Multiplication par les fréquences exactes
        f0_sop = sop[:, 0, :] * torch.from_numpy(mf0[:, :, 1]).to(assigned_saliences.device)
        f0_alto = alto[:, 0, :] * torch.from_numpy(mf0[:, :, 2]).to(assigned_saliences.device)
        f0_tenor = tenor[:, 0, :] * torch.from_numpy(mf0[:, :, 3]).to(assigned_saliences.device)
        f0_bass = bass[:, 0, :] * torch.from_numpy(mf0[:, :, 4]).to(assigned_saliences.device)
        
        f0s = torch.stack((f0_sop, f0_alto, f0_tenor, f0_bass), dim=1) # [batch_size, n_sources, n_frames]
            
        return f0s, None
    
            
    def forward(self, assigned_saliences):
        if self.method == 'sigmoid':
            return self.sigmoid(assigned_saliences)
        elif self.method == 'reconstruction':
            return self.reconstruction(assigned_saliences)
        elif self.method == 'amplitude':
            return self.amplitude(assigned_saliences)
        elif self.method == 'ste':
            return self.ste(assigned_saliences)
        elif self.method == 'softmax':
            return self.softmax(assigned_saliences)
        elif self.method == 'threshold':
            return self.threshold(assigned_saliences)
        else:
            raise ValueError('Unknown method: %s' % self.method)


#----------------- Straight Through Estimator ------------------------------------------------------------------------------------------
# Test of Straight Through Estimator to make the thresholding operation differentiable
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh):
        return (input > thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output, -1000, 1000), None
    
    
class StraightThroughEstimator(nn.Module):
    def __init__(self, thresh):
        super(StraightThroughEstimator, self).__init__()
        self.thresh = thresh
        
    def forward(self, x):
        x = STEFunction.apply(x, self.thresh)
        return x


# -------- Unsupervised Model for Source Separation ----------------------------------------------------------------------------------------------
class SourceFilterMixtureAutoencoder2(_Model):

    """Autoencoder that encodes a mixture of n voices into synthesis parameters
    from which the mixture is re-synthesised. Synthesis of each voice is done with a
    source filter model """

    def __init__(self,
                 n_harmonics=101,
                 filter_order=20,
                 fft_size=512,
                 hop_size=256,
                 n_samples=64000,
                 return_sources=False,
                 harmonic_roll_off=12,
                 estimate_noise_mag=False,
                 f_ref=200,  # for harmonics roll off
                 encoder='SeparationEncoderSimple',
                 encoder_hidden_size=256,
                 embedding_size=128,
                 decoder_hidden_size=512,
                 decoder_output_size=512,
                 n_sources=2,
                 bidirectional=True,
                 voiced_unvoiced_diff=True,
                 F0Extractor=None,
                 F0Extractor_trainable=False,
                 F0Assigner=None,
                 F0_models_trainable=False,
                 method='sigmoid',
                 ):

        super().__init__()

        if harmonic_roll_off == -1:
            # estimate roll off
            output_splits=(('harmonic_amplitude', 1),
                           ('noise_gain', 1),
                           ('line_spectral_frequencies', filter_order + 1),
                           ('harmonic_roll_off', 1))
        else:
            output_splits=(('harmonic_amplitude', 1),
                           ('noise_gain', 1),
                           ('line_spectral_frequencies', filter_order + 1))

        # attributes
        self.return_sources = return_sources
        self.n_harmonics = n_harmonics
        self.output_splits = output_splits
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.return_synth_controls = False
        self.harmonic_roll_off = harmonic_roll_off
        self.f_ref = torch.tensor(f_ref, dtype=torch.float32)
        self.estimate_noise_mag = estimate_noise_mag
        self.return_lsf = False
        self.voiced_unvoiced_diff = voiced_unvoiced_diff
        self.n_sources = n_sources
        self.audio_length = n_samples // 16000
        self.F0Extractor = F0Extractor
        self.F0Assigner = F0Assigner
        self.F0_models_trainable = F0_models_trainable
        self.F0Extractor_trainable = F0Extractor_trainable
        
        # neural networks
        overlap = hop_size / fft_size

        if encoder == 'MixEncoderSimple':
            self.encoder = nc.MixEncoderSimple(fft_size=fft_size, overlap=overlap,
                                               hidden_size=encoder_hidden_size, embedding_size=embedding_size,
                                               n_sources=n_sources, bidirectional=bidirectional)

        self.decoder = nc.SynthParameterDecoderSimple(z_size=embedding_size,
                                                      hidden_size=decoder_hidden_size,
                                                      output_size=decoder_output_size,
                                                      bidirectional=bidirectional)
        self.dense_outs = torch.nn.ModuleList([torch.nn.Linear(decoder_output_size, v[1]) for v in output_splits])

        if self.harmonic_roll_off == -2:
            self.gru_roll_off = torch.nn.GRU(decoder_output_size, 1, batch_first=True)
        if self.estimate_noise_mag:
            self.gru_noise_mag = torch.nn.GRU(decoder_output_size, 40, batch_first=True)

        # synth
        self.source_filter_synth = synths.SourceFilterSynth2(n_samples=n_samples,
                                                             sample_rate=16000,
                                                             n_harmonics=n_harmonics,
                                                             audio_frame_size=fft_size,
                                                             hp_cutoff=500,
                                                             f_ref=f_ref,
                                                             estimate_voiced_noise_mag=estimate_noise_mag)
        
        self.method = method
        self.mf0Extract_from_salience = mf0Extract_from_salience(method=self.method,
                                                                 sigmoid_factor=1000,
                                                                 thresh=[0.23,0.17,0.15,0.17])

    @classmethod
    def from_config(cls, config: dict):
        keys = config.keys()
        filter_order = config['filter_order'] if 'filter_order' in keys else 10
        harmonic_roll_off = config['harmonic_roll_off'] if 'harmonic_roll_off' in keys else 12
        f_ref = config['f_ref_source_spec'] if 'f_ref_source_spec' in keys else 500
        encoder = config['encoder'] if 'encoder' in keys else 'SeparationEncoderSimple'
        encoder_hidden_size = config['encoder_hidden_size'] if 'encoder_hidden_size' in keys else 256
        embedding_size = config['embedding_size'] if 'embedding_size' in keys else 128
        decoder_hidden_size = config['decoder_hidden_size'] if 'decoder_hidden_size' in keys else 512
        decoder_output_size = config['decoder_output_size'] if 'decoder_output_size' in keys else 512
        n_sources = config['n_sources'] if 'n_sources' in keys else 2
        estimate_noise_mags = config['estimate_noise_mags'] if 'estimate_noise_mags' in keys else False
        bidirectional = not config['unidirectional'] if 'unidirectional' in keys else True
        voiced_unvoiced_diff = not config['voiced_unvoiced_same_noise'] if 'voiced_unvoiced_same_noise' in keys else True
        return_sources = config['return_sources'] if 'return_sources' in keys else False
        method = config['method'] if 'method' in keys else 'sigmoid'

        return cls(filter_order=filter_order,
                   fft_size=config['nfft'],
                   hop_size=config['nhop'],
                   harmonic_roll_off=harmonic_roll_off,
                   estimate_noise_mag=estimate_noise_mags,
                   f_ref=f_ref,
                   encoder=encoder,
                   encoder_hidden_size=encoder_hidden_size,
                   embedding_size=embedding_size,
                   decoder_hidden_size=decoder_hidden_size,
                   decoder_output_size=decoder_output_size,
                   n_sources=n_sources,
                   bidirectional=bidirectional,
                   voiced_unvoiced_diff=voiced_unvoiced_diff,
                   return_sources=return_sources,
                   method=method,
                   )

    def forward(self, audio, f0_hz, masks=None, plot_figures=False):
        # audio [batch_size, n_samples]
        # f0_hz [batch_size, n_freq_frames, n_sources] : tensor qui stack des tensors     

        if self.F0Extractor is not None: # mf0s extraction is learnt jointly with the rest of the model
        
            if self.F0_models_trainable: # saliences extraction or assignment need is trainable

                if self.F0Extractor_trainable:
                    salience_maps = self.F0Extractor(audio)
                    # print('F0Extractor is trainable, salience_maps are extracted with grad()')
                else:
                    with torch.no_grad(): salience_maps = self.F0Extractor(audio)
                    # print('F0Extractor is not trainable, salience_maps are extracted with no_grad()')
                    
                # From Saliences map to saliences assignment
                assigned_saliences = self.F0Assigner(salience_maps) # .eval() to be sure that the model is in inference mode, we do not update batchnorm  
                if masks is not None: assigned_saliences = assigned_saliences * masks; # print('ATTENTION MASK') # Attention à cette ajout, c'était un simple test
                
                # from assigned salience to mf0s
                mf0s, assigned_saliences_rec = self.mf0Extract_from_salience.forward(assigned_saliences) # [batch_size, n_sources, n_frames]                
                
                mf0s = mf0s.transpose(1, 2)  # [batch_size, n_frames, n_sources]
                f0_hz = mf0s               
                f0_hz = f0_hz.clamp(min=1e-5)
                                                
            else: 
                with torch.no_grad():
                    # With Hcqt from Pytorch (nnAudio) - Salience_maps extraction
                    salience_maps = self.F0Extractor(audio) # .eval() to be sure that the model is in inference mode, we do not update batchnorm        
                    
                    # From Saliences map to saliences assignment
                    assigned_saliences = self.F0Assigner(salience_maps) # .eval() to be sure that the model is in inference mode, we do not update batchnorm  
                    if masks is not None: assigned_saliences = assigned_saliences * masks; print('ATTENTION MASK') # Attention à cette ajout, c'était un simple test
                    
                    # from assigned salience to mf0s
                    mf0s, assigned_saliences_rec = self.mf0Extract_from_salience.forward(assigned_saliences) # [batch_size, n_sources, n_frames]                  
                    
                    mf0s = mf0s.transpose(1, 2)  # [batch_size, n_frames, n_sources]
                    f0_hz = mf0s            
                    f0_hz = f0_hz.clamp(min=1e-5)
                    
                
        z = self.encoder(audio, f0_hz)  # [batch_size, n_frames, n_sources, embedding_size], f0_hz, est un argument non utilisé dans l'encoder

        batch_size, n_frames, n_sources, embedding_size = z.shape

        f0_hz = f0_hz.transpose(1, 2)  # [batch_size, n_sources, n_freq_frames]
        f0_hz = torch.reshape(f0_hz, (batch_size*n_sources, -1))  # [batch_size * n_sources, n_freq_frames]

        f0_hz = core.resample(f0_hz, n_frames)

        if self.voiced_unvoiced_diff:
            # use different noise models for voiced and unvoiced frames (this option was not used in the experiments)
            voiced_unvoiced = torch.where(f0_hz > 1., torch.tensor(1., device=f0_hz.device),
                                                  torch.tensor(0., device=f0_hz.device))[:, :, None]
        else:
            # one noise model (this option was used in the experiments for the paper)
            voiced_unvoiced = torch.ones_like(f0_hz)[:, :, None]

        f0_hz = f0_hz[:, :, None]  # [batch_size * n_sources, n_frames, 1]

        z = z.permute(0, 2, 1, 3)
        z = z.reshape((batch_size*n_sources, n_frames, embedding_size))

        x = self.decoder(f0_hz, z)


        outputs = {}
        for layer, (key, _) in zip(self.dense_outs, self.output_splits):
            outputs[key] = layer(x)

        if self.harmonic_roll_off == -1:
            harmonic_roll_off = core.exp_sigmoid(outputs['harmonic_roll_off'], max_value=20.)
        elif self.harmonic_roll_off == -2:
            # constant value for roll off is GRU output of last frame through exponential sigmoid activation
            harmonic_roll_off = core.exp_sigmoid(4 * self.gru_roll_off(x)[0][:, -1, :], max_value=15., exponent=2.)
            harmonic_roll_off = torch.ones_like(outputs['harmonic_amplitude']) * harmonic_roll_off[:, :, None]
        else:
            harmonic_roll_off = torch.ones_like(outputs['harmonic_amplitude']) * self.harmonic_roll_off

        if self.estimate_noise_mag:
            noise_mag = self.gru_noise_mag(x)[0][:, -1, :]
            noise_mag = noise_mag[:, None, :]
        else:
            noise_mag = None

        # return synth controls for insights into how the input sources are reconstructed
        if self.return_synth_controls:
            return self.source_filter_synth.get_controls(outputs['harmonic_amplitude'],
                                                         harmonic_roll_off,
                                                         f0_hz,
                                                         outputs['noise_gain'],
                                                         voiced_unvoiced,
                                                         outputs['line_spectral_frequencies'],
                                                         noise_mag)


        # apply synthesis model
        signal = self.source_filter_synth(outputs['harmonic_amplitude'],
                                          harmonic_roll_off,
                                          f0_hz,
                                          outputs['noise_gain'],
                                          voiced_unvoiced,
                                          outputs['line_spectral_frequencies'],
                                          noise_mag)

        sources = torch.reshape(signal, (batch_size, n_sources, -1))
        mix = torch.sum(sources, dim=1)

        if self.return_sources:
            if self.F0Extractor is not None:
                if self.F0_models_trainable:
                    if self.method == 'reconstruction' or self.method == 'ste':
                        return mix, sources, salience_maps, assigned_saliences, assigned_saliences_rec, f0_hz
                    else:
                        return mix, sources, salience_maps, assigned_saliences, f0_hz
                else:
                    if self.method == 'reconstruction' or self.method == 'ste':
                        return mix, sources, salience_maps, assigned_saliences, assigned_saliences_rec, f0_hz
                    else:
                        return mix, sources, salience_maps, assigned_saliences, f0_hz
            else:
                return mix, sources, _, _, f0_hz
        if self.return_lsf:
            lsf = core.lsf_activation(outputs['line_spectral_frequencies'])
            return mix, lsf
        return mix




# -------- U-Net Baselines -------------------------------------------------------------------------------------------

class NormalizeSpec(torch.nn.Module):
    def __init__(self):
        super(NormalizeSpec, self).__init__()

    def forward(self, spectrogram):
        """
        Input: spectrograms
              (nb_samples, nb_channels, nb_bins, nb_frames)
        Returns: normalized spectrograms (divided by their respective max)
              (nb_samples, nb_channels, nb_bins, nb_frames)
        """
        max_values, max_idx = torch.max(spectrogram, dim=2, keepdim=True)
        max_values, max_idx = torch.max(max_values, dim=3, keepdim=True)

        max_values[max_values == 0, ...] = 1

        norm_spec = spectrogram / max_values
        return norm_spec


class ConditionGenerator(torch.nn.Module):

    """
    Process f0 information in a more efficient way
    """
    def __init__(self,
                 n_fft,
                 overlap):
        super().__init__()

        self.n_fft = n_fft
        self.overlap = overlap
        self.harmonic_synth = synths.Harmonic(n_samples=64000)

        in_features = int(n_fft//2 + 1)

        self.linear_gamma_1 = torch.nn.Linear(in_features, in_features)
        self.linear_gamma_2 = torch.nn.Linear(in_features, in_features)

        self.linear_beta_1 = torch.nn.Linear(in_features, in_features)
        self.linear_beta_2 = torch.nn.Linear(in_features, in_features)

    def forward(self, f0_hz):

        # f0_hz with shape [batch_size, n_frames, 1] contains f0 in Hz per frame for target source
        batch_size, n_frames, _ = f0_hz.shape
        device = f0_hz.device
        harmonic_amplitudes = torch.ones((batch_size, n_frames, 1), dtype=torch.float32, device=device)
        harmonic_distribution = torch.ones((batch_size, n_frames, 101), dtype=torch.float32, device=device)
        harmonic_signal = self.harmonic_synth(harmonic_amplitudes, harmonic_distribution, f0_hz)

        harmonic_mag = spectral_ops.compute_mag(harmonic_signal, self.n_fft, self.overlap, pad_end=True, center=True)
        harmonic_mag = harmonic_mag.transpose(1, 2)  # [batch_size, n_frames, n_features]

        gamma = self.linear_gamma_1(harmonic_mag)
        gamma = torch.tanh(gamma)
        gamma = self.linear_gamma_2(gamma)
        gamma = torch.relu(gamma)
        gamma = gamma.transpose(1, 2)  # [batch_size, n_features, n_frames]

        beta = self.linear_beta_1(harmonic_mag)
        beta = torch.tanh(beta)
        beta = self.linear_beta_2(beta)
        beta = torch.relu(beta)
        beta = beta.transpose(1, 2)  # [batch_size, n_features, n_frames]

        return beta, gamma


def process_f0(f0, f_bins, n_freqs):
    freqz = np.zeros((f0.shape[0], f_bins.shape[0]))
    haha = np.digitize(f0, f_bins) - 1
    idx2 = haha < n_freqs
    haha = haha[idx2]
    freqz[range(len(haha)), haha] = 1
    atb = filters.gaussian_filter1d(freqz.T, 1, axis=0, mode='constant').T
    min_target = np.min(atb[range(len(haha)), haha])
    atb = atb / min_target
    atb[atb > 1] = 1
    return atb

def grid_to_bins(grid, start_bin_val, end_bin_val):
    """Compute the bin numbers from a given grid
    """
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins


def gaussian_kernel1d(sigma, truncate=4.0):
    """
    Computes a 1-D Gaussian convolution kernel.

    Args:
        sigma: standard deviation
        truncate: Truncate the filter at this many standard deviations.

    Returns:
        phi_x: Gaussian kernel

    """

    radius = int(truncate * sigma + 0.5)

    exponent_range = np.arange(1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    return phi_x



class ConditionGeneratorOriginal(torch.nn.Module):

    """
    Process the f0 information exactly as done in "Petermann et al.,
    DEEP LEARNING BASED SOURCE SEPARATION APPLIED TO CHOIRENSEMBLES, ISMIR 2020"
    """

    def __init__(self,
                 n_fft,
                 overlap):
        super().__init__()

        self.n_fft = n_fft
        self.overlap = overlap

        self.gaussian_kernel = torch.tensor(gaussian_kernel1d(sigma=1.), dtype=torch.float32)[None, None, :]

        self.conv1 = torch.nn.Conv1d(361, 16, kernel_size=10, stride=1, padding=4)
        self.conv2 = torch.nn.Conv1d(16, 64, kernel_size=10, stride=1, padding=4)
        self.conv3 = torch.nn.Conv1d(64, 256, kernel_size=10, stride=1, padding=4)

        self.linear_gamma = torch.nn.Linear(256, 513)
        self.linear_beta = torch.nn.Linear(256, 513)


    def forward(self, f0_hz):

        # f0_hz with shape [batch_size, n_frames, 1] contains f0 in Hz per frame for target source
        batch_size, n_frames, _ = f0_hz.shape
        device = f0_hz.device

        # compute bin index for each f0 value
        k = torch.round(torch.log2(f0_hz/32.7 + 1e-8) * 60) + 1
        k = torch.where(k < 0, torch.tensor(0., device=device), k)
        k = k.type(torch.long)

        f0_one_hot = torch.zeros((batch_size, n_frames, 361), device=device, dtype=torch.float32)
        ones = torch.ones_like(k, device=device, dtype=torch.float32)
        f0_one_hot.scatter_(dim=2, index=k, src=ones)

        padding = self.gaussian_kernel.shape[-1] // 2
        f0_one_hot = f0_one_hot.reshape((batch_size * n_frames, 361))[:, None, :]

        f0_blured = torch.nn.functional.conv1d(f0_one_hot, self.gaussian_kernel.to(device), padding=padding)
        f0_blured = f0_blured.reshape((batch_size, n_frames, -1))
        f0_blured = f0_blured / f0_blured.max(dim=2, keepdim=True)[0]
        f0_blured = f0_blured.transpose(1, 2)  # [batch_size, n_channels, n_frames]

        f0_blured = torch.nn.functional.pad(f0_blured, pad=(0, 1))
        x = self.conv1(f0_blured)
        x = torch.nn.functional.pad(x, pad=(0, 1))
        x = self.conv2(x)
        x = torch.nn.functional.pad(x, pad=(0, 1))
        x = self.conv3(x)  # [batch_size, 256, n_frames]

        x = x.transpose(1, 2)  # [batch_size, n_frames, 256]

        beta = self.linear_beta(x)
        beta = beta.transpose(1, 2)
        gamma = self.linear_gamma(x)
        gamma = gamma.transpose(1, 2)

        return beta, gamma



class BaselineUnet(_Model):

    def __init__(
            self,
            n_fft=1024,
            n_hop=512,
            nb_channels=1,
            sample_rate=16000,
            power=1,
            original=False
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)

        Output: Power/Mag Spectrogram
                (nb_samples, nb_bins, nb_frames, nb_channels)
        """

        super().__init__()

        self.return_mask = False

        self.normalize = NormalizeSpec()

        self.n_fft = n_fft
        self.overlap = 1 - n_hop / n_fft

        self.register_buffer('sample_rate', torch.tensor(sample_rate))
        #self.transform = torch.nn.Sequential(self.stft, self.spec, self.normalize)

        if original:
            self.condition_generator = ConditionGeneratorOriginal(n_fft=n_fft, overlap=self.overlap)
        else:
            self.condition_generator = ConditionGenerator(n_fft=n_fft, overlap=self.overlap)


        # Define the network components
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2), padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(True)
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(True)
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=(5, 5), stride=(2, 2), padding=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(True)
        )
        self.deconv1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv1_BAD = torch.nn.Sequential(
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Dropout2d(0.5)
        )
        self.deconv2 = torch.nn.ConvTranspose2d(512, 128, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv2_BAD = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.Dropout2d(0.5)
        )
        self.deconv3 = torch.nn.ConvTranspose2d(256, 64, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv3_BAD = torch.nn.Sequential(
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.Dropout2d(0.5)
        )
        self.deconv4 = torch.nn.ConvTranspose2d(128, 32, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv4_BAD = torch.nn.Sequential(
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            #torch.nn.Dropout2d(0.5)
        )
        self.deconv5 = torch.nn.ConvTranspose2d(64, 16, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv5_BAD = torch.nn.Sequential(
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            #torch.nn.Dropout2d(0.5)
        )
        self.deconv6 = torch.nn.ConvTranspose2d(32, 1, kernel_size=(5, 5), stride=(2, 2), padding=2)


    @classmethod
    def from_config(cls, config: dict):
        keys = config.keys()
        original = config['original_cu_net'] if 'original_cu_net' in keys else False
        return cls(
                   n_fft=config['nfft'],
                   n_hop=config['nhop'],
                   sample_rate=config['samplerate'],
                   original=original
                   )

    def forward(self, x):

        mix = x[0]  # mix [batch_size, n_samples]
        f0_info = x[1]  #

        beta, gamma = self.condition_generator(f0_info)

        mix_mag = spectral_ops.compute_mag(mix, self.n_fft, self.overlap, pad_end=True, center=True)[:, None, :, :]

        # input must have shape  (batch_size, nb_channels, nb_bins, nb_frames)
        mix_mag_normalized = self.normalize(mix_mag)

        mix_mag_conditioned = mix_mag_normalized * gamma[:, None, :, :] + beta[:, None, :, :]

        conv1_out = self.conv1(mix_mag_conditioned)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        deconv1_out = self.deconv1(conv6_out, output_size=conv5_out.size())
        deconv1_out = self.deconv1_BAD(deconv1_out)
        deconv2_out = self.deconv2(torch.cat([deconv1_out, conv5_out], 1), output_size=conv4_out.size())
        deconv2_out = self.deconv2_BAD(deconv2_out)
        deconv3_out = self.deconv3(torch.cat([deconv2_out, conv4_out], 1), output_size=conv3_out.size())
        deconv3_out = self.deconv3_BAD(deconv3_out)
        deconv4_out = self.deconv4(torch.cat([deconv3_out, conv3_out], 1), output_size=conv2_out.size())
        deconv4_out = self.deconv4_BAD(deconv4_out)
        deconv5_out = self.deconv5(torch.cat([deconv4_out, conv2_out], 1), output_size=conv1_out.size())
        deconv5_out = self.deconv5_BAD(deconv5_out)
        deconv6_out = self.deconv6(torch.cat([deconv5_out, conv1_out], 1), output_size=mix_mag.size())
        mask = torch.sigmoid(deconv6_out)

        y_hat = mask * mix_mag

        # at test time, either return the mask and multiply with complex mix STFT or compute magnitude estimates
        # for all sources and build soft masks with them and multiply then with complex mix STFT
        if self.return_mask:
            return mask
        else:
            return y_hat


def cuesta_model_test():
    
    # -------------------------------- Librosa --------------------------------
    
    # load audio file and compute hcqt
    # pump = data.create_pump_object()
    # features = data.compute_pump_features(pump,'/home/ids/chouteau/umss/Datasets/BC/mixtures_4_sources/1_BC001_part12_satb.wav')
    # input_hcqt = features['dphase/mag'][0]
    # input_dphase = features['dphase/dphase'][0]
    
    # # reshape hcqt and dphase to be compatible with the model
    # input_hcqt = input_hcqt.transpose(2, 1, 0)[np.newaxis, :, :, :]
    # input_dphase = input_dphase.transpose(2, 1, 0)[np.newaxis, :, :, :]
    
    # torch_input1 = torch.from_numpy(input_hcqt[:, :, :, 0:5000].astype('float32'))
    # torch_input2 = torch.from_numpy(input_dphase[:, :, :, 0:5000].astype('float32'))
    
    # cuesta_model = F0Extractor(trained_cuesta=True)
    # cuesta_model = cuesta_model.eval()
    
    # predicted_output = cuesta_model(torch_input1, torch_input2)
    
    # print(predicted_output.shape)
    
    # est_times, est_freqs, peak_tresh_mat = data.pitch_activations_to_mf0(predicted_output[0, :, :].detach().cpu().numpy(), 0.5)

    # # rearrange output
    # for i, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
    #     if any(fqs <= 0):
    #         est_freqs[i] = np.array([f for f in fqs if f > 0])
    
    # f0_assigned_old, f0_assigned_new = f0_assignement(est_freqs, audio_length=10, n_sources=4)

    # print(f0_assigned_old[0:3], f0_assigned_old[-4:-1])
    
    # output_path = './test_fig/librosa_hcqt_output.csv'
    # data.save_multif0_output(est_times, est_freqs, output_path)
    
    # plt.imshow(predicted_output[0, :, :].detach().cpu().numpy(), aspect='auto', origin='lower')
    # plt.savefig('./test_fig/test_librosa_hcqt.png')
    
    
    # -------------------------------- Torch --------------------------------
    
    # definition of the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load the model
    cuesta_model = F0Extractor(trained_cuesta=True)
    cuesta_model = cuesta_model.eval()
    cuesta_model = cuesta_model.to(device)
    
    F0Assigner = nn.Sequential(
        # Input size (in_channels, n_freq_bins, n_frames) = (1, 360, ...)
        # output size (out_channels, n_source, n_frames)
        
        nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1)),
        nn.MaxPool2d(kernel_size=(4, 1)),
        nn.ReLU(),
        
        nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1)),
        nn.MaxPool2d(kernel_size=(2, 1)),
        nn.ReLU(),
        
        nn.Conv2d(1, 1, kernel_size=(42, 1), padding=(0, 0)),
    )
    
    
    # define the resampler to 22050 Hz
    resampler = torchaudio.transforms.Resample(44100, 16000)
    resampler = resampler.to(device)
    
    # load audio file and resample it to 22050 Hz
    audio, sr = torchaudio.load('/home/ids/chouteau/umss/Datasets/BC/mixtures_4_sources/1_BC001_part12_satb.wav')
    audio = resampler(audio)
    audio = audio.to(device)
    print(audio.shape)
    
    predicted_output = cuesta_model(audio)
    
    # Test to observe the peak threshold matrix
    est_times, est_freqs, peak_thresh_mat = data.pitch_activations_to_mf0(predicted_output[0, :, :].detach().cpu().numpy(), 0.5)
    plt.imshow(peak_thresh_mat, aspect='auto', origin='lower')
    plt.savefig('test_fig/peak_thresh_mat.png')
    
    # rearrange output => remove negative frequencies if any (should not happen)
    for i, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
        if any(fqs <= 0):
            est_freqs[i] = np.array([f for f in fqs if f > 0])
    
    print(est_freqs[0:3], est_freqs[-4:-1])
    
    # F0 assignment to each source
    f0_assigned_old, f0_assigned_new = f0_assignement(est_freqs, audio_length=10, n_sources=4)

    print(f0_assigned_old)
    
    # Test to observe the salience map obtained from the F0 assignment
    salience_map_reconstruct = data.mf0_assigned_to_salience_map(est_times, f0_assigned_new, peak_thresh_mat)
    
    plt.imshow(salience_map_reconstruct, aspect='auto', origin='lower')
    plt.savefig('test_fig/salience_reconstruct.png')
    
    output_path = './test_fig/torch_hcqt_output.csv'
    data.save_multif0_output(est_times, est_freqs, output_path)
    
    plt.imshow(predicted_output[0, :, :].detach().cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.savefig('test_fig/test_torch_hcqt.png')
    
    
    # Test the f0 assigner model to remove the non differentiable part
    # f0_assigner_output = F0Assigner(predicted_output[None, :, :, :])
    # print(f0_assigner_output.shape)
    
    # for k, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
    #     if any(fqs <= 0):
    #         # est_freqs[k] = np.array([f for f in fqs if f > 0])
    #         est_freqs[k] = torch.tensor([f for f in fqs if f > 0])
    #     else:
    #         est_freqs[k] = np.array(fqs)
    #         est_freqs[k] = torch.tensor(fqs)

    # # F0 assignment to each source
    # # f0_assigned = f0_assignement(est_freqs, audio_length=10, n_sources=2)
    # f0_assigned = f0_assignement_torch(est_freqs, audio_length=10, n_sources=2)


if __name__ == "__main__":
    
    # torch.random.manual_seed(0)
    # # model = SourceFilterMixtureAutoencoder2(harmonic_roll_off=-2, estimate_noise_mag=True, bidirectional=False)
    # # audio = torch.rand((16, 64000))
    # # f0 = torch.rand((16, 125, 2))
    # # out = model(audio, f0)
    # # print(out.shape)

    # model = BaselineUnet(n_fft=1024, n_hop=256, original=True)
    # mix = torch.rand((16, 64000))
    # info = torch.rand((16, 254, 1)) * 500
    # out = model((mix, info))
    # print(out.shape)
    
    import torchaudio
    cuesta_model_test()