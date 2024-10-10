from spconv.pytorch import functional as Fsp
import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel
from spconv.pytorch.hash import HashTable
import numpy as np
from cumm import tensorview as tv
import open3d as o3d
from config import config as cfg
import cumm
import torch.nn.functional as F
from os.path import join
from debug import tensor_to_ply,tensorboard_launcher
import MinkowskiEngine as ME
from torch.utils.tensorboard import SummaryWriter

class PointCloud3DCNN(nn.Module):
    ENC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    DEC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    def __init__(self, batch_size, in_channels, out_channels, dimension, n_depth=4):
        super(PointCloud3DCNN, self).__init__()
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS
        
        self.D = dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.upsample_feat_size = 128
        
        self.device = cfg.device
        self.num_point_features = 4
        self.max_num_points_per_voxel = 3
        self.alpha = 0.0
        self.Encoder1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=self.in_channels, kernel_size=3, stride=2, out_channels=enc_ch[0], dimension=self.D),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiReLU()
        )
        self.Encoder2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=enc_ch[0], out_channels=enc_ch[1], kernel_size=3, stride=2, dimension=self.D),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU()
        )
        self.Encoder3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=enc_ch[1], out_channels=enc_ch[2], kernel_size=3, stride=2, dimension=self.D),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU()
        )

        self.Encoder4 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=enc_ch[2], out_channels=enc_ch[3], kernel_size=3, stride=2, dimension=self.D),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU()
        )
        self.Encoder5 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=enc_ch[3], out_channels=enc_ch[4], kernel_size=3, stride=2, dimension=self.D),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU()
        )
        

        self.Decoder5 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_channels=dec_ch[4], out_channels=dec_ch[3], kernel_size=(3, 3, 3, 1), stride=2, dimension=self.D),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU()
        )
        self.occu5 = nn.Sequential(
            ME.MinkowskiConvolution(self.upsample_feat_size, 1, kernel_size=1, bias=True, dimension=self.D),
            ME.MinkowskiSigmoid()

        )

        self.Decoder4 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_channels=dec_ch[3], out_channels=dec_ch[2], kernel_size=(3, 3, 3, 1), stride=2, dimension=self.D),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU()
        )
        self.occu4 = nn.Sequential(
            ME.MinkowskiConvolution(self.upsample_feat_size, 1, kernel_size=1, bias=True, dimension=self.D),
            ME.MinkowskiSigmoid()

        )
        self.Decoder3 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_channels=dec_ch[2], out_channels=dec_ch[1], kernel_size=(3, 3, 3, 1), stride=2, expand_coordinates=True, dimension=self.D),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU()
        )
        self.occu3 = nn.Sequential(
            ME.MinkowskiConvolution(self.upsample_feat_size, 1, kernel_size=1, bias=True, dimension=self.D),
            ME.MinkowskiSigmoid()

        )
        self.Decoder2 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_channels=dec_ch[1], out_channels=dec_ch[0], kernel_size=(3, 3, 3, 1), stride=2, expand_coordinates=True, dimension=self.D),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU()
        )
        self.occu2 = nn.Sequential(
            ME.MinkowskiConvolution(self.upsample_feat_size, 1, kernel_size=1, bias=True, dimension=self.D),
            ME.MinkowskiSigmoid()
        )

        self.Decoder1 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_channels=dec_ch[0], out_channels=self.out_channels, kernel_size=(3, 3, 3, 1), stride=2, expand_coordinates=True, dimension=self.D),
            ME.MinkowskiSigmoid()
            # ME.MinkowskiBatchNorm(self.in_channels),
            # ME.MinkowskiReLU()
        )
        self.occu1 = nn.Sequential(
            ME.MinkowskiConvolution(self.upsample_feat_size, 1, kernel_size=1, bias=True, dimension=self.D),
            ME.MinkowskiSigmoid()
        )
        self.occu0 = nn.Sequential(
            ME.MinkowskiConvolution(self.upsample_feat_size, 1, kernel_size=1, bias=True, dimension=self.D),
            ME.MinkowskiSigmoid()
        )
        self.conv_feat5 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=dec_ch[4], out_channels=self.upsample_feat_size, kernel_size=1, dimension=self.D),
            ME.MinkowskiBatchNorm(self.upsample_feat_size),
            ME.MinkowskiReLU()
        )        

        self.conv_feat4 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=dec_ch[3], out_channels=self.upsample_feat_size, kernel_size=1, dimension=self.D),
            ME.MinkowskiBatchNorm(self.upsample_feat_size),
            ME.MinkowskiReLU()
        )
        self.conv_feat3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=dec_ch[2], out_channels=self.upsample_feat_size, kernel_size=1, dimension=self.D),
            ME.MinkowskiBatchNorm(self.upsample_feat_size),
            ME.MinkowskiReLU()
        )        
        self.conv_feat2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=dec_ch[1], out_channels=dec_ch[2], kernel_size=1, dimension=self.D),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(in_channels=dec_ch[2], out_channels=self.upsample_feat_size, kernel_size=1, dimension=self.D),
            ME.MinkowskiBatchNorm(self.upsample_feat_size),
            ME.MinkowskiReLU()
        )        
        self.conv_feat1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=dec_ch[0], out_channels=dec_ch[1], kernel_size=1, dimension=self.D),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(in_channels=dec_ch[1], out_channels=dec_ch[2], kernel_size=1, dimension=self.D),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(in_channels=dec_ch[2], out_channels=self.upsample_feat_size, kernel_size=1, dimension=self.D),
            ME.MinkowskiBatchNorm(self.upsample_feat_size),
            ME.MinkowskiReLU()
        )   
        self.conv_feat0 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=self.out_channels, out_channels=dec_ch[1], kernel_size=1, dimension=self.D),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(in_channels=dec_ch[1], out_channels=dec_ch[2], kernel_size=1, dimension=self.D),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(in_channels=dec_ch[2], out_channels=self.upsample_feat_size, kernel_size=1, dimension=self.D),
            ME.MinkowskiBatchNorm(self.upsample_feat_size),
            ME.MinkowskiReLU()
        )    
        self.pruning = ME.MinkowskiPruning()
        self.weight_initialization()
        
    def get_target(self, out, target_key, iter, epoch, num_layers, kernel_size=1):
        with torch.no_grad():
            # out = self.cls_process(out)
            
            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            cm = out.coordinate_manager
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0],
            )
            coords = cm.get_coordinates(strided_target_key)

            # print("coords : ", out.dense(min_coordinate=torch.tensor([0, 0, 0, 0], dtype=torch.int32))[0].shape)
            ## for debugging
            # batch_idx = coords[:, 0]
            # coords = coords[:, 1:4]
            # # tensorboard_launcher(coords[batch_idx == 0], iter, [0, 1.0, 0], f"target_{num_layers}")
            # # if iter == 1:
            # #     tensorboard_launcher(coords[batch_idx == 0], epoch, [0, 1.0, 0], f"target_{num_layers}_epoch")
            # if (epoch + 1) % cfg.debug_epoch == 0:
            #     tensorboard_launcher(coords[batch_idx == 0], iter, [0, 1.0, 0], f"target_{num_layers}_epoch", epoch_writer)


            kernel_map = cm.kernel_map(
                out.coordinate_map_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1,
            )
            for k, curr_in in kernel_map.items():
                target[curr_in[0].long()] = 1

        return target, coords
    def encode(self, sparse_tensor):
        enc_feat = []
        enc_feat.append(sparse_tensor)
        enc_0 = self.Encoder1(sparse_tensor) # 20 x 60 x 60
        enc_feat.append(enc_0)
        enc_1 = self.Encoder2(enc_0) # 10 x 30 x 30
        enc_feat.append(enc_1)
        enc_2 = self.Encoder3(enc_1) # 5 x 15 x 15
        enc_feat.append(enc_2)
        enc_3 = self.Encoder4(enc_2) # 3 x 8 x 8
        enc_feat.append(enc_3)
        enc_4 = self.Encoder5(enc_3) 
        enc_feat.append(enc_4)

        return enc_feat
        
    def forward(self, sparse_tensor, target_key, is_train, iter, epoch):
        probs = []
        enc_feat = self.encode(sparse_tensor)
        pyramid_output = None
        num_layers = len(enc_feat)
        
        outputs = []
        targets = []
        classifications = []
        keep_buf = []
        pred_keep_buf = []
        
        for layer_idx in reversed(range(num_layers)):
            conv_feat_layer = self.get_layer('conv_feat', layer_idx)
            conv_occu_layer = self.get_layer('occu', layer_idx)
            if layer_idx is not 0:
                dec = self.get_layer('Decoder', layer_idx)
            curr_feat = enc_feat[layer_idx]

            if pyramid_output is not None:
                assert pyramid_output.tensor_stride == curr_feat.tensor_stride
                curr_feat = curr_feat + pyramid_output 
            feat = conv_feat_layer(curr_feat)
            pred_occu = conv_occu_layer(feat)
            
            target, coords_ = self.get_target(curr_feat, target_key, iter, epoch, layer_idx)
            # print("coords : ", curr_feat.dense(min_coordinate=torch.tensor([0, 0, 0, 0], dtype=torch.int32))[0].shape)
            ## for debugging
            batch_idx_ = coords_[:, 0]
            coords_ = coords_[:, 1:4]
            # tensorboard_launcher(coords[batch_idx == 0], iter, [0, 1.0, 0], f"target_{num_layers}")
            # if iter == 1:
            #     tensorboard_launcher(coords[batch_idx == 0], epoch, [0, 1.0, 0], f"target_{num_layers}_epoch")
            coords = pred_occu.C
            batch_idx = coords[:, 0]
            coords = coords[:, 1:4]
            # tensorboard_launcher(coords[batch_idx == 0], iter, [1.0, 0, 0], f"prob_{layer_idx}")
            # if iter == 1:
            #     tensorboard_launcher(coords[batch_idx == 0], epoch, [1.0, 0, 0], f"prob_{layer_idx}_epoch")
            if (epoch + 1) % cfg.debug_epoch == 0:
                epoch_writer = SummaryWriter(join(cfg.BASE_LOGDIR, f"{epoch}"))
                tensorboard_launcher(coords_[batch_idx_ == 0], iter, [0.0, 0, 1.0], f"target_{layer_idx}_epoch", epoch_writer)
                tensorboard_launcher(coords[batch_idx == 0], iter, [1.0, 0, 0], f"prob_{layer_idx}_epoch", epoch_writer)
                epoch_writer.close()

            pred_keep = (pred_occu.F > 0.8).squeeze(-1)
            gt_keep = target
            # keep = (1 - self.alpha) * gt_keep + self.alpha * pred_keep.squeeze(-1) == 1
            # mask = torch.rand_like(pred_keep) < self.alpha
            # gt_keep[mask.squeeze(-1)] = (pred_keep[mask] > 0.8).squeeze(-1)
            keep = pred_keep
            keep += gt_keep
            # if (epoch + 1) % 5 == 0:
            #     self.alpha += 0.2
            
            if torch.any(keep) and layer_idx is not 0:
                # Prune and upsample
                pyramid_output = dec(self.pruning(curr_feat, keep)) # torch.Size([2, 12, 40, 120, 120, 1])
                # print("coords : ", pyramid_output.dense(min_coordinate=torch.tensor([0, 0, 0, 0], dtype=torch.int32))[0].shape)

                # Generate final feature for current level
                final_pruned = self.pruning(curr_feat, keep)
            else:
                # pyramid_output = None
                final_pruned = None
            
            # Post processing
            classifications.insert(0, pred_occu.F)
            targets.insert(0, target)
            outputs.insert(0, final_pruned)
            keep_buf.insert(0, gt_keep)
            pred_keep_buf.insert(0, pred_keep)

        # if pyramid_output is None:
        #     raise ValueError("pyramid_output is None")
        
        
        preds, batch_coords = self.postprocess(pyramid_output)
        preds = preds.view(self.batch_size, -1, self.max_num_points_per_voxel)
        # preds = preds[:, :, :3]
        
        # batch_coords = final_pruned.decomposed_coordinates
        # preds = self.get_coordinates(final_pruned)        
        
        # min_coord = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
        # dense_tensor = pyramid_output.dense(min_coordinate=min_coord)
        # decoding = self.conv(dense_tensor[0].squeeze(-1)) # torch.Size([2, 1, 56, 120, 120])
        
        return preds, classifications, targets, batch_coords[0][:, :3], pred_keep_buf, keep_buf

    # def postprocess(self, preds):
    #     all_preds = []
    #     batch_coords, batch_feats = preds.decomposed_coordinates_and_features
        
    #     ## padding
    #     batch_counts = torch.zeros(len(batch_coords), device=preds.device)
    #     for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
    #         batch_counts[b] = coords.shape[0]
    #     max_num_points = batch_counts.max().int()
    #     for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
    #         feats = feats.view(-1, self.num_point_features, 3)
    #         feats = feats[: ,:, :3]
    #         coords = coords[:, [2, 1, 0]]
    #         voxel_centers = (coords.float() * torch.tensor([0.05, 0.05, 0.05]).to(preds.device)) + torch.tensor([-3.0, -3.0, -1.0]).to(preds.device)
    #         pred = torch.where(feats == 0, torch.zeros_like(feats), (voxel_centers.unsqueeze(1).to(self.device) + feats*torch.tensor([0.05, 0.05, 0.05]).to(self.device)))
    #         preds = pred.view(-1, 3 * self.num_point_features)
            
    #         padding_size = max_num_points - preds.shape[0]
    #         if padding_size > 0:
    #             padding = torch.zeros((padding_size, preds.shape[1]), device=preds.device)
    #             preds = torch.cat([preds, padding], dim=0)
    #         all_preds.append(preds)
    #     all_preds = torch.stack(all_preds, dim=0)    # (batch_size, max_num_points, 9) 
    #     return all_preds
    
    
    def postprocess(self, preds):
        all_preds = []
        batch_coords, batch_feats = preds.decomposed_coordinates_and_features
        
        ## padding
        batch_counts = torch.zeros(len(batch_coords), device=preds.device)
        for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
            batch_counts[b] = coords.shape[0]
        max_num_points = batch_counts.max().int()
        for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
            coords = coords[:, [2, 1, 0]]
            voxel_centers = (coords.float() * torch.tensor([0.05, 0.05, 0.05]).to(preds.device)) + torch.tensor([-3.0, -3.0, -1.0]).to(preds.device)
            # pred = torch.where(feats == 0, torch.zeros_like(feats), (voxel_centers.unsqueeze(1).to(self.device) + feats*torch.tensor([0.05, 0.05, 0.05]).to(self.device)))
            preds = voxel_centers.view(-1, 3)
            
            padding_size = max_num_points - preds.shape[0]
            if padding_size > 0:
                padding = torch.zeros((padding_size, preds.shape[1]), device=preds.device)
                preds = torch.cat([preds, padding], dim=0)
            all_preds.append(preds)
        all_preds = torch.stack(all_preds, dim=0)    # (batch_size, max_num_points, 9) 
        return all_preds , batch_coords

    
    def get_layer(self, layer_name, layer_idx):
        layer = f'{layer_name}{layer_idx}'
        if hasattr(self, layer):
            return getattr(self, layer)
        else:
            print(f"Layer {layer} does not exist")
            return None
        
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
          
    def cls_process(self, preds):
        all_preds = []
        batch_coords, batch_feats = preds.decomposed_coordinates_and_features
        
        ## padding
        batch_counts = torch.zeros(len(batch_coords), device=preds.device)
        for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
            batch_counts[b] = coords.shape[0]
            
        max_num_points = batch_counts.max().int()
        for i in range(len(batch_coords)):
            # Separate batch indices and the actual coordinates
            coords = batch_coords[i]  # Shape: [N, D+1], where D+1 includes the batch index
            batch_idx = coords[:, 0]  # The batch indices (first column)
            actual_coords = coords[:, 1:]  # The remaining D dimensions for coordinates (e.g., x, y, z)

            feats = batch_feats[i]  # Shape: [N, F] (features for each point)
            
            padding_size = max_num_points - feats.shape[0]
            if padding_size > 0:
                padding = torch.zeros((padding_size, feat.shape[1]), device=preds.device)
                feat = torch.cat([feat, padding], dim=0)
            all_preds.append(preds)
            
            
            
    def get_coords(self, preds):
        batch_coords = final_pruned.decomposed_coordinates
        batch_counts = torch.zeros(len(batch_coords), device=preds.device)
        for b, (coords) in enumerate((batch_coords)):
            batch_counts[b] = coords.shape[0]
        max_num_points = batch_counts.max().int()

        for b, (coords) in enumerate((batch_coords)):
            coords = coords[:, [2, 1, 0]]
            voxel_centers = (coords.float() * torch.tensor([0.05, 0.05, 0.05]).to(preds.device)) + torch.tensor([-3.0, -3.0, -1.0]).to(preds.device)


        


        
