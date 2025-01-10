from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import ConfigDict
from mmdet.core import bbox2roi, bbox_overlaps
from mmdet.models.builder import HEADS
from mmfewshot.detection.models.roi_heads.meta_rcnn_roi_head import MetaRCNNRoIHead
from .supervised_contrastive_loss import MetaSupervisedContrastiveLoss
from .roi_feature_storage import RoIFeatureStorage

class ContrastiveLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels)
        self.fc2 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.normalize(x, dim=1)
        return x
    
class WeaklySelector(nn.Module):

    def __init__(self, in_channels: int, num_classes: int, num_select: dict):
        """
        inputs: dictionary contain torch.Tensors, which comes from backbone
                [Tensor1(hidden feature1), Tensor2(hidden feature2)...]
                Please note that if len(features.size) equal to 3, the order of dimension must be [B,S,C],
                S mean the spatial domain, and if len(features.size) equal to 4, the order must be [B,C,H,W]

        """
        super(WeaklySelector, self).__init__()

        self.num_select = num_select
        self.classifier = nn.Linear(in_channels, num_classes)

    # def select(self, logits, l_name):
    #     """
    #     logits: [B, S, num_classes]
    #     """
    #     probs = torch.softmax(logits, dim=-1)
    #     scores, _ = torch.max(probs, dim=-1)
    #     _, ids = torch.sort(scores, -1, descending=True)
    #     sn = self.num_select[l_name]
    #     s_ids = ids[:, :sn]
    #     not_s_ids = ids[:, sn:]
    #     return s_ids.unsqueeze(-1), not_s_ids.unsqueeze(-1)

    def forward(self, x: Tensor):
        """
        x : Tensor with shape [N, C, H, W]
            
        """
        
        N, C, H, W = x.size()
        x = x.view(N, C, H*W).permute(0, 2, 1).contiguous()
        logits = self.classifier(x)
        
        probs = torch.softmax(logits, dim=-1)
        selections = []
        preds_1 = []
        preds_0 = []
        num_select = self.num_select
        for bi in range(logits.size(0)):
            max_ids, _ = torch.max(probs[bi], dim=-1)
            confs, ranks = torch.sort(max_ids, descending=True)
            sf = x[bi][ranks[:num_select]]
            nf = x[bi][ranks[num_select:]]  # calculate
            selections.append(sf) # [num_selected, C]
            preds_1.append(logits[bi][ranks[:num_select]])
            preds_0.append(logits[bi][ranks[num_select:]])
            
        selections = torch.stack(selections)
        preds_1 = torch.stack(preds_1)
        preds_0 = torch.stack(preds_0)

        return selections, preds_1, preds_0
    
    def loss_funtion(self, selected_logits, not_selected_logits, labels, sf_weight=1.0, nsf_weight=1.0):
        B, S1, C = selected_logits.size()
        selected_logits = selected_logits.view(-1, C).contiguous()
        loss_s = nn.CrossEntropyLoss()(selected_logits, labels.unsqueeze(1).repeat(1, S1).flatten(0))

        B, S2, C = not_selected_logits.size()
        not_selected_logits = not_selected_logits.view(-1, C).contiguous()
        n_preds = nn.Tanh()(not_selected_logits)
        n_labels = torch.zeros([B * S2, C]) - 1
        n_labels = n_labels.to(selected_logits.device)
        loss_n = nn.MSELoss()(n_preds, n_labels)

        return loss_s * sf_weight + loss_n * nsf_weight

@HEADS.register_module()
class VFARoIHead(MetaRCNNRoIHead):

    def __init__(self, 
                 mlp_head_channels=512,
                 num_classes=0,
                 num_novel=0,
                 *args, 
                 **kargs) -> None:
        super().__init__(*args, **kargs)

        self.num_novel = num_novel
        self.num_classes = num_classes
        # self.contrastive_layer = ContrastiveLayer(in_channels=2048, out_channels=mlp_head_channels)
        # self.contrastive_loss = MetaSupervisedContrastiveLoss(temperature=0.2,
        #                                                       iou_threshold=0.8,
        #                                                       loss_weight=0.5,
        #                                                       reweight_type='none')

    def _bbox_forward_train(self, query_feats: List[Tensor],
                            support_feats: List[Tensor],
                            sampling_results: object,
                            query_img_metas: List[Dict],
                            query_gt_bboxes: List[Tensor],
                            query_gt_labels: List[Tensor],
                            support_gt_labels: List[Tensor]) -> Dict:
        """Forward function and calculate loss for box head in training.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            sampling_results (obj:`SamplingResult`): Sampling results.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each query
                image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y]
                format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each box of query images.
            support_gt_labels (list[Tensor]): Class indices corresponding to
                each box of support images.

        Returns:
            dict: Predicted results and losses.
        """
        query_rois = bbox2roi([res.bboxes for res in sampling_results])
        query_roi_feats = self.extract_query_roi_feat(query_feats, query_rois)
        support_feat = self.extract_support_feats(support_feats)[0]

        
        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  query_gt_bboxes,
                                                  query_gt_labels,
                                                  self.train_cfg)

        (labels, label_weights, bbox_targets, bbox_weights) = bbox_targets

        # Code Here

        # Calculate proposal ious
        proposal_ious = []
        for res in sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        proposal_ious = torch.cat(proposal_ious, dim=0)

        multi_query_roi_feats = query_roi_feats
        multi_proposal_ious = proposal_ious
        multi_labels = labels
        
        # Features contrasitve learning
        # multi_query_contrast_feats = self.contrastive_layer(multi_query_roi_feats)
        # support_contrast_feats = self.contrastive_layer(support_feat)

        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': []}
        batch_size = len(query_img_metas)
        num_sample_per_imge = query_roi_feats.size(0) // batch_size
        bbox_results = None
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            
            # class agnostic aggregation
            # random_index = np.random.choice(
            #     range(query_gt_labels[img_id].size(0)))
            # random_query_label = query_gt_labels[img_id][random_index]

            # # vfa old code
            # random_index = np.random.choice(
            #     range(len(support_gt_labels)))
            # random_query_label = support_gt_labels[random_index]
            # for i in range(support_feat.size(0)):
            #     if support_gt_labels[i] == random_query_label:
            #         bbox_results = self._bbox_forward(
            #             query_roi_feats[start:end],
            #             support_feat[i].unsqueeze(0))
            #         single_loss_bbox = self.bbox_head.loss(
            #             bbox_results['cls_score'], bbox_results['bbox_pred'],
            #             query_rois[start:end], labels[start:end],
            #             label_weights[start:end], bbox_targets[start:end],
            #             bbox_weights[start:end])
            #         for key in single_loss_bbox.keys():
            #             loss_bbox[key].append(single_loss_bbox[key])

            s_feat_ids = []
            if self.num_novel != 0:
                b_class = list(range(0, self.num_classes - self.num_novel))
                n_class = list(range(self.num_novel, self.num_classes))
                random_index = np.random.choice(
                    range(query_gt_labels[img_id].size(0)))
                random_query_label = query_gt_labels[img_id][random_index]
                # random_index = np.random.choice(range(len(support_gt_labels)))
                # random_query_label = support_gt_labels[random_index]
                random_base_label = np.random.choice(b_class)
                random_support_label = np.random.choice(n_class)

                for i in range(support_feat.size(0)):
                    if support_gt_labels[i] == random_query_label:
                        s_feat_ids.append(i)

                    # if support_gt_labels[i] == random_base_label:
                    #     s_feat_ids.append(i)
                    
                    if support_gt_labels[i] == random_support_label:
                        s_feat_ids.append(i)
            else:
                random_index = np.random.choice(range(len(support_gt_labels)))
                random_query_label = support_gt_labels[random_index]

                for i in range(support_feat.size(0)):
                    if support_gt_labels[i] == random_query_label:
                        s_feat_ids.append(i)

            for i in s_feat_ids:
                bbox_results = self._bbox_forward(
                    query_roi_feats[start:end],
                    support_feat[i].unsqueeze(0))
                single_loss_bbox = self.bbox_head.loss(
                    bbox_results['cls_score'], bbox_results['bbox_pred'],
                    query_rois[start:end], labels[start:end],
                    label_weights[start:end], bbox_targets[start:end],
                    bbox_weights[start:end])
                for key in single_loss_bbox.keys():
                    loss_bbox[key].append(single_loss_bbox[key])

        if bbox_results is not None:
            for key in loss_bbox.keys():
                if key == 'acc':
                    loss_bbox[key] = torch.cat(loss_bbox['acc']).mean()
                else:
                    loss_bbox[key] = torch.stack(
                        loss_bbox[key]).sum() / batch_size

        # meta classification loss
        if self.bbox_head.with_meta_cls_loss:
            # input support feature classification
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        supp_labels = torch.cat(support_gt_labels)
        # supervised contrastive loss
        # loss_contrast = self.contrastive_loss(
        #     multi_query_contrast_feats,
        #     support_contrast_feats,
        #     multi_labels,
        #     supp_labels,
        #     multi_proposal_ious
        # )
        loss_contrast = {'loss_contrast': loss_contrast}
        loss_bbox.update(loss_contrast)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
    
    def extract_query_roi_feat(self, feats: List[Tensor],
                               rois: Tensor) -> Tensor:
        """Extracting query BBOX features, which is used in both training and
        testing.
        Args:
            feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            rois (Tensor): shape with (bs*128, 5).
        Returns:
            Tensor: RoI features with shape (N, C).
        """
        roi_feats = self.bbox_roi_extractor(
            feats[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        return roi_feats

    def extract_support_feats(self, feats: List[Tensor]) -> List[Tensor]:
        """Forward support features through shared layers.
        Args:
            feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
        Returns:
            list[Tensor]: List of support features, each item
                with shape (N, C).
        """
        out = []
        if self.with_shared_head:
            for lvl in range(len(feats)):
                out.append(self.shared_head.forward_support(feats[lvl]))
        else:
            out = feats
        return out

    def _bbox_forward(self, query_roi_feats: Tensor,
                      support_roi_feats: Tensor) -> Dict:
        """Box head forward function used in both training and testing.

        Args:
            query_roi_feats (Tensor): Query roi features with shape (N, C).
            support_roi_feats (Tensor): Support features with shape (1, C).

        Returns:
             dict: A dictionary of predicted results.
        """
        # feature aggregation
        roi_feats = self.aggregation_layer(
            query_feat=query_roi_feats.unsqueeze(-1).unsqueeze(-1),
            support_feat=support_roi_feats.view(1, -1, 1, 1))[0]
        cls_score, bbox_pred = self.bbox_head(
            roi_feats.squeeze(-1).squeeze(-1), query_roi_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def simple_test_bboxes(
            self,
            query_feats: List[Tensor],
            support_feats_dict: Dict,
            query_img_metas: List[Dict],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """Test only det bboxes without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list
                with shape (num_boxes, 4) and with shape (num_boxes, )
                in second list. The length of both lists should be equal
                to batch_size.
        """
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)

        rois = bbox2roi(proposals)

        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)
        cls_scores_dict, bbox_preds_dict = {}, {}
        num_classes = self.bbox_head.num_classes
        for class_id in support_feats_dict.keys():
            support_feat = support_feats_dict[class_id]
            bbox_results = self._bbox_forward(
                query_roi_feats, support_feat)
            cls_scores_dict[class_id] = \
                bbox_results['cls_score'][:, class_id:class_id + 1]
            bbox_preds_dict[class_id] = \
                bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]
            # the official code use the first class background score as final
            # background score, while this code use average of all classes'
            # background scores instead.
            if cls_scores_dict.get(num_classes, None) is None:
                cls_scores_dict[num_classes] = \
                    bbox_results['cls_score'][:, -1:]
            else:
                cls_scores_dict[num_classes] += \
                    bbox_results['cls_score'][:, -1:]
        cls_scores_dict[num_classes] /= len(support_feats_dict.keys())
        cls_scores = [
            cls_scores_dict[i] if i in cls_scores_dict.keys() else
            torch.zeros_like(cls_scores_dict[list(cls_scores_dict.keys())[0]])
            for i in range(num_classes + 1)
        ]
        bbox_preds = [
            bbox_preds_dict[i] if i in bbox_preds_dict.keys() else
            torch.zeros_like(bbox_preds_dict[list(bbox_preds_dict.keys())[0]])
            for i in range(num_classes)
        ]
        cls_score = torch.cat(cls_scores, dim=1)
        bbox_pred = torch.cat(bbox_preds, dim=1)

        # split batch bbox prediction back to each image
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels