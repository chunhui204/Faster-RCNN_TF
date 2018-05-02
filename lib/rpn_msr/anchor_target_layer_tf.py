# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform
import pdb

DEBUG = False
"""
用于生成anchor, 筛选anchor，
"""
def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, data, _feat_stride = [16,], anchor_scales = [4 ,8, 16, 32]):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    #生成9个anchor， ratio={0.5,1,2}, scale=[8,16,32]
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]

    if DEBUG:
        print 'anchors:'
        print _anchors
        print 'anchor shapes:'
        print np.hstack((
            _anchors[:, 2::4] - _anchors[:, 0::4],
            _anchors[:, 3::4] - _anchors[:, 1::4],
        ))
        _counts = cfg.EPS
        _sums = np.zeros((1, 4))
        _squared_sums = np.zeros((1, 4))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0

    # allow boxes to sit over the edge by a small amount
    _allowed_border =  0
    # map of shape (..., H, W)
    #height, width = rpn_cls_score.shape[1:3]
    #im_info:height, width
    im_info = im_info[0]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap

    assert rpn_cls_score.shape[0] == 1, \
        'Only single item batches are supported'

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]

    if DEBUG:
        print 'AnchorTargetLayer: height', height, 'width', width
        print ''
        print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
        print 'scale: {}'.format(im_info[2])
        print 'height, width: ({}, {})'.format(height, width)
        print 'rpn: gt_boxes.shape', gt_boxes.shape
        print 'rpn: gt_boxes', gt_boxes
"""
相对于(0,0)位置的anchors在原图的偏移量，需要将其在feature map中相对于(0,0)的偏移量×16。
使用np.meshgrid生成所有的x,y偏移值的组合，以600*1000的图片为例，feature map为39*46，shift_x， shift_y都是39*46的
"""
    # 1. Generate proposals from bbox deltas and shifted anchors
    #width=39, 产生39个偏移值
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    #np.ravel将array展成一维，shifts:39*46,4
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors#9
    K = shifts.shape[0]#39*46=2496
    #利用+的广播模式，使得每个anchor(对应A)和每个偏移值（对应K）进行进行求和，
    #all_anchor: (1,K*A,4), 是2496*9个anchor映射到origin image的坐标值
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    #total_anchors=2496
    total_anchors = int(K * A)

    # only keep anchors inside the image
    #筛选anchor：去掉超出边界的anchor，_allowed_border为超出的阈值，代码设置为0，
    #经过这一步anchor从20000-->6000, inds_inside:边界内anchor的索引
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]

    if DEBUG:
        print 'total_anchors', total_anchors
        print 'inds_inside', len(inds_inside)

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    if DEBUG:
        print 'anchors.shape', anchors.shape
    #np.empty用于开辟内存
    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    #bbox_overlaps计算的是anchor与GT box之间的IOU,得到（N，K）数组，分别是anchor num和gt num
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    #找到每个anchor对应IOU最大的gt box
    argmax_overlaps = overlaps.argmax(axis=1)
    #每个anchor对应最大的IOU的值
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    #每个gt box对应IOU最大的anchor
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    #gt对应最大的IOU值
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    #这句其实没用，效果和上上句一样
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    #RPN_CLOBBER_POSITIVES =1用于惩罚正样本，如果正样本的anchor比较多的话， =0惩罚负样本
    #其实都会执行labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0，只是顺序不一样，当惩罚正样本
    #会在labels[gt_argmax_overlaps] = 1之后让IOU《0.3的成为负样本，这样显然会比反顺序负样本数更多。
    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        #cfg.TRAIN.RPN_NEGATIVE_OVERLAP=0.3
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    #gt对应的IOU最大的anchor一定是positive
    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    #cfg.TRAIN.RPN_POSITIVE_OVERLAP=0.7
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
#这样anchor还是很多，以下是每张图的anchor数量为RPN_BATCHSIZE（256），而且pos:neg=1:1， RPN_FG_FRACTION：postive anchor比例
#从postive anchor随机挑选128个
    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1
#随机挑选neg的128个， 当正样本不足128，使用负样本进行填充
    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
        #print "was %s inds, disabling %s, now %s inds" % (
            #len(bg_inds), len(disable_inds), np.sum(labels == 0))
#生成offset的标签值， np.zeros只用于占用空间，应该可以用np.empty代替
#offset是anchor在origin image的坐标与gt在origin image的坐标间的offset
    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    #_compute_targets具体实现了x1y1x2y2到offset的转变
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
#论文loss中的p(i), 正样本为1，负样本为0
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
#在cfg文件里面，cfg.TRAIN.RPN_POSITIVE_WEIGHT为－１，因此这里是对正负样本的权重都除以样本总数，相当于实现了1/Ｎreg的功能。
    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                            np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    if DEBUG:
        _sums += bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means:'
        print means
        print 'stdevs:'
        print stds

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    if DEBUG:
        print 'rpn: max max_overlap', np.max(max_overlaps)
        print 'rpn: num_positive', np.sum(labels == 1)
        print 'rpn: num_negative', np.sum(labels == 0)
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    # labels
    #pdb.set_trace()
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    #assert bbox_inside_weights.shape[2] == height
    #assert bbox_inside_weights.shape[3] == width

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    #assert bbox_outside_weights.shape[2] == height
    #assert bbox_outside_weights.shape[3] == width

    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights



def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5
#bbox_transform根据anchor和gt box生成offset的标签值，（x* - x_a)/w_a...,见论文offset变换
    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
