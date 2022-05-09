from __future__ import print_function, absolute_import
from collections import OrderedDict
import torch
import torch.nn.functional as F
from reid.feature_extraction import extract_cnn_feature, extract_cnn_feature_with_tnorm
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from .rerank import re_ranking


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(
            type(tensor)))
    return tensor


def extract_features_per_cam(model, data_loader, norm=True):
    model.eval()
    per_cam_features_without = {}
    per_cam_features_norm = {}
    per_cam_fname = {}
    print("Start extract features per camera")
    for imgs, fnames, _, camid in tqdm(data_loader):
        camid = list(camid)
        for cam in camid:
            cam = cam.item()
            if cam not in per_cam_features_without.keys():
                per_cam_features_without[cam] = []
                per_cam_fname[cam] = []
                per_cam_features_norm[cam] = []
        with torch.no_grad():
            outputs = extract_cnn_feature(model, imgs, norm=False)

            fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
            norm_outputs = outputs.div(fnorm.expand_as(outputs))
        if norm:
            for fname, output, cam in zip(fnames, norm_outputs, camid):
                cam = cam.item()
                per_cam_features_norm[cam].append(output)
                per_cam_fname[cam].append(fname)
        else:
            for fname, output, output_without, cam in zip(
                    fnames, norm_outputs, outputs, camid):
                cam = cam.item()
                per_cam_features_norm[cam].append(output)
                per_cam_fname[cam].append(fname)
                per_cam_features_without[cam].append(output_without)
    if norm:
        return per_cam_features_norm, per_cam_fname
    else:
        return per_cam_features_norm, per_cam_features_without, per_cam_fname


def extract_features_cross_cam(model, data_loader, norm=True, bn_neck=False):
    model.eval()
    cross_cam_features = []
    cross_cam_features_without = []
    cross_cam_fnames = []
    cross_cam_distribute = []
    cams = []
    cam_number = len(model.classifier)
    print("Start extract features cross camera")
    for imgs, fnames, _, camid in tqdm(data_loader):
        with torch.no_grad():
            outputs = extract_cnn_feature(model, imgs, norm=False)
            fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
            norm_outputs = outputs.div(fnorm.expand_as(outputs))
            if bn_neck:
                outputs = model.bn_neck(outputs)
            for i in range(cam_number):
                x = model.classifier[i](outputs)
                if i == 0:
                    distribute = F.softmax(x.data, dim=1)
                else:
                    distribute_tmp = F.softmax(x.data, dim=1)
                    distribute = torch.cat((distribute, distribute_tmp), dim=1)
        if norm:
            for fname, output, cam, dis in zip(fnames, norm_outputs, camid,
                                               distribute):
                cam = cam.item()
                cross_cam_fnames.append(fname)
                cross_cam_features.append(output)
                cams.append(cam)
                cross_cam_distribute.append(dis.cpu().numpy())
        else:
            for fname, output, output_without, cam, dis in zip(
                    fnames, norm_outputs, outputs, camid, distribute):
                cam = cam.item()
                cross_cam_fnames.append(fname)
                cross_cam_features.append(output)
                cross_cam_features_without.append(output_without)
                cams.append(cam)
                cross_cam_distribute.append(dis.cpu().numpy())
    if norm:
        return cross_cam_features, cross_cam_fnames, cross_cam_distribute, cams
    else:
        return cross_cam_features, cross_cam_features_without, cross_cam_fnames, cross_cam_distribute, cams


def extract_features_cross_cam_with_tnorm(model, data_loader):
    model.eval()
    cross_cam_features = []
    cross_cam_fnames = []
    cross_cam_distribute = []
    cams = []
    cam_number = len(model.classifier)
    print("Start extract features cross camera")
    for imgs, fnames, _, camid in tqdm(data_loader):

        with torch.no_grad():
            for i in range(cam_number):
                t = extract_cnn_feature_with_tnorm(model,
                                                   imgs,
                                                   camid,
                                                   i,
                                                   norm=False)
                if i == 0:
                    tmp = t
                else:
                    tmp = tmp + t
                x = model.classifier[i](t)
                if i == 0:
                    distribute = F.softmax(x.data, dim=1)
                else:
                    distribute_tmp = F.softmax(x.data, dim=1)
                    distribute = torch.cat((distribute, distribute_tmp), dim=1)
            norm_outputs = F.normalize(tmp, p=2, dim=1)

        for fname, output, cam, dis in zip(fnames, norm_outputs, camid,
                                           distribute):
            cam = cam.item()
            cross_cam_fnames.append(fname)
            cross_cam_features.append(output)
            cams.append(cam)
            cross_cam_distribute.append(dis.cpu().numpy())
    return cross_cam_features, cross_cam_fnames, cross_cam_distribute, cams


def jaccard_sim_cross_cam(cross_cam_distribute):
    print(
        "Start calculate jaccard similarity cross camera, this step may cost a lot of time"
    )
    n = cross_cam_distribute.size(0)
    jaccard_sim = torch.zeros((n, n))
    for i in range(n):
        distribute = cross_cam_distribute[i]
        abs_sub = torch.abs(distribute - cross_cam_distribute)
        sum_distribute = distribute + cross_cam_distribute
        intersection = (sum_distribute - abs_sub).sum(dim=1) / 2
        union = (sum_distribute + abs_sub).sum(dim=1) / 2
        jaccard_sim[i, :] = intersection / union
    return to_numpy(jaccard_sim)


def cluster_cross_cam(cross_cam_dist,
                      cross_cam_fname,
                      eph,
                      linkage="average",
                      cams=None,
                      mix_rate=0.,
                      jaccard_sim=None,
                      n_clusters=None):
    cluster_results = OrderedDict()
    print("Start cluster cross camera according to distance")
    if mix_rate > 0:
        assert jaccard_sim is not None, "if mix_rate > 0, the jaccard sim is needed"
        assert cams is not None, "if mix_rate > 0, the cam is needed"
        n = len(cross_cam_fname)
        cams = np.array(cams).reshape((n, 1))
        expand_cams = np.tile(cams, n)
        mask = np.array(expand_cams != expand_cams.T, dtype=np.float32)
        cross_cam_dist -= mask * jaccard_sim * mix_rate
    cross_cam_dist = re_ranking(cross_cam_dist)
    if n_clusters is None:
        tri_mat = np.triu(cross_cam_dist, 1)
        tri_mat = tri_mat[np.nonzero(tri_mat)]
        tri_mat = np.sort(tri_mat, axis=None)
        top_num = np.round(eph * tri_mat.size).astype(int)
        eps = tri_mat[top_num]
        print(eps)
    else:
        eps = None

    Ag = AgglomerativeClustering(n_clusters=n_clusters,
                                 affinity="precomputed",
                                 linkage=linkage,
                                 distance_threshold=eps)
    labels = Ag.fit_predict(cross_cam_dist)
    print(len(set(labels)))
    tem = {}
    relabel = 0
    for fname, label in zip(cross_cam_fname, labels):
        if label not in tem.keys():
            tem[label] = []
        tem[label].append(fname)
    for label, names in tem.items():
        if len(names) > 1:
            for name in names:
                cluster_results[name] = relabel
            relabel += 1
    return cluster_results


def distance_cross_cam(features, use_cpu=False):
    print("Start calculate pairwise distance cross camera")
    n = len(features)
    x = torch.cat(features)
    x = x.view(n, -1)
    if use_cpu:
        dist = 1 - np.matmul(x.cpu().numpy(), x.cpu().numpy().T)
    else:
        dist = 1 - torch.mm(x, x.t())

    return to_numpy(dist)


def distane_per_cam(per_cam_features):
    per_cam_dist = {}
    print("Start calculate pairwise distance per camera")
    for k, features in per_cam_features.items():

        n = len(features)
        x = torch.cat(features)
        x = x.view(n, -1)

        per_cam_dist[k] = 1 - torch.mm(x, x.t())
    return per_cam_dist


def cluster_per_cam(per_cam_dist, per_cam_fname, eph, linkage="average", n_clusters=None):
    cluster_results = {}
    print("Start cluster per camera according to distance")
    for k, dist in per_cam_dist.items():
        cluster_results[k] = OrderedDict()

        # handle the number of samples is small
        dist = dist.cpu().numpy()
        n = dist.shape[0]
        if n < eph:
            eph = n // 2
            # double the number of samples
            dist = np.tile(dist, (2, 2))
            per_cam_fname[k] = per_cam_fname[k] + per_cam_fname[k]

        dist = re_ranking(dist)
        if n_clusters is None:
            tri_mat = np.triu(dist, 1)
            tri_mat = tri_mat[np.nonzero(tri_mat)]
            tri_mat = np.sort(tri_mat, axis=None)
            top_num = np.round(eph * tri_mat.size).astype(int)
            eps = tri_mat[top_num]
            # eps = tri_mat[:top_num].mean()
            print(eps)
        else:
            eps = None

        Ag = AgglomerativeClustering(n_clusters=n_clusters,
                                     affinity="precomputed",
                                     linkage=linkage,
                                     distance_threshold=eps)
        # Ag = DBSCAN(eps=eps, min_samples=3, metric='precomputed')

        labels = Ag.fit_predict(dist)
        print(len(set(labels)))
        tem = {}
        relabel = 0
        for fname, label in zip(per_cam_fname[k], labels):
            if label != -1:
                if label not in tem.keys():
                    tem[label] = []
                tem[label].append(fname)
        for label, names in tem.items():
            if len(names) > 1:
                for name in names:
                    cluster_results[k][name] = relabel
                relabel += 1
        # for fname, label in zip(per_cam_fname[k], labels):
        #     if label != -1:
        #         cluster_results[k][fname] = label
    return cluster_results


def get_intra_cam_cluster_result(model, data_loader, eph, linkage, n_clusters=None):
    per_cam_features, per_cam_fname = extract_features_per_cam(
        model, data_loader)
    per_cam_dist = distane_per_cam(per_cam_features)
    cluster_results = cluster_per_cam(per_cam_dist, per_cam_fname, eph,
                                      linkage, n_clusters=n_clusters)
    return cluster_results


def get_inter_cam_cluster_result(model,
                                 data_loader,
                                 eph,
                                 linkage,
                                 mix_rate=0.,
                                 use_cpu=False,
                                 n_clusters=None):
    features, fnames, cross_cam_distribute, cams = extract_features_cross_cam(
        model, data_loader)

    cross_cam_distribute = torch.Tensor(np.array(cross_cam_distribute)).cuda()

    cross_cam_dist = distance_cross_cam(features, use_cpu=use_cpu)
    if mix_rate > 0:
        jaccard_sim = jaccard_sim_cross_cam(cross_cam_distribute)
    else:
        jaccard_sim = None

    cluster_results = cluster_cross_cam(
        cross_cam_dist,
        fnames,
        eph,
        linkage=linkage,
        cams=cams,
        mix_rate=mix_rate,
        jaccard_sim=jaccard_sim,
        n_clusters=n_clusters
    )
    return cluster_results


def get_inter_cam_cluster_result_tnorm(model,
                                       data_loader,
                                       eph,
                                       linkage,
                                       mix_rate=0.,
                                       use_cpu=False):
    features, fnames, cross_cam_distribute, cams = extract_features_cross_cam_with_tnorm(
        model, data_loader)

    cross_cam_distribute = torch.Tensor(np.array(cross_cam_distribute)).cuda()
    cross_cam_dist = distance_cross_cam(features, use_cpu=use_cpu)

    if mix_rate > 0:
        jaccard_sim = jaccard_sim_cross_cam(cross_cam_distribute)
    else:
        jaccard_sim = None

    cluster_results = cluster_cross_cam(
        cross_cam_dist,
        fnames,
        eph,
        linkage=linkage,
        cams=cams,
        mix_rate=mix_rate,
        jaccard_sim=jaccard_sim,
    )
    return cluster_results
