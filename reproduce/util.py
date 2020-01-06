import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import time
import os

import pickle
import pandas as pd
# import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.ticker import FormatStrFormatter
# sns.set_style("white")


def get_bias(d1, d2, d3=None):
    z = d1 + d2
    bias = 0
    try:
        bias = abs(d1/z - 1/2)
    except FloatingPointError:
        pass
    return bias

def get_bias2(d1, d2, d3=None):
    m = abs(d1 - d2)
    z = d1 + d2
    bias = 0
    try:
        bias = m / z**2
    except FloatingPointError:
        pass
    return bias


def get_bias3(d1, d2, d3):
    m = d1 - d2
    z = (d1 + d2)
    bias = 0
    try:
        bias = m / z
    except FloatingPointError:
        pass
    return bias




class Test:
    def __init__(self, paths):
        self.count = 0
        self.map_obj_to_idx = dict()
        self.map_idx_to_obj = dict()
        self.count_bias = None
        self.bias = None
        self.conf = None
        self.low_conf_triplet_tensor = None
        self.conf_mat = None
        self.dist_mat = None

        [self.dist_path, self.conf_path, self.cooccur_path, self.label_path, self.pred_label_path] = paths

        self.build_dict(self.dist_path)

        self.bias_avg = None

        self.conf_tensor = None



    def reset_path(self, paths):
        [self.dist_path, self.conf_path, self.cooccur_path, self.label_path, self.pred_label_path] = paths


    def build_dict(self, dist_path):
        with open(dist_path, 'r') as f:
            for line in f:
                tokens = line.split(',')
                obj1 = tokens[0]
                obj2 = tokens[1]
                if obj1 != 'object1' and obj1 != 'object' and obj1 not in self.map_obj_to_idx:
                    self.map_obj_to_idx[obj1] = self.count
                    self.count += 1
                if obj2 != 'object2' and obj2 != 'object' and obj2 not in self.map_obj_to_idx:
                    self.map_obj_to_idx[obj2] = self.count
                    self.count += 1
            self.map_idx_to_obj = {v:k for k,v in self.map_obj_to_idx.items()}


    # Build 2d distance matrix
    def build_dist_mat(self):
        dist_mat = np.zeros([self.count, self.count])
        with open(self.dist_path, 'r') as f:
            for line in f:
                tokens = line.rstrip('\n').split(',')
                if tokens[0] != 'object1' and tokens[0] != 'object' and tokens[0] != 'image_id':
                    idx1 = self.map_obj_to_idx[tokens[0]]
                    idx2 = self.map_obj_to_idx[tokens[1]]
                    dist = float(tokens[2])
                    dist_mat[idx1, idx2] = dist
                    dist_mat[idx2, idx1] = dist
        self.dist_mat = dist_mat

        return dist_mat

    def draw_hist(self):
        assert self.dist_mat is not None
        dist = self.dist_mat.ravel()

        N = dist.shape[0]
        cutoff = int(np.ceil(N * 0.1))
        cutoff_dist = np.sort(dist)[cutoff]
        print('mean:', np.mean(dist), 'standard deviation:', np.std(dist), 'top 10% cutoff:', f'{cutoff_dist:.4f}')
        print('distance between mean and 20% cutoff in terms of std:', f'{(np.mean(dist)-cutoff_dist)/np.std(dist):.4f}')
        # the histogram of the data
        plt.hist(dist, 50, density=True, facecolor='g', alpha=0.75)
        plt.plot([cutoff_dist for _ in range(15)], [i for i in range(15)], 'r--')
        plt.xlabel('distance')
        plt.ylabel('frequency')
        plt.show()

    # Build 2d directional confusion matrix
    def build_conf_mat(self):
        conf_mat = np.zeros([self.count, self.count])
        with open(self.conf_path, 'r') as f:
            for line in f:
                tokens = line.rstrip('\n').split(',')
                if tokens[0] != 'object1' and tokens[0] != 'object':
                    idx1 = self.map_obj_to_idx[tokens[0]]
                    idx2 = self.map_obj_to_idx[tokens[1]]
                    conf = float(tokens[2])
                    conf_mat[idx1, idx2] = conf
        self.conf_mat = conf_mat
        return conf_mat

    def build_conf_tensor(self):
        conf_count_tensor = np.zeros([self.count, self.count, self.count])
        with open(self.label_path, 'r') as f_label, open(self.pred_label_path, 'r') as f_pred:
            for i, (label_line, pred_line) in enumerate(zip(f_label, f_pred)):
                if i > 0:
                    true_tokens = label_line.strip().split(',')[1].split(';')
                    pred_tokens = pred_line.strip().split(',')[1].split(';')

                    if (len(true_tokens) == 1 and true_tokens[0] == '') or (len(pred_tokens) == 1 and pred_tokens[0] == ''):
                        continue
                    # print(label_line.strip().split(',')[0], 'true', true_tokens, 'pred', pred_tokens)
                    common_tokens = list(set(true_tokens) and set(pred_tokens))
                    for true_token in true_tokens:
                        if true_token not in pred_tokens:
                            for common_token in common_tokens:
                                if common_token != true_token:
                                    for pred_token in pred_tokens:
                                        x = self.map_obj_to_idx[true_token]
                                        y = self.map_obj_to_idx[pred_token]
                                        z = self.map_obj_to_idx[common_token]
                                        conf_count_tensor[x, y, z] += 1


        base = conf_count_tensor.sum(axis=0)[np.newaxis, :, :]
        base = base + 1e-10

        conf_tensor = conf_count_tensor / base

        return conf_tensor





    # Build 2d coocurrence matrix
    def build_cooccur_mat(self):
        cooccur_mat = None
        if self.cooccur_path:
            cooccur_mat = np.zeros([self.count, self.count])
            with open(self.cooccur_path, 'r') as f:
                for line in f:
                    tokens = line.rstrip('\n').split(',')
                    if tokens[0] != 'object1' and tokens[0] != 'object':
                        idx1 = self.map_obj_to_idx[tokens[0]]
                        idx2 = self.map_obj_to_idx[tokens[1]]
                        occur_times = int(tokens[2])
                        cooccur_mat[idx1, idx2] = occur_times
                        cooccur_mat[idx2, idx1] = occur_times
        return cooccur_mat


    def set_low_conf_triplet_tensor(self, dist_mat, conf_mat, t1):
        assert self.dist_mat is not None
        assert self.conf_mat is not None

        self.low_conf_triplet_tensor = np.zeros([self.count, self.count, self.count])
        # Set some threshold to eliminate cases when no confusions happen or base becomes 0
        t2 = 0.00001
        for i in range(self.count):
            for j in range(i+1, self.count):
                for k in range(self.count):
                    if (conf_mat[i, k] > t1 or conf_mat[j, k] > t1 or conf_mat[k, i] > t1 or conf_mat[k, j] > t1) and dist_mat[i, k] > t2 and dist_mat[j, k] > t2:
                        self.low_conf_triplet_tensor[i, j, k] = 1

    '''
    Get 3D matrix for bias, training bias, and confusion
    '''
    def get_bias_and_conf(self, dist_mat, conf_mat, cooccur_mat, cal_count_bias, mode):
        assert self.low_conf_triplet_tensor is not None
        count_bias = np.zeros([self.count, self.count, self.count])
        bias = np.zeros([self.count, self.count, self.count])
        conf = np.zeros([self.count, self.count, self.count])

        d_1d = dist_mat.ravel()
        # cutoff = int(d_1d.shape[0]*0.1)
        # cutoff_val = np.sort(d_1d)[cutoff]

        cutoff_val = np.mean(d_1d)-np.std(d_1d)

        # Set some threshold to eliminate cases when no confusions happen or base becomes 0
        start = time.time()
        for i in range(self.count):
            if i % 50 == 0:
                print(i,'/',self.count, time.time()-start)
            for j in range(i+1, self.count):
                for k in range(self.count):
                    if mode == 'normal':
                        if self.low_conf_triplet_tensor[i, j, k] > 0:
                            if cal_count_bias:
                                count_bias[i, j, k] = get_bias(cooccur_mat[i, k], cooccur_mat[j, k])
                            if dist_mat[i, k] < cutoff_val or dist_mat[j, k] < cutoff_val:
                                bias[i, j, k] = get_bias(dist_mat[i, k], dist_mat[j, k])
                            else:
                                bias[i, j, k] = 0
                            conf[i, j, k] = abs(conf_mat[i, k] + conf_mat[k, i] - conf_mat[j, k] - conf_mat[k, j])

                            # conf[i, j, k] = abs(conf_mat[i, k] - conf_mat[j, k])
                            # conf[i, j, k] = abs(conf_mat[k, i] - conf_mat[k, j])

                            # conf[i, j, k] = (abs(conf_mat[i, k] - conf_mat[j, k]) + abs(conf_mat[k, i] - conf_mat[k, j]))/2
                            # conf[i, j, k] = np.abs(np.min([conf_mat[i, k], conf_mat[k, i]]) - np.min([conf_mat[j, k], conf_mat[k, j]]))
                    elif mode == 'conditional':
                        bias[i, j, k] = get_bias(dist_mat[i, k], dist_mat[j, k], dist_mat[i, j])
                        conf[i, j, k] = self.conf_tensor[i, j, k]

        return count_bias, bias, conf

    def get_pair_list(self, t1=0.0, cal_count_bias=False, filename=None, mode='normal', set_new_low_conf_triplet_tensor=True):

        dist_mat = self.build_dist_mat()
        conf_mat = self.build_conf_mat()
        cooccur_mat = None

        if mode == 'conditional':
            self.conf_tensor = self.build_conf_tensor()

        if cal_count_bias:
            cooccur_mat = self.build_cooccur_mat()

        if set_new_low_conf_triplet_tensor:
            self.set_low_conf_triplet_tensor(dist_mat, conf_mat, t1)

        count_bias, bias, conf = self.get_bias_and_conf(dist_mat, conf_mat, cooccur_mat, cal_count_bias, mode)

        self.count_bias = count_bias
        self.bias = bias
        self.conf = conf

        # Calculate the average of each pair
        count_bias_avg = count_bias.sum(axis=2)/(self.count-2)
        bias_avg = bias.sum(axis=2)/(self.count-2)
        conf_avg = conf.sum(axis=2)/(self.count-2)

        self.bias_avg = bias_avg
        self.conf_avg = conf_avg

        # Build two lists of pairs for plot
        bias_conf_list = []
        count_bias_conf_list = []
        trip_conf_list = []

        # outliers = []

        for i in range(self.count):
            for j in range(i+1, self.count):
                # if bias_avg[i, j] > 0.072 and conf_avg[i, j] < 0.048:
                #     outliers.append((conf_avg[i, j]-bias_avg[i, j], conf_avg[i, j], bias_avg[i, j], self.map_idx_to_obj[i], self.map_idx_to_obj[j]))

                bias_conf_list.append((conf_avg[i, j], bias_avg[i, j]))
                count_bias_conf_list.append((conf_avg[i, j], count_bias_avg[i, j]))
        # print('max:', np.max(self.dist_mat))
        # for i in range(self.count):
        #     for j in range(self.count):
        #         for k in range(self.count):
        #             trip_conf_list.append((self.conf[i, j, k], self.bias[i, j, k]))

        #print(sorted(outliers))

        if filename:
            print('start building conf_list')
            start = time.time()
            with open(filename, 'w') as f:
                for i in range(self.count):
                    if i % int(self.count/20) == 0:
                        print(i, '/', self.count, time.time()-start)
                    for j in range(i+1, self.count):
                        for k in range(self.count):
                            f.write(','.join([str(conf[i, j, k]), str(bias[i, j, k]), self.map_idx_to_obj[i], self.map_idx_to_obj[j], self.map_idx_to_obj[k]])+'\n')

        return np.array(bias_conf_list), count_bias_conf_list, trip_conf_list

    def analyze(self, obj1, obj2, filename, dic=None):
        # print(self.map_obj_to_idx.keys())
        i = self.map_obj_to_idx[obj1]
        j = self.map_obj_to_idx[obj2]
        if i > j:
            i, j = j, i
        if dic:
            obj1 = dic[obj1]
            obj2 = dic[obj2]

        filename += ','+obj1+','+obj2

        l = []
        with open(filename, 'w') as f_w:
            f_w.write(','.join(['obj3, conf_score, bias_score, dist1, dist2, conf13, conf23, conf31, conf32'])+'\n')
            for k in range(self.count):
                obj3 = self.map_idx_to_obj[k]
                if dic:
                    obj3 = dic[obj3]
                c = self.conf[i,j,k]
                b = self.bias[i,j,k]
                dik = self.dist_mat[i,k]
                djk = self.dist_mat[j,k]
                cik = self.conf_mat[i,k]
                cjk = self.conf_mat[j,k]
                cki = self.conf_mat[k,i]
                ckj = self.conf_mat[k,j]

                l.append((obj3, c, b, dik, djk, cik, cjk, cki, ckj))
            l = sorted(l, key=lambda t:t[2])
            #l = sorted(l, key=lambda t:t[3]-t[4])
            for t in l:
                s = f'{t[0]}, {t[1]:.4f}, {t[2]:.4f}, {t[3]:.4f}, {t[4]:.4f}, {t[5]:.4f}, {t[6]:.4f}, {t[7]:.4f}, {t[8]:.4f}\n'
                f_w.write(s)

    def find_most_biased_pairs(self, k, k2, filename):
        top_biased = np.argsort(self.bias_avg, axis=None)
        inds_list = []
        top_list = []
        for tr in reversed(top_biased[-k:]):
            inds = np.unravel_index(tr, self.bias_avg.shape)
            inds_list.append(inds)

            bias = self.bias_avg[inds[0], inds[1]]
            conf = self.conf_avg[inds[0], inds[1]]

            inds = [self.map_idx_to_obj[p] for p in inds]
            inds = inds + [bias, conf]
            # print(inds)
            top_list.append(inds)

        with open(filename, 'w') as f_out:
            for inds in inds_list:
                third_inds_list = reversed(np.argsort(self.bias[inds[0], inds[1]])[-k2:])
                x = inds[0]
                y = inds[1]
                for third_ind in third_inds_list:
                    z = third_ind
                    inds = [x, y, z]
                    inds = [self.map_idx_to_obj[p] for p in inds]

                    info = [self.bias[x, y, z], self.conf_mat[x, z], self.conf_mat[y, z], self.conf_mat[z, x], self.conf_mat[z, y]]
                    info_str = [f'{d:.4f}' for d in info]
                    print(inds + info_str)
                    f_out.write(','.join(inds)+'\n')



    def find_most_biased_triplets(self, k):
        top_biased = np.argsort(self.bias, axis=None)
        top_list = []
        for tr in reversed(top_biased[-k:]):
            inds = np.unravel_index(tr, self.conf.shape)
            inds = [self.map_idx_to_obj[p] for p in inds]
            print(inds)
            top_list.append(inds)











def plot(pair_list, title, xlabel, ylabel, save_name=None, xlim=None, ylim=None, dot=None, fitting=True):
    x_1 = [p[0] for p in pair_list]
    x_2 = [p[1] for p in pair_list]


    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()


    print(scipy.stats.spearmanr(x_1, x_2))
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


    ax.scatter(x_1, x_2, s=2)
    if dot:
        ax.plot(dot[0], dot[1], marker='o', markersize=5, color="red")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)



    if fitting:
        X = np.array(x_1)
        Y = np.array(x_2)
        Z = np.polyfit(X, Y, 1)
        p = np.poly1d(Z)
        ax.plot(X,p(X),"r--")
    plt.show()


    save_dir = 'imgs/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if save_name:
         fig.savefig(save_dir+save_name+'.pdf', bbox_inches='tight')


#
# def save(filename, conf_list):
#     start = time.time()
#     count = len(conf_list)
#     with open(filename, 'w') as f:
#         for i, q in enumerate(sorted(conf_list)):
#             if i % int(count/20) == 0:
#                 print(i, '/', count, time.time()-start)
#             assert len(q) == 5
#             f.write(','.join([str(q[0]), str(q[1]), q[2], q[3], q[4]])+'\n')
#
#
# def load_nus_mapping(filepath):
#     dic = dict()
#     with open(filepath, 'r') as f_r:
#         for line in f_r:
#             k, v = line.split(',')
#             dic[k] = v.rstrip('\n')
#     return dic
#
# def visualize(filepath, title=None, edge_len=60, save=True):
#
#     df = pd.read_csv(filepath)
#     labels, X = df.values[:, 0], df.values[:, 1:]
#     X_embedded = TSNE(n_components=2).fit_transform(X)
#
#     X, Y = X_embedded[:,0], X_embedded[:,1]
#     title = 't-SNE embedding of ImageNet'
#
#     fig, ax = plt.subplots(figsize=(edge_len, edge_len))
#     ax.scatter(X, Y)
#     ax.set_title(title)
#
#     for i, txt in enumerate(labels):
#         ax.annotate(txt, (X[i], Y[i]))
#
#     plt.show()
#
#     if save:
#         fig.savefig('saved_data/'+title+'.pdf', bbox_inches='tight')
#
#
# def compare_cluster(pca_path, conf_path, labels, kmeanlabels, labels_to_idx, target):
#
#     df_conf = pd.read_csv(conf_path)
#
#     df_pca = pd.read_csv(pca_path)
#
#     objects, embeddings = df_pca.values[:, 0], df_pca.values[:, 1:]
#
#     obj_embed = dict()
#     for i in range(objects.shape[0]):
#         obj_embed[objects[i]] = embeddings[i]
#
#     group1 = []
#     group2 = []
#
#     for i, lab in enumerate(kmeanlabels):
#         if lab == target:
#             group1.append(i)
#         else:
#             group2.append(i)
#
#     n = len(labels)
#     conf_mat = np.zeros([n, n])
#     dist_mat = np.zeros([n, n])
#
#     for data in df_conf.values:
#         obj1, obj2, conf = data
#         idx1, idx2 = labels_to_idx[obj1], labels_to_idx[obj2]
#
#         conf_mat[idx1, idx2] = conf
#         dist_mat[idx1, idx2] = np.linalg.norm(obj_embed[obj1] - obj_embed[obj2])
#
#     intra_conf_list = []
#     inter_conf_list = []
#
#     intra_dist_list = []
#     inter_dist_list = []
#
#     for a in group1:
#         for b in group1:
#             intra_conf_list.append(conf_mat[a, b]+conf_mat[b, a])
#             intra_dist_list.append(dist_mat[a, b]+dist_mat[b, a])
#         for c in group2:
#             inter_conf_list.append(conf_mat[a, c]+conf_mat[c, a])
#             inter_dist_list.append(dist_mat[a, c]+dist_mat[c, a])
#
#     intra_conf = np.mean(intra_conf_list)
#     inter_conf = np.mean(inter_conf_list)
#     intra_dist = np.mean(intra_dist_list)
#     inter_dist = np.mean(inter_dist_list)
#
#     return intra_conf, inter_conf, intra_dist, inter_dist
#
#
# def get_scores(m, x_th, y_th):
#     m1 = m[m[:, 0]>x_th]
#     m0 = m[m[:, 0]<=x_th]
#
#     m11 = m1[m1[:, 1]>y_th].shape[0]
#     m10 = m1[m1[:, 1]<=y_th].shape[0]
#     m01 = m0[m0[:, 1]>y_th].shape[0]
#     # m00 = m0[m0[:, 1]<=y_th].shape[0]
#
#     precision = 0
#     recall = 0
#     if m01 + m11 > 0:
#         precision = m11 / (m01 + m11)
#     if m10 + m11 > 0:
#         recall = m11 / (m10 + m11)
#     F1 = 2*precision*recall/(precision+recall)
#
#     return m11, m10, precision, recall, F1
#
# def get_precision_recall(bias_conf_list, x_th, y_th_list):
#
#     precision_list = []
#     recall_list = []
#     TP_list = []
#     FP_list = []
#     F1_list = []
#
#     m = np.array(bias_conf_list)
#     m1 = m[m[:, 0]>x_th]
#     m0 = m[m[:, 0]<=x_th]
#     for y_th in y_th_list:
#         # Fix confusion
#
#
#         m11 = m1[m1[:, 1]>y_th].shape[0]
#         m10 = m1[m1[:, 1]<=y_th].shape[0]
#         m01 = m0[m0[:, 1]>y_th].shape[0]
#         # m00 = m0[m0[:, 1]<=y_th].shape[0]
#         # print(m.shape[0], m1.shape[0], m0.shape[0], m11, m01, x_th, y_th)
#         precision = 0
#         recall = 0
#         if m01 + m11 > 0:
#             precision = m11 / (m01 + m11)
#         if m10 + m11 > 0:
#             recall = m11 / (m10 + m11)
#         if precision == 0 and recall == 0:
#             F1 = 0
#         else:
#             F1 = 2*precision*recall/(precision+recall)
#
#         # print(y_th, precision, recall, m11)
#
#         precision_list.append(f'{precision:.4f}')
#         recall_list.append(f'{recall:.4f}')
#         F1_list.append(F1)
#         TP_list.append(str(m11))
#         FP_list.append(str(m01))
#
#     y_th_str_list = [str(y_th) for y_th in y_th_list]
#
#     print('y TP FP precision recall F1 ')
#     i = np.argmax(F1_list)
#     print(f'{float(y_th_str_list[i]):.4f}', '&', TP_list[i], '&', FP_list[i], '&', precision_list[i], '&', recall_list[i], '&', f'{F1_list[i]:.4f}')
#     # for i in range(len(y_th_list)):
#     #     print(y_th_str_list[i], precision_list[i], recall_list[i], F1_list[i], TP_list[i], FP_list[i])
#
#     # print()
#     # print(' & '+'threshold'+' & '+' & '.join(y_th_str_list)+' \\\\')
#     # print(' & '+'precision'+' & '+' & '.join(precision_list)+ ' \\\\')
#     # print(' & '+'recall'+' & '+' & '.join(recall_list)+ ' \\\\')
#     # print(' & '+'\\#TP' +' & '+' & '.join(TP_list)+ ' \\\\')
#
#     return y_th_str_list[i], TP_list[i], FP_list[i], precision_list[i], recall_list[i], F1_list[i]
#
# def get_random_precision_recall(conf_list, x_th, num):
#     np.random.seed(0)
#     conf_list = np.random.permutation(conf_list)
#     pred_pos = conf_list[:num]
#     pred_neg = conf_list[num:]
#
#     m11 = pred_pos[pred_pos > x_th].shape[0]
#     m10 = pred_neg[pred_neg > x_th].shape[0]
#     m01 = pred_pos[pred_pos <= x_th].shape[0]
#
#     precision = 0
#     recall = 0
#     F1 = 0
#     if m01 + m11 > 0:
#         precision = m11 / (m01 + m11)
#     if m10 + m11 > 0:
#         recall = m11 / (m10 + m11)
#     if  precision+recall > 0:
#         F1 = 2*precision*recall/(precision+recall)
#
#     # print(y_th, precision, recall, m11)
#     print('TP FP precision recall F1 ')
#     print('-', '&', int(m11), '&', int(m10), '&', f'{precision:.4f}', '&', f'{recall:.4f}', '&', f'{F1:.4f}')
#
#
#
# '''
# Strong Baseline using confusion disparity in the validation set to predict that in the test set.
# '''
# def get_val_conf_precision_recall(test_conf_avg, val_conf_avg, test_conf_top_num, val_conf_top_num, val_conf_top_percentage):
#
#     test_conf_avg_1d = test_conf_avg.ravel()
#     val_conf_avg_1d = val_conf_avg.ravel()
#
#     test_conf_avg_1d = np.argsort(test_conf_avg_1d)
#     val_conf_avg_1d = np.argsort(val_conf_avg_1d)
#
#     top_test_conf_inds = test_conf_avg_1d[-test_conf_top_num:]
#     bottom_test_conf_inds = test_conf_avg_1d[:-test_conf_top_num]
#     top_val_conf_inds = val_conf_avg_1d[-val_conf_top_num:]
#     bottom_val_conf_inds = val_conf_avg_1d[:-val_conf_top_num]
#
#     F1 = get_single_precision_recall(top_test_conf_inds, bottom_test_conf_inds, top_val_conf_inds, bottom_val_conf_inds, val_conf_top_percentage)
#
#     return F1






def get_single_precision_recall_wrapper(predicted_label_conf_bias_list, conf_top_percentage, top_percentage, verbose=False):
    '''
    predicted_label_conf_bias_list:  A ndarray with size 2 * number consisting of the confusion and bias value for every pair.
    conf_top_percentage: the ground-truth confusion percentage threshold.
    top_percentages: x-axis percentage currently used.
    '''
    N = len(predicted_label_conf_bias_list)
    bias_list = predicted_label_conf_bias_list[:, 1]
    conf_list = predicted_label_conf_bias_list[:, 0]
    sorted_bias_list = np.argsort(bias_list)
    sorted_conf_list = np.argsort(conf_list)

    conf_top_num = int(np.ceil(N*conf_top_percentage))
    top_conf_inds = sorted_conf_list[-conf_top_num:]
    bottom_conf_inds = sorted_conf_list[:-conf_top_num]

    top_num = int(np.ceil(N*top_percentage))
    # print(conf_top_percentage, top_percentage)
    recall_ab = 0
    precision_ab = 0
    if top_num > 0:
        top_bias_inds = sorted_bias_list[-top_num:]
        bottom_bias_inds = sorted_bias_list[:-top_num]

        recall_ab, precision_ab = get_single_precision_recall(top_conf_inds, bottom_conf_inds, top_bias_inds, bottom_bias_inds, top_percentage, verbose)
        if verbose:
            print('random baseline')
            N = len(predicted_label_conf_bias_list)
            m11 = int(top_percentage * conf_top_percentage * N)
            m01 = int(top_percentage * (1-conf_top_percentage) * N)
            precision = conf_top_percentage
            recall = top_percentage
            F1 = 2*precision*recall/(precision+recall)
            print(','.join([f'{top_percentage:.2f}', str(m11), str(m01), f'{precision:.3f}', f'{recall:.3f}', f'{F1:.3f}']))

    return recall_ab, precision_ab




def get_single_precision_recall(top_conf_inds, bottom_conf_inds, top_bias_inds, bottom_bias_inds, top_percentage, verbose=False):
    '''
    top_conf_inds: inds that are true.
    bottom_conf_inds: inds that are false.
    top_bias_inds: inds that are positive.
    bottom_bias_inds: inds that are negative.
    top_percentage: the current percentage of positive rate.
    '''

    top_conf_inds = set(top_conf_inds)
    bottom_conf_inds = set(bottom_conf_inds)
    top_bias_inds = set(top_bias_inds)
    bottom_bias_inds = set(bottom_bias_inds)

    m11 = len(top_conf_inds & top_bias_inds)
    m10 = len(top_conf_inds & bottom_bias_inds)
    m01 = len(bottom_conf_inds & top_bias_inds)

    precision = 0
    recall = 0
    F1 = 0
    if m01 + m11 > 0:
        precision = m11 / (m01 + m11)
    if m10 + m11 > 0:
        recall = m11 / (m10 + m11)
    if  precision+recall > 0:
        F1 = 2*precision*recall/(precision+recall)


    if verbose:
        print(','.join([f'{top_percentage:.2f}', str(m11), str(m01), f'{precision:.3f}', f'{recall:.3f}', f'{F1:.3f}']))




    return recall, precision


def calculate_AUCEC(x_list, y_list):
    '''
    x_list: a list of x-axis percentage currently used.
    y_list: a list of recall.
    '''
    s = 0
    for i in range(len(y_list)-1):
        interval = (x_list[i+1]-x_list[i])
        s += interval * (y_list[i]+y_list[i+1])/2
    return s

def get_AUCEC_gain(conf_top_percentage, top_percentages, predicted_label_conf_bias_list):
    '''
    conf_top_percentage: the ground-truth confusion percentage threshold.
    top_percentages: a list of x-axis percentage currently used.
    predicted_label_conf_bias_list:  A ndarray with size 2 * number consisting of the confusion and bias value for every pair.
    '''
    recall_precision_ab_list = []
    for top_percentage in top_percentages:
        recall_ab, precision_ab = get_single_precision_recall_wrapper(predicted_label_conf_bias_list, conf_top_percentage, top_percentage, verbose=False)
        recall_precision_ab_list.append((recall_ab, precision_ab))
    recall_precision_ab_list = np.array(recall_precision_ab_list)
    AUCEC_gain = (calculate_AUCEC(top_percentages, recall_precision_ab_list[:, 0]) - 1/2) / (1/2)


    return AUCEC_gain, recall_precision_ab_list

def draw_CE_comparison(x_list, recall_precision_ab_list_per_dataset, MODE_recall_precision_ab_list_per_dataset, top_std_cutoff_per_dataset, AUCEC_gain_per_dataset, datasets, dataset_names, conf_top_percentage):
    '''
    Draw a 2 x 4 cost-effective graphs.
    INPUT:
    x_list: a list of different percentages.
    recall_precision_ab_list_per_dataset: dictionary with key to be dataset folder name and corresponding list of (recall, precision) pairs at different percentage.
    MODE_recall_precision_ab_list_per_dataset: similar to recall_precision_ab_list_per_dataset, but for MODE baseline and does not have imsitu
    top_std_cutoff_per_dataset: dictionary with key to be dataset folder name and the corresponding percentage of top 1std cutoff.

    AUCEC_gain_per_dataset:

    datasets: a list of dataset folder names.
    datasets: a list of dataset names.
    conf_top_percentage: the ground-truth confusion percentage threshold.
    '''
    title = 'bias_top_'+str(conf_top_percentage)
    y0_list = [np.min([x, conf_top_percentage])/conf_top_percentage for x in x_list]




    y1_list = [x for x in x_list]

    fig, axs = plt.subplots(2, 4, figsize=(40, 20), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0, 'wspace': 0})
    # fig.suptitle(title)
    for i in range(2):
        for j in range(4):
            ax = axs[i][j]
            ind = i*4+j
            if ind < len(datasets):
                dataset = datasets[ind]
                dataset_name = dataset_names[ind]
                y2_list = recall_precision_ab_list_per_dataset[dataset][:, 0]

                y3_list = None
                if dataset != 'imsitu':
                    y3_list = MODE_recall_precision_ab_list_per_dataset[dataset][:, 0]

                print((calculate_AUCEC(x_list, y0_list)-calculate_AUCEC(x_list, y2_list))/calculate_AUCEC(x_list, y2_list))

                ax.set_ylabel('% of errors found', fontsize=60)
                if j == 2:
                    ax.set_xlabel('% of pairs inspected', fontsize=60)
                else:
                    ax.set_xlabel('')


                ax.plot([top_std_cutoff_per_dataset[dataset] for _ in range(10)], np.linspace(0, 1, 10), linestyle='-', linewidth=8, color='red')

                ax.plot(x_list, y0_list, label='optimal', linestyle=':', linewidth=8, color='green')
                ax.plot(x_list, y1_list, label='random', linestyle='--', linewidth=8, color='blue')
                ax.plot(x_list, y2_list, label='DeepInspect', linewidth=8, color='orange')

                if dataset != 'imsitu':
                    ax.plot(x_list, y3_list, label='MODE', linestyle='-.', linewidth=8, color='purple')


                ax.text(0.08, 0, dataset_name, size=42)

                # pc_1percent = precison_recall_at_1percent_per_dataset[dataset]
                # pc_1std = precison_recall_at_1std_per_dataset[dataset]
                AUCEC = AUCEC_gain_per_dataset[dataset]

                # ax.text(0.6, 0.7, '('+f'{pc_1percent[1]:.3f}'+','+f'{pc_1percent[0]:.3f}'+')', size=26)
                # ax.text(0.6, 0.5, '('+f'{pc_1std[1]:.3f}'+','+f'{pc_1std[0]:.3f}'+')', size=26)
                ax.text(0.4, 0.5, 'Gain='+f'{AUCEC*100:.1f}'+'%', size=45)

                ax.tick_params(axis='x', labelsize=35)
                ax.tick_params(axis='y', labelsize=35)

                ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
                ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
                ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
                if ind == 7:
                    ax.legend(prop={'size': 35}, framealpha=0.5)

    for ax in axs.flat:
        ax.label_outer()

    plt.savefig('saved_ce_figures'+'/'+'bias_top_1std'+'.pdf', format='pdf', dpi=1000, bbox_inches = 'tight')
    plt.show()

def get_val_and_test_path(dataset, err_type='type2'):
    '''
    Get the paths of all the necessary files.
    The paths are quite local so they may subject to change.
    '''

    prepend_path = 'Paper_deepInspect/experiment/ase_extra'
    dataset_path = os.path.join(prepend_path, dataset)

    multi_label = False
    if dataset in ['coco', 'coco_gender', 'nus']:
        multi_label = True

    cooccur_path = None

    if not multi_label:
        err_type = 'type1'
        cal_count_bias = False

    dist_path = os.path.join(dataset_path, 'neuron_distance_from_predicted_labels_test_90.csv')

    val_predicted_label_dist_path = os.path.join(dataset_path, 'test_predicted_labels_10.csv')
    test_predicted_label_dist_path = os.path.join(dataset_path, 'test_predicted_labels_90.csv')

    val_conf_type1_path = os.path.join(dataset_path, 'objects_directional_type1_confusion_test_10.csv')
    test_conf_type1_path = os.path.join(dataset_path, 'objects_directional_type1_confusion_test_90.csv')

    if multi_label == True:
        val_conf_type2_path = os.path.join(dataset_path, 'objects_directional_type2_confusion_test_10.csv')
        test_conf_type2_path = os.path.join(dataset_path, 'objects_directional_type2_confusion_test_90.csv')
        cooccur_path = os.path.join(dataset_path, 'concurrence_count_90.csv')

    val_label_path = os.path.join(dataset_path, 'test_labels_10.csv')
    test_label_path = os.path.join(dataset_path, 'test_labels_90.csv')

    val_pred_path = os.path.join(dataset_path, 'test_predicted_labels_10.csv')
    test_pred_path = os.path.join(dataset_path, 'test_predicted_labels_90.csv')

    if err_type == 'type1':
        val_conf_path = val_conf_type1_path
        test_conf_path = test_conf_type1_path
    elif err_type == 'type2':
        val_conf_path = val_conf_type2_path
        test_conf_path = test_conf_type2_path

    paths_val = [dist_path, val_conf_path, cooccur_path, val_label_path, val_pred_path]
    paths_test = [dist_path, test_conf_path, cooccur_path, test_label_path, test_pred_path]

    return paths_val, paths_test
