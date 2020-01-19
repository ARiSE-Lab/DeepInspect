'''
Reproduce results in paper.
Table3 and Figure6
'''
import os
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib.ticker import FormatStrFormatter
from util import Test, plot, draw_CE_comparison, get_val_and_test_path, get_AUCEC_gain, get_single_precision_recall_wrapper

def evaluate_confusion_bugs(datasets):
    
    #datasets = ["cifar100"]
    name = {}
    name["coco"] = "COCO"
    name["coco_gender"] = "COCO gender"
    name["nus"] = "NUS"
    name["cifar100"] = "CIFAR-100"
    name["robust_cifar10_small"] = "Robust CIFAR-10 Small"
    name["robust_cifar10_large"] = "Robust CIFAR-10 Large"
    name["robust_cifar10_resnet"] = "Robust CIFAR-10 ResNet"
    name["tiny_imagenet"] = "Tiny ImageNet"
    name["imsitu"] = "imSitu"
    name["imagenet"] = "ImageNet"
    folder = '../data/'
    folder2 = '../data/'
    #datasets = datasets[1:]
    conf_top_percentage = 0.1
    
    if len(datasets) == 8:
        # initialize graph drawing parameters. only draw graph when run on all datasets
        x = np.linspace(0, 1, 40)
        m = {}
        y = {}
        o = {}
        r = {}
        c = {}
        gain = {}
        gain2 = {}
        for d in datasets:
            y[d] = []
            o[d] = []
            r[d] = []
            m[d] = []

    multi = ["coco", "coco_gender", "imsitu"]
    for d in datasets:
        print("---------------------------")
        print("dataset: " + d)
        if d != "imsitu":
            with open(folder2 + d + '/weight_matrix.pickle', 'rb') as f:
                weight_matrix = pickle.load(f, encoding='latin1')
        d_test_confusion = {}  # directional type1/type2 confusion
        n_test_confusion = {}  # non-directional type1/type2 confusion

        truth = []
        truth_value = []
        distance = {}
        our_positive = []
        our_positive_value = []
        random_positive = []
        if d in multi:
            confusion_file = '/objects_directional_type2_confusion_test_90.csv'
            if d != "imsitu":
                with open(folder2 + d + '/id2object.pickle', 'rb') as f:
                    id2object = pickle.load(f)

                with open(folder2 + d + '/object2id.pickle', 'rb') as f:
                    object2id = pickle.load(f)
        else:
            confusion_file = '/objects_directional_type1_confusion_test_90.csv'
        distance_file = "/neuron_distance_from_predicted_labels_test_90.csv"

        if d == "coco" or d == "cifar100":

            # using neuron coverage threshold 0.5
            distance_file = "/neuron_distance_from_predicted_labels_test_90.csv"
            # neuron coverage threshold used in computing probability matrix can be selected based on file name
            # distance_file =
            # "/neuron_distance_from_predicted_labels_test_90_0.6.csv" # using
            # neuron coverage threshold 0.6

        if d != "imsitu":
            dist = euclidean_distances(weight_matrix)
            MODE_dist = {}
            for i in range(weight_matrix.shape[0]):
                for j in range(weight_matrix.shape[0]):
                    if i == j:
                        continue
                    if d in multi:
                        if (id2object[i], id2object[j]) in MODE_dist or (
                                id2object[j], id2object[i]) in MODE_dist:
                            continue
                        MODE_dist[(id2object[i], id2object[j])] = dist[i][j]
                    else:
                        if (str(i), str(j)) in MODE_dist or (
                                str(j), str(i)) in MODE_dist:
                            continue
                        MODE_dist[(str(i), str(j))] = dist[i][j]

        with open(folder + d + confusion_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count = line_count + 1
                    continue
                d_test_confusion[(row[0], row[1])] = float(row[2])

        for pair in d_test_confusion.keys():
            if (pair[1], pair[0]) in n_test_confusion:
                continue
            if (pair[1], pair[0]) in d_test_confusion:
                # mean
                n_test_confusion[pair] = (
                    d_test_confusion[pair] + d_test_confusion[(pair[1], pair[0])]) * 1.0 / 2
            else:
                n_test_confusion[pair] = d_test_confusion[pair]

        for key, value in [(k, n_test_confusion[k]) for k in sorted(
                n_test_confusion, key=n_test_confusion.get, reverse=True)]:
            truth.append(key)
            truth_value.append(value)

        with open(folder + d + distance_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count = line_count + 1
                    continue
                distance[(row[0], row[1])] = float(row[2])

        for key, value in [(k, distance[k]) for k in sorted(
                distance, key=distance.get, reverse=False)]:
            our_positive.append(key)
            our_positive_value.append(value)

        for key in truth:
            if key not in distance and (key[1], key[0]) not in distance:
                our_positive.append(key)
        if d != "imsitu":
            MODE_positive = []
            MODE_positive_value = []
            for key, value in [(k, MODE_dist[k]) for k in sorted(
                    MODE_dist, key=MODE_dist.get, reverse=False)]:
                MODE_positive.append(key)
                MODE_positive_value.append(value)

        total = len(truth)
        print("total number of object pairs: " + str(total))

        # ground truth threshold
        threshold = np.mean(truth_value) + np.std(truth_value)
        d_threshold = np.mean(our_positive_value) - \
            np.std(our_positive_value)  # mean-1std threshold
        if d != "imsitu":
            mode_d_threshold = np.mean(
                MODE_positive_value) - np.std(MODE_positive_value)

        truth_n = 0  # count of ground truth
        temp_truth = {}  # ground truth
        for ti in range(len(truth_value)):
            if truth_value[ti] > threshold:
                truth_n = truth_n + 1
                temp_truth[truth[ti]] = 1
                temp_truth[tuple(reversed(truth[ti]))] = 1

        print(
            "count of ground truth buggy pairs if using 'mean-1std' as threshold " +
            str(truth_n))
        print("buggy pairs rate in ground truth if using 'mean-1std' as threshold : " +
              str(truth_n * 1.0 / total))

        print("using top 1% as predictions ")
        pred_n = int(total * 1.0 * 0.01)

        tp = 0
        fp = 0
        fn = 0
        tn = 0
        mode_tp = 0
        mode_fp = 0

        for p in our_positive[:pred_n]:  # using top 1% as predictions
            if p in temp_truth:
                tp = tp + 1
            else:
                fp = fp + 1

        if d != "imsitu":
            # print(MODE_positive[:pred_n])
            # print(temp_truth)
            for p in MODE_positive[:pred_n]:  # using top 1% as predictions
                if p in temp_truth:
                    mode_tp += 1
                else:
                    mode_fp += 1
        if pred_n == 0:
            print("-")
        else:
            print("deepinspect: tp: " + str(tp))
            print("deepinspect: fp: " + str(fp))
            print("deepinspect: precision: " + str(tp / pred_n))
            print("deepinspect: recall: " + str(tp / truth_n))

            if d != "imsitu":
                print("MODE: tp: " + str(mode_tp))
                print("MODE: fp: " + str(mode_fp))
                print("MODE: precision: " + str(mode_tp / pred_n))
                print("MODE: recall: " + str(mode_tp / truth_n))

            print("Random: tp: " + str(int(truth_n / 100)))
            print("Random: fp: " + str(pred_n - int(truth_n / 100)))
            print("Random: precision: " + str(int(truth_n / 100) / pred_n))
            print("Random: recall: " + str(int(truth_n / 100) / truth_n))

        print("using dist < mean-1std as predictions")
        pred_n2 = 0  # using dist < mean-1st as predictions
        mode_pred_n2 = 0
        for t in our_positive_value:
            if t < d_threshold:
                pred_n2 = pred_n2 + 1
        if d != "imsitu":
            for t in MODE_positive_value:
                if t < mode_d_threshold:
                    mode_pred_n2 = mode_pred_n2 + 1
        cutoff2 = pred_n2 * 1.0 / len(our_positive)
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        mode_tp = 0
        mode_fp = 0

        for p in our_positive[:pred_n2]:  # using dist < mean-1std as predictions
            if p in temp_truth:
                tp = tp + 1
            else:
                fp = fp + 1

        if d != "imsitu":
            # print(MODE_positive[:pred_n])
            # print(temp_truth)
            # using dist < mean-1std as predictions
            for p in MODE_positive[:mode_pred_n2]:
                if p in temp_truth:
                    mode_tp += 1
                else:
                    mode_fp += 1

        print("deepinspect: tp: " + str(tp))
        print("deepinspect: fp: " + str(fp))
        print("deepinspect: precision: " + str(tp / pred_n2))
        print("deepinspect: recall: " + str(tp / truth_n))

        if d != "imsitu":
            print("MODE: tp: " + str(mode_tp))
            print("MODE: fp: " + str(mode_fp))
            print("MODE: precision: " + str(mode_tp / mode_pred_n2))
            print("MODE: recall: " + str(mode_tp / truth_n))

        print("Random: tp: " + str(int(truth_n * cutoff2)))
        print("Random: fp: " + str(pred_n2 - int(truth_n * cutoff2)))
        print("Random: precision: " + str(int(truth_n * cutoff2) / pred_n2))
        print("Random: recall: " + str(int(truth_n * cutoff2) / truth_n))

        # AUCEC figure 6 data
        if len(datasets) == 8:

            for j in x:
                pred_n = int(total * 1.0 * j)
                #truth_n = int(total * 1.0 * conf_top_percentage)

                tp = 0
                fp = 0
                fn = 0
                tn = 0
                mode_tp = 0
                mode_fp = 0

                for p in our_positive[:pred_n]:  # using top (100*j)% as predictions
                    if p in temp_truth:
                        tp = tp + 1
                    else:
                        fp = fp + 1
                if d != "imsitu":
                    # print(MODE_positive[:pred_n])
                    # print(temp_truth)
                    # using top (100*j)% as predictions
                    for p in MODE_positive[:pred_n]:
                        if p in temp_truth:
                            mode_tp += 1
                        else:
                            mode_fp += 1
                if truth_n == 0:
                    y[d].append(0)
                    if d != "imsitu":
                        m[d].append(0)
                else:
                    y[d].append(tp / truth_n)
                    if d != "imsitu":
                        m[d].append(mode_tp / truth_n)

                # y[d].append(tp*1.0/truth_n)
                if pred_n < truth_n:
                    o[d].append(pred_n / truth_n)
                else:
                    o[d].append(1)
                r[d].append(j)

            cutoff1 = 0.01
            cutoff2 = pred_n2 * 1.0 / len(our_positive)
            print("prediction cutoff: " + str(cutoff2))
            c[d] = cutoff2

            area1 = 0
            area2 = 0
            area3 = 0
            area4 = 0
            for i in range(len(r[d])):
                area1 = area1 + y[d][i] * 1.0 / 39
            for i in range(len(r[d])):
                area2 = area2 + r[d][i] * 1.0 / 39
            for i in range(len(r[d])):
                area3 = area3 + o[d][i] * 1.0 / 39
            if d != "imsitu":
                for i in range(len(r[d])):
                    area4 = area4 + m[d][i] * 1.0 / 39
                gains2 = (area1 - area4) * 1.0 / area4
                gain2[d] = gains2
            gains = (area1 - area2) * 1.0 / area2

            print("gain: " + str(gains))
            gain[d] = gains

    if len(datasets) == 8:
        draw_AUCEC(datasets, name, x, o, r, y, m, c, gain, gain2)
'''
Draw AUCEC graphs.
INPUT:
x_list: a list of different percentages.
recall_ab_list_per_dataset: dictionary with key to be dataset folder name and corresponding list of recall (the percentage of bugs found) at different percentage.
datasets: a list of dataset folder names.
datasets: a list of dataset names.
conf_top_percentage: the ground-truth confusion percentage threshold.
'''
def draw_AUCEC(datasets, name, x, o, r, y, m, c, gain, gain2):

    #title = 'confusion percentage at '+str(conf_top_percentage)

    fig, axs = plt.subplots(2, 4, figsize=(40, 20), sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    # fig.suptitle(title)
    for i in range(2):
        for j in range(4):
            ax = axs[i][j]
            ind = i * 4 + j
            if ind < len(datasets):
                dataset = datasets[ind]
                dataset_name = name[datasets[ind]]
                if ind == 0 or ind == 4:
                    ax.set_ylabel('% of errors found', fontsize=60)
                else:
                    ax.set_ylabel('')

                if j == 2:
                    ax.set_xlabel('% of pairs inspected', fontsize=60)
                else:
                    ax.set_xlabel('')
                ax.tick_params(axis='x', labelsize=35)
                ax.tick_params(axis='y', labelsize=35)
                ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
                ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
                ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
                ax.plot([c[dataset] for _ in range(10)],
                        np.linspace(0, 1, 10), 'r--')
                ax.plot(
                    x,
                    o[dataset],
                    label='optimal',
                    linestyle=':',
                    linewidth=8,
                    color='green')
                ax.plot(
                    x,
                    r[dataset],
                    label='random',
                    linestyle='--',
                    linewidth=8,
                    color='blue')
                ax.plot(
                    x,
                    y[dataset],
                    label='DeepInspect',
                    linewidth=8,
                    color='orange')
                if dataset != 'imsitu':
                    ax.plot(
                        x,
                        m[dataset],
                        label='MODE',
                        linestyle='-.',
                        linewidth=8,
                        color='purple')
                    ax.text(
                        0.23,
                        0.5,
                        'Gain wrt MODE =' +
                        f'{gain2[dataset] * 100:.1f}' +
                        '%',
                        size=30)
                ax.text(
                    0.23,
                    0.6,
                    'Gain wrt random =' +
                    f'{gain[dataset] * 100:.1f}' +
                    '%',
                    size=30)
                ax.text(0.08, 0, dataset_name, size=42)
                #gain_v = "%.1f" % round(gain[dataset]*100,1)
                #ax.text(0.4, 0.5, 'Gain='+gain_v+'%', size=45)
                if ind == 7:
                    ax.legend(prop={'size': 35}, framealpha=0.5)

    for ax in axs.flat:
        ax.label_outer()

    # plt.savefig(title+'.pdf')
    print("AUCEC graph has been saved in confusion_std.pdf.")
    plt.savefig('confusion_std.pdf', format='pdf', dpi=1000, bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', dest='datasets', type=str, default='all', help='dataset to use')
    args = parser.parse_args()

    datasets = args.datasets
    assert datasets in [
        "all",
        "coco",
        "coco_gender",
        "cifar100",
        "robust_cifar10_small",
        "robust_cifar10_large",
        "robust_cifar10_resnet",
        "imsitu",
        "imagenet"]
    if datasets == 'all':
        datasets = [
        "coco",
        "coco_gender",
        "cifar100",
        "robust_cifar10_small",
        "robust_cifar10_large",
        "robust_cifar10_resnet",
        "imsitu",
        "imagenet"]
    else:
        datasets = [datasets]
    evaluate_confusion_bugs(datasets)
