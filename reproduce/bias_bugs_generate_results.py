import argparse
import numpy as np
from scipy.stats import percentileofscore


from util import draw_CE_comparison, get_AUCEC_gain, get_single_precision_recall_wrapper



def get_AUCEC_gain_and_recall_precision_ab_list(datasets, top_percentages, use_MODE):

    recall_precision_ab_list_per_dataset = dict()
    top_std_cutoff_per_dataset = dict()
    AUCEC_gain_per_dataset = dict()

    for dataset in datasets:
        print(dataset)
        use_MODE_str = ''
        if use_MODE:
            if dataset == 'imsitu':
                continue
            use_MODE_str = '_MODE'

        test_filepath = '../data/'+dataset+'/saved_data_object'+'/'+dataset+'_test'+use_MODE_str
        predicted_label_conf_bias_list = np.load(test_filepath+'.npy')
        conf_list = np.sort(predicted_label_conf_bias_list[:, 0])
        cutoff_val = np.mean(conf_list)+np.std(conf_list)
        conf_top_percentage = 1 - (percentileofscore(conf_list, cutoff_val) / 100)

        bias_list = np.sort(predicted_label_conf_bias_list[:, 1])
        cutoff_val_bias = np.mean(bias_list)+np.std(bias_list)
        top_percentage = 1 - (percentileofscore(bias_list, cutoff_val_bias) / 100)


        AUCEC, recall_precision_ab_list = get_AUCEC_gain(conf_top_percentage, top_percentages, predicted_label_conf_bias_list)


        recall_precision_ab_list_per_dataset[dataset] = recall_precision_ab_list
        top_std_cutoff_per_dataset[dataset] = top_percentage

        AUCEC_gain_per_dataset[dataset] = AUCEC
    return conf_top_percentage, recall_precision_ab_list_per_dataset, top_std_cutoff_per_dataset, AUCEC_gain_per_dataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', dest='datasets', type=str, default='all', help='dataset to use')
    args = parser.parse_args()

    datasets = args.datasets
    assert datasets in ["all", "coco","coco_gender","cifar100","robust_cifar10_small","robust_cifar10_large","robust_cifar10_resnet","imsitu","imagenet"]
    if datasets == 'all':
        datasets = ["coco","coco_gender","cifar100","robust_cifar10_small","robust_cifar10_large","robust_cifar10_resnet","imsitu","imagenet"]
    else:
        datasets = [datasets]

    # datasets support one of the following or all of them: ["coco","coco_gender","cifar100","robust_cifar10_small","robust_cifar10_large","robust_cifar10_resnet","imsitu","imagenet"]

    for dataset in datasets:
        print("---------------------------")
        print('dataset:', dataset)


        # DeepInspect
        test_filepath = '../data/'+dataset+'/saved_data_object'+'/'+dataset+'_test'
        predicted_label_conf_bias_list = np.load(test_filepath+'.npy')

        # Mode
        if dataset != "imsitu":
            test_filepath += '_MODE'
            predicted_label_conf_bias_list_mode = np.load(test_filepath+'.npy')


        N = len(predicted_label_conf_bias_list)
        conf_list = np.sort(predicted_label_conf_bias_list[:, 0])
        cutoff_val = np.mean(conf_list)+np.std(conf_list)
        conf_top_percentage = 1 - (percentileofscore(conf_list, cutoff_val) / 100)


        print("total number of object pairs: " + str(N))
        print("count of ground truth buggy pairs if using 'mean+1std' as threshold: " + str(int(N*conf_top_percentage)))
        print("buggy pairs rate in ground truth if using 'mean+1std' as threshold: " + str(conf_top_percentage))


        # top 1 %
        print('using avg_bias > 1% as predictions')
        if dataset in ["robust_cifar10_small","robust_cifar10_large","robust_cifar10_resnet"]:
            print('-')
        else:
            top_percentage = 0.01

            m11, m01, precision, recall = get_single_precision_recall_wrapper(predicted_label_conf_bias_list, conf_top_percentage, top_percentage)

            print("deepinspect: tp: " + str(m11))
            print("deepinspect: fp: " + str(m01))
            print("deepinspect: precision: " + str(precision))
            print("deepinspect: recall: " + str(recall))

            if dataset != "imsitu":
                m11_mode, m01_mode, precision_mode, recall_mode = get_single_precision_recall_wrapper(predicted_label_conf_bias_list_mode, conf_top_percentage, top_percentage)

                print("MODE: tp: " + str(m11_mode))
                print("MODE: fp: " + str(m01_mode))
                print("MODE: precision: " + str(precision_mode))
                print("MODE: recall: " + str(recall_mode))

            print("Random: tp: " + str(int(top_percentage * conf_top_percentage * N)))
            print("Random: fp: " + str(int(top_percentage * (1-conf_top_percentage) * N)))
            print("Random: precision: " + str(conf_top_percentage))
            print("Random: recall: " + str(top_percentage))


        # mean + stdev
        print('using avg_bias > mean+1std as predictions')
        bias_list = np.sort(predicted_label_conf_bias_list[:, 1])
        cutoff_val_bias = np.mean(bias_list)+np.std(bias_list)
        top_percentage = 1 - (percentileofscore(bias_list, cutoff_val_bias) / 100)

        m11, m01, precision, recall = get_single_precision_recall_wrapper(predicted_label_conf_bias_list, conf_top_percentage, top_percentage)

        print("deepinspect: tp: " + str(m11))
        print("deepinspect: fp: " + str(m01))
        print("deepinspect: precision: " + str(precision))
        print("deepinspect: recall: " + str(recall))

        if dataset != "imsitu":
            m11_mode, m01_mode, precision_mode, recall_mode = get_single_precision_recall_wrapper(predicted_label_conf_bias_list_mode, conf_top_percentage, top_percentage)

            print("MODE: tp: " + str(m11_mode))
            print("MODE: fp: " + str(m01_mode))
            print("MODE: precision: " + str(precision_mode))
            print("MODE: recall: " + str(recall_mode))

        print("Random: tp: " + str(int(top_percentage * conf_top_percentage * N)))
        print("Random: fp: " + str(int(top_percentage * (1-conf_top_percentage) * N)))
        print("Random: precision: " + str(conf_top_percentage))
        print("Random: recall: " + str(top_percentage))


    num_of_intervals = 40
    top_percentages = np.linspace(0, 1, num_of_intervals)

    conf_top_percentage, recall_precision_ab_list_per_dataset, top_std_cutoff_per_dataset, AUCEC_gain_per_dataset = get_AUCEC_gain_and_recall_precision_ab_list(datasets, top_percentages, False)

    _, MODE_recall_precision_ab_list_per_dataset, _, MODE_AUCEC_gain_per_dataset = get_AUCEC_gain_and_recall_precision_ab_list(datasets, top_percentages, True)


    AUCEC_gain_per_dataset_dual = dict()

    for dataset in datasets:
        if dataset == 'imsitu':
            continue
        aucec_over_random = AUCEC_gain_per_dataset[dataset]
        MODE_aucec_over_random = MODE_AUCEC_gain_per_dataset[dataset]
        aucec = aucec_over_random*(1/2)+1/2
        MODE_aucec = MODE_aucec_over_random*(1/2)+1/2

        aucec_over_MODE = (aucec - MODE_aucec) / MODE_aucec
        AUCEC_gain_per_dataset_dual[dataset] = (aucec_over_MODE, aucec_over_random)

    if 'imsitu' in datasets:
        AUCEC_gain_per_dataset_dual['imsitu'] = (None, AUCEC_gain_per_dataset['imsitu'])


    datasets_names_mappings = {"coco":'COCO' ,"coco_gender":'COCO gender',"cifar100":'CIFAR-100',"robust_cifar10_small":'Robust CIFAR-10 Small',"robust_cifar10_large":'Robust CIFAR-10 Large',"robust_cifar10_resnet":'Robust CIFAR-10 ResNet',"imsitu":'imSitu',"imagenet":'ImageNet'}

    dataset_names = [datasets_names_mappings[d] for d in datasets]

    if len(datasets) == 8:
        draw_CE_comparison(top_percentages, recall_precision_ab_list_per_dataset, MODE_recall_precision_ab_list_per_dataset, top_std_cutoff_per_dataset, AUCEC_gain_per_dataset_dual, datasets, dataset_names, conf_top_percentage)
