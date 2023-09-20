import os

def testset(args, test_results ):

    if args.data_augmentation:
        if not os.path.exists(args.save_test_result):
            os.mkdir(args.save_test_result)
    if not args.data_augmentation:
        if not os.path.exists(args.save_conventional_test_result):
            os.mkdir(args.save_conventional_test_result)

    if args.data_augmentation:
        save_test_result_specific = f'{args.save_test_result}/{args.model_name}_lr_{args.lr}_vqlr_{args.vq_lr}_codebook_{args.num_codebook_vectors}_seed_{args.seed}'
    if not args.data_augmentation:
        save_test_result_specific = f'{args.save_conventional_test_result}/model_{args.model_name}_lr_{args.lr}_seed_{args.seed}'
    
    if not os.path.exists(save_test_result_specific):
        os.mkdir(save_test_result_specific)

    dataset_name = dataset_name_mapping[args.dataset]

    if args.dataset in ['mnli', 'mnli-mm']:
        pred_map = {0: 'neutral', 1: 'entailment', 2: 'contradiction'}
        with open(os.path.join(save_test_result_specific, "%s.tsv" % (dataset_name)), 'w') as pred_fh:
            pred_fh.write("index\tprediction\n")
            # split_idx = 0
            for idx, pred in enumerate(test_results):
                pred = pred_map[pred]
                pred_fh.write("%d\t%s\n" % (idx, pred))

    else:
        with open(os.path.join(save_test_result_specific, "%s.tsv" % (dataset_name)), 'w') as pred_fh:
            pred_fh.write("index\tprediction\n")
            for idx, pred in enumerate(test_results):
                if args.dataset in ['qnli', 'rte']:
                    pred_map = {0: 'not_entailment', 1: 'entailment'}
                    pred = pred_map[pred]
                    pred_fh.write('%d\t%s\n' % (idx, pred))
                else:
                    pred_fh.write("%d\t%d\n" % (idx, pred))

    # with open(os.path.join(save_test_result_specific, "results.tsv"), 'a') as results_fh: # aggregate results easily
    #     run_name = args.run_dir.split('/')[-1]
    #     all_metrics_str = ', '.join(['%s: %.3f' % (metric, score) for \
    #                                 metric, score in val_results.items()])
    #     results_fh.write("%s\t%s\n" % (run_name, all_metrics_str))


dataset_name_mapping = {
    'cola': 'CoLA',
    'sst-2': 'SST-2',
    'qqp': 'QQP',
    'mrpc': 'MRPC',
    'mnli': 'MNLI',
    'mnli-mm': 'MNLI-mm',
    'qnli': 'QNLI',
    'rte': 'RTE',
    'wnli': 'WNLI',
}