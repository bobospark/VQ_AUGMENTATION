import os


def testset(args, test_results ):
    if not os.path.exists(args.save_test_result):
        os.mkdir(args.save_test_result)

    save_test_result_specific = f'{args.save_test_result}/model_{args.model_name}_lr_{args.lr}_vqlr_{args.vq_lr}_codebook_{args.num_codebook_vectors}_seed_{args.seed}'
    
    
    if not os.path.exists(save_test_result_specific):
        os.mkdir(save_test_result_specific)

    # all_results[task] = (val_results, test_results, save_path)
    # for idx, task_preds in enumerate(test_results): # write predictions for each task
        #if 'mnli' not in eval_task:
        #    continue
        # idxs_and_preds = [(idx, pred) for pred, idx in zip(task_preds[0], task_preds[1])]
        # idxs_and_preds.sort(key=lambda x: x[0])
    if 'mnli' is args.dataset:
        pred_map = {0: 'neutral', 1: 'entailment', 2: 'contradiction'}
        with open(os.path.join(save_test_result_specific, "%s-m.tsv" % (args.dataset)), 'w') as pred_fh:
            pred_fh.write("index\tprediction\n")
            split_idx = 0
            for idx, pred in test_results[:9796]:
                pred = pred_map[pred]
                pred_fh.write("%d\t%s\n" % (split_idx, pred))
                split_idx += 1
        with open(os.path.join(save_test_result_specific, "%s-mm.tsv" % (args.dataset)), 'w') as pred_fh:
            pred_fh.write("index\tprediction\n")
            split_idx = 0
            for idx, pred in test_results[9796:9796+9847]:
                pred = pred_map[pred]
                pred_fh.write("%d\t%s\n" % (split_idx, pred))
                split_idx += 1
        with open(os.path.join(save_test_result_specific, "diagnostic.tsv"), 'w') as pred_fh:
            pred_fh.write("index\tprediction\n")
            split_idx = 0
            for idx, pred in test_results[9796+9847:]:
                pred = pred_map[pred]
                pred_fh.write("%d\t%s\n" % (split_idx, pred))
                split_idx += 1
    else:
        with open(os.path.join(save_test_result_specific, "%s.tsv" % (dataset_name_mapping[args.dataset])), 'w') as pred_fh:
            pred_fh.write("index\tprediction\n")
            for idx, pred in enumerate(test_results):
                if args.dataset is 'sts-b':
                    pred_fh.write("%d\t%.3f\n" % (idx, pred))
                elif args.dataset is 'rte':
                    pred = 'entailment' if pred else 'not_entailment'
                    pred_fh.write('%d\t%s\n' % (idx, pred))
                elif args.dataset is 'squad':
                    pred = 'entailment' if pred else 'not_entailment'
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