import os

def testset(args, test_results ):

    if args.data_augmentation:
        if not os.path.exists(f'{args.save_test_result}(batch_{args.batch_size})'):
            os.mkdir(f'{args.save_test_result}(batch_{args.batch_size})')
        if not os.path.exists(f'{args.save_test_result}(batch_{args.batch_size})/rate_{args.rate_of_real}'):
            os.mkdir(f'{args.save_test_result}(batch_{args.batch_size})/rate_{args.rate_of_real}')
            
    if not args.data_augmentation:
        if not os.path.exists(f'{args.save_conventional_test_result}(batch_{args.batch_size})'):
            os.mkdir(f'{args.save_conventional_test_result}(batch_{args.batch_size})')
        

    if args.data_augmentation:
        save_test_result_specific = f'{args.save_test_result}(batch_{args.batch_size})/rate_{args.rate_of_real}/{args.model_name}_lr_{args.lr}_vqlr_{args.vq_lr}_codebook_{args.num_codebook_vectors}_rate_{args.rate_of_real}_batch_{args.batch_size}_seed_{args.seed}'
    if not args.data_augmentation:
        save_test_result_specific = f'{args.save_conventional_test_result}(batch_{args.batch_size})/model_{args.model_name}_lr_{args.lr}_batch_{args.batch_size}_seed_{args.seed}'
    
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


dataset_name_mapping = {
    'cola': 'CoLA',
    'sst-2': 'SST-2',
    'qqp': 'QQP',
    'mrpc': 'MRPC',
    'mnli': 'MNLI-m',
    'mnli-mm': 'MNLI-mm',
    'qnli': 'QNLI',
    'rte': 'RTE',
    'wnli': 'WNLI',
}

