import argparse
from typing import Dict
import os

import torch
from torch import optim

from datasets import TemporalDataset
from optimizers import TKBCOptimizer
from models import MuTP
from regularizers import N3, Lambda3, ScaleRegularizer


parser = argparse.ArgumentParser(
    description="MuTP"
)
parser.add_argument(
    '--dataset', type=str, default='ICEWS14',
    help="Dataset name"
)

parser.add_argument(
    '--model', default='MuTP', type=str,
    help="Model Name"
)
parser.add_argument(
    '--max_epochs', default=500, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=10, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=2000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--scale_reg', default=0., type=float,
    help="Multi-scale regularizer strength (entropy + L3)"
)
parser.add_argument(
    '--n_scales', default=1, type=int,
    help="Number of scales for multi-scale time decomposition"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)
parser.add_argument(
    '--gpu', default=0, type=int,
    help="GPU device ID to use (default: 0)"
)


args = parser.parse_args()

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
            """
            aggregate metrics for missing lhs and rhs
            :param mrrs: d
            :param hits:
            :return:
            """
            m = (mrrs['lhs'] + mrrs['rhs']) / 2.
            h = (hits['lhs'] + hits['rhs']) / 2.
            return {'MRR': m, 'hits@[1,3,10]': h}

def learn(model=args.model,
          dataset=args.dataset,
          rank=args.rank,
          learning_rate = args.learning_rate,
          batch_size = args.batch_size, 
          emb_reg=args.emb_reg, 
          time_reg=args.time_reg,
          scale_reg=args.scale_reg,
          n_scales=args.n_scales
         ):

    if torch.cuda.is_available():
        if args.gpu >= torch.cuda.device_count():
            print(f"Warning: GPU {args.gpu} not available, using GPU 0")
            args.gpu = 0
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)} (GPU {args.gpu})")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    root = 'results/'+ dataset +'/' + model
    modelname = model
    datasetname = dataset


    PATH=os.path.join(
        root,
        'rank{:.0f}/lr{:.4f}/batch{:.0f}/emb_reg{:.5f}/time_reg{:.5f}/n_scales{:.0f}/scale_reg{:.5f}/'.format(
            rank, learning_rate, batch_size, emb_reg, time_reg, n_scales, scale_reg
        )
    )
    
    dataset = TemporalDataset(dataset, device=device)
    
    sizes = dataset.get_shape()
    model = {
        'MuTP': MuTP(sizes, rank, no_time_emb=args.no_time_emb, n_scales=n_scales)
    }[model]
    model = model.to(device)


    opt = optim.Adagrad(model.parameters(), lr=learning_rate)

    print("Start training process: ", modelname, "on", datasetname, "using", "rank =", rank, "lr =", learning_rate, "emb_reg =", emb_reg, "time_reg =", time_reg, "n_scales =", n_scales, "scale_reg =", scale_reg)

    emb_reg = N3(emb_reg)
    time_reg = Lambda3(time_reg)
    scale_reg_obj = ScaleRegularizer() if scale_reg > 0 and hasattr(model, 'rel_scale_weights') else None
  
    try:
        os.makedirs(PATH)
    except FileExistsError:
        pass
    patience = 0
    mrr_std = 0

    curve = {'train': [], 'valid': [], 'test': []}

    for epoch in range(args.max_epochs):
        print("[ Epoch:", epoch, "]")
        examples = torch.from_numpy(
            dataset.get_train().astype('int64')
        )

        model.train()

        optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, opt,
            batch_size=batch_size,
            device=device,
            scale_regularizer=scale_reg_obj,
            scale_reg_weight=scale_reg
        )

        optimizer.epoch(examples)
       
        if epoch < 0 or (epoch + 1) % args.valid_freq == 0:

            if dataset.interval: 
                valid, test = [
                    avg_both(*dataset.eval(model, split, -1))
                    for split in ['valid', 'test']
                ]
                # 对于interval数据集，train评估使用50000个样本
                train = avg_both(*dataset.eval(model, 'train', 50000))
                print("valid: ", valid['MRR'])
                print("test: ", test['MRR'])
                print("train: ", train['MRR'])

            else:
                valid, test, train = [
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['valid', 'test', 'train']
                ]
                print("valid: ", valid['MRR'])
                print("test: ", test['MRR'])
                print("train: ", train['MRR'])

            # Save results in format: [Epoch:X]-TEST : {...} VALID : {...} TRAIN : {...}
            f = open(os.path.join(PATH, 'result.txt'), 'a+')
            f.write("\n[Epoch:{}]-TEST : {} VALID : {} TRAIN : {}".format(epoch, test, valid, train))
            f.close()
            
            # early-stop with patience
            mrr_valid = valid['MRR']
            if mrr_valid < mrr_std:
               patience += 1
               if patience >= 10:
                  print("Early stopping ...")
                  break
            else:
               patience = 0
               mrr_std = mrr_valid
               torch.save(model.state_dict(), os.path.join(PATH, modelname+'.pkl'))

            curve['valid'].append(valid)
            curve['test'].append(test)
            curve['train'].append(train)
    
            print("\t TRAIN: ", train)
            print("\t VALID : ", valid)
            print("\t TEST : ", test)

    # Load best model (based on validation set) and evaluate on test set
    checkpoint_path = os.path.join(PATH, modelname+'.pkl')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print("\nLoading best model (based on validation set) for final test evaluation...")
    
    results = avg_both(*dataset.eval(model, 'test', -1))
    print("\n\nTEST : ", results)
    f = open(os.path.join(PATH, 'result.txt'), 'a+')
    f.write("\n\nTEST : ")
    f.write(str(results))
    f.write("\n")
    f.close()

if __name__ == '__main__':
    learn()


