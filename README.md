# Self-attention For Graph Problem

## Procedure to Generate data, Training and evaluation:
 
### Step1. Generate Data(training data generated while training) 
    python generate_data.py --problem op --name validation --seed 4321 -f

    python generate_data.py --problem op --name test --seed 1234 -f

### Step2. Training
    For graph size 20
    python run.py --graph_size 20 --baseline rollout --run_name 'op_dist20_rollout'

    For graph size 50
    python run.py --graph_size 50 --baseline rollout --run_name 'op_dist50_rollout'

    For graph size 100
    python run.py --graph_size 100 --baseline rollout --run_name 'op_dist100_rollout'

### Step3. Evaluation

    python eval.py data/op/op_dist20_test_seed1234.pkl --model pretrained/op_dist_20 --decode_strategy sample --width 1280 --eval_batch_size 1

**Note:**

1. Before Step2, data needs to be generated first. So, please run step1 commands before training(step2) and evaluating(step3). And once data generation is done, step2 and step3 can be run independently.

2. To run this code, pytorch enviroment needs to be created with packages available in environment.yml to create environment. Code is compatible with both CPU and GPU

## Acknowledgements

Thanks to [wouterkool](https://github.com/wouterkool/attention-learn-to-route) for getting me started with encoder and decoder implementation, most of the code taken over from this repository.