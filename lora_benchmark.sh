#!/bin/bash

# HOSTNAMES MASTER_ADDR MASTER_PORT COUNT_NODE are coming from the main script
H=`hostname`
RANK=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`

cd /fsx/ethankim/trlx
source /fsx/ethankim/trlx/env/bin/activate
echo $RANK
export RANK=0
# --main_process_ip $MASTER_ADDR
echo $COUNT_NODE
export COUNT_NODE=1


# model_path EleutherAI/pythia-160m

###
#    delta_type: lora
#             modified_modules: "all"
#             lora_r: 8
#             lora_alpha: 16
# #             lora_dropout: 0.0
# #
# export delta_type=lora
# export lora_r=8
# export model_path=EleutherAI/pythia-1.4B

# args='{\"train\": {\"model_path\": \"EleutherAI/pythia-2.8b\",\"delta_type\": \"lora\",\"tokenizer_path\": \"EleutherAI/pythia-2.8b\"}}'
echo $args

accelerate launch --num_processes $((8 * $COUNT_NODE)) --num_machines $COUNT_NODE --config_file /fsx/ethankim/trlx/configs/accelerate/zero2-bf16.yaml /fsx/ethankim/trlx/examples/ppo_sentiments.py lora_config.json
# accelerate launch --num_processes 7 --config_file configs/accelerate/zero2-bf16.yaml examples/hh/ppo_hh.py "$args"