# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import os
import sys
from typing import List

import torch
from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import TRLConfig, default_ppo_config


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_ppo_config().to_dict(), hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * 256,
        config=config,
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
from dataclasses import dataclass, field, asdict

from transformers import HfArgumentParser


@dataclass
class ModelArgs:
    num_layers_unfrozen: int = 5
    #learning_rate: float = field(default=1.0e-5)
    model_path: str = field(default="EleutherAI/pythia-410M")
    tokenizer_path: str = field(default="EleutherAI/gpt-neox-20b")


@dataclass
class LoraArgs:
    lora_r: int = field(default=8)
    lora_alpha: float = field(default=16)
    delta_type: str = field(default="lora")

@dataclass
class TrainingArgs:
    batch_size: int = 2



if __name__ == "__main__":
    # use hf-argument-parser to parse hparams
    import torch 
    # access local rank from env as integer
    import os 
    rank = os.environ.get("LOCAL_RANK", 0)
    print(rank)
    # torch.cuda.empty_cache()
    # torch.cuda.set_per_process_memory_fraction(0.9, device=f"cuda:{rank}")
    parser = HfArgumentParser((LoraArgs, ModelArgs, TrainingArgs))
    lora_args, model_args, training_args = parser.parse_args_into_dataclasses()
    hparams = asdict(model_args)
    hparams["delta_kwargs"] = asdict(lora_args)
    training_args = asdict(training_args)
    hparams = {"model": hparams, "train": training_args}

    import json
    print(json.dumps(hparams))
    pprint(hparams)
    # main(hparams)
