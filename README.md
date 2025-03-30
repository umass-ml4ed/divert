# DiVERT: Distractor Generation with Variational Errors Represented as Text for Math Multiple-choice Questions

This repo contains the code for the paper, <a href="https://aclanthology.org/2024.emnlp-main.512//">DiVERT: Distractor Generation with Variational Errors Represented as Text for Math Multiple-choice Questions</a>, by Nigel Fernandez\*, Alexander Scarlatos\*, Wanyong Feng, Simon Woodhead, and Andrew Lan, published at EMNLP 2024 (\*equal contribution).

If you use our code, or find this work helpful for your research, please cite us:
```
@inproceedings{fernandez-etal-2024-divert,
    title = "{D}i{VERT}: Distractor Generation with Variational Errors Represented as Text for Math Multiple-choice Questions",
    author = "Fernandez, Nigel  and
      Scarlatos, Alexander  and
      Feng, Wanyong  and
      Woodhead, Simon  and
      Lan, Andrew",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.512/",
    doi = "10.18653/v1/2024.emnlp-main.512",
    pages = "9063--9081"
}
```

## Setup

Create Python environment:
```
python3 -m venv er
source er/bin/activate
python3 -m pip install -r code/utils/requirements.txt
```

## Data Processing

Create data splits:
```
python3 code/utils/create_err_datasets.py
```

## Finetuning

e_given_s
```
python -m code.finetune.train\
    'task_name="e_given_s"'
```


e_given_s_d
```
python -m code.finetune.train\
    'task_name="e_given_s_d"'
```


d_given_s_e
```
python -m code.finetune.train\
    'task_name="d_given_s_e"'
```



## Distractor generation

```
python -m code.distractorgen.train
```



## Evaluation

->Baseline p(d|s) (beam search with 100 beams (to maintain fairness of overgenerating 100 distractors in variational pipeline) and pick top 10 beams)
```
python -m code.finetune.test_distractor_gen\
    'task_name="d_given_s"'\
    'model_name="meta-math/MetaMath-Mistral-7B"'\
    testing=true\
    'wandb_run_name="NAME"'\
    test_batch_size=32\
    num_distractor_samples=10\
    num_beams=100
```

->Variational pipeline (diverse Beam search errors + standard beam search distractors + rank using distractor beam scores only)
```
python -m code.distractorgen.test\
    'model_name="meta-math/MetaMath-Mistral-7B"'\
    testing=true\
    'wandb_run_name="NAME"'\
    test_batch_size=32\
    num_distractor_samples=10\
    num_error_samples=10\
    rank=true\
    rank_with_distractor_scores_only=true\
    diverse_beam_search_error_gen=true\
    diversity_penalty=1.0
```

## Prompting Baselines

### Setup

First, clone the repo for prompting-based distractor generation: https://github.com/umass-ml4ed/prompt_distractor_generation_NAACL

From the DiVERT repo, prepare and move the data to the prompting repo:
```
python -m utils.prompting_format_data
mv data/eedi_* <path to prompting repo>/data/
```

### Inference and Evaluation

Run the following in the prompting repo.

#### Zero-Shot
```
python run.py openAI.model=gpt-4o prompt.type=zero_shot_all_info_error retriever.type=none prompt.num_distractors=10
python evaluation.py analysis/zero_shot_all_info_error_ndis10_gpt-4o_none_q.csv 10
```

#### kNN
```
python run.py openAI.model=gpt-4o prompt.type=distractor_all_info_with_errors retriever.type=KNN retriever.encodingPattern=q+a+f prompt.num_distractors=10
python evaluation.py analysis/distractor_all_info_with_errors_ndis10_gpt-4o_KNN_q+a+f.csv 10
```