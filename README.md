# Knowledge Graph Policy Network

This is our Pytorch implementation for the paper:

>Xiang Wang, Yaokun Xu, Xiangnan He, Yixin Cao, Meng Wang and Tat-Seng Chua (2020). Reinforced Negative Sampling over Knowledge Graph for Recommendation. [Paper in ACM DL](https://dl.acm.org/doi/abs/10.1145/3366423.3380098) or [Paper in arXiv](https://arxiv.org/abs/2003.05753). In WWW'2020, Taipei, Taiwan, China, April 20–24, 2020.

Author: Dr. Xiang Wang (xiangwang at u.nus.edu) and Mr. Yaokun Xu (xuyaokun98 at gmail.com)

## Introduction
Knowledge Graph Policy Network (KGPolicy) is a new negative sampling framework tailored to knowledge-aware personalized recommendation. Exploiting rich connections of knowledge graph, KGPolicy is able to discover high-quality (i.e., informative and factual) items as negative training instances, thus providing better recommendation.


## Citation 
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{KGPolicy20,
  author    = {Xiang Wang and
               Yaokun Xu and
               Xiangnan He and
               Yixin Cao and
               Meng Wang and
               Tat{-}Seng Chua},
  title     = {Reinforced Negative Sampling over Knowledge Graph for Recommendation},
  booktitle = {{WWW}},
  year      = {2020}
}
```
## Reproducibility
To demonstrate the reporducibility of the best performance reported in our paper and faciliate researchers for development and testing purpose, we provide the instructions as follows. Later, we will release the other baselines.

### 1. Data and Source Code
We follow our previous work, KGAT, and you can get the detailed information about the datasets in [KGAT](https://github.com/xiangwang1223/knowledge_graph_attention_network).

i. Create a new directory for this repo
```bash
➜ mkdir KG-Policy
➜ cd KG-Policy
```

ii. Get dataset and pretrain model
```bash
➜ wget https://github.com/xiangwang1223/kgpolicy/releases/download/v1.0/Data.zip
➜ unzip Data.zip
```

iii. Get source code
```bash
➜ git clone https://github.com/xiangwang1223/kgpolicy.git
```

### 2. Environment
Please use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) to manage the environment.

i. Switch to source code dir 
```
➜ cd kgpolicy
```

ii. Create a new environment
```bash
➜ conda create -n geo python=3.6
➜ conda activate geo
```

iii. Ensure python version is `3.6`. Then install all requirements for this project.
```bashsetup.sh
➜ bash setup.sh
```

Note that: Sometimes there is mismatch between `cuda` version and `torch_geometric` version. If you encounter this problem, please try to install a correct `cuda` version. If you prefer to install all dependences by yourself, please ensure `torch_geometric` is properly installed. After doing these, now it's ready to train KG-Policy model.

### 3. Train
i. Train KG-Policy on `last-fm`. Also, KG-Policy can be trained on other two datasets, `amazon-book` and `yelp2018`, check it out in `Data`.
```bash
➜ python main.py
```

To run on other two datasets
```bash
➜ python main.py --regs 1e-4 --dataset yelp2018 --model_path model/best_yelp.ckpt 
➜ python main.py --regs 1e-4 --dataset amazon-book --model_path model/best_ab.ckpt 
```

Note that: The default `regs` is `1e-5`, while we use `1e-4` as `regs` when training `amazon-book` and `yelp2018`. There are also some others parameters can be tuned for a better performance, check it out at `common/config/parser.py`.

### 4. Experiment result
To be consistent to our KGAT, we use the same evaluation metrics (i.e., `Recall@K` and `NDCG@K`), use the same codes released in [KGAT](https://github.com/xiangwang1223/knowledge_graph_attention_network), and report them in our KGPolicy paper. We note that this implementation of `NDCG@K` is different from the [standard definition](https://en.wikipedia.org/wiki/Discounted_cumulative_gain), while they reflect similar trendings. Hence here we also report the results in terms of the standard `NDCG@K*` and please check the implementation at `common/test.py`.

i. Dataset: last-fm

|    Model    | Recall@20 | NDCG@20 |
| :---------: | :-------: | :----------: |
|     RNS     |  0.0687   |    0.0584    |  
|     DNS     |  0.0874   |    0.0746    | 
|    IRGAN    |  0.0755   |    0.0627    | 
|   KG-Policy |  0.0957   |    0.0837    |

ii. Dataset: yelp2018

|    Model    | Recall@20 | NDCG@20 |
| :---------: | :-------: | :----------: |
|     RNS     |  0.0465   |    0.0298    | 
|     DNS     |  0.0666   |    0.0429    | 
|    IRGAN    |  0.0538   |    0.0342    | 
|   KG-Policy |  0.0746   |    0.0489    |         

iii. Dataset: amazon-book

|    Model    | Recall@20 | NDCG@20 |
| :---------: | :-------: | :----------: |
|     RNS     |  0.1239   |    0.0647    |
|     DNS     |  0.1460   |    0.0775    |
|    IRGAN    |  0.1330   |    0.0693    |
|   KG-Policy |  0.1609   |    0.0890    |

## Acknowledgement
Any scientific publications that use our codes and datasets should cite the following paper as the reference:
```
@inproceedings{KGPolicy20,
  author    = {Xiang Wang and
               Yaokun Xu and
               Xiangnan He and
               Yixin Cao and
               Meng Wang and
               Tat{-}Seng Chua},
  title     = {Reinforced Negative Sampling over Knowledge Graph for Recommendation},
  booktitle = {{WWW}},
  year      = {2020}
}
```

Nobody guarantees the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set. The data set may be used for any research purposes under the following conditions:
* The user must acknowledge the use of the data set in publications resulting from the use of the data set.
* The user may not redistribute the data without separate permission.
* The user may not try to deanonymise the data.
* The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from us.

## Funding Source Acknowledgement

This research is supported by the National Research Foundation, Singapore under its International Research Centres in Singapore Funding Initiative. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore.
