# KG-Policy

Reinforced Negative Sampling over Knowledge Graph for Recommendation


---
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Data and Source Code

Create a new directory for this repo
```bash
➜ mkdir KG-Policy
➜ cd KG-Policy
```

Get dataset and pretrain model
```bash
➜ wget https://github.com/xiangwang1223/kgpolicy/releases/download/v1.0/Data.zip
➜ unzip Data.zip
```

Get source code
```bash
➜ git clone https://github.com/xiangwang1223/kgpolicy.git
```

### Environment

Please use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) to manage the environment.

Switch to source code dir 
```
➜ cd kgpolicy
```


Create a new environment
```bash
➜ conda create -n geo python=3.6
➜ conda activate geo
```

Ensure python version is `3.6`.

Then install all requirements for this project.
```bashsetup.sh
➜ bash setup.sh
```

Sometimes there is mismatch between `cuda` version and `torch_geometric` version. If you encounter this problem, please try to install a correct `cuda` version or leave us a message.

If you prefer to install all dependences by yourself, please ensure `torch_geometric` is properly installed.

After doing these, now it's ready to train KG-Policy model.

### Train

Train KG-Policy on `last-fm`
```bash
➜ python main.py
```
Also, KG-Policy can be trained on other two datasets, `amazon-book` and `yelp2018`, check it out in `Data`.

To run on other two datasets
```bash
➜ python main.py --regs 1e-4 --dataset yelp2018 --model_path model/best_yelp.ckpt 
➜ python main.py --regs 1e-4 --dataset amazon-book --model_path model/best_ab.ckpt 
```

Note that the default `regs` is `1e-5`, while we use `1e-4` as `regs` when training `amazon-book` and `yelp2018`. There are also some others parameters can be tuned for a better performance, check it out at `common/config/parser.py`.

### Experiment result

dataset: last-fm

|    Model    | Recall@20 | NDCG@20 |
| :---------: | :-------: | :----------: |
|     RNS     |  0.0687   |    0.0584    |  
|     DNS     |  0.0874   |    0.0746    | 
|    IRGAN    |  0.0755   |    0.0627    | 
|   KG-Policy |  0.0957   |    0.0837    |

dataset: yelp2018

|    Model    | Recall@20 | NDCG@20 |
| :---------: | :-------: | :----------: |
|     RNS     |  0.0465   |    0.0298    | 
|     DNS     |  0.0666   |    0.0429    | 
|    IRGAN    |  0.0538   |    0.0342    | 
|   KG-Policy |  0.0746   |    0.0489    |         

dataset: amazon-book

|    Model    | Recall@20 | NDCG@20 |
| :---------: | :-------: | :----------: |
|     RNS     |  0.1239   |    0.0647    |
|     DNS     |  0.1460   |    0.0775    |
|    IRGAN    |  0.1330   |    0.0693    |
|   KG-Policy |  0.1609   |    0.0890    |    

The results are different from what reported in our paper, as we corrected how to calculate `NDCG`. Check it at `common/test.py`. 

Also, we use pretrained model in KG-Policy training process. The pretrained model is coming from `DNS`, a variant of `BPR-MF`. We implement this algorithm in another repo. 
