# On the Sentence Embeddings from Pre-trained Language Models

This is a TensorFlow implementation of the following [paper](https://arxiv.org):

```
On the Sentence Embeddings from Pre-trained Language Models
Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang, Yiming Yang, Lei Li
EMNLP 2020
```

Please contact bohanl1@cs.cmu.edu if you have any questions.

## Requirements

* Python >= 3.6
* TensorFlow >= 1.14

## Preparation

### Pretrained BERT models
```bash
export BERT_PREMODELS="../bert_premodels"
mkdir ${BERT_PREMODELS}; cd ${BERT_PREMODELS}
# then download the pre-trained BERT models from https://github.com/google-research/bert
# ...

ls ${BERT_PREMODELS}/uncased_L-12_H-768_A-12 # base
ls ${BERT_PREMODELS}/uncased_L-24_H-1024_A-16 # large
```

### GLUE
```bash
export GLUE_DIR="../glue_data"
python download_glue_data.py --data_dir=${GLUE_DIR}
```

### SentEval
```bash
cd ..
git clone https://github.com/facebookresearch/SentEval
```

### Build dir for cached tfrecord files
```bash
export CACHED_DIR="../cached_data"
mkdir ${CACHED_DIR}
```

## Usage

### Fine-tune BERT with NLI supervision (optional)
```bash
export RANDOM_SEED=1234
export CUDA_VISIBLE_DEVICES=0
export BERT_NAME="large"
export TASK_NAME="ALLNLI"
export OUTPUT_PARENT_DIR="../exp"
unset INIT_CKPT
bash scripts/train_siamese.sh train \
"--exp_name=exp_${BERT_NAME}_${RANDOM_SEED} \
--num_train_epochs=1.0 \
--learning_rate=2e-5 \
--train_batch_size=16 \
--cached_dir=${CACHED_DIR} \
--use_xla"


# evaluation
export RANDOM_SEED=1234
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=STS-B
export BERT_NAME=large
export OUTPUT_PARENT_DIR="../exp"
export INIT_CKPT=${OUTPUT_PARENT_DIR}/exp_${BERT_NAME}_${RANDOM_SEED}/model.ckpt-60108
export CACHED_DIR=${OUTPUT_PARENT_DIR}/cached_data
export EXP_NAME=exp_${BERT_NAME}_${RANDOM_SEED}_eval
bash scripts/train_siamese.sh predict \
"--exp_name=${EXP_NAME} \
 --cached_dir=${CACHED_DIR} \
 --sentence_embedding_type=avg \
 --flow=0 --flow_loss=0 \
 --num_examples=0 \
 --num_train_epochs=1e-10"
```

### Unsupervised learning of flow-based generative models
```bash
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=STS-B
export BERT_NAME=large
export OUTPUT_PARENT_DIR="../exp"
export INIT_CKPT=${OUTPUT_PARENT_DIR}/exp_large_t_ALLNLI_ep_1.00_lr_2.00e-05_bsz_16/model.ckpt-60108
export CACHED_DIR=${OUTPUT_PARENT_DIR}/cached_data
bash scripts/train_siamese.sh train \
"--exp_name_prefix=exp \
 --cached_dir=${CACHED_DIR} \
 --sentence_embedding_type=avg-last-2 \
 --flow=1 --flow_loss=1 \
 --num_examples=0 \
 --num_train_epochs=1.0 \
 --flow_learning_rate=1e-3 \
 --use_full_for_training=1"

# evaluation
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=STS-B
export BERT_NAME=large
export OUTPUT_PARENT_DIR="../exp"
export INIT_CKPT=${OUTPUT_PARENT_DIR}/exp_large_t_ALLNLI_ep_1.00_lr_2.00e-05_bsz_16/model.ckpt-60108
export CACHED_DIR=${OUTPUT_PARENT_DIR}/cached_data
export EXP_NAME=exp_t_STS-B_ep_1.00_lr_5.00e-05_e_avg-last-2_f_11_1.00e-03_allsplits
bash scripts/train_siamese.sh predict \
"--exp_name=${EXP_NAME} \
 --cached_dir=${CACHED_DIR} \
 --sentence_embedding_type=avg-last-2 \
 --flow=1 --flow_loss=1 \
 --num_examples=0 \
 --num_train_epochs=1.0 \
 --flow_learning_rate=1e-3 \
 --use_full_for_training=1"
```

## Reference

```
@inproceedings{li2020emnlp,
    title = {On the Sentence Embeddings from Pre-trained Language Models},
    author = {Bohan Li and Hao Zhou and Junxian He and Mingxuan Wang and Yiming Yang and Lei Li},
    booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    month = {November},
    year = {2020}
}

```

## Acknowledgements

A large portion of this repo is borrowed from the following projects:
- https://github.com/google-research/bert
- https://github.com/zihangdai/xlnet
- https://github.com/tensorflow/tensor2tensor


