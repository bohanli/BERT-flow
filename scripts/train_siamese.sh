#!/bin/bash
CURDIR=$(cd $(dirname $0); cd ..; pwd)

export BERT_DIR=${BERT_PREMODELS}/uncased_L-12_H-768_A-12

if [[ $BERT_NAME == "large-wwm" ]];then
  export BERT_DIR=${BERT_PREMODELS}/wwm_uncased_L-24_H-1024_A-16
elif [[ $BERT_NAME == "large" ]];then
  export BERT_DIR=${BERT_PREMODELS}/uncased_L-24_H-1024_A-16
else
  export BERT_DIR=${BERT_PREMODELS}/uncased_L-12_H-768_A-12
fi

if [ -z "$INIT_CKPT" ]; then
    export INIT_CKPT=$BERT_DIR/bert_model.ckpt
fi

if [ -z "$TASK_NAME" ]; then
    export TASK_NAME="STS-B"
fi


if [[ $1 == "train" ]];then
    echo "train"

    exec python3 ${CURDIR}/run_siamese.py \
      --task_name=${TASK_NAME} \
      --do_train=true \
      --do_eval=true \
      --data_dir=${GLUE_DIR}/${TASK_NAME} \
      --vocab_file=${BERT_DIR}/vocab.txt \
      --bert_config_file=${BERT_DIR}/bert_config.json \
      --init_checkpoint=${INIT_CKPT} \
      --max_seq_length=64 \
      --output_parent_dir=${OUTPUT_PARENT_DIR} \
      ${@:2}
elif [[ $1 == "eval" ]];then
    echo "eval"
    python3 ${CURDIR}/run_siamese.py \
      --task_name=${TASK_NAME} \
      --do_eval=true \
      --data_dir=${GLUE_DIR}/${TASK_NAME} \
      --vocab_file=${BERT_DIR}/vocab.txt \
      --bert_config_file=${BERT_DIR}/bert_config.json \
      --init_checkpoint=${INIT_CKPT} \
      --max_seq_length=64 \
      --output_parent_dir=${OUTPUT_PARENT_DIR} \
      ${@:2}
elif [[ $1 == "predict" ]];then
    echo "predict"
    python3 ${CURDIR}/run_siamese.py \
      --task_name=${TASK_NAME} \
      --do_predict=true \
      --data_dir=${GLUE_DIR}/${TASK_NAME} \
      --vocab_file=${BERT_DIR}/vocab.txt \
      --bert_config_file=${BERT_DIR}/bert_config.json \
      --init_checkpoint=${INIT_CKPT} \
      --max_seq_length=64 \
      --output_parent_dir=${OUTPUT_PARENT_DIR} \
      ${@:2}

    python3 scripts/eval_stsb.py \
        --glue_path="GLUE_DIR" \
        --task_name=${TASK_NAME} \
        --pred_path=${OUTPUT_PARENT_DIR}/${EXP_NAME}/test_results.tsv \
        --is_test=1

elif [[ $1 == "predict_pool" ]];then
    echo "predict_dev"
    python3 ${CURDIR}/run_siamese.py \
      --task_name=${TASK_NAME} \
      --do_predict=true \
      --data_dir=${GLUE_DIR}/${TASK_NAME} \
      --vocab_file=${BERT_DIR}/vocab.txt \
      --bert_config_file=${BERT_DIR}/bert_config.json \
      --max_seq_length=64 \
      --output_parent_dir=${OUTPUT_PARENT_DIR} \
      --predict_pool=True \
      ${@:2}

elif [[ $1 == "predict_dev" ]];then
    echo "predict_dev"
    python3 ${CURDIR}/run_siamese.py \
      --task_name=${TASK_NAME} \
      --do_predict=true \
      --data_dir=${GLUE_DIR}/${TASK_NAME} \
      --vocab_file=${BERT_DIR}/vocab.txt \
      --bert_config_file=${BERT_DIR}/bert_config.json \
      --max_seq_length=64 \
      --output_parent_dir=${OUTPUT_PARENT_DIR} \
      --do_predict_on_dev=True \
      --predict_pool=True \
      ${@:2}
  
elif [[ $1 == "predict_full" ]];then
    echo "predict_dev"
    python3 ${CURDIR}/run_siamese.py \
      --task_name=${TASK_NAME} \
      --do_predict=true \
      --data_dir=${GLUE_DIR}/${TASK_NAME} \
      --vocab_file=${BERT_DIR}/vocab.txt \
      --bert_config_file=${BERT_DIR}/bert_config.json \
      --max_seq_length=64 \
      --output_parent_dir=${OUTPUT_PARENT_DIR} \
      --do_predict_on_full=True \
      --predict_pool=True \
      ${@:2}

elif [[ $1 == "do_senteval" ]];then
    echo "do_senteval"
    python3 ${CURDIR}/run_siamese.py \
      --task_name=${TASK_NAME} \
      --do_senteval=true \
      --data_dir=${GLUE_DIR}/${TASK_NAME} \
      --vocab_file=${BERT_DIR}/vocab.txt \
      --bert_config_file=${BERT_DIR}/bert_config.json \
      --init_checkpoint=${INIT_CKPT} \
      --max_seq_length=64 \
      --output_parent_dir=${OUTPUT_PARENT_DIR} \
      ${@:2}

else
    echo "NotImplementedError"
fi





