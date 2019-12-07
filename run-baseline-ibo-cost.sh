#!/bin/bash

#ROOT_DIR=`cd ..&&pwd`
ROOT_DIR=./
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR
#DATA_DIR=$ROOT_DIR'/datadir'
DATA_DIR=$ROOT_DIR'data'
OUT_DIR=$ROOT_DIR'results'

#module load cuda/9.0
module load intel/17.0.4 python3/3.6.3
module load cuda/10.0 cudnn/7.6.2 nccl/2.4.7
#module load python/3.6.2
#source $ROOT_DIR/../env/bin/activate
#module load tensorflow/1.12.0-python3.6-gcc5

export CUDA_VISIBLE_DEVICES=0
#CACHE_PATH=/tmp/nv-$DATE
#mkdir $CACHE_PATH
#export CUDA_CACHE_PATH=$CACHE_PATH

DATASETS=( conll2002 conll2002)
EXPS=( ned)
TRAINS=( ned.train)
TESTS=( ned.testa)
DEVS=( ned.testb)
VOCAB_SIZES=( 20000 35000 35000 35000 )
STRATEGIES=( Random Uncertainty Diversity )

POLICY_PATH=$ROOT_DIR/policy/en_bi_policy.h5

EMBEDING_FILE=$DATA_DIR/"emb/twelve.table4.multiCCA.window_5+iter_10+size_40+threads_16.normalized"

index=$1
QUERY_STRATEGY=$1
DATASET_NAME=${DATASETS[$index]}
EXP_NAME=${EXPS[$index]}
TRAIN_FILE=${TRAINS[$index]}
DEV_FILE=${DEVS[$index]}
TESTS_FILE=${TESTS[$index]}
TEXT_DATA_DIR=$DATA_DIR'/ner/'$DATASET_NAME

echo ${TRAIN_FILE}
echo ${STRATEGIES[$QUERY_STRATEGY]}
#    for QUERY_STRATEGY in "${STRATEGIES[@]}"; do
        OUTPUT=$OUT_DIR/ibo-${DATASET_NAME}_${EXP_NAME}_${STRATEGIES[$QUERY_STRATEGY]}_${DATE}
        mkdir -p $OUTPUT
        echo "RUN BASELINE ON dataset ${DATASET_NAME} experiment name ${EXP_NAME} query strategy ${STRATEGIES[$QUERY_STRATEGY]}"
        cd $SRC_PATH && python3 AL-crf-baselines-cost.py --root_dir $ROOT_DIR --dataset_name $DATASET_NAME \
            --train_file $TEXT_DATA_DIR/$TRAIN_FILE --dev_file $TEXT_DATA_DIR/$DEV_FILE \
            --test_file $TEXT_DATA_DIR/$TESTS_FILE \
            --policy_path $POLICY_PATH\
            --word_vec_file $EMBEDING_FILE --timesteps 20 \
            --output $OUTPUT --label_data_size 200 --annotation_budget 3 \
            --initial_training_size 0 --vocab_size 20000 --query_strategy ${STRATEGIES[$QUERY_STRATEGY]} --ibo_scheme
#    done

rm -r -f $CACHE_PATH
