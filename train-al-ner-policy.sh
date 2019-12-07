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
EXPS=( eps ned)
TRAINS=( esp.train ned.train)
TESTS=( esp.testa ned.testa)
DEVS=( esp.testb ned.testb)
VOCAB_SIZES=( 20000 35000 35000 35000 )
STRATEGIES=( Random Uncertainty Diversity )

index=$1
policy_idx=$2
DATASET_NAME=${DATASETS[$index]}
EXP_NAME=${EXPS[$index]}
TRAIN_FILE=${TRAINS[$index]}
DEV_FILE=${DEVS[$index]}
TESTS_FILE=${TESTS[$index]}
EMBEDING_FILE=$DATA_DIR"/emb/twelve.table4.multiCCA.window_5+iter_10+size_40+threads_16.normalized"
TEXT_DATA_DIR=$DATA_DIR'/ner/'$DATASET_NAME
vocab=${VOCAB_SIZES[$index]}

OUTPUT=$OUT_DIR/${DATASET_NAME}_${EXP_NAME}_${DATE}
mkdir -p $OUTPUT
echo "DREAM TRANSFER AL POLICY ${POLICY_NAME} with policy path ${POLICY_PATH} on dataset ${DATASET_NAME} experiment name ${EXP_NAME} "
cd $SRC_PATH && python3 AL-crf-simulation.py --root_dir $ROOT_DIR --dataset_name $DATASET_NAME  \
    --train_file $TEXT_DATA_DIR/$TRAIN_FILE --dev_file $TEXT_DATA_DIR/$DEV_FILE \
    --test_file $TEXT_DATA_DIR/$TESTS_FILE \
    --word_vec_file $EMBEDING_FILE --episodes 2 --timesteps 5 \
    --output $OUTPUT --label_data_size 100 --annotation_budget 50 \
    --initial_training_size 0 --vocab_size $vocab --ibo_scheme
#rm -r -f $CACHE_PATH
