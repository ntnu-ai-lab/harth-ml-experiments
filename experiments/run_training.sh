#! /bin/bash

function usage() {
    cat <<USAGE

    Usage: $0 [-a approach] [-d dataset_path]

    Options:
	-a, --approach:      Which approach to use (xgb,rf,svm,knn,cnn,lstm,mr_cnn)
        -d, --dataset_path:  Path of the HARTH dataset
USAGE
    exit 1
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

SKIP_VERIFICATION=false
TAG=

while [ "$1" != "" ]; do
    case $1 in
    -a | --approach)
        shift
        model=$1
        ;;
    -d | --dataset_path)
        shift 
	dataset_path=$1
        ;;
    -h | --help)
        usage
        ;;
    *)
        usage
        exit 1
        ;;
    esac
    shift
done

if [ "$model" == "xgb" ]; then
    approach_path=traditional_machine_learning
    config_path=$approach_path/params/xgb_50hz/
elif [ "$model" == "svm" ]; then
    approach_path=traditional_machine_learning
    config_path=$approach_path/params/svm_50hz/
elif [ "$model" == "rf" ]; then
    approach_path=traditional_machine_learning
    config_path=params/rf_50hz/
elif [ "$model" == "knn" ]; then
    approach_path=traditional_machine_learning
    config_path=$approach_path/params/knn_50hz/
elif [ "$model" == "cnn" ]; then
    approach_path=deep_learning
    config_path=$approach_path/params/cnn_50hz/
elif [ "$model" == "lstm" ]; then
    approach_path=deep_learning
    config_path=$approach_path/params/lstm_50hz/
elif [ "$model" == "mr_cnn" ]; then
    approach_path=deep_learning
    config_path=$approach_path/params/inc_cnn_50hz/
else
    echo "Error: Unknown model $model." 
    echo "Allowed models: xgb, svm, rf, knn, cnn, lstm, mr_cnn"
fi

echo "Start training "$model...
config_path=$(realpath $config_path)
dataset_path=$(realpath $dataset_path)
python $approach_path/train.py -p $config_path -d $dataset_path
