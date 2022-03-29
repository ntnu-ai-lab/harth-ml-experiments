#! /bin/bash

function usage() {
    cat <<USAGE

    Usage: $0 [-c config_path] [-d dataset_path]

    Options:
	-c, --config_path:   Path to config file (e.g. traditional_machine_learning/params/xgb_50hz/config.yml)
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
    -c | --config_path)
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

approach_path=$(echo $model | cut -d/ -f1)
echo "Start training using model: "$model ...
config_path=$(realpath $model)

if [ "$dataset_path" == "" ]; then
	python $approach_path/train.py -p $config_path
else
	dataset_path=$(realpath $dataset_path)
	python $approach_path/train.py -p $config_path -d $dataset_path
fi;
