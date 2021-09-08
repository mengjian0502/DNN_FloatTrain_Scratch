PYTHON="/home/mengjian/anaconda3/bin/python"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=vgg7
model_ref=vgg7_torch
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD

wd=0.0
lr=0.05

save_path="./save/${model}_${dataset}/${model}_lr${lr}_wd${wd}/"
log_file="${model}_lr${lr}_wd${wd}_train.log"

# pretrained_model="./save/vgg7/vgg7_lr0.05_wd0.0001/model_best.pth.tar"

$PYTHON -W ignore main.py --dataset ${dataset} \
    --data_path ./dataset/ \
    --model ${model} \
    --model_ref ${model_ref} \
    --save_path ${save_path} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr  ${lr} \
    --weight_decay ${wd} \
    --schedule 60 120 \
    --gammas 0.1 0.1 \
    --batch_size ${batch_size} \