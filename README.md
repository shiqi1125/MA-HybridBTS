

if you want to train my model, you can run
python main.py --exp-name "CKD" --devices 0 --dataset-folder "dataset/" --batch-size 1 --workers 1 --lr 1e-4 --end-epoch 300 --mode "train"

if you want to evaluate the model, you can run
python main.py --mode test --dataset-folder dataset --exp-name CKD --devices 0
