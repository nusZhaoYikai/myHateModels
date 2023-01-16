

CUDA_VISIBLE_DEVICES="0"
python main.py --resume_training --plm hateBERT --use_kg True --data_balance True --batch_size 24 --learning_rate 2e-5 --epochs 20 --seed 1000

echo ""
date
