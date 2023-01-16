epochs_num=20
# run paper bert
python run_kbert_cls.py --use_postag --epochs_num "$epochs_num" --batch_size 48

echo ""
date
