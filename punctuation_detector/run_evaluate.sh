THRESHOLD=$1
SEN_SIZE=90
SHIFT=30
FEAT_PATH="/media/mohamed/24E8F168E8F1389E/UbuntuFiles/sen_segmentation/feat"
SUFFIX="manual_test"

python align_seq.py tmp_dev_target.txt tmp_dev_hyp.txt $SEN_SIZE $SHIFT $FEAT_PATH $SUFFIX

python get_bounds.py $THRESHOLD > dev_hyp_tmp.txt
python evaluate_dev.py dev_hyp_tmp.txt

#rm -rf dev_hyp_tmp.txt tmp_dev_target.txt tmp_dev_hyp.txt
