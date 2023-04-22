NUM_GPUS=$1
export PORT=9487
export MASTER_PORT=9487

set -e
start=`date +%s`
>&2 echo "---------------------" 
>&2 echo "cascade_original.json" 
>&2 echo "---------------------" 
echo "---------------------" 
echo "cascade_original.json" 
echo "---------------------" 

# cascade_original.json
bash tools/dist_test.sh configs/mva2023/cascade_rcnn_r50_fpn_40e_coco_nwd_finetune.py final/cascade_rcnn_r50_fpn_40e_coco_nwd_finetune/latest.pth $NUM_GPUS --format-only --eval-options jsonfile_prefix=cascade_original

>&2 echo "---------------------" 
>&2 echo "intern_h_public_nosahi.json" 
>&2 echo "---------------------" 
echo "---------------------" 
echo "intern_h_public_nosahi.json" 
echo "---------------------" 
# intern_h_public_nosahi.json
bash tools/dist_test.sh configs/mva2023/cascade_mask_internimage_h_fpn_40e_nwd_finetune.py final/internimage_h_nwd/latest.pth $NUM_GPUS --format-only --eval-options jsonfile_prefix=intern_h_public_nosahi

>&2 echo "---------------------" 
>&2 echo "intern_xl_public_nosahi_randflip.json" 
>&2 echo "---------------------" 
echo "---------------------" 
echo "intern_xl_public_nosahi_randflip.json" 
echo "---------------------" 
# intern_xl_public_nosahi_randflip.json
bash tools/dist_test.sh configs/mva2023/cascade_mask_internimage_xl_fpn_40e_nwd_finetune.py final/internimage_xl_nwd/latest.pth $NUM_GPUS --format-only --eval-options jsonfile_prefix=intern_xl_public_nosahi_randflip

>&2 echo "---------------------" 
>&2 echo "intern_h_public_nosahi_randflip.json" 
>&2 echo "---------------------" 
echo "---------------------" 
echo "intern_h_public_nosahi_randflip.json" 
echo "---------------------" 
# intern_h_public_nosahi_randflip.json
bash tools/dist_test.sh configs/mva2023/cascade_mask_internimage_h_fpn_40e_nwd_finetune_tta_randflip.py final/internimage_h_nwd/latest.pth $NUM_GPUS --format-only --eval-options jsonfile_prefix=intern_h_public_nosahi_randflip

>&2 echo "---------------------" 
>&2 echo "centernet_slicing_01.json" 
>&2 echo "---------------------" 
echo "---------------------" 
echo "centernet_slicing_01.json" 
echo "---------------------" 
# centernet_slicing_01.json
mpirun --allow-run-as-root -np $NUM_GPUS python tools/sahi_evaluation_ompi.py configs/mva2023_baseline/centernet_resnet18_140e_coco_inference.py \
			final/baseline_centernet/latest.pth \
			data/mva2023_sod4bird_private_test/images/ \
			data/mva2023_sod4bird_private_test/annotations/private_test_coco_empty_ann.json \
			--out-file-name centernet_slicing_01.json \
			--score-threshold 0.1 \
			--crop-size 512 \
			--overlap-ratio 0.2

>&2 echo "---------------------" 
>&2 echo "results_interImage.json" 
>&2 echo "---------------------" 
echo "---------------------" 
echo "results_interImage.json" 
echo "---------------------" 
#results_interImage.json
mpirun --allow-run-as-root -np $NUM_GPUS python tools/sahi_evaluation_ompi.py configs/mva2023/cascade_mask_internimage_xl_fpn_finetune.py \
			final/internimage_xl_no_nwd/latest.pth \
		    data/mva2023_sod4bird_private_test/images/ \
		    data/mva2023_sod4bird_private_test/annotations/private_test_coco_empty_ann.json \
		    --out-file-name results_interImage.json

>&2 echo "---------------------" 
>&2 echo "cascade_nwd_paste_howard_0604.json" 
>&2 echo "---------------------" 
echo "---------------------" 
echo "cascade_nwd_paste_howard_0604.json" 
echo "---------------------" 
# cascade_nwd_paste_howard_0604.json
mpirun --allow-run-as-root -np $NUM_GPUS python tools/sahi_evaluation_ompi.py configs/cascade_rcnn_mva2023/cascade_rcnn_r50_fpn_20e_coco_finetune_nwd_paste.py \
				final/cascade_nwd_paste_howard/latest.pth \
		    data/mva2023_sod4bird_private_test/images/ \
		    data/mva2023_sod4bird_private_test/annotations/private_test_coco_empty_ann.json \
		    --out-file-name cascade_nwd_paste_howard_0604.json

>&2 echo "---------------------" 
>&2 echo "cascade_rcnn_sticker_61_2.json" 
>&2 echo "---------------------" 
echo "---------------------" 
echo "cascade_rcnn_sticker_61_2.json" 
echo "---------------------" 
# cascade_rcnn_sticker_61_2.json
mpirun --allow-run-as-root -np $NUM_GPUS python tools/sahi_evaluation_ompi.py configs/cascade_rcnn_mva2023/cascade_rcnn_r50_fpn_40e_coco_finetune_sticker.py \
			final/cascade_rcnn_r50_fpn_40e_coco_finetune_sticker/latest.pth \
		    data/mva2023_sod4bird_private_test/images/ \
		    data/mva2023_sod4bird_private_test/annotations/private_test_coco_empty_ann.json \
		    --out-file-name cascade_rcnn_sticker_61_2.json

>&2 echo "---------------------" 
>&2 echo "cascade_mask_internimage_xl_fpn_20e_nwd_finetune_merged_train.json" 
>&2 echo "---------------------" 
echo "---------------------" 
echo "cascade_mask_internimage_xl_fpn_20e_nwd_finetune_merged_train.json" 
echo "---------------------" 
# cascade_mask_internimage_xl_fpn_20e_nwd_finetune_merged_train.json
mpirun --allow-run-as-root -np $NUM_GPUS python tools/sahi_evaluation_ompi.py configs/mva2023/cascade_mask_internimage_xl_fpn_20e_nwd_finetune_merged_train.py \
			final/cascade_mask_internimage_xl_fpn_20e_nwd_finetune_merged_train/latest.pth \
		    data/mva2023_sod4bird_private_test/images/ \
		    data/mva2023_sod4bird_private_test/annotations/private_test_coco_empty_ann.json \
		    --out-file-name cascade_mask_internimage_xl_fpn_20e_nwd_finetune_merged_train.json

>&2 echo "---------------------" 
>&2 echo "cascade_mask_internimage_h_fpn_40e_nwd_finetune.json" 
>&2 echo "---------------------" 
echo "---------------------" 
echo "cascade_mask_internimage_h_fpn_40e_nwd_finetune.json" 
echo "---------------------" 
# cascade_mask_internimage_h_fpn_40e_nwd_finetune.json
mpirun --allow-run-as-root -np $NUM_GPUS python tools/sahi_evaluation_ompi.py configs/mva2023/cascade_mask_internimage_h_fpn_40e_nwd_finetune.py \
			final/internimage_h_nwd/latest.pth \
		    data/mva2023_sod4bird_private_test/images/ \
		    data/mva2023_sod4bird_private_test/annotations/private_test_coco_empty_ann.json \
			--crop-size 512 \
		    --out-file-name cascade_mask_internimage_h_fpn_40e_nwd_finetune.json



>&2 echo "---------------------" 
>&2 echo "-------Ensemble------" 
>&2 echo "---------------------" 
echo "---------------------" 
echo "-------Ensemble------" 
echo "---------------------" 

mv cascade_original.bbox.json ensemble/cascade_original.json
mv intern_h_public_nosahi.bbox.json ensemble/intern_h_public_nosahi.json
mv intern_xl_public_nosahi_randflip.bbox.json ensemble/intern_xl_public_nosahi_randflip.json
mv intern_h_public_nosahi_randflip.bbox.json ensemble/intern_h_public_nosahi_randflip.json
mv centernet_slicing_01.json ensemble/centernet_slicing_01.json
mv results_interImage.json ensemble/results_interImage.json
mv cascade_nwd_paste_howard_0604.json ensemble/cascade_nwd_paste_howard_0604.json
mv cascade_rcnn_sticker_61_2.json ensemble/cascade_rcnn_sticker_61_2.json
mv cascade_mask_internimage_xl_fpn_20e_nwd_finetune_merged_train.json ensemble/cascade_mask_internimage_xl_fpn_20e_nwd_finetune_merged_train.json
mv cascade_mask_internimage_h_fpn_40e_nwd_finetune.json ensemble/cascade_mask_internimage_h_fpn_40e_nwd_finetune.json

pushd ensemble
python ensemble.py ../data/mva2023_sod4bird_private_test/annotations/private_test_coco_empty_ann.json
zip results_team1.zip results.json
popd
cp ensemble/results_team1.zip ./

>&2 echo "---------------------" 
>&2 echo "!!!!!  FINISH  !!!!!" 
>&2 echo "---------------------" 
echo "---------------------" 
echo "!!!!!  FINISH  !!!!!" 
echo "---------------------" 

end=`date +%s`

echo "Elapsed Time: $(($end-$start)) seconds"