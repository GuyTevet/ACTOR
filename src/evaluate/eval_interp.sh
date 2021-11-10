## declare an array variable
interp_type=("nearest" "linear" "cubic")
interp_ratio=(2 4 6 8 10 12 14 16)

cpt_path=pretrained_models/humanact12/checkpoint_5000.pth.tar

python -m src.evaluate.evaluate_cvae $cpt_path --batch_size 64 --niter 1

## now loop through the above array
for t in "${interp_type[@]}"
do
  for r in "${interp_ratio[@]}"
  do
    python -m src.evaluate.evaluate_cvae $cpt_path --batch_size 64 --niter 1  --interp_type $t --interp_ratio $r
  done
done

# You can access them using echo "${arr[0]}", "${arr[1]}" also