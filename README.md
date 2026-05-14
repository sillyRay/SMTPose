dataurl：
https://drive.google.com/file/d/1WAsg6ZWSMzA9IcePw9pEI-sy3g8T8opi/view?usp=drive_link
https://drive.google.com/file/d/1Y784e6gFjUppWhIJVF-sF5Aa9bD5rz1Z/view?usp=drive_link

datastruct：
checkpoint/submit/best_cpn/model.bin
data/data_3d_h36m.npz

train Command：
python run_h36m.py --cfg ./cfg/submit/t_7_dim_4.yaml --gpu 0
python run_h36m.py --cfg ./cfg/submit/t_27_dim_4.yaml --gpu 0

test Command：
python run_h36m.py --cfg ./cfg/submit/t_7_dim_4.yaml --eval --checkpoint ./checkpoint/submit/best_cpn/model.bin --gpu 0 --eval_n_frames 7 --eval_n_views 4 
--eval_batch_size 50 --n_frames 7
python run_h36m.py --cfg ./cfg/submit/t_27_dim_4.yaml --eval --checkpoint ./checkpoint/submit/best_gt/model.bin --gpu 0 --eval_n_frames 1 7 --eval_n_view
s 4 --eval_batch_size 500 --n_frames 27
