rem Window script to run Ml model Pix2PixTensorFlow 2.0 developed by amrutha
set INPUT_DIR="D:\Zhaochen\ML_training_data_augmentation\splittedData_demo"
set CODEDEVELOPMENTMODE=False
set MODE=train
set IMDATASETTYPE=grainBoundary
set IMDATASETMATHTYPE=logarithmic
set SUMMARY_FREQ=500
set SEED=10
set MAX_STEPS=2000000
set MAX_EPOCHS=5
set LOG_HIST_WEIGHTS_FREQUENCY=20
set PROGRESS_FREQ=100
set TRACE_FREQ=100
set DISPLAY_FREQ=250
set SAVE_FREQ=50
set LR=1e-4
set LRDESCRIMINATOR=1e-4
set IMTYPE=.png
set BETA1=0.99
set L1_WEIGHT=0.0
set BATCH_SIZE=1
set GAN_WEIGHT=100.0
set L2_WEIGHT=0.1
set OUTPUT_GROUP_DIR=MeltPoolSim
set FILENAMEPREFIX=grainBoundary-demorun-zhaochenSystem-4_DLR_1e-4_GAN_1.0_L2_100.0_L1_0
set CUDA_DEVICE=0
set COMMENT="1st_grainBoundary-demorun-zhaochenSystem-4_DLR_1e-4_GAN_100.0_L2_0.1_L1_0.0"


echo  pix2pixTensorFlow2.0.py --input_dir %INPUT_DIR% --mode %MODE% --seed %SEED% ^
                        --codeDevelopmentMode %CODEDEVELOPMENTMODE% --imtype %IMTYPE% --max_steps %MAX_STEPS% --max_epochs %MAX_EPOCHS% ^
                               --summary_freq %SUMMARY_FREQ% --log_hist_weights_frequency %LOG_HIST_WEIGHTS_FREQUENCY% ^
                                --imDataSetType %IMDATASETTYPE% --imDatasetMathType %IMDATASETMATHTYPE% --progress_freq %PROGRESS_FREQ% --trace_freq %TRACE_FREQ% ^
                                --display_freq %DISPLAY_FREQ% --save_freq %SAVE_FREQ% ^
                                --lr %LR% --lrDescriminator %LRDESCRIMINATOR% ^
                              --l1_weight %L1_WEIGHT% --gan_weight %GAN_WEIGHT% --l2_weight %L2_WEIGHT% ^
                               --output_group_dir %OUTPUT_GROUP_DIR% --fileNamePrefix %FILENAMEPREFIX% ^
                               --comment %COMMENT% --beta1 %BETA1% --batch_size %BATCH_SIZE% --cuda_device %CUDA_DEVICE%

python pix2pixTensorFlow2.0.py --input_dir %INPUT_DIR% --mode %MODE% --seed %SEED% ^
                        --codeDevelopmentMode %CODEDEVELOPMENTMODE% --max_steps %MAX_STEPS% --max_epochs %MAX_EPOCHS% ^
                               --summary_freq %SUMMARY_FREQ% --log_hist_weights_frequency %LOG_HIST_WEIGHTS_FREQUENCY% ^
                                --imDataSetType %IMDATASETTYPE% --imDatasetMathType %IMDATASETMATHTYPE% --progress_freq %PROGRESS_FREQ% --trace_freq %TRACE_FREQ% ^
                                --display_freq %DISPLAY_FREQ% --save_freq %SAVE_FREQ% ^
                                --lr %LR% --lrDescriminator %LRDESCRIMINATOR% ^
                              --l1_weight %L1_WEIGHT% --gan_weight %GAN_WEIGHT% --l2_weight %L2_WEIGHT% ^
                               --output_group_dir %OUTPUT_GROUP_DIR% --fileNamePrefix %FILENAMEPREFIX% ^
                               --comment %COMMENT% --beta1 %BETA1% --batch_size %BATCH_SIZE% --cuda_device %CUDA_DEVICE%

