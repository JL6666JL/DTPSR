CUDA_VISIBLE_DEVICES="1,2,3" accelerate launch train.py  \
--output_dir /data2/jianglei/HoliSDiP/experiments/test  \
--enable_xformers_memory_efficient_attention \
--train_batch_size=4 \
--gradient_accumulation_steps=2  