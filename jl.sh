CUDA_VISIBLE_DEVICES="0" accelerate launch train.py  \
--output_dir experiments/test  \
--enable_xformers_memory_efficient_attention \
--train_batch_size=4 \
--gradient_accumulation_steps=2 