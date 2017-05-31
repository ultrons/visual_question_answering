python main.py \
       --phase=train \
       --cnn_model=vgg16 \
       --cnn_model_file=./tfmodels/vgg16.tfmodel \
       --load_cnn_model \
       --attention=gru \
       --init_embed_with_glove  2>&1 | tee log_$$

        
       
