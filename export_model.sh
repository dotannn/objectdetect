export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path configs/faster_rcnn_resnet101_tools.config \
    --trained_checkpoint_prefix training/model.ckpt-6423 \
    --output_directory frcnn_resnet101_tools