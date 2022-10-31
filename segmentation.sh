for fold in 0 1 2 3 4; #0 1 2 3 4 #solo se tienen 2 clases (original y val) por lo tanto solo se usan 0, 1 fold
do
    echo "fold ${fold}"
    ##Train segmentation model
    # python3 -m kuzushiji.segment.main \
    #    --output-dir _runs/segment-fold${fold} --fold ${fold} \
    #    --model fasterrcnn_resnet50_fpn #--model fasterrcnn_resnet50_fpn #fasterrcnn_resnet152_fpn 
    #    --pretrained True
    #    #--model fasterrcnn_resnet152_fpn
    # Out-of-fold predictions
    # python3 -m kuzushiji.segment.main \
    #     --output-dir _runs/segment-fold${fold}/imgs --fold ${fold} \
    #     --model fasterrcnn_resnet50_fpn \
    #     --resume _runs/segment-fold${fold}/model_best.pth \
    #     --test-only 
    # #     ##--model fasterrcnn_resnet152_fpn \
    # Test predictions
    python3 -m kuzushiji.segment.main \
        --output-dir _runs/segment-fold${fold}/imgs --fold ${fold} \
        --model fasterrcnn_resnet50_fpn \
        --resume _runs/segment-fold${fold}/model_best.pth \
        --submission
        ##--model fasterrcnn_resnet152_fpn \
done

#TODO: separar original dataset en train test