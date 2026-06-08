CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --dataset_db "SevenScenesRetrieval(scene='{}', split='train')" \
    --dataset_q "SevenScenesRetrieval(scene='{}', split='test')" \
    --dataset_relpose "SevenScenesRelpose(scene='{}', pair_id={}, resolution={})" \
    --scene "chess" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --dataset_db "SevenScenesRetrieval(scene='{}', split='train')" \
    --dataset_q "SevenScenesRetrieval(scene='{}', split='test')" \
    --dataset_relpose "SevenScenesRelpose(scene='{}', pair_id={}, resolution={})" \
    --scene "fire" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --dataset_db "SevenScenesRetrieval(scene='{}', split='train')" \
    --dataset_q "SevenScenesRetrieval(scene='{}', split='test')" \
    --dataset_relpose "SevenScenesRelpose(scene='{}', pair_id={}, resolution={})" \
    --scene "heads" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --dataset_db "SevenScenesRetrieval(scene='{}', split='train')" \
    --dataset_q "SevenScenesRetrieval(scene='{}', split='test')" \
    --dataset_relpose "SevenScenesRelpose(scene='{}', pair_id={}, resolution={})" \
    --scene "office" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --dataset_db "SevenScenesRetrieval(scene='{}', split='train')" \
    --dataset_q "SevenScenesRetrieval(scene='{}', split='test')" \
    --dataset_relpose "SevenScenesRelpose(scene='{}', pair_id={}, resolution={})" \
    --scene "pumpkin" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --dataset_db "SevenScenesRetrieval(scene='{}', split='train')" \
    --dataset_q "SevenScenesRetrieval(scene='{}', split='test')" \
    --dataset_relpose "SevenScenesRelpose(scene='{}', pair_id={}, resolution={})" \
    --scene "redkitchen" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --dataset_db "SevenScenesRetrieval(scene='{}', split='train')" \
    --dataset_q "SevenScenesRetrieval(scene='{}', split='test')" \
    --dataset_relpose "SevenScenesRelpose(scene='{}', pair_id={}, resolution={})" \
    --scene "stairs" \
    --topk 10 \
