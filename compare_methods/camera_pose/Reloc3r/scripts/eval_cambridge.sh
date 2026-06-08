CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --dataset_db "CambridgeRetrieval(scene='{}', split='train')" \
    --dataset_q "CambridgeRetrieval(scene='{}', split='test')" \
    --dataset_relpose "CambridgeRelpose(scene='{}', pair_id={}, resolution={})" \
    --scene "GreatCourt" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --dataset_db "CambridgeRetrieval(scene='{}', split='train')" \
    --dataset_q "CambridgeRetrieval(scene='{}', split='test')" \
    --dataset_relpose "CambridgeRelpose(scene='{}', pair_id={}, resolution={})" \
    --scene "KingsCollege" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --dataset_db "CambridgeRetrieval(scene='{}', split='train')" \
    --dataset_q "CambridgeRetrieval(scene='{}', split='test')" \
    --dataset_relpose "CambridgeRelpose(scene='{}', pair_id={}, resolution={})" \
    --scene "OldHospital" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --dataset_db "CambridgeRetrieval(scene='{}', split='train')" \
    --dataset_q "CambridgeRetrieval(scene='{}', split='test')" \
    --dataset_relpose "CambridgeRelpose(scene='{}', pair_id={}, resolution={})" \
    --scene "ShopFacade" \
    --topk 10 \

CUDA_VISIBLE_DEVICES=0 python eval_visloc.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --dataset_db "CambridgeRetrieval(scene='{}', split='train')" \
    --dataset_q "CambridgeRetrieval(scene='{}', split='test')" \
    --dataset_relpose "CambridgeRelpose(scene='{}', pair_id={}, resolution={})" \
    --scene "StMarysChurch" \
    --topk 10 \
