set -e

PRETRAINED_CLIPMOL="/share/home/xiabing/ssc_jobs/MOLDE/best_model/fineture/epoch_12.pth"
PRETRAINED_CLIPMOL_XL="/share/home/xiabing/ssc_jobs/MOLDE/best_model/fineture/epoch_40.pth"
SMILES_VOCAB="/share/home/xiabing/ssc_jobs/MOLDE/data/static_tokenizer_smiles.txt"
INCHI_VOCAB="/share/home/xiabing/ssc_jobs/MOLDE/data/static_tokenizer_inchi.txt"
GPU=1
WHICH="${1:-all}"

# -----------------------------------------------------------------------------
# 1) ClipMol + ALLCCS  (val RMSE 4.4345, ckpt best-epoch=88-v2)
#    data: 48-adduct one-hot ALLCCS split
# -----------------------------------------------------------------------------
if [[ "$WHICH" == "all" || "$WHICH" == "allccs" ]]; then
    echo ">>> [1/4] ClipMol + ALLCCS"
    python finetune_fusion_allccs.py \
        --input_type smiles \
        --task_type regression \
        --dataset_name ALLCCS \
        --measure_name ccs \
        --data_root ./data/processed_data_allccs \
        --pretrained_ckpt "$PRETRAINED_CLIPMOL" \
        --smiles_vocab "$SMILES_VOCAB" \
        --inchi_vocab "$INCHI_VOCAB" \
        --batch_size 64 \
        --epochs 100 \
        --metric rmse \
        --hidden_dim 768 \
        --num_layers 2 \
        --learning_rate 0.0002962285256680979 \
        --dropout 0.020888055667686534 \
        --use_bn \
        --gpu $GPU

    # -------------------------------------------------------------------------
    # 2) ClipMol-XL + ALLCCS  (val RMSE 4.3188, G_R2 0.9944 / G_RMSE 4.3820,
    #    G_MedARE 0.9443; ckpt best-epoch=98-v14, logs/ALLCCS/version_290)
    #    source row: result/ALLCCS_inchi2026_xlresults.csv
    #    data: same 48-adduct one-hot split as [1/4]
    # -------------------------------------------------------------------------
    echo ">>> [2/4] ClipMol-XL + ALLCCS"
    python finetune_fusion_allccs_class.py \
        --input_type smiles \
        --task_type regression \
        --dataset_name ALLCCS \
        --measure_name ccs \
        --data_root ./data/processed_data_allccs \
        --pretrained_ckpt "$PRETRAINED_CLIPMOL_XL" \
        --smiles_vocab "$SMILES_VOCAB" \
        --inchi_vocab "$INCHI_VOCAB" \
        --batch_size 64 \
        --epochs 100 \
        --metric rmse \
        --hidden_dim 768 \
        --num_layers 2 \
        --learning_rate 0.0004163534959912372 \
        --dropout 0.02897834387912429 \
        --use_bn \
        --gpu $GPU
fi

# -----------------------------------------------------------------------------
# 3) ClipMol + METLIN  (val RMSE 4.9320, logs_fusion/METLIN/version_117)
# -----------------------------------------------------------------------------
if [[ "$WHICH" == "all" || "$WHICH" == "metlin" ]]; then
    echo ">>> [3/4] ClipMol + METLIN"
    python finetune_fusion_metlin.py \
        --input_type smiles \
        --task_type regression \
        --dataset_name METLIN \
        --measure_name ccs \
        --data_root ./data/processed_data2026 \
        --pretrained_ckpt "$PRETRAINED_CLIPMOL" \
        --smiles_vocab "$SMILES_VOCAB" \
        --inchi_vocab "$INCHI_VOCAB" \
        --batch_size 32 \
        --epochs 100 \
        --metric rmse \
        --hidden_dim 512 \
        --num_layers 1 \
        --learning_rate 3.92288721155035e-05 \
        --dropout 0.4050416600337977 \
        --use_bn \
        --gpu $GPU

    # -------------------------------------------------------------------------
    # 4) ClipMol-XL + METLIN  (val RMSE 4.8982, logs/METLIN/version_500)
    #    Original run used ./data/processed_data42 (no longer present);
    #    processed_data2026 is the METLIN split that remains in data/.
    # -------------------------------------------------------------------------
    echo ">>> [4/4] ClipMol-XL + METLIN"
    python finetune_fusion_metlin.py \
        --input_type smiles \
        --task_type regression \
        --dataset_name METLIN \
        --measure_name ccs \
        --data_root ./data/processed_data2026 \
        --pretrained_ckpt "$PRETRAINED_CLIPMOL_XL" \
        --smiles_vocab "$SMILES_VOCAB" \
        --inchi_vocab "$INCHI_VOCAB" \
        --batch_size 32 \
        --epochs 100 \
        --metric rmse \
        --hidden_dim 512 \
        --num_layers 1 \
        --learning_rate 3.92288721155035e-05 \
        --dropout 0.4050416600337977 \
        --use_bn \
        --gpu $GPU
fi

# =============================================================================
# Superseded ALLCCS + ClipMol-XL config (not used above):
#   result/ALLCCS_inchi2026_xl_results_class.csv, val RMSE 4.2892,
#   ckpt best-epoch=99-v24
#   --batch_size 32 --hidden_dim 768 --num_layers 3
#   --learning_rate 0.0002898779299723908 --dropout 0.05501240167286835
# =============================================================================

echo "Done."
