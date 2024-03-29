#!/bin/bash

source activate pos-bert
echo "HOSTNAME: " `hostname`
echo "ALLOCATED GPU: " $CUDA_VISIBLE_DEVICES

run() {
    echo "===== Running $2 on $1 ====="
    python tester.py --test_lang=$1 --model_path=./models/$2
}

run_pos() {
    run "abq_atb" $1
    run "gun_thomas" $1
    run "ta_ttb" $1
    run "tl_ugnayan" $1
    run "bm_crb" $1
    run "id_gsd" $1
    run "te_mtg" $1
    run "wbp_ufal" $1
    run "bxr_bdt" $1
    run "id_pud" $1
    run "th_pud" $1
    run "wo_wtb" $1
    run "eu_bdt" $1
    run "pcm_nsc" $1
    run "tl_trg" $1
    run "yo_ytb" $1
}

run_qa() {
    run "finnish" $1    # split 1
    run "korean" $1     # split 1
    run "bengali" $1    # split 1
    run "arabic" $1     # split 1
    run "russian" $1    # split 2
    run "indonesian" $1 # split 2
    run "telugu" $1     # split 2
    run "swahili" $1    # split 2
}

run_pos $1
# run_qa $1
