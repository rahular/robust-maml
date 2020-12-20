#!/bin/bash

#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --output=logs/test.out
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanrtx
#SBATCH --mem=60GB

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

run_ner() {
    run "ady" $1
    run "chy" $1
    run "lo" $1
    run "qu" $1
    run "xmf" $1
    run "av" $1
    run "ik" $1
    run "nah" $1
    run "te" $1
    run "zh-classical" $1
    run "chr" $1
    run "kn" $1
    run "nv" $1
    # run "th" $1
    run "zh-yue" $1
}

run_qa() {
    run "finnish" $1
    run "korean" $1
#    run "russian" $1
#    run "indonesian" $1
#    run "telugu" $1
    run "bengali" $1
#    run "swahili" $1
#    run "english" $1
    run "arabic" $1
}

################# POS #################
# only head
# run_pos "2020-10-23_02-32-04-TXEV"  # MTL
# run_pos "2020-10-29_17-50-03-1O48"  # ML-uniform
# run_pos "2020-10-29_17-36-46-BZ0A"  # ML-minmax
# run_pos "2020-11-08_22-20-23-9LDM"  # ML-minmax-alcgd
# run_pos "2020-11-09_12-14-27-T6YB"  # ML-constrained
# run_pos "2020-11-10_11-53-41-IFP8"  # ML-constrained-alcgd

# warm start
# run_pos "2020-11-09_01-55-52-OKKT"  # MTL
# run_pos "2020-11-10_15-22-00-R2FT"  # ML-uniform
# run_pos "2020-11-10_15-22-07-DJP9"  # ML-minmax
# run_pos "2020-11-10_16-19-45-N66W"  # ML-minmax-alcgd
# run_pos "2020-11-10_15-23-49-95T3"  # ML-constrained
# run_pos "2020-11-10_16-19-44-3WUX"  # ML-constrained-alcgd

# run_pos "2020-12-04_02-01-51-FLYJ"  # ML-minmax-alcgd (small classifier)

################# NER #################
# only head
# run_ner "2020-11-12_09-13-09-9UT4"  # MTL
# run_ner "2020-11-11_21-11-21-ZYOM"  # ML-uniform
# run_ner "2020-11-12_01-13-16-6ELR"  # ML-minmax
# run_ner "2020-11-12_01-53-22-84UL"  # ML-minmax-alcgd
# run_ner "2020-11-12_01-53-23-K33C"  # ML-constrained
# run_ner "2020-11-12_02-53-40-7PGM"  # ML-constrained-alcgd

# only head with randomized data
# run_ner "2020-11-17_13-55-22-6CG4"  # MTL
# run_ner "2020-11-17_23-30-54-9T82"  # ML-uniform
# run_ner "2020-11-18_11-26-53-RNJZ"  # ML-minmax
# run_ner "2020-11-18_13-57-01-429O"  # ML-minmax-alcgd
# run_ner "2020-11-18_19-51-02-38EZ"  # ML-constrained
# run_ner "2020-11-18_20-46-05-YICS"  # ML-constrained-alcgd

# only head with 50-15 split
# run_ner "2020-12-04_16-10-10-8NAR"    # MTL
# run_ner "2020-12-04_17-23-09-8HDW"    # ML-minmax-alcgd
# run_ner "2020-12-06_23-29-56-ONL2"    # ML-uniform
# run_ner "2020-12-06_23-30-06-NX9Q"    # ML-constrained
# run_ner "2020-12-06_23-29-56-4M2N"    # ML-minmax
# run_ner "2020-12-06_23-30-25-8J1F"    # ML-minmax-alcgd
# run_ner "2020-12-06_23-31-13-AGS6"    # ML-constrained-alcgd

################# QA #################
# only head
# run_qa "2020-12-01_12-11-53-1BRT"  # baseline (train all and test all)
run_qa "2020-12-15_11-56-22-E9PV"   # baseline (finetuned encoder + all langs)
# run_qa "2020-12-07_01-04-52-X3SY"  # MTL (finetuned encoder)
