#!/bin/bash

run() {
    echo "===== Running $2 on $1 ====="
    python tester.py --test_lang=$1 --model_path=./models/$2
}

run_qa() {
    run "finnish" $1
    run "korean" $1
    run "russian" $1
    run "indonesian" $1
    run "telugu" $1
    run "bengali" $1
    run "swahili" $1
    run "english" $1
    run "arabic" $1
}

run_qa "2020-12-01_12-11-53-1BRT"
