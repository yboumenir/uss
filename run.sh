#!/bin/bash

mkdir -p "./checkpoints"

MODEL1_CKPT="./checkpoints/ss_model=resunet30,querynet=at_soft,data=full,devices=8,step=1000000.ckpt"
MODEL1_YAML="./checkpoints/ss_model=resunet30,querynet=at_soft,data=full.yaml"
MODEL2_CKPT="./checkpoints/ss_model=resunet30,querynet=emb,data=balanced,devices=1,steps=1000000.ckpt"

URL_MODEL1_CKPT="https://huggingface.co/RSNuts/Universal_Source_Separation/resolve/main/uss_material/ss_model%3Dresunet30%2Cquerynet%3Dat_soft%2Cdata%3Dfull%2Cdevices%3D8%2Cstep%3D1000000.ckpt"
URL_MODEL1_YAML="https://huggingface.co/RSNuts/Universal_Source_Separation/resolve/main/uss_material/ss_model%3Dresunet30%2Cquerynet%3Dat_soft%2Cdata%3Dfull.yaml?download=true"
URL_MODEL2_CKPT="https://huggingface.co/RSNuts/Universal_Source_Separation/resolve/main/uss_material/ss_model%3Dresunet30%2Cquerynet%3Demb%2Cdata%3Dbalanced%2Cdevices%3D1%2Csteps%3D1000000.ckpt"

# ---------------------------
#  AUDIO INPUT HANDLING
# ---------------------------

DEFAULT_AUDIO="./resources/harry_potter.flac"
INPUT_FILE="$1"

if [ -z "$INPUT_FILE" ]; then
    echo "No input file provided. Using default: $DEFAULT_AUDIO"
    AUDIO_FILE="$DEFAULT_AUDIO"
else
    BASENAME=$(basename "$INPUT_FILE")
    NAME="${BASENAME%.*}"
    EXT="${BASENAME##*.}"
    EXT_LOWER=$(echo "$EXT" | tr '[:upper:]' '[:lower:]')

    OUTFILE="/tmp/${NAME}.flac"
    mkdir -p /tmp

    # Audio-only formats we accept directly:
    case "$EXT_LOWER" in
        flac)
            echo "Copying FLAC file to /tmp..."
            cp "$INPUT_FILE" "$OUTFILE"
            AUDIO_FILE="$OUTFILE"
            ;;
        mp3|wav|ogg|m4a|aac)
            echo "Converting audio file $INPUT_FILE → $OUTFILE"
            ffmpeg -y -i "$INPUT_FILE" -ac 2 -sample_fmt s16 -ar 44100 "$OUTFILE"
            AUDIO_FILE="$OUTFILE"
            ;;
        *)
            # Assume any unknown extension *might* be video — FFmpeg will confirm
            echo "Attempting to extract audio from video file $INPUT_FILE → $OUTFILE"
            ffmpeg -y -i "$INPUT_FILE" -vn -ac 2 -sample_fmt s16 -ar 44100 "$OUTFILE"

            if [ $? -ne 0 ]; then
                echo "Error: FFmpeg cannot process this file type: $INPUT_FILE"
                exit 1
            fi

            AUDIO_FILE="$OUTFILE"
            ;;
    esac
fi

# ---------------------------
#  CHECKPOINT DOWNLOAD
# ---------------------------

if [ ! -f "$MODEL1_CKPT" ]; then
    wget -O "$MODEL1_CKPT" "$URL_MODEL1_CKPT"
fi

if [ ! -f "$MODEL1_YAML" ]; then
    wget -O "$MODEL1_YAML" "$URL_MODEL1_YAML"
fi

if [ ! -f "$MODEL2_CKPT" ]; then
    wget -O "$MODEL2_CKPT" "$URL_MODEL2_CKPT"
fi

# ---------------------------
#  RUN INFERENCE
# ---------------------------

PYTHONPATH=~/source/uss CUDA_VISIBLE_DEVICES=0 python uss/inference.py \
    --audio_path="$AUDIO_FILE" \
    --levels 1 2 3 \
    --config_yaml="$MODEL1_YAML" \
    --checkpoint_path="$MODEL1_CKPT"
