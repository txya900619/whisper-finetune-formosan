import argparse
import os
from pathlib import Path

import torch
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

import evaluate

EVAL_CONFIG_TO_LANGUAGE = {
    "阿美": "ami",
    "賽德克": "sdq",
    "太魯閣": "trv",
    "排灣": "pwn",
}

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            "Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )


def get_text_column_names(column_names):
    if "text" in column_names:
        return "text"
    elif "sentence" in column_names:
        return "sentence"
    elif "normalized_text" in column_names:
        return "normalized_text"
    elif "transcript" in column_names:
        return "transcript"
    elif "transcription" in column_names:
        return "transcription"


whisper_norm = BasicTextNormalizer()


def normalise(batch):
    batch["norm_text"] = get_text(batch).replace(", ", " ")
    return batch


def main(args):
    if args.is_public_repo == False:
        os.system(f"mkdir -p {args.temp_ckpt_folder}")
        ckpt_dir_parent = str(Path(args.ckpt_dir).parent)
        os.system(f"cp {ckpt_dir_parent}/added_tokens.json {ckpt_dir_parent}/normalizer.json \
        {ckpt_dir_parent}/preprocessor_config.json {ckpt_dir_parent}/special_tokens_map.json \
        {ckpt_dir_parent}/tokenizer_config.json {ckpt_dir_parent}/merges.txt \
        {ckpt_dir_parent}/vocab.json {args.ckpt_dir}/config.json  {args.ckpt_dir}/model.safetensors \
        {args.ckpt_dir}/generation_config.json \
        {args.ckpt_dir}/training_args.bin {args.temp_ckpt_folder}")
        model_id = args.temp_ckpt_folder
    else:
        model_id = args.hf_model

    processor = WhisperProcessor.from_pretrained(
        model_id, language=args.language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
    )

    text_column_name = get_text_column_names(dataset.column_names)
    dataset = dataset.map(
        lambda x: {text_column_name: x[text_column_name].replace(" ,", ",")}, num_proc=8
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(normalise, num_proc=2)
    # dataset = dataset.filter(
    #     is_target_text_in_range, input_columns=[text_column_name], num_proc=2
    # )

    predictions = []
    references = []
    norm_predictions = []
    norm_references = []

    def map_to_pred(batch):
        audios = batch["audio"]
        input_features = processor(
            [audio["array"] for audio in audios],
            sampling_rate=audios[0]["sampling_rate"],
            return_tensors="pt",
        ).input_features

        references.extend(batch[text_column_name])
        norm_references.extend(batch["norm_text"])

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features.to("cuda", dtype=torch.bfloat16),
                prompt_ids=processor.get_prompt_ids(
                    EVAL_CONFIG_TO_LANGUAGE[args.config],
                    return_tensors="pt",
                ).cuda(),
                language=args.language,
                task="transcribe",
            )
        transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        predictions.extend(transcriptions)
        norm_predictions.extend(
            [transcription.replace(", ", " ") for transcription in transcriptions]
        )
        return {}

    dataset.map(map_to_pred, batched=True, batch_size=args.batch_size)

    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)
    cer = cer_metric.compute(references=references, predictions=predictions)
    cer = round(100 * cer, 2)
    norm_wer = wer_metric.compute(
        references=norm_references, predictions=norm_predictions
    )
    norm_wer = round(100 * norm_wer, 2)
    norm_cer = cer_metric.compute(
        references=norm_references, predictions=norm_predictions
    )
    norm_cer = round(100 * norm_cer, 2)

    print("\nWER : ", wer)
    print("CER : ", cer)
    print("\nNORMALIZED WER : ", norm_wer)
    print("NORMALIZED CER : ", norm_cer)

    os.system(f"mkdir -p {args.output_dir}")
    dset = args.dataset.replace("/", "_") + "_" + args.config + "_" + args.split
    op_file = args.output_dir + "/" + dset
    if args.is_public_repo:
        op_file = op_file + "_" + args.hf_model.replace("/", "_")
    else:
        op_file = op_file + "_" + args.ckpt_dir.split("/")[-1].replace("/", "_")
    result_file = open(op_file, "w")
    result_file.write("\nWER: " + str(wer) + "\n")
    result_file.write("CER: " + str(cer) + "\n")
    result_file.write("\nNORMALIZED WER: " + str(norm_wer) + "\n")
    result_file.write("NORMALIZED CER: " + str(norm_cer) + "\n\n\n")

    for ref, hyp in zip(norm_references, norm_predictions):
        result_file.write("REF: " + ref + "\n")
        result_file.write("HYP: " + hyp + "\n")
        result_file.write(
            "------------------------------------------------------" + "\n"
        )
    result_file.close()

    if args.is_public_repo == False:
        os.system(f"rm -r {args.temp_ckpt_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--is_public_repo",
        required=False,
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="If the model is available for download on huggingface.",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        required=False,
        default="openai/whisper-tiny",
        help="Huggingface model name. Example: openai/whisper-tiny",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=False,
        default=".",
        help="Folder with the pytorch_model.bin file",
    )
    parser.add_argument(
        "--temp_ckpt_folder",
        type=str,
        required=False,
        default="temp_dir",
        help="Path to create a temporary folder containing the model and related files needed for inference",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=False,
        default="hi",
        help="Two letter language code for the transcription language, e.g. use 'hi' for Hindi. This helps initialize the tokenizer.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="mozilla-foundation/common_voice_11_0",
        help="Dataset from huggingface to evaluate the model on. Example: mozilla-foundation/common_voice_11_0",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="hi",
        help="Config of the dataset. Eg. 'hi' for the Hindi split of Common Voice",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=False,
        default="test",
        help="Split of the dataset. Eg. 'test'",
    )
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="The device to run the pipeline on. -1 for CPU, 0 for the first GPU (default) and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="predictions_dir",
        help="Output directory for the predictions and hypotheses generated.",
    )

    args = parser.parse_args()
    main(args)
