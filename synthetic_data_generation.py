import os
import time
import pandas as pd
from datasets import Dataset
from unsloth.dataprep import SyntheticDataKit
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig

# --- Unsloth and vLLM Setup ---
generator = SyntheticDataKit.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
)

generator.prepare_qa_generation(
    output_folder="qna_dataset",
    temperature=0.7,
    top_p=0.95,
    overlap=64,
    max_generation_tokens=512,

    # Configure for LMStudio (OpenAI-compatible endpoint)
    llm_provider="openai",
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="not-required",
    model_name="local-model"
)

# --- Document Parsing and Chunking for multiple local directories ---
# The original notebook processed a single online HTML file.
# We need to process local markdown files from specified directories.

input_dirs = ["odforce_scrapMD", "houdini_docs_mkdown"]
all_source_files = []

# Create the output directory for ingested text if it doesn't exist
ingested_text_output_dir = "qna_dataset/output_text"
os.makedirs(ingested_text_output_dir, exist_ok=True)

print(f"Starting ingestion from directories: {input_dirs}")
for doc_dir in input_dirs:
    dir_source_files = []
    for root, _, files in os.walk(doc_dir):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                target_file_name = os.path.join(ingested_text_output_dir, f"{doc_dir.replace('/', '_')}_{os.path.splitext(file)[0]}.txt")
                try:
                    with open(full_path, 'r', encoding='utf-8') as infile, open(target_file_name, 'w', encoding='utf-8') as outfile:
                        outfile.write(infile.read())
                    print(f"Processed and copied: {full_path} to {target_file_name}")
                    dir_source_files.append(target_file_name)
                except Exception as e:
                    print(f"Error processing file {full_path}: {e}")
    all_source_files.extend(dir_source_files)

print(f"Collected source files for chunking: {len(all_source_files)}")

all_chunked_files = []
if not all_source_files:
    print("No source files found to chunk. Exiting generation part.")
else:
    print("Chunking data...")
    for source_file_path in all_source_files:
        try:
            filenames = generator.chunk_data(source_file_path)
            print(f"Chunked {source_file_path}: {len(filenames)} chunks - {filenames[:3]}")
            all_chunked_files.extend(filenames)
        except Exception as e:
            print(f"Error chunking file {source_file_path}: {e}")

if not all_chunked_files:
    print("No chunked files to process for Q&A generation. Exiting.")
else:
    print(f"Total chunked files to process: {len(all_chunked_files)}")
    files_to_process_for_qa = all_chunked_files[:min(5, len(all_chunked_files))]
    print(f"Will generate Q&A for {len(files_to_process_for_qa)} chunk(s): {files_to_process_for_qa}")

    generated_qa_pair_files = []
    for chunk_file_path in files_to_process_for_qa:
        base_chunk_name = os.path.basename(chunk_file_path)
        output_qa_json_filename = os.path.join(
            "qna_dataset/generated",
            f"{os.path.splitext(base_chunk_name)[0]}_qa_pairs.json"
        )
        os.makedirs(os.path.dirname(output_qa_json_filename), exist_ok=True)

        config_file_path = "qna_dataset/synthetic_data_kit_config.yaml"

        qa_command = (
            f"synthetic-data-kit -c {config_file_path} "
            f"create {chunk_file_path} "
            f"--num-pairs 25 --type qa "
            f"--output-file {output_qa_json_filename}"
        )
        print(f"Executing: {qa_command}")
        return_code = os.system(qa_command)
        if return_code == 0:
            print(f"Content saved to {output_qa_json_filename}")
            generated_qa_pair_files.append(output_qa_json_filename)
        else:
            print(f"Error generating QA for {chunk_file_path}. Return code: {return_code}")
        time.sleep(2)

    # --- Convert to Finetuning Format ---
    final_ft_files = []
    if not generated_qa_pair_files:
        print("No QA pair files were generated. Skipping conversion to finetuning format.")
    else:
        print("Converting QA pairs to finetuning format...")
        for qa_json_file in generated_qa_pair_files:
            base_qa_json_name = os.path.basename(qa_json_file)
            output_ft_json_filename = os.path.join(
                "qna_dataset/final",
                f"{os.path.splitext(base_qa_json_name)[0]}_ft.json"
            )
            os.makedirs(os.path.dirname(output_ft_json_filename), exist_ok=True)

            save_as_command = (
                f"synthetic-data-kit -c {config_file_path} "
                f"save-as {qa_json_file} -f ft "
                f"--output-file {output_ft_json_filename}"
            )
            print(f"Executing: {save_as_command}")
            return_code = os.system(save_as_command)
            if return_code == 0:
                print(f"Converted to ft format and saved to {output_ft_json_filename}")
                final_ft_files.append(output_ft_json_filename)
            else:
                print(f"Error converting {qa_json_file} to FT format. Return code: {return_code}")

    if not final_ft_files:
        print("No finetuning files available. Cannot proceed with dataset loading and training.")
        dataset = None
    else:
        print("Loading formatted data for finetuning...")
        try:
            conversations = pd.concat([
                pd.read_json(name) for name in final_ft_files
            ]).reset_index(drop=True)
            dataset = Dataset.from_pandas(conversations)
            print(f"Dataset loaded with {len(dataset)} entries.")
            if len(dataset) > 0:
                print("First entry:", dataset[0])
                print("Last entry:", dataset[-1])
            else:
                print("Dataset is empty.")
        except Exception as e:
            print(f"Error loading final FT files into dataset: {e}")
            dataset = None

generator.cleanup()
print("vLLM server cleanup called.")

if dataset and len(dataset) > 0:
    print("Starting fine-tuning process...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B",
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched=True,)
    if len(dataset) > 0:
      print("Formatted dataset first entry:", dataset[0])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="qna_dataset/training_output",
        ),
    )

    print("Starting SFTTrainer training...")
    trainer_stats = trainer.train()
    print("Training finished.")
    print(f"Trainer stats: {trainer_stats.metrics}")

    print("Running inference example...")
    messages_infer = [
        {"role": "user", "content": "What is Houdini?"},
    ]
    inputs_infer = tokenizer.apply_chat_template(
        messages_infer,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(input_ids=inputs_infer, streamer=text_streamer,
                       max_new_tokens=256, temperature=0.1)
    print("\nInference example finished.")

    print("Saving LoRA model...")
    model.save_pretrained("qna_dataset/lora_model")
    tokenizer.save_pretrained("qna_dataset/lora_model")
    print("LoRA model saved to qna_dataset/lora_model")

else:
    print("Dataset is empty or not loaded. Skipping training, inference, and model saving.")

print("Script finished.") 