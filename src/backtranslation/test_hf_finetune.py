"""

Training script implementing backtranslation for autoformalization.
Author: Jakob Nordhagen

"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch


class BacktranslationTrainer(Trainer):
    def __init__(self, *args, formal_to_informal_model, informal_to_formal_model, tokenizer, **kwargs):
        super().__init__(*args, **kwargs)
        self.formal_to_informal_model = formal_to_informal_model
        self.informal_to_formal_model = informal_to_formal_model
        self.tokenizer = tokenizer  

    def informalize(self, inputs, max_length=512):
        model = self.formal_to_informal_model
        # Tokenize inputs
        inputs_tokenized = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
        
        # Generate outputs
        with torch.no_grad():
            model.eval()
            outputs = model.generate(**inputs_tokenized, max_length=max_length)
        
        # Decode outputs to text
        informal_examples = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return informal_examples

    def training_step(self, model, inputs):
        inputs = self._prepare_inputs(inputs)

        # Generate synthetic IL given FL
        with torch.no_grad():  # Ensure this part doesn't update weights
            self.formal_to_informal_model.eval()
            synthetic_informal = self.informalize(inputs)

        # Backtranslate: IL -> FL
        backtranslated_outputs = self.informal_to_formal_model(**synthetic_informal)
        loss = self.compute_loss(model, backtranslated_outputs)  # Could change to custom method
        return loss


def process_datasets(dataset_path):
    """
    Loads and preprocesses datasets.
    """
    dataset = load_dataset('json', data_files={'train': dataset_path}, split='train')
    theorems = [thm['code'] for lst in dataset['premises'] for thm in lst if thm['code'].startswith('theorem')]
    dataset = Dataset.from_dict({'formal': theorems})

    eval_dataset = load_dataset('hoskinson-center/proofnet')
    return dataset, eval_dataset


def main():
    model_name = 'gpt2'
    formal_to_informal_model = GPT2LMHeadModel.from_pretrained(model_name)
    informal_to_formal_model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    dataset_path = "data/corpus.jsonl"
    train_dataset, eval_dataset = process_datasets(dataset_path)

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )

    trainer = BacktranslationTrainer(
        model=informal_to_formal_model,  # Assuming this is the model you are training
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formal_to_informal_model=formal_to_informal_model,
        informal_to_formal_model=informal_to_formal_model,
        tokenizer=tokenizer
    )

    trainer.train()


if __name__ == '__main__':
    main()