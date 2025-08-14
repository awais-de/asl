import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data_loading.aslg_pc12_dataset import ASLGPC12Dataset
from src.utils.metrics import compute_bleu, compute_rouge, compute_exact_match
from src.utils.helpers import (
    get_latest_run_id,
    load_run_metadata,
    save_run_metadata,
    Artifact,
    add_artifact_to_metadata,
)
from src.utils.artifact_names import ASLG_PC12_TOKENIZED_PT


class TextToGlossModel(pl.LightningModule):
    def __init__(self, model_name='t5-small', lr=3e-4, tokenized_path=None):
        super().__init__()
        self.save_hyperparameters()

        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.lr = lr
        self.tokenized_path = tokenized_path

        self.val_preds = []
        self.val_labels = []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        val_loss = outputs.loss
        self.log('val_loss', val_loss, prog_bar=True)

        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=50
        )
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        self.val_preds.extend(preds)
        self.val_labels.extend(labels)

        return val_loss

    def on_validation_epoch_end(self):
        if self.val_preds and self.val_labels:
            bleu_score = compute_bleu(self.val_preds, self.val_labels)
            rouge_scores = compute_rouge(self.val_preds, self.val_labels)
            exact_match = compute_exact_match(self.val_preds, self.val_labels)

            self.log('val_bleu', bleu_score, prog_bar=True)
            self.log('val_exact_match', exact_match, prog_bar=True)
            for k, v in rouge_scores.items():
                self.log(f'val_rouge_{k}', v, prog_bar=False)

        self.val_preds = []
        self.val_labels = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        dataset = ASLGPC12Dataset(data_path=None, tokenized_path=self.tokenized_path)
        return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        dataset = ASLGPC12Dataset(data_path=None, tokenized_path=self.tokenized_path)
        return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


def main(args=None):
    parser = argparse.ArgumentParser(description="Train T5 Text-to-Gloss model on ASLGPC12")
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs to use, 0 for CPU")
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--tokenized_path', type=str, help="Path to the tokenized dataset (.pt)")
    parser.add_argument('--checkpoint_dir', type=str, default='artifacts', help="Checkpoint save directory")
    parsed_args = parser.parse_args(args)

    # Resolve tokenized_path
    if parsed_args.tokenized_path:
        tokenized_path = Path(parsed_args.tokenized_path)
        run_id = get_latest_run_id()
        run_metadata = load_run_metadata(run_id)
    else:
        run_id = get_latest_run_id()
        run_metadata = load_run_metadata(run_id)
        tokenized_path_str = run_metadata["artifacts"].get(ASLG_PC12_TOKENIZED_PT)
        if not tokenized_path_str:
            raise RuntimeError("Tokenized dataset path not found in run metadata and not provided via --tokenized_path")
        tokenized_path = Path(tokenized_path_str)

    checkpoint_dir = Path(parsed_args.checkpoint_dir)

    model = TextToGlossModel(
        model_name='t5-small',
        lr=parsed_args.lr,
        tokenized_path=tokenized_path,
    )

    checkpoint_paths = []

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename='text_to_gloss-epoch={epoch:02d}-val_loss={val_loss:.4f}',
        save_top_k=3,
        mode='min'
    )

    orig_save_ckpt = checkpoint_callback._save_checkpoint

    def _tracking_save_checkpoint(*args, **kwargs):
        result = orig_save_ckpt(*args, **kwargs)
        latest_ckpt = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
        if latest_ckpt and latest_ckpt not in checkpoint_paths:
            checkpoint_paths.append(latest_ckpt)
        return result

    checkpoint_callback._save_checkpoint = _tracking_save_checkpoint

    if parsed_args.gpus == 0:
        accelerator = 'cpu'
        devices = 1
    else:
        accelerator = 'auto'
        devices = parsed_args.gpus

    trainer = pl.Trainer(
        max_epochs=parsed_args.max_epochs,
        accelerator='mps' if torch.backends.mps.is_available() else accelerator,
        devices=1 if torch.backends.mps.is_available() else devices,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        default_root_dir="logs"
    )

    trainer.fit(model)

    if checkpoint_paths:
        checkpoint_info = []
        for path_str in checkpoint_paths:
            path = Path(path_str)
            val_loss_str = path.name.split('val_loss=')[-1].replace('.ckpt', '')
            try:
                val_loss = float(val_loss_str)
            except ValueError:
                val_loss = None
            checkpoint_info.append({
                "filepath": str(path),
                "val_loss": val_loss
            })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = checkpoint_dir / f"checkpoints_info_{timestamp}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(checkpoint_info, f, indent=4)
        print(f"[INFO] Saved checkpoint info to: {output_file}")

        ckpt_info_artifact = Artifact(
            name=output_file.name,
            type="json",
            run_id=run_id,
            use_run_folder=False,
        )
        add_artifact_to_metadata(run_metadata, ckpt_info_artifact)
        save_run_metadata(run_id, run_metadata)


if __name__ == "__main__":
    main()
