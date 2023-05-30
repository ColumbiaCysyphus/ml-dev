# Spring 2023: Fine-Tuning BERT

## 0. Fine-Tuning Template
This folder provides BERT fine-tuning templates for Masked Language Modeling (MLM) and sequential classification using TensorFlow and PyTorch. It also includes a template script for extracting sentences from policy documents in PDF format.

## 1. Fine-Tuning BERT
Include three approaches to fine-tuning:
- Original size model: All parameters are trainable.
- Reduced size model: The model's capacity is reduced by decreasing the number of attention heads, hidden layers, and hidden size.
- Model with frozen layers: Gradually unfreeze encoder layers starting from the top.

## 2. Model Evaluation
The fine-tuned models are evaluated by measuring the cosine similarity of embeddings within recommendation texts, within non-recommendation texts, and between these two classes. The results are compared to the original BERT model.

## 3. Extraction Workflow
The workflow from the previous summer (2022) is replicated to identify sentences whose embeddings are closest to the cluster centroids of recommendation texts.

## 4. Final Results
This folder presents the final results of the fine-tuning process, including tables and presentations.