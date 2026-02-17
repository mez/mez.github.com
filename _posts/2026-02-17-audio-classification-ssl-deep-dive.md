---
layout: post
title: "Audio Classification: A Weekend of Experiments (FAAAAHH!)"
description: "A weekend experiment exploring CNNs, Transformers, and SSL techniques on environmental sound classification."
tags: [Deep Learning, Audio, Self-Supervised Learning]
---

I went down an audio classification rabbit hole. What started as "how hard can ESC-50 be?" turned into a weekend of building CNNs, Transformers, and finally understanding why everyone keeps talking about self-supervised learning.

This post documents everything I tried, what worked, what didn't, and the lessons learned.

---

## The Dataset: ESC-50

The [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) contains:
- **2,000 audio clips** (5 seconds each)
- **50 classes** of environmental sounds
- **5 folds** for cross-validation

With only 40 samples per class, this is a challenging dataset that rewards good representations over brute-force memorization.

---

## The Supervised Baseline

### ResNet-34 from Scratch

Classic approach: convert audio to mel spectrograms and treat them as images for a CNN.

```python
batch_size = 32
learning_rate = 1e-3
optimizer = AdamW(weight_decay=0.01)
scheduler = OneCycleLR
```

**Result: 80.5% accuracy**

### Test-Time Augmentation

I noticed validation performance was inconsistent, so I added TTA, averaging predictions across multiple augmented versions of each test sample.

```python
def predict_with_tta(model, spectrogram, n_augments=5):
    predictions = []
    for _ in range(n_augments):
        aug_spec = apply_augmentation(spectrogram)
        pred = model(aug_spec)
        predictions.append(pred)
    return torch.stack(predictions).mean(dim=0)
```

**Result: 83.5% accuracy** (+3% from TTA alone!)

---

## Transfer Learning Detour

Before diving into SSL, I tried transfer learning with ImageNet-pretrained EfficientNet-B0.

**Result: 81.5% accuracy**

This performed *worse* than my from-scratch ResNet. ImageNet features (edges, textures, objects) don't transfer perfectly to spectrograms. Good to know.

---

## Self-Supervised Learning Experiments

This is where things got interesting. I wanted to understand *why* and *how* SSL works, not just use pretrained models.

### SimCLR (Contrastive Learning)

The idea: learn representations by pulling augmented views of the same audio together while pushing different audios apart.

```
Audio → Mel Spectrogram → ResNet Encoder → Projection Head → Contrastive Loss
```

**Key components:**
- **NT-Xent Loss:** Normalized Temperature-scaled Cross Entropy
- **Projection Head:** 512 → 512 → 128 (discarded after pretraining)
- **Temperature:** 0.5

**Result: 82.5% accuracy**

But here's the catch: the contrastive task itself reached **98% accuracy during pretraining**.

The model was solving a nearly trivial task.

Environmental sounds in ESC-50 are already highly separable in the spectral domain (dog bark vs chainsaw vs rain, etc.). With relatively mild augmentations, the identity of each sound barely changes. That means the encoder can solve the contrastive objective using coarse cues (energy bands, temporal envelope) instead of learning robust, invariant representations.

In other words, the model learned:

*"these sounds are different"*

instead of:

*"these two transformed versions of the same sound are meaningfully the same"*

**Lesson:** Contrastive learning only works when the task is hard enough. You need augmentations and negatives that force the model to learn invariances, not shortcuts.

Even with this limitation, the learned representation still reached 82.5% accuracy, which is competitive with standard supervised CNN baselines on ESC-50.

### Masked Prediction (BEATs-style)

Inspired by BERT and BEATs: hide patches of the spectrogram and predict discrete tokens.

> **Note:** This was a simplified, educational implementation to understand the BEATs framework, not a full reproduction. The real BEATs includes iterative tokenizer refinement, larger models (90M+ params), and training on millions of samples. My goal was to grasp the core concepts: patch embeddings, masking, and discrete token prediction.

```
Spectrogram → Patch Embedding → Encoder → Predict Masked Tokens
```

**The approach:**
1. Split spectrogram into patches (16×16)
2. Randomly mask 40% of patches
3. Predict the cluster ID of masked patches

**The tokenizer (codebook):**
- K-means clustering on spectrogram patches
- Each patch gets assigned to nearest centroid
- Model predicts which cluster the masked patch belongs to

```python
# Tokenizing a patch
distances = ||patch - codebook||²  # Distance to each centroid
token = argmin(distances)          # Closest centroid = token ID
```

**Result: 74.5% accuracy** (with CNN encoder)

Underperformed the supervised baseline. Hmm.

### Transformer-BEATs

I tried a full Transformer encoder instead of CNN.

**Architecture:**
- 6 Transformer layers
- 256 embedding dimension
- 8 attention heads
- ~5M parameters

**Result: 40% accuracy** (nearly random, FAAAHH!!)

This was humbling. Transformers need *massive* amounts of data. With only 1,600 training samples, it couldn't learn meaningful patterns. CNNs have stronger inductive biases that help with limited data.

---

## The BEATs Revelation

After all my experiments, I tried Microsoft's BEATs model, pretrained on AudioSet (2 million+ clips).

I tested two approaches:

**Frozen encoder:** Only train a new classifier head on top.
**Result: 94.50% accuracy**

**Fine-tuned with differential learning rates:**

```python
param_groups = [
    {'params': model.encoder.parameters(), 'lr': 1e-5},    # Slow for pretrained
    {'params': model.classifier.parameters(), 'lr': 1e-3}  # Fast for new head
]
```

**Result: 95.25% accuracy** 🎯

The frozen approach gets you 94.5% with minimal compute. The AudioSet pretraining is *that* good. Fine-tuning squeezes out another 0.75%, which matters if you're chasing leaderboards but honestly? Frozen is probably fine for most use cases.

Even though I knew I probably wouldn't make a dent against the pretrained model, I had to try it.

---

## Things I Had to Figure Out

**What are "tokens" in audio?** In BEATs, tokens are discrete IDs representing audio patterns. K-means clustering groups similar spectrogram patches, and each patch gets a cluster ID. It's like creating a vocabulary of audio building blocks.

**Why iterate on the tokenizer?** Clever trick: the tokenizer and model improve each other. Each iteration creates more meaningful tokens, forcing the model to learn finer distinctions.

**Why two learning rates?** Differential rates balance preservation and adaptation. Without this, you either destroy pretrained features (high LR) or the classifier never converges (low LR).

---

## Final Results

<table style="width:100%; border-collapse: collapse; margin: 1.5rem 0;">
  <thead>
    <tr style="background: #2a2a2a; color: #fff;">
      <th style="padding: 12px 15px; text-align: left;">Method</th>
      <th style="padding: 12px 15px; text-align: center;">Accuracy</th>
      <th style="padding: 12px 15px; text-align: left;">Approach</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background: #e8f5e9;">
      <td style="padding: 12px 15px; font-weight: bold;">BEATs (fine-tuned)</td>
      <td style="padding: 12px 15px; text-align: center; font-weight: bold;">95.25%</td>
      <td style="padding: 12px 15px;">Pretrained + differential LR</td>
    </tr>
    <tr style="background: #f1f8e9;">
      <td style="padding: 12px 15px;">BEATs (frozen)</td>
      <td style="padding: 12px 15px; text-align: center;">94.50%</td>
      <td style="padding: 12px 15px;">Pretrained, classifier only</td>
    </tr>
    <tr style="background: #fff;">
      <td style="padding: 12px 15px;">ResNet-34 + TTA</td>
      <td style="padding: 12px 15px; text-align: center;">83.50%</td>
      <td style="padding: 12px 15px;">Supervised baseline</td>
    </tr>
    <tr style="background: #f5f5f5;">
      <td style="padding: 12px 15px;">SimCLR</td>
      <td style="padding: 12px 15px; text-align: center;">82.50%</td>
      <td style="padding: 12px 15px;">Contrastive SSL</td>
    </tr>
    <tr style="background: #fff;">
      <td style="padding: 12px 15px;">EfficientNet-B0</td>
      <td style="padding: 12px 15px; text-align: center;">81.50%</td>
      <td style="padding: 12px 15px;">ImageNet transfer</td>
    </tr>
    <tr style="background: #f5f5f5;">
      <td style="padding: 12px 15px;">CNN-BEATs</td>
      <td style="padding: 12px 15px; text-align: center;">74.50%</td>
      <td style="padding: 12px 15px;">Masked prediction</td>
    </tr>
    <tr style="background: #ffebee;">
      <td style="padding: 12px 15px;">Transformer-BEATs</td>
      <td style="padding: 12px 15px; text-align: center;">40.00%</td>
      <td style="padding: 12px 15px;">Tomfoolery 💀</td>
    </tr>
  </tbody>
</table>

The gap between my from-scratch SSL attempts and the pretrained BEATs tells the whole story.

---

## Takeaways

- **Scale matters enormously for SSL.** My weekend experiments couldn't match 2M-sample pretraining, but they taught me *how* these methods work.
- **Contrastive learning needs hard negatives.** Trivial pretext tasks don't force useful representations.
- **Transformers are data-hungry.** CNNs win on small datasets.
- **TTA is free accuracy.** 3% boost for minimal effort.

This is why I love jumping between domains: contrastive learning from NLP, masked prediction from BERT, spectrogram tricks from signal processing. The dots connect in unexpected ways.

The real takeaway? Running these experiments taught me more than any tutorial ever could. Sometimes you have to build the thing yourself to understand why it works, and why it doesn't.

What surprised me most is that self-supervised learning doesn't automatically produce meaningful representations. It will happily learn shortcuts if the task allows it. Designing the right objective turns out to matter just as much as the model itself.
