
## **Project 1: Implementing the Transformer from Scratch**

- [ ] **Task 1: Setup the Environment**
  - [ ] Install Python and required libraries: `numpy`, `torch`, `matplotlib`
  - [ ] Create a virtual environment and organize project folders (`data/`, `models/`, `scripts/`, etc.)

- [ ] **Task 2: Build Core Components**
  - [ ] Implement **Scaled Dot-Product Attention**
  - [ ] Implement **Multi-Head Attention** (parallel attention heads)
  - [ ] Implement **Position-wise Feedforward Network**
  - [ ] Implement **Positional Encoding** (sine/cosine pattern)

- [ ] **Task 3: Construct Encoder and Decoder Blocks**
  - [ ] Create one **Encoder Block** (Multi-Head Attention + Feedforward + LayerNorm + Residuals)
  - [ ] Create one **Decoder Block** (Masked Self-Attention + Cross-Attention + Feedforward)
  - [ ] Stack multiple encoder and decoder layers

- [ ] **Task 4: Assemble the Full Transformer**
  - [ ] Add input/output embedding layers
  - [ ] Integrate positional encodings
  - [ ] Add final linear and softmax output layers
  - [ ] Ensure all tensor dimensions match correctly

- [ ] **Task 5: Training and Testing**
  - [ ] Create a toy dataset (e.g., English → English identity translation)
  - [ ] Train for a few epochs and log loss values
  - [ ] Test model predictions with greedy decoding
  - [ ] Save trained weights

- [ ] **Task 6: Visualization and Debugging**
  - [ ] Plot **attention maps** for specific tokens
  - [ ] Inspect embedding and layer weights
  - [ ] Verify gradients and ensure stable training


---

## **Project 2: Using a Pretrained Transformer (Fine-Tuning)**

- [ ] **Task 1: Setup**
  - [ ] Install Hugging Face libraries: `transformers`, `datasets`, `evaluate`
  - [ ] Load a pretrained model (`bert-base-uncased`, `t5-small`, etc.)

- [ ] **Task 2: Data Preparation**
  - [ ] Tokenize text data using model’s tokenizer
  - [ ] Prepare attention masks and labels
  - [ ] Split dataset into train/test

- [ ] **Task 3: Fine-Tuning**
  - [ ] Use `Trainer` API or a custom training loop
  - [ ] Monitor loss and evaluation metrics
  - [ ] Save fine-tuned checkpoints

- [ ] **Task 4: Inference and Evaluation**
  - [ ] Run inference on new text inputs
  - [ ] Decode token IDs to readable outputs
  - [ ] Visualize **attention weights** for interpretability
  - [ ] Compare model performance before and after fine-tuning