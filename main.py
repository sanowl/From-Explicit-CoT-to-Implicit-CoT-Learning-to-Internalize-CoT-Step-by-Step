from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from datasets import load_dataset
import math
import random
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

class CoTDataset(Dataset):
 def __init__(self, problems: List[str], solutions: List[str], cot_steps: List[str], tokenizer: GPT2Tokenizer, max_length: int = 512) -> None:
  self.problems: List[str] = problems
  self.solutions: List[str] = solutions
  self.cot_steps: List[str] = cot_steps
  self.tokenizer: GPT2Tokenizer = tokenizer
  self.max_length: int = max_length

 def __len__(self) -> int:
  return len(self.problems)

 def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
  problem: str = self.problems[idx]
  solution: str = self.solutions[idx]
  cot: str = self.cot_steps[idx]

  input_ids: List[int] = self.tokenizer.encode(problem, max_length=self.max_length, truncation=True)
  cot_ids: List[int] = self.tokenizer.encode(cot, max_length=self.max_length, truncation=True)
  output_ids: List[int] = self.tokenizer.encode(solution, max_length=self.max_length, truncation=True)

  return {
   'input_ids': torch.tensor(input_ids, dtype=torch.long),
   'cot_ids': torch.tensor(cot_ids, dtype=torch.long),
   'output_ids': torch.tensor(output_ids, dtype=torch.long)
  }

class StepwiseInternalization:
 def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, learning_rate: float, delta: int, lambda_smooth: float, total_steps: int) -> None:
  self.model: GPT2LMHeadModel = model
  self.tokenizer: GPT2Tokenizer = tokenizer 
  self.optimizer: optim.AdamW = optim.AdamW(model.parameters(), lr=learning_rate)
  self.delta: int = delta
  self.lambda_smooth: float = lambda_smooth
  self.total_steps: int = total_steps
  self.scaler: GradScaler = GradScaler()

 def removal_schedule(self, step: int) -> int:
  return math.floor((self.delta * step) / self.total_steps)

 def removal_smoothing(self) -> int:
  o: int = 0
  while random.random() < math.exp(-self.lambda_smooth):
   o += 1
  return o

 def train_step(self, batch: Dict[str, torch.Tensor], step: int) -> float:
  self.optimizer.zero_grad()

  input_ids: torch.Tensor = batch['input_ids'].cuda()
  cot_ids: torch.Tensor = batch['cot_ids'].cuda()
  output_ids: torch.Tensor = batch['output_ids'].cuda()

  tokens_to_remove: int = self.removal_schedule(step) + self.removal_smoothing()
  cot_truncated: torch.Tensor = cot_ids[:, tokens_to_remove:]

  full_input: torch.Tensor = torch.cat([input_ids, cot_truncated, output_ids], dim=1)
  labels: torch.Tensor = full_input.clone()
  labels[:, :input_ids.size(1)] = -100  

  with autocast():
   outputs: Any = self.model(full_input, labels=labels)
   loss: torch.Tensor = outputs.loss

  self.scaler.scale(loss).backward()
  self.scaler.step(self.optimizer)
  self.scaler.update()

  return loss.item()

 def train(self, dataloader: DataLoader, val_dataloader: DataLoader, num_epochs: int) -> None:
  self.model.train()
  global_step: int = 0
  for epoch in range(num_epochs):
   epoch_loss: float = 0.0
   for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
    loss: float = self.train_step(batch, global_step)
    epoch_loss += loss

    if self.removal_schedule(global_step) > self.removal_schedule(global_step - 1):
     self.optimizer = optim.AdamW(self.model.parameters(), lr=self.optimizer.param_groups[0]['lr'])

    global_step += 1

   print(f"Epoch {epoch+1} average loss: {epoch_loss / len(dataloader)}")

   # Evaluate on validation set
   val_accuracy: float = self.evaluate(val_dataloader)
   print(f"Validation accuracy: {val_accuracy}")

 def evaluate(self, dataloader: DataLoader) -> float:
  self.model.eval()
  correct: int = 0
  total: int = 0
  with torch.no_grad():
   for batch in dataloader:
    input_ids: torch.Tensor = batch['input_ids'].cuda()
    output_ids: torch.Tensor = batch['output_ids'].cuda()

    generated: torch.Tensor = self.model.generate(input_ids, max_length=input_ids.size(1) + output_ids.size(1))
    generated = generated[:, input_ids.size(1):]

    correct += (generated == output_ids).all(dim=1).sum().item()
    total += input_ids.size(0)

  return correct / total

 def inference(self, problem: str) -> str:
  self.model.eval()
  with torch.no_grad():
   input_ids: torch.Tensor = self.tokenizer.encode(problem, return_tensors='pt').cuda()
   output: torch.Tensor = self.model.generate(input_ids, max_length=100, num_return_sequences=1)
   return self.tokenizer.decode(output[0], skip_special_tokens=True)

def create_multiplication_dataset(n_digits: int, num_examples: int) -> Tuple[List[str], List[str], List[str]]:
 problems: List[str] = []
 solutions: List[str] = []
 cot_steps: List[str] = []
 for _ in range(num_examples):
  a: int = random.randint(10**(n_digits-1), 10**n_digits - 1)
  b: int = random.randint(10**(n_digits-1), 10**n_digits - 1)
  problem: str = f"{a} * {b}"
  solution: str = str(a * b)
  
  cot: str = f"Let's solve {a} * {b} step by step:\n"
  partial_products: List[int] = []
  for i, digit in enumerate(str(b)[::-1]):
   partial: int = a * int(digit)
   cot += f"{i+1}) {a} * {digit} = {partial}\n"
   partial_products.append(partial * (10**i))
  cot += f"Now, let's sum up all partial products:\n"
  cot += " + ".join(map(str, partial_products)) + f" = {solution}"

  problems.append(problem)
  solutions.append(solution)
  cot_steps.append(cot)

 return problems, solutions, cot_steps

def load_gsm8k_dataset() -> Tuple[List[str], List[str], List[str]]:
 dataset: Any = load_dataset("gsm8k", "main")
 problems: List[str] = dataset["train"]["question"] + dataset["test"]["question"]
 solutions: List[str] = dataset["train"]["answer"] + dataset["test"]["answer"]
 cot_steps: List[str] = [solution.split("####")[0].strip() for solution in solutions]
 solutions = [solution.split("####")[1].strip() for solution in solutions]
 return problems, solutions, cot_steps

def main() -> None:
 # Load larger model
 model_name: str = 'gpt2-medium'  # or 'gpt2-large' for even better performance
 tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(model_name)
 model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_name).cuda()

 # Prepare datasets
 datasets: Dict[str, Tuple[List[str], List[str], List[str]]] = {}
 for size in [4, 5, 7, 9]:
  problems, solutions, cot_steps = create_multiplication_dataset(size, 808000 // 4)  # 808k total examples
  datasets[f'{size}x{size}'] = (problems, solutions, cot_steps)

 # Add GSM8K dataset
 gsm8k_problems, gsm8k_solutions, gsm8k_cot_steps = load_gsm8k_dataset()
 datasets['gsm8k'] = (gsm8k_problems, gsm8k_solutions, gsm8k_cot_steps)

 # Create dataloaders
 dataloaders: Dict[str, Dict[str, DataLoader]] = {}
 for name, (problems, solutions, cot_steps) in datasets.items():
  dataset: CoTDataset = CoTDataset(problems, solutions, cot_steps, tokenizer)
  dataloaders[name] = {
   'train': DataLoader(dataset[:int(0.8*len(dataset))], batch_size=32, shuffle=True),
   'val': DataLoader(dataset[int(0.8*len(dataset)):], batch_size=32, shuffle=False)
  }

 configs: List[Dict[str, Any]] = [
  {'lr': 1e-5, 'delta': 4, 'lambda_smooth': 2},
  {'lr': 5e-5, 'delta': 8, 'lambda_smooth': 4},
  {'lr': 1e-4, 'delta': 16, 'lambda_smooth': 8},
 ]

 for name, loaders in dataloaders.items():
  print(f"Training on {name} dataset")
  for config in configs:
   print(f"Using hyperparameters: {config}")
   trainer: StepwiseInternalization = StepwiseInternalization(
    model=model,
    tokenizer=tokenizer,
    learning_rate=config['lr'],
    delta=config['delta'],
    lambda_smooth=config['lambda_smooth'],
    total_steps=len(loaders['train']) * 200  # 200 epochs
   )
   trainer.train(loaders['train'], loaders['val'], num_epochs=200)

   # Evaluate on all datasets
   for eval_name, eval_loaders in dataloaders.items():
    accuracy: float = trainer.evaluate(eval_loaders['val'])
    print(f"Accuracy on {eval_name} dataset: {accuracy}")

  # Inference example
  if name.endswith('x'):
   size: int = int(name.split('x')[0])
   a: int = random.randint(10**(size-1), 10**size - 1)
   b: int = random.randint(10**(size-1), 10**size - 1)
   test_problem: str = f"{a} * {b}"
  else:
   test_problem = random.choice(gsm8k_problems)
  result: str = trainer.inference(test_problem)
  print(f"Problem: {test_problem}")
  print(f"Model's solution: {result}")

if __name__ == "__main__":
 main()