import json
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, HfFolder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
# import numpy as np
# import evaluate
 
dataset = None
eval_dataset = None
train_dataset = None
local = True

if local:
    caminho_arquivo = r'C:\Projetos\\'
else:
    caminho_arquivo = r'/content/drive/MyDrive/Notebooks/'

token_hugging_face = "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"
local_arquivo = rf"{caminho_arquivo}local"
input_file = rf'{caminho_arquivo}trn.json'                  # Arquivo de entrada
output_file = rf'{caminho_arquivo}dado_formatado.json'     # Arquivo de saída formatado
nome_modelo = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(nome_modelo)

print("Verificar se a GPU está disponível")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

HfFolder.save_token(token_hugging_face)

def carrega_dataset():
   return load_dataset('json',data_files=rf'{caminho_arquivo}dado_formatado.json')

def carrega_dataset_treino(dataset):
    dataset_split = dataset['train'].train_test_split(test_size=0.2)
    treino_dataset = dataset_split["train"]
    return treino_dataset

def carrega_dataset_teste(dataset):
     dataset_split = dataset['train'].train_test_split(test_size=0.2)
     teste_dataset = dataset_split["test"]
     return teste_dataset
    
    
def gera_dataset_json(input_file, output_file):
    dado_formatado = []

    # Abrir o arquivo de entrada que contém o JSON
    with open(input_file, 'r', encoding='utf-8') as infile:
        # Lê cada linha (cada linha é um JSON separado)
        contador = 0
        print("quantidade de linhas no arquivo")
        for line in infile:
            try:
                contador += 1
                
                # Carregar o JSON da linha
                data = json.loads(line)
                
                # Extrair o título (title) e o conteúdo (content)
                title = data.get('title', '').strip()
                content = data.get('content', '').strip()
                          
                # Ignorar entradas onde título ou conteúdo estão vazios
                if title and content:
                    prompt = f"Title: {title}" 
                    completion = f"Product: {content}"
                    # Adiciona o prompt e completion formatados ao dataset final
                    dado_formatado.append({"input_text": prompt, "output_text": completion})
            
                if contador >= 100000:
                    break
                # if contador >= 50000:
                #     break
            
            except json.JSONDecodeError:
                print(f"Erro ao ler a linha: {line}")
    
    # Salvar o dataset formatado em um novo arquivo JSONL
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in dado_formatado:
            outfile.write(json.dumps(item) + '\n')


    print(f"Dataset formatado salvo em: {output_file}")

def funcao_tokenizar_somente_dataset(dt):
    inputs = funcao_tokenizar(dt)
    tokenizer.save_pretrained(local_arquivo)
    return inputs

def funcao_tokenizar(dt):
    tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
    inputs = tokenizer(dt['input_text'], max_length=512, truncation=True, padding="max_length")
    targets = tokenizer(dt['output_text'], max_length=128, truncation=True, padding="max_length")
    inputs['labels'] = targets['input_ids']
    tokenizer.save_pretrained(local_arquivo)
    return inputs

# Exemplo de uso

gera_dataset_json(input_file, output_file)
dataset = carrega_dataset()
train_dataset = carrega_dataset_treino(dataset)
eval_dataset = carrega_dataset_teste(dataset)

# Aplicar a tokenização ao dataset
tokenized_datasets = dataset.map(funcao_tokenizar_somente_dataset, batched=True)
print("treino dataset")
tokenized_train_dataset = train_dataset.map(funcao_tokenizar, batched=True)
print("teste dataset")
tokenized_eval_dataset = eval_dataset.map(funcao_tokenizar, batched=True)

# Definir argumentos de treinamento
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
)

# Carregar o modelo de linguagem
model = AutoModelForSeq2SeqLM.from_pretrained(nome_modelo)
model.to(device)

# Criar o Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_train_dataset,
    eval_dataset = tokenized_eval_dataset
)

print("Treinar o modelo")
trainer.train()

print("Salvar o modelo fine-tunado")

trainer.save_model(rf"{local_arquivo}")