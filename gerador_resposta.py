import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Verificar se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o tokenizer e o modelo fine-tunado
#tokenizer = AutoTokenizer.from_pretrained(r"C:\Projetos\local")
#model = AutoModelForSeq2SeqLM.from_pretrained(r"C:\Projetos\local")
tokenizer = AutoTokenizer.from_pretrained(r"C:\Projetos\local")
model = AutoModelForSeq2SeqLM.from_pretrained(r"C:\Projetos\local")

# Mover o modelo para a GPU (se disponível)
model = model.to(device)

# Função para gerar resumo a partir de um título
def gerar_resumo(titulo):
    # Preparar o input para o modelo (tokenização)
    inputs = tokenizer(f"Title: {titulo}", return_tensors="pt", max_length=512, truncation=True).to(device)
  
    # Gerar a resposta (resumo)
    outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=100, early_stopping=True, no_repeat_ngram_size=2)
    # Decodificar o output (transformar tokens de volta em texto)
    resumo = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resumo

# Exemplo de uso: gerar o resumo baseado no título fornecido
titulo = "Worship with Don Moen [VHS]"
resumo_gerado = gerar_resumo(titulo)

print("Resumo gerado:", resumo_gerado)
