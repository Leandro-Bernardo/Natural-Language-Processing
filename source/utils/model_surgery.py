import torch
from collections import OrderedDict
import sys

# --- CONFIGURE AQUI ---
# 1. Caminho para o checkpoint antigo (com 'input_layer' e 'layer_1')
OLD_CKPT_PATH = "D:/Natural-Language-Processing/checkpoints/subj_classifier/subj_classifier_v3/wobbly-sweep-587.ckpt"
# 2. Caminho para salvar o novo checkpoint (corrigido)
NEW_CKPT_PATH = "D:/Natural-Language-Processing/checkpoints/subj_classifier/subj_classifier_v3/wobbly-sweep-5872.ckpt"
# -----------------------

# Mapeamento exato com base no seu log de erro
# (Chave Antiga no Checkpoint) -> (Chave Nova no Modelo)
# O seu log indica esta correspondência:
key_map = {
    'model.input_layer.0': 'model.body.0',
    'model.input_layer.1': 'model.body.1',
    # Note que as camadas 2 e 3 (provavelmente ReLU/Dropout)
    # não têm parâmetros, por isso pulamos para a 4
    'model.layer_1.0': 'model.body.4',
    'model.layer_1.1': 'model.body.5'
}

def perform_surgery():
    print(f"Carregando checkpoint antigo de: {OLD_CKPT_PATH}")

    try:
        # Carrega o checkpoint (pode ser um dict do Lightning)
        checkpoint = torch.load(OLD_CKPT_PATH, map_location="cpu")
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em {OLD_CKPT_PATH}")
        print("Por favor, verifique o caminho 'OLD_CKPT_PATH' no script.")
        sys.exit(1)

    # O state_dict real dos pesos está geralmente sob a chave 'state_dict'
    if 'state_dict' not in checkpoint:
        print("ERRO: O checkpoint não parece ser um arquivo do PyTorch Lightning (falta a chave 'state_dict').")
        sys.exit(1)

    old_state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()

    print("Iniciando a cirurgia de renomeação...")

    for old_key, value in old_state_dict.items():
        new_key = old_key
        found_map = False

        # Verifica se a chave precisa ser mapeada
        for map_from, map_to in key_map.items():
            if old_key.startswith(map_from):
                # Ex: 'model.input_layer.0.weight' -> 'model.body.0.weight'
                suffix = old_key[len(map_from):] # Pega o '.weight' ou '.bias'
                new_key = map_to + suffix
                found_map = True
                break

        # Ignora a 'criterion.pos_weight' do checkpoint antigo
        if 'criterion.pos_weight' in old_key:
            print(f"  Ignorando (obsoleto): {old_key}")
            continue

        # Adiciona a chave (antiga ou nova) ao novo dicionário
        new_state_dict[new_key] = value
        if found_map:
            print(f"  Mapeado: {old_key} -> {new_key}")
        # else:
            # print(f"  Mantido: {old_key}") # Descomente para ver as chaves mantidas

    # Substitui o state_dict antigo pelo novo, corrigido
    checkpoint['state_dict'] = new_state_dict

    # Salva o novo checkpoint
    torch.save(checkpoint, NEW_CKPT_PATH)
    print(f"\nCirurgia completa! Novo checkpoint salvo em: {NEW_CKPT_PATH}")

if __name__ == "__main__":
    perform_surgery()