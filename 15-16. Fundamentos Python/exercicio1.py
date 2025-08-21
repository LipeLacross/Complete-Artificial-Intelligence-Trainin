# exercicio1.py
# Função que calcula a amplitude de uma lista

def amplitude(lista):
    return max(lista) - min(lista)

if __name__ == "__main__":
    valores = [3, 7, 10, 2, 15]
    print("Lista:", valores)
    print("Amplitude:", amplitude(valores))
