# exercicio2.py
# Função que imprime string na vertical

def imprime_vertical(texto):
    for letra in texto:
        print(letra)

if __name__ == "__main__":
    palavra = "python"
    print(f"Palavra: {palavra}")
    imprime_vertical(palavra)
