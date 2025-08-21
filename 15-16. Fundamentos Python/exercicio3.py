# exercicio3.py
# Programa que calcula valor do transporte com base no peso

def calcular_valor_transporte(peso):
    if peso <= 10:
        return "O valor será de R$ 50,00"
    elif 11 <= peso <= 20:
        return "O valor será de R$ 80,00"
    else:
        return "Transporte não aceito"

if __name__ == "__main__":
    # Testes automáticos
    for p in [5, 12, 25]:
        print(f"Peso {p}kg -> {calcular_valor_transporte(p)}")

    # Ou leitura do usuário:
    # peso = int(input("Digite o peso da carga (kg): "))
    # print(calcular_valor_transporte(peso))
