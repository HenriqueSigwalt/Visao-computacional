from functions import *

letras = identifyDict("trabalho/banco_de_imagens/fonte_mercosul.png")
for i in range(5):
    placa = platePrepare(f"trabalho/banco_de_imagens/nivel_1/placa{i+1}.jpg")
    result = readPlate(placa,letras,1)
    if len(result)!=7:
        placa = retryPlate(f"trabalho/banco_de_imagens/nivel_1/placa{i+1}.jpg")
        result = readPlate(placa,letras,2)
    print(f"Resultado placa {i+1}: "+result)
for i in range(7):
    placa = platePrepare(f"trabalho/banco_de_imagens/nivel_2/placa{i+6}.jpg")
    result = readPlate(placa,letras,1)
    if len(result)!=7:
        placa = retryPlate(f"trabalho/banco_de_imagens/nivel_2/placa{i+6}.jpg")
        result = readPlate(placa,letras,2)
    print(f"Resultado placa {i+6}: "+result)