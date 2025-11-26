from functions import *

letras = identifyDict("trabalho/banco_de_imagens/fonte_mercosul.png")
#for i in range(5):
#    placa = platePrepare(f"trabalho/banco_de_imagens/nivel_1/placa{i+1}.jpg")
#    result=readPlate(placa,letras)
#    print(f"Resultado placa {i+1}: "+result)
for i in range(1):
    placa = platePrepare(f"trabalho/banco_de_imagens/nivel_2/placa{i+9}.jpg")
    result=readPlate(placa,letras)
    print(f"Resultado placa {i+6}: "+result)