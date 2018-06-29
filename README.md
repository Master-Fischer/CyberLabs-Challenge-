# CyberLabs-Challenge-
Desafio da empresa CyberLabs para criar um modelo que pretende distinguir Cervejas de Refrigerantes por meio do programa Keras e Tensorflow.

Bibliotecas Usadas:

-Keras com backend Tensorflow

-Numpy 

-Opencv

O dataset foi obtido pelo Google Images.

Foram criadas duas redes neurais:

 1) Arquitetura LeNet com tamanho das imagens (28,28).
 
Foi possível obter uma “accuracy” de 78% com o modelo LeNet, porém devido ao tamanho das imagens (28,28) a rede não consegue extrair “features” suficientes devido as poucas camadas. 

Instruções:

1) Baixar, e unzip dataset (link no final deste documento)

2) Colocar ambos os codigos no mesmo diretorio que o dataset junto com o pyimagesearch

3) 

  - Rodar train_soda-beer_network.py para treinar rede LeNet

  - Rodar test_beer_soda.py para avaliar a rede neural
  
  
5) Para testar, rodar test_beer_soda.py com uma imagem no mesmo diretorio e adicionar no local no codigo o nome da Imagem. 

Link para dataset: https://www.dropbox.com/sh/nar7fst140c2xd8/AAB_gesJG5MQkvR033oM5bZ9a?dl=0


