# CyberLabs-Challenge-
Desafio da empresa CyberLabs para criar um modelo que pretende distinguir Cervejas de Refrigerantes por meio do programa Keras e Tensorflow.

Bibliotecas Usadas:

-keras com backend Tensorflow

-Numpy 

-Opencv

Instruções:

1) Baixar, e unzip dataset (link no final deste documento)

2) Colocar ambos os codigos no mesmo diretorio que o dataset

3) Baixar weights ja treinados para VGG16 e colocar no mesmo diretorio que o codigo e dataset

4) 

  - Rodar Beer_Soda.py para treinar rede LeNet

  - Rodar Beers_Sodas_VGG.py para treinar rede VGG16
  
  
5) Para testar rodar ou test_vgg.py ou test_lenet.py com o argumento -i caminho_para_foto


O data set foi obtido pelo Google Images usando Faktun Batch Download Images[Extensão do Chrome].

Foram criadas duas redes neurais:

 1)Arquitetura LeNet com tamanho das imagens (100,100).
 
 2)Arquitetura VGG16*.
 
 *Na rede que utiliza a Arquitetura VGG16, apenas as últimas 10 camadas foram treinadas, as primeiras 6 foram usadas os mesmos weights da rede VGG treinda no dataset ImageNet.

Foi possível obter uma “accuracy” de 78% com o modelo LeNet, porém devido ao tamanho das imagens (100,100) a rede não consegue extrair “features” suficientes devido as poucas camadas.

Com o Modelo VGG16 foi possível obter uma “accuracy” de 84%, esta poderia ser melhorada se fosse usado um dataset maior e melhor do que o obtido com a extensão do Google Chrome.

Link para dataset: https://www.dropbox.com/s/6zwhu1mklfa2lnh/Dataset.rar?dl=0

Link para weights já treinados (Rede usando aquitetura VGG16): https://www.dropbox.com/s/7uy4iwcvk8xc5tn/weights_array.npy?dl=0
