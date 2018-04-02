# CyberLabs-Challenge-
Desafio da empresa CyberLabs de criar um modelo que classifica Cervejas e Refrigerantes usando Keras e Tensorflow

O data set foi obtido pelo Google Images usando Faktun Batch Download Images[Extensao do Chrome]

Foi criada duas redes neurais:

 1)Arquitetura LeNet com tamanho das imagens (100,100)
 
 2)Arquitetura VGG16*
 
 *Na rede que utiliza a arquitetura VGG16 apenas as ultimas 10 camadas foram treinadas as primeiras 6 foram usadas os mesmos weights da rede VGG treinda no dataset ImageNet

Foi possivel obter uma accuracy de 78% com o modelo LeNet porem devido ao tamnho das imagens (100,100) a rede nao consegue extrair 'features' suficientes devido as poucas camadas

Com o Modelo VGG16 foi possivel obter uma accuracy de 85%, essa accuracy poderia der melhorada se fosse usado um dataset maior e melhor do que o obtido e com a extensao do Google Chrome

Link para dataset: https://www.dropbox.com/s/6zwhu1mklfa2lnh/Dataset.rar?dl=0

Link para weights ja treinados (Rede usando aquitetura VGG16): https://www.dropbox.com/s/7uy4iwcvk8xc5tn/weights_array.npy?dl=0
