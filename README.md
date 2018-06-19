# Backpropagation-Starcraft-Bot
Repositorio para un proyecto de Topicos de Inteligencia Artificial UNSA 2018

# Especificaciones
- AMD A8 2.7Ghz
- 4gb RAM
- 1000Gb HDD
- AMD RADEON 1GB GC

# Red Neuronal Artificial con Backpropagation
## Requisitos
- C++
- gnuplot

## Como utilizar
- Compilar: g++ -std=c++17 NeuralNetwork.cpp -o nn2
- Ejecutar: ./nn2
- Parametros de entrada: Numero de iteraciones
## Modificar la estructura de la red:
- En el archivo main.cpp 
- Modificar el parametro n_hidden_layer_sizes con una lista de inicializacion { xi, xi+1 ... xi+n} donde xi es el numero de neuronas en la capa hidden i.
## Caracteristicas
- Utiliza backpropagation y Gradiente descendiente
- Funciones de activacion: Relu, Identity, Logistic
- Batch_size: 100
- Training: 80%, Testing: 20%
## Fuentes de inspiracion
- https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neural_network/multilayer_perceptron.py
## Ejemplo:
- Base de datos: Iris
- Resultados: en la carpeta de screenshots

# Configuracion del escenario de STARCRAFT con BWAPI
## Pasos para la instalacion
Windows 7+

1.- Instalacion
	- Instalar el Visual Studio 2013
	- Instalar el starcraft: Brood War
	- Actualizar el Starcraft: Brood war a 1.16.1
	- Instalar Bwapi

## Configuracion del proyecto
2.	- Compilar ExampleProjects.sln en directorio de instalacion BWAPI
	- Construir ExampleAIModule en modo RELEASE
	- Copiar ExampleAIModule.dll a bwapi-data/AI dentro de la carpeta de BWAPI
3. Ejecutar Starcraft mediante ChaosLauncher.exe
4. Ejecutar un juego para probar el funcionamiento
5. Alternativa: https://github.com/adakitesystems/DropLauncher/blob/master/INSTRUCTIONS.md

## Resultados
Resultados de la instalacion en la carpeta de Screenshots

# Fuentes:
- https://bwapi.github.io/
- https://github.com/tscmoo/bwheadless
- https://visualstudio.microsoft.com/es/vs/older-downloads/
- https://starcraft.com/en-gb/
- DropLauncher: https://github.com/adakitesystems/DropLauncher/releases
- Bot de prueba: https://sscaitournament.com/index.php?action=scores
