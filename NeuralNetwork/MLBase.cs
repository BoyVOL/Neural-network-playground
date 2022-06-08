using System;

namespace DeepLearning
{
    /// <summary>
    /// базовый класс, представляющий один слой нейросети
    /// </summary>
    public class NNLayer{

        /// <summary>
        /// Веса связей между нейронами предыдущего и данного слоя.
        /// </summary>
        protected float[,] Weights;

        /// <summary>
        /// Буфер для выхода нейронного слоя
        /// </summary>
        protected float[] Output;

        /// <summary>
        /// Инициализация слоя
        /// </summary>
        /// <param name="inputSize">Количество входов</param>
        /// <param name="outputSize">Количество выходов</param>
        public NNLayer(int inputSize, int outputSize){
            Weights = new float[inputSize,outputSize];
            Output = new float[outputSize];
            ResetWeights();
        }

        /// <summary>
        /// Функция для вычисления ответа нейрона в соответствии со входным значением
        /// </summary>
        /// <param name="x">вход функции</param>
        /// <returns></returns>
        public virtual float SigmoidFunction(float x){
            return (float)(1/(1+Math.Exp(-x)));
        }

        /// <summary>
        /// Метод, возвращающий число входов слоя
        /// </summary>
        /// <returns></returns>
        public int GetInputSize(){
            return Weights.GetLength(0);
        }

        /// <summary>
        /// Метод, возвращающий количество выходов слоя
        /// </summary>
        /// <returns></returns>
        public int GetOutputSize(){
            return Weights.GetLength(1);
        }

        /// <summary>
        /// Метод для переопределения стартовых весов слоя.
        /// </summary>
        /// <returns></returns>
        public virtual void ResetWeights(){
            for (int inp = 0; inp < GetInputSize(); inp++)
            {
                for (int outp = 0; outp < GetOutputSize(); outp++)
                {
                    Weights[inp,outp]= 0;
                }
            }
        }

        /// <summary>
        /// Метод для переопределения стартовых весов слоя.
        /// </summary>
        /// <returns></returns>
        public virtual void ResetWeights(Random rand){
            for (int inp = 0; inp < GetInputSize(); inp++)
            {
                for (int outp = 0; outp < GetOutputSize(); outp++)
                {
                    Weights[inp,outp]= (float)(rand.NextDouble()-0.5);
                }
            }
        }

        /// <summary>
        /// Метод, вычисляющий выход одного нейрона
        /// </summary>
        /// <param name="input">массив входа нейрона</param>
        /// <param name="id">индекс нейрона в матрице весов</param>
        /// <returns></returns>
        protected float SingleNeuronOutput(float[] input, int id){            
                float value = 0;
                //для каждого выходного нейрона собираем сумму его входов, помноженных на вес слоя
                for (int inp = 0; inp < GetInputSize(); inp++)
                {
                    value += input[inp]*Weights[inp,id];
                }
                return SigmoidFunction(value);
        }

        /// <summary>
        /// Метод, генерирующий выходной массив значений слоя
        /// </summary>
        /// <param name="input"></param>
        public void GenerateOutput(float[] input){
            if(input.Length != GetInputSize())
                throw new Exception("input array size does not match array size. Input = "+input.Length+", Layer input = "+GetInputSize());
            //обновляем массив выходов слоя
            Output = new float[GetOutputSize()];
            //проходим по всем выходным нейронам
            for (int outp = 0; outp < GetOutputSize(); outp++)
            {
                Output[outp] = SingleNeuronOutput(input, outp);
            }
        }
    }

    /// <summary>
    /// Класс, отображающий целую нейросеть
    /// </summary>
    public class NeuralNetwork{

        /// <summary>
        /// Массив 
        /// </summary>
        public NNLayer[] Layers;

        /// <summary>
        /// Вычисление входа по каждой "лапке" логистического нейрона
        /// </summary>
        /// <param name="w">вес входа</param>
        /// <param name="x">передаваемое значение</param>
        /// <param name="b">смещение</param>
        /// <returns></returns>
        public virtual float SingleLogInput(float w, float x, float b = 0){
            return w*x-b;
        }
    }
}