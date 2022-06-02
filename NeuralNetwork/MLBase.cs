using System;

namespace DeepLearning
{
    /// <summary>
    /// базовый класс, представляющий один слой нейросети
    /// </summary>
    public class NNLayer{

        /// <summary>
        /// Количество входов слоя
        /// </summary>
        public readonly int InputSize;

        /// <summary>
        /// Количество выходов слоя
        /// </summary>
        public readonly int OutputSize;

        /// <summary>
        /// Веса связей между нейронами предыдущего и данного слоя.
        /// </summary>
        protected float[,] Weights;

        /// <summary>
        /// Конструктор класса
        /// </summary>
        /// <param name="inputSize">Количество входов</param>
        /// <param name="outputSize">Количество выходов</param>
        public NNLayer(int inputSize, int outputSize){
            InputSize = inputSize;
            OutputSize = outputSize;
            Weights = new float[inputSize,outputSize];
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < OutputSize; j++)
                {
                    Weights[i,j]=0;
                }
            }
        }

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

        /// <summary>
        /// Функция для вычисления ответа нейрона в соответствии со входным значением
        /// </summary>
        /// <param name="x">вход функции</param>
        /// <returns></returns>
        public virtual double SigmoidFunction(float x){
            return 1/(1+Math.Exp(-x));
        }


    }

}