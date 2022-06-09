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
        /// Буффер для входа нейронного слоя
        /// </summary>
        protected float[] Input;

        /// <summary>
        /// Буффер ошибок нейронного слоя
        /// </summary>
        protected float[] Error;

        /// <summary>
        /// Инициализация слоя
        /// </summary>
        /// <param name="inputSize">Количество входов</param>
        /// <param name="outputSize">Количество выходов</param>
        public NNLayer(int inputSize, int outputSize){
            Weights = new float[inputSize,outputSize];
            Output = new float[outputSize];
            Error = new float[outputSize];
            ResetWeights();
        }

        /// <summary>
        /// Функция для вычисления ответа нейрона в соответствии со входным значением
        /// </summary>
        /// <param name="x">вход функции</param>
        /// <returns></returns>
        public virtual float AcivationFunct(float x){
            return (float)(1/(1+Math.Exp(-x)));
        }

        /// <summary>
        /// Вычисление производной функции активации с заданным значением
        /// </summary>
        /// <param name="x">значение функции</param>
        /// <returns></returns>
        public virtual float AcivationDeriv(float x){
            return x*(1-x);
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
        /// <param name="id">индекс нейрона в матрице весов</param>
        /// <returns></returns>
        protected float SingleNeuronOutput(int id){            
                float value = 0;
                //для каждого выходного нейрона собираем сумму его входов, помноженных на вес слоя
                for (int inp = 0; inp < GetInputSize(); inp++)
                {
                    value += Input[inp]*Weights[inp,id];
                }
                return AcivationFunct(value);
        }

        /// <summary>
        /// Метод, генерирующий выходной массив значений слоя
        /// </summary>
        public void GenerateOutput(){
            //обновляем массив выходов слоя
            Output = new float[GetOutputSize()];
            //проходим по всем выходным нейронам
            for (int outp = 0; outp < GetOutputSize(); outp++)
            {
                Output[outp] = SingleNeuronOutput(outp);
            }
        }

        /// <summary>
        /// Метод для задания входных значений нейрона
        /// </summary>
        /// <param name="input">массив входных значений</param>
        public void SetInput(float[] input){
            if(input.Length != GetInputSize())
                throw new Exception("input array size does not match array size. Input = "+input.Length+", Layer input = "+GetInputSize());
            for (int i = 0; i < GetInputSize(); i++)
            {
                Input[i] = input[i];
            }
        }
        
        /// <summary>
        /// Метод для корректировки веса одного нейрона на основе вычисленной ошибки
        /// </summary>
        /// <param name="id">индекс нейрона</param>
        /// <param name="speed">коэффициент изменения весов</param>
        protected void AdjustWSingle(int id, float speed){
            for (int inp = 0; inp < GetInputSize(); inp++)
            {
                Weights[inp,id] -= speed*Error[id]*Input[inp];
            }
        }

        /// <summary>
        /// Метод для корректировки весов всех нейронов слоя
        /// </summary>
        /// <param name="speed">коэффициент изменения весов</param>
        public void AdjustW(float speed){
            for (int i = 0; i < GetOutputSize(); i++)
            {
                AdjustWSingle(i,speed);
            }
        }

        /// <summary>
        /// Метод для вычисления взвешанной суммы ошибок на следующем за этим нейроном слоем
        /// </summary>
        /// <param name="id">индекс нейрона</param>
        /// <param name="NextWeights">веса следующего слоя</param>
        /// <param name="NextErrors">Ошибки следующего слоя</param>
        /// <returns></returns>
        protected float WeightedErrorOfNeuron(int id, float[,]NextWeights, float[] NextErrors){
            if(NextWeights.GetLength(1) != NextErrors.Length)
                throw new Exception("Output size of passed weight matrix does not match with errors size. Weight Matr = "+NextWeights.GetLength(1)+", Errors = "+NextErrors.Length);
            float Result = 0;
            for (int i = 0; i < NextWeights.GetLength(1); i++)
            {
                Result += NextWeights[id,i]+NextErrors[i];
            }
            return Result;
        }

        /// <summary>
        /// Метод для вычисления значения ошибки одного нейрона
        /// </summary>
        /// <param name="ErrorVal">Либо взвешанные ошибки предыдущих нейронов, либо производная функции вычисления расхождения (если последний слой)</param>
        /// <param name="id"></param>
        /// <returns></returns>
        protected void CalculateErrorSingle(float ErrorVal, int id){
            Error[id] = ErrorVal*AcivationDeriv(Output[id]);
        }

        /// <summary>
        /// Метод для заполнения всего массива ошибок слоя
        /// </summary>
        /// <param name="NextErrors"></param>
        public void FillErrors(float[,]NextWeights, float[] NextErrors){
            for (int i = 0; i < GetOutputSize(); i++)
            {
                CalculateErrorSingle(WeightedErrorOfNeuron(i,NextWeights,NextErrors),i);
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