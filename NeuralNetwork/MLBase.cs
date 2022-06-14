using System;
using System.Collections.Generic;
using Godot;

namespace DeepLearning
{
    /// <summary>
    /// Абстрактный класс активационных функций для использования в построении нейросетей
    /// </summary>
    public abstract class ActivationFunct{

        /// <summary>
        /// Вычисление результата функции активации по входному значению
        /// </summary>
        /// <param name="x">входное значение</param>
        /// <returns></returns>
        public abstract float ActivationVal(float x);

        /// <summary>
        /// Вычисление производной функции активации по входному значению
        /// </summary>
        /// <param name="x">входное значение</param>
        /// <returns></returns>
        public abstract float ActivationDeriv(float x);
    }

    public class SigmoidFunct: ActivationFunct{
        public override float ActivationVal(float x)
        {
            return (float)(1/(1+Math.Exp(-x)));
        }

        public override float ActivationDeriv(float x)
        {
            return ActivationVal(x)*(1-ActivationVal(x));
        }
    }

    /// <summary>
    /// Абстрактный базовый класс для всех нейронных слоёв
    /// </summary>
    public abstract class NNLayer{

        /// <summary>
        /// Веса связей между нейронами предыдущего и данного слоя.
        /// </summary>
        protected float[,] Weights;

        /// <summary>
        /// Буфер для выхода нейрона до использования функции активации
        /// </summary>
        protected float[] Output;

        /// <summary>
        /// Буффер для входа нейронного слоя
        /// </summary>
        protected float[] Input;

        /// <summary>
        /// Свойство, хранящее функцию активации
        /// </summary>
        protected ActivationFunct Funct;
        
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
        /// Возвращает массив выхода
        /// </summary>
        /// <returns></returns>
        public float[] GetOutput(){
            return Output;
        }
        
        /// <summary>
        /// Метод, возвращающий буффер выхода нейрона, прогнанный через функцию активации
        /// </summary>
        /// <returns></returns>
        public float[] GetActivOutput(){
            float[] Result = new float[GetOutputSize()];
            for (int i = 0; i < GetOutputSize(); i++)
            {
                Result[i] = Funct.ActivationVal(Output[i]);
            }
            return Result;
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
                    Weights[inp,outp]= (float)(rand.NextDouble()*2-1);
                }
            }
        }

        /// <summary>
        /// Метод, выдающий вывод одного нейрона до прогона его через функцию активации
        /// </summary>
        /// <param name="id"></param>
        /// <returns></returns>
        protected float SingleNeuronWeightedInput(int id){
                float value = 0;
                //для каждого выходного нейрона собираем сумму его входов, помноженных на вес слоя
                for (int inp = 0; inp < GetInputSize(); inp++)
                {
                    value += Input[inp]*Weights[inp,id];
                }
                return value;
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
        /// Метод, генерирующий выходной массив значений слоя
        /// </summary>
        public void GenerateOutput(){
            //обновляем массив выходов слоя
            Output = new float[GetOutputSize()];
            //проходим по всем выходным нейронам
            for (int outp = 0; outp < GetOutputSize(); outp++)
            {
                Output[outp] = SingleNeuronWeightedInput(outp);
            }
        }
        
        public abstract void AdjustWeights();

        /// <summary>
        /// Метод для строкового представления весов слоя
        /// </summary>
        /// <returns></returns>
        public string WeightsToString(){
            string result = "";
            for (int i = 0; i < GetInputSize(); i++)
            {
                for (int j = 0; j < GetOutputSize(); j++)
                {
                    result += Weights[i,j]+" ";
                }
                result += "\n";
            }
            return result;
        }
    
        public string InputToString(){
            string result = "";
                for (int j = 0; j < GetInputSize(); j++)
                {
                    result += Input[j]+" ";
                }
            return result;
        }
    
        public string OutputToString(){
            float[] ActivOutput = GetActivOutput();
            string result = "";
                for (int j = 0; j < GetOutputSize(); j++)
                {
                    result += ActivOutput[j]+" ";
                }
            return result;
        }
    
        public virtual string StateToString(){
            string result = "";
            result += "Weights = "+WeightsToString();
            result += "Input = "+InputToString();
            result += "Output = "+OutputToString();
            return result;
        }
    }

    /// <summary>
    /// базовый класс, представляющий один слой нейросети
    /// </summary>
    public class BackPropLayer : NNLayer{

        /// <summary>
        /// массив, хранящий предыдущие исправления весов
        /// </summary>
        protected float[,] PrevCorr;

        /// <summary>
        /// Буффер ошибок нейронного слоя
        /// </summary>
        protected float[] Error;

        protected float speed = 1;

        /// <summary>
        /// Инициализация слоя
        /// </summary>
        /// <param name="inputSize">Количество входов</param>
        /// <param name="outputSize">Количество выходов</param>
        public BackPropLayer(int inputSize, int outputSize, ActivationFunct funct, Random rand = null){
            Funct = funct;
            Weights = new float[inputSize,outputSize];
            PrevCorr = new float[inputSize,outputSize];
            Input = new float[inputSize];
            Output = new float[outputSize];
            Error = new float[outputSize];
            if(rand!=null){
                ResetWeights(rand);
            } else {
                ResetWeights();
            }
        }

        /// <summary>
        /// Метод для задания скорости обучения слоя
        /// </summary>
        /// <param name="val">новая скорость обучения</param>
        public void SetSpeed(float val){
            speed = val;
        }

        /// <summary>
        /// Метод для корректировки веса одного нейрона на основе вычисленной ошибки
        /// </summary>
        /// <param name="id">индекс нейрона</param>
        /// <param name="speed">коэффициент изменения весов</param>
        /// <param name="InCoeff">коэффициент инерции изменений</param>
        protected void AdjustWSingle(int id, float speed){
            for (int inp = 0; inp < GetInputSize(); inp++)
            {
                Weights[inp,id] -= speed*Error[id]*Input[inp];
            }
        }

        /// <summary>
        /// Метод для корректировки весов всех нейронов слоя. для корректировки должна быть задана ошибка
        /// </summary>
        /// <param name="speed">коэффициент изменения весов</param>
        /// <param name="InCoeff">коэффициент инерции изменений</param>
        public override void AdjustWeights(){
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
        protected float WeightedErrorOfNeuron(int id, BackPropLayer Next){
            if(Next.GetOutputSize() != Next.Error.Length)
                throw new Exception("Output size of passed weight matrix does not match with errors size. Weight Matr = "+Next.GetOutputSize()+", Errors = "+ Next.Error.Length);
            float Result = 0;
            for (int i = 0; i < Next.GetOutputSize(); i++)
            {
                Result += Next.Weights[id,i]*Next.Error[i];
            }
            return Result;
        }

        /// <summary>
        /// Метод для вычисления значения ошибки одного нейрона
        /// </summary>
        /// <param name="ErrorVal">Либо взвешанные ошибки предыдущих нейронов, либо производная функции вычисления расхождения (если последний слой)</param>
        /// <param name="id"></param>
        /// <returns></returns>
        protected void MultiplyByDeriv(float ErrorVal, int id){
            Error[id] = ErrorVal*Funct.ActivationDeriv(Output[id]);
        }

        /// <summary>
        /// Метод для заполнения всего массива ошибок слоя
        /// </summary>
        /// <param name="NextErrors"></param>
        public void FillErrors(BackPropLayer Next){
            for (int i = 0; i < GetOutputSize(); i++)
            {
                MultiplyByDeriv(WeightedErrorOfNeuron(i,Next),i);
            }
        }

        /// <summary>
        /// Метод для заполнения массива ошибок слоя. Перегрузка для заданных изначально значений массива
        /// </summary>
        public void FillErrors(float[]ErrorVals){   
            if(GetOutputSize() != ErrorVals.Length)
                throw new Exception("ErrorVal size of does not match with output size. Oputput size = "+GetOutputSize()+", Error size = "+ErrorVals.Length);
                     
            for (int i = 0; i < GetOutputSize(); i++)
            {
                MultiplyByDeriv(ErrorVals[i],i);
            }
        }

        public string ErrorToString(){
            string result = "";
                for (int j = 0; j < GetOutputSize(); j++)
                {
                    result += Error[j]+" ";
                }
            return result;
        }
    
        public override string StateToString(){
            string result = base.StateToString();
            result += "Error = "+ErrorToString();
            return result;
        }
    }

    /// <summary>
    /// Класс простой нейросети с обратным распространением ошибки
    /// </summary>
    public class BackPropNetwork{

        /// <summary>
        /// Массив 
        /// </summary>
        protected List<BackPropLayer> Layers = new List<BackPropLayer>();

        /// <summary>
        /// Конструктор класса, задающий начальные данные первого слоя
        /// </summary>
        /// <param name="Inputs"></param>
        /// <param name="Outputs"></param>
        public BackPropNetwork(int Inputs, int Outputs, ActivationFunct funct, Random Rand = null){
            Layers.Add(new BackPropLayer(Inputs,Outputs,funct,Rand));
        }

        /// <summary>
        /// Добавление нового слоя поверх последнего
        /// </summary>
        /// <param name="Outputs">количество выходов последнего слоя</param>
        /// <param name="funct">функция активации</param>
        /// <param name="Rand">рандом, если есть возможность</param>
        public void AddLayer(int Outputs, ActivationFunct funct, Random Rand = null){
            Layers.Add(new BackPropLayer(Layers[Layers.Count-1].GetOutputSize(),Outputs,funct,Rand));
        }

        /// <summary>
        /// Удаление верхнего слоя нейросети
        /// </summary>
        public void RemoveLayer(){
            Layers.RemoveAt(Layers.Count-1);
        }

        /// <summary>
        /// Метод, выдающий строковое предсавление структуры и состояния нейросети
        /// </summary>
        /// <returns></returns>
        public string StructToString(){
            string result = "";
            foreach (BackPropLayer item in Layers)
            {
                result += item.StateToString();
            }
            return result;
        }

        /// <summary>
        /// Метод, задающий вход первого слоя нейросети
        /// </summary>
        /// <param name="Array"></param>
        public void SetInput(float [] Array){
            Layers[0].SetInput(Array);
        }

        public void MoveForward(){
            for (int i = 0; i < Layers.Count; i++)
            {
                Layers[i].GenerateOutput();
                if(i < Layers.Count-1){
                    Layers[i+1].SetInput(Layers[i].GetActivOutput());
                }
            }
        }

        public BackPropLayer GetLastLayer(){
            return Layers[Layers.Count-1];
        }

        public BackPropLayer GetFirstLayer(){
            return Layers[0];
        }

        public void CalcError(float[] Expected){
            BackPropLayer LastLayer = GetLastLayer();
            float[] Errors = LastLayer.GetActivOutput();
            for (int i = 0; i < Errors.Length; i++)
            {
                Errors[i] = Errors[i]-Expected[i];
            }
            LastLayer.FillErrors(Errors);
        }

        /// <summary>
        /// Метод, продвигающий ошибку назад по сети
        /// </summary>
        public void BackProp(){
            for (int i = Layers.Count-1; i > 0; i--)
            {
                Layers[i-1].FillErrors(Layers[i]);
            }
        }

        public void AdjustWeights(){
            for (int i = 0; i < Layers.Count; i++)
            {
                Layers[i].AdjustWeights();
            }
        }
    }
}