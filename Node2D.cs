using Godot;
using System;
using DeepLearning;

public class Node2D : Godot.Node2D
{
    Random Rand = new Random();
    BackPropLayer TestLayer;
    BackPropLayer TestLayer2;

    BackPropNetwork Net;

    /// <summary>
    /// Тестовая функция, которая заставляет слой нейросети выдавать тот же массив, что подавался на вход
    /// </summary>
    public void LearnEcho(float [] Array,float[] Expected){
        TestLayer.AdjustWeights();
        TestLayer2.AdjustWeights();
        TestLayer.SetInput(Array);
        TestLayer.GenerateOutput();
        TestLayer2.SetInput(TestLayer.GetActivOutput());
        TestLayer2.GenerateOutput();
        float[] Error = new float[] {TestLayer2.GetActivOutput()[0]-Expected[0],TestLayer2.GetActivOutput()[1]-Expected[1]};
        TestLayer2.FillErrors(Error);
        TestLayer.FillErrors(TestLayer2);
    }

    public void NetLearn(float [] Array,float[] Expected){
        Net.SetInput(Array);
        Net.MoveForward();
        Net.CalcError(Expected);
        Net.BackProp();
        Net.AdjustWeights();
    }

    public void NetworkClassTest(){
        Net = new BackPropNetwork(2,3,new SigmoidFunct(), Rand);
        Net.AddLayer(2,new SigmoidFunct(), Rand);
        Net.SetInput(new float[]{0,1});
        Net.MoveForward();
        Net.CalcError(new float[]{0,1});
        GD.Print(Net.StructToString());
        Net.BackProp();
        Net.AdjustWeights();
        GD.Print(Net.StructToString());
    }

    public void NetLayersLearning(){
        Net = new BackPropNetwork(2,4,new SigmoidFunct(), Rand);
        Net.AddLayer(2,new SigmoidFunct(), Rand);
        GD.Print("HelloWorld");
        for (int i = 0; i < 10000; i++)
        {
            GD.Print(i);
            NetLearn(new float[] {0f,1f},new float[] {0,0});
            NetLearn(new float[] {1,0},new float[] {0.3f,0});
            NetLearn(new float[] {1,1},new float[] {0.6f,0});
            NetLearn(new float[] {0,0},new float[] {0.9f,0});
        }
        NetLearn(new float[] {0f,1f},new float[] {1,0});
        GD.Print("\n \n State1 = ",Net.StructToString());
            NetLearn(new float[] {1,0},new float[] {1,0});
        GD.Print("\n \n State2 = ",Net.StructToString());
            NetLearn(new float[] {1,1},new float[] {0,1});
        GD.Print("\n \n State3 = ",Net.StructToString());
            NetLearn(new float[] {0,0},new float[] {0,1});
        GD.Print("\n \n State4 = ",Net.StructToString());
            NetLearn(new float[] {0,0},new float[] {0.5f,0.5f});
        GD.Print("\n \n Test State = ",Net.StructToString());
    }

    public void SeparateLayersLearning(){
        TestLayer = new BackPropLayer(2,4,new SigmoidFunct(), Rand);
        TestLayer2 = new BackPropLayer(4,2,new SigmoidFunct(), Rand);
        GD.Print("HelloWorld");
        for (int i = 0; i < 10000; i++)
        {
            GD.Print(i);
            LearnEcho(new float[] {0f,1f},new float[] {0,0});
            LearnEcho(new float[] {1,0},new float[] {0.3f,0});
            LearnEcho(new float[] {1,1},new float[] {0.6f,0});
            LearnEcho(new float[] {0,0},new float[] {0.9f,0});
        }
            LearnEcho(new float[] {0f,1f},new float[] {1,0});
        GD.Print("Layer1 = ",TestLayer.StateToString());
        GD.Print("Layer2 = ",TestLayer2.StateToString());
            LearnEcho(new float[] {1,0},new float[] {1,0});
        GD.Print("Layer1 = ",TestLayer.StateToString());
        GD.Print("Layer2 = ",TestLayer2.StateToString());
            LearnEcho(new float[] {1,1},new float[] {0,1});
        GD.Print("Layer1 = ",TestLayer.StateToString());
        GD.Print("Layer2 = ",TestLayer2.StateToString());
            LearnEcho(new float[] {0,0},new float[] {0,1});
        GD.Print("Layer1 = ",TestLayer.StateToString());
        GD.Print("Layer2 = ",TestLayer2.StateToString());
            LearnEcho(new float[] {0,0},new float[] {0.5f,0.5f});
        GD.Print("Layer1 = ",TestLayer.StateToString());
        GD.Print("Layer2 = ",TestLayer2.StateToString());
    }
    // Declare member variables here. Examples:
    // private int a = 2;
    // private string b = "text";

    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
    {
        //SeparateLayersLearning();
        NetLayersLearning();
    }

//  // Called every frame. 'delta' is the elapsed time since the previous frame.
//  public override void _Process(float delta)
//  {
//      
//  }
}
