using Godot;
using System;
using DeepLearning;

public class Node2D : Godot.Node2D
{
    Random Rand = new Random();
    NNLayer TestLayer;
    NNLayer TestLayer2;

    /// <summary>
    /// Тестовая функция, которая заставляет слой нейросети выдавать тот же массив, что подавался на вход
    /// </summary>
    public void LearnEcho(float [] Array,float[] Expected){
        TestLayer.AdjustW(1,0.5f);
        TestLayer2.AdjustW(1,0.5f);
        TestLayer.SetInput(Array);
        TestLayer.GenerateOutput();
        TestLayer2.SetInput(TestLayer.GetOutput());
        TestLayer2.GenerateOutput();
        float[] Error = new float[] {TestLayer2.GetOutput()[0]-Expected[0],TestLayer2.GetOutput()[1]-Expected[1]};
        TestLayer2.FillErrors(Error);
        TestLayer.FillErrors(TestLayer2);
    }
    // Declare member variables here. Examples:
    // private int a = 2;
    // private string b = "text";

    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
    {
        TestLayer = new NNLayer(2,4,Rand);
        TestLayer2 = new NNLayer(4,2,Rand);
        GD.Print("HelloWorld");
        for (int i = 0; i < 1000; i++)
        {
            GD.Print(i);
            LearnEcho(new float[] {0f,1f},new float[] {1,0});
            LearnEcho(new float[] {1,0},new float[] {1,0});
            LearnEcho(new float[] {1,1},new float[] {0,1});
            LearnEcho(new float[] {0,0},new float[] {0,1});
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
    }

//  // Called every frame. 'delta' is the elapsed time since the previous frame.
//  public override void _Process(float delta)
//  {
//      
//  }
}
