using Godot;
using System;
using DeepLearning;

public class Node2D : Godot.Node2D
{
    Random Rand = new Random();
    BackPropLayer TestLayer;
    BackPropLayer TestLayer2;

    BackPropNetwork Net;

    public void NetLearn(float [] Array,float[] Expected){
        Net.SetInput(Array);
        Net.MoveForward();
        Net.CalcError(Expected);
        Net.BackProp();
        Net.AdjustWeights();
    }

    public void NetLayersLearning(){
        Net = new BackPropNetwork(2,4,new SigmoidFunct(), Rand);
        Net.AddLayer(2,new SigmoidFunct(), Rand);
        float[] Result;
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
        Result = Net.GetOutput();
        GD.Print("\n \n Out 1 = ",Result);
            NetLearn(new float[] {1,0},new float[] {1,0});
        Result = Net.GetOutput();
        GD.Print("\n \n Out 2 = ",Result);
            NetLearn(new float[] {1,1},new float[] {0,1});
        Result = Net.GetOutput();
        GD.Print("\n \n Out 3 = ",Result);
            NetLearn(new float[] {0,0},new float[] {0,1});
        Result = Net.GetOutput();
        GD.Print("\n \n Out 4 = ",Result);
            NetLearn(new float[] {0,0},new float[] {0.5f,0.5f});
        Result = Net.GetOutput();
        GD.Print("\n \n Test Out = ",Result);
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
