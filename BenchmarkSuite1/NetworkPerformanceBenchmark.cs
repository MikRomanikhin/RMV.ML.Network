using System;
using System.Collections.Generic;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;

using RMV.ML.Network.Common;
using RMV.ML.Network.Domain;

namespace RMV.ML.Network.Benchmarks;

[MemoryDiagnoser]
public class NetworkPerformanceBenchmark
{
    private Net _network = null !;
    private double[][] _testInputs = null !;
    private double[][] _testOutputs = null !;
    private DataSet _trainingData = null !;
    private const int TestSize = 100;
    [GlobalSetup]
    public void Setup()
    {
        var settings = new AppSettings
        {
            Input = 784,
            Hidden = new[]
            {
                128,
                64
            },
            Output = 10,
            Rate = 0.01,
            Momentum = 0.9,
            Iterations = 10,
            Batch = 32,
            Print = 1,
            Train = "train.csv",
            Test = "test.csv",
            Errors = "errors.csv"
        };
        _network = new Net(settings);
        _network.Connect();
        _network.Initialize();
        _testInputs = new double[TestSize][];
        _testOutputs = new double[TestSize][];
        var random = new Random(42);
        for (int i = 0; i < TestSize; i++)
        {
            _testInputs[i] = new double[784];
            for (int j = 0; j < 784; j++)
            {
                _testInputs[i][j] = random.NextDouble();
            }

            _testOutputs[i] = new double[10];
            int correctClass = random.Next(10);
            _testOutputs[i][correctClass] = 1.0;
        }

        var sourceList = new List<double[]>();
        var targetList = new List<double[]>();
        for (int i = 0; i < 1000; i++)
        {
            sourceList.Add(_testInputs[i % TestSize]);
            targetList.Add(_testOutputs[i % TestSize]);
        }

        _trainingData = new DataSet(sourceList, targetList);
    }

    [Benchmark]
    public void ForwardPass()
    {
        for (int i = 0; i < TestSize; i++)
        {
            _network.Training(_trainingData);
        }
    }

    [Benchmark]
    public void BackwardPass()
    {
        for (int i = 0; i < TestSize; i++)
        {
            _network.Training(_trainingData);
        }
    }

    [Benchmark]
    public void FullTrainingCycle()
    {
        for (int i = 0; i < 50; i++)
        {
            _network.Training(_trainingData);
        }

        _network.Update(50);
    }
}