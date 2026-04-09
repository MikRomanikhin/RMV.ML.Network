using System.Diagnostics;

using MathNet.Numerics.LinearAlgebra;

using RMV.ML.Network.Common;

namespace RMV.ML.MNIST.CS;

/// <summary>
///  
/// </summary>
class ControllerOpt
{
	AppSettings? settings;
	const int LINE = 40;

	public void Run()
	{
		var timer = new Stopwatch();
		timer.Start();

		settings = ConfigManager.GetRoot<AppSettings>() ?? throw new Exception( "Failed to load configuration" );

		var parser = new Parser( settings.Output );
		var trainSet = parser.Run( File.ReadAllLines( settings.TrainPath ) );

		var testSet = parser.Run( File.ReadAllLines( settings.TestPath ) );
		var valX = Matrix<double>.Build.DenseOfRowArrays( testSet.Source );
		var valT = Matrix<double>.Build.DenseOfRowArrays( testSet.Target );

		Console.WriteLine( $"Loaded data sets. Time:{timer.Elapsed.TotalSeconds:f2} sec" );

		var network = new MultiLayerNet( settings );

		for( int i = 0; i < settings.Iterations; i++ )
		{
			var (xBatch, tBatch) = trainSet.GetRandomTrain( settings.Batch );			

			network.Update( xBatch, tBatch );	

			double loss = network.Loss( xBatch, tBatch );

			double trainAcc = network.Accuracy( xBatch, tBatch );
			
			Console.WriteLine( $"iter:{i} loss:{loss:F4}  train:{trainAcc:F4}  Time={timer.Elapsed:hh\\:mm\\:ss}" );
		
			if( i % settings.Epoch == 0 ) // validation testing at configured intervals
			{				
				double validAcc = network.Accuracy( valX, valT );
				Console.WriteLine( $"iter:{i}  validation:{validAcc:F4}  Time={timer.Elapsed:hh\\:mm\\:ss}" );
				Console.WriteLine( new string( '-', LINE ) );
			}
		}
		
		double valAcc = network.Accuracy( valX, valT );
		Console.WriteLine( new string( '-', LINE ) );
		Console.WriteLine( $"validation:{valAcc:F4}  Time={timer.Elapsed:hh\\:mm\\:ss}" );
		Console.WriteLine( new string( '-', LINE ) );
	}
	
}
