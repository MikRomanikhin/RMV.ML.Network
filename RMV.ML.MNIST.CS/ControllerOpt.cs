using System.Diagnostics;

using MathNet.Numerics.LinearAlgebra;

using RMV.ML.Network.Common;

namespace RMV.ML.MNIST.CS;

/// <summary>
/// Controller for optimizing the neural network using various optimization algorithms.
/// </summary>
class ControllerOpt
{
	AppSettings? settings;
	const int LINE = 40;

	/// <summary>
	/// Trains and evaluates a neural network using the configured training and validation data sets.
	/// </summary>
	/// <remarks>Loads training and test data, initializes the neural network, and performs iterative training. 
	/// Validation is performed at configured intervals, and the best validation results are saved to a file.
	/// Progress and results are written to the console.</remarks>	
	public void Run()
	{
		var timer = new Stopwatch();
		timer.Start();

		double maxAccuracy = double.MinValue;

		settings = ConfigManager.GetRoot<AppSettings>() ?? throw new Exception( "Failed to load configuration" );

		var parser = new Parser( settings.Output );
		var trainSet = parser.Run( File.ReadAllLines( settings.TrainPath ) );

		var testSet = parser.Run( File.ReadAllLines( settings.TestPath ) );
		var valX = Matrix<double>.Build.DenseOfRowArrays( testSet.Source );
		var valT = Matrix<double>.Build.DenseOfRowArrays( testSet.Target );

		Console.WriteLine( $"Loaded data sets. Time:{timer.Elapsed.TotalSeconds:f2} sec" );

		var network = new MultiLayerNet( settings );

		int stagnation = 0;
		int bestIter = 0;

		for( int i = 0; i < settings.Iterations; i++ )
		{
			var (xBatch, tBatch) = trainSet.GetRandomTrain( settings.Batch );			

			network.Update( xBatch, tBatch );   // update the network weights based on the current batch

			double loss = network.Loss( xBatch, tBatch ); // calculate the loss for the current batch

			var ( trainAcc, _, _) = network.Accuracy( xBatch, tBatch );  // training accuracy for the current batch

			if( i % settings.Print == 0 )
				Console.WriteLine( $"iter:{i} loss:{loss:F4}  accuracy:{trainAcc:F4}  Time={timer.Elapsed:hh\\:mm\\:ss}" );
		
			if( i % settings.Epoch == 0 ) // validation testing at configured intervals
			{				
				var( testAcc, errors, indexes) = network.Accuracy( valX, valT ); // validation accuracy, errors, and indexes

				Console.WriteLine( $"iter:{i}  validation:{testAcc:F4}  Time={timer.Elapsed:hh\\:mm\\:ss}" );
				Console.WriteLine( new string( '-', LINE ) );

				if( testAcc > maxAccuracy ) // if the accuracy is better than the best so far, save the errors to a file
				{ 
					maxAccuracy = testAcc;
					stagnation = 0;
					bestIter = i;
					File.WriteAllText( settings.ErrorPath, errors.Join() ?? string.Empty );
					File.WriteAllText( settings.IndexPath, indexes.Join() ?? string.Empty );					
					continue;
				}

				if( ++stagnation > settings.Stagnation ) // no improvement in validation accuracy - stop training
				{
					Console.WriteLine( $"Stopping at {i} due to stagnation. Accuracy: {maxAccuracy:F4} at iteration {bestIter}. Time={timer.Elapsed:hh\\:mm\\:ss}" );
					break;
				}						
			}
		}		
	}
	
}
