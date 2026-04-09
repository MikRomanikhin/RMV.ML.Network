using System.Diagnostics;

using MathNet.Numerics.LinearAlgebra;

using RMV.ML.Network.Common;

namespace RMV.ML.MNIST.CS;

/// <summary>
/// 
/// </summary>
class Controller2Layer
{
	AppSettings? settings;

	public void Run()
	{
		var timer = new Stopwatch();
		timer.Start();

		settings = ConfigManager.GetRoot<AppSettings>() ?? throw new Exception( "Failed to load configuration" );

		var parser = new Parser( settings.Output );
		var trainSet = parser.Run( File.ReadAllLines( settings.TrainPath ) );
		var testSet = parser.Run( File.ReadAllLines( settings.TestPath ) );

		Console.WriteLine( $"Loaded data sets. Time:{timer.Elapsed.TotalSeconds:f2} sec" );

		var network = new TwoLayerNet( settings );

		for( int i = 0; i < settings.Iterations; i++ )
		{
			var (xBatch, tBatch) = trainSet.GetRandomTrain( settings.Batch );
			var (xTest, tTest) = trainSet.GetRandomTrain( settings.Batch );

			var grad = network.Gradient( xBatch, tBatch ); // compute gradients via backprop

			// apply parameter updates: params[key] -= learning_rate * grad[key]
			ApplyGradUpdate( network.Params, grad, "W1" );
			ApplyGradUpdate( network.Params, grad, "b1" );
			ApplyGradUpdate( network.Params, grad, "W2" );
			ApplyGradUpdate( network.Params, grad, "b2" );

			double loss = network.Loss( xBatch, tBatch );
			
			double trainAcc = network.Accuracy( xBatch, tBatch );
			double testAcc = network.Accuracy( xTest, tTest );
			Console.WriteLine( $"iter:{i} loss:{loss:F4}  train_acc:{trainAcc:F4}  test_acc:{testAcc:F4}  Time={timer.Elapsed:hh\\:mm\\:ss}" );
		
			if( i % settings.Epoch == 0 ) // Perform validation testing at configured intervals
			{
				var valX = Matrix<double>.Build.DenseOfRowArrays( testSet.Source );
				var valT = Matrix<double>.Build.DenseOfRowArrays( testSet.Target );

				double validAcc = network.Accuracy( valX, valT );
				Console.WriteLine( $"iter:{i}  validation:{validAcc:F4}  Time={timer.Elapsed:hh\\:mm\\:ss}" );
			}
		}
	}



	/// <summary>
	/// Applies the gradient update to the specified parameter in the network.
	/// </summary>
	/// <param name="paramDict">The dictionary containing the network parameters.</param>
	/// <param name="gradDict">The dictionary containing the gradients.</param>
	/// <param name="key">The key of the parameter to update.</param>
	void ApplyGradUpdate( Dictionary<string, object> paramDict, Dictionary<string, object> gradDict, string key )
	{
		if( !paramDict.TryGetValue( key, out var p ) ) return;
		if( !gradDict.TryGetValue( key, out var g ) ) return;

		switch( (p, g) )
		{
			case (Matrix<double> pM, Matrix<double> gM ):
				UpdateMatrixInPlace( pM, gM );
				break;
			case (Vector<double> pV, Vector<double> gV ):
				UpdateVectorInPlace( pV, gV );
				break;
			default:
				throw new InvalidOperationException( $"Unsupported param/grad types for key '{key}'." );
		}
	}

	/// <summary>
	/// Updates the elements of a matrix in place using the corresponding gradients
	/// </summary>
	/// <param name="p">The matrix to be updated.</param>
	/// <param name="g">The gradient matrix.</param>
	void UpdateMatrixInPlace( Matrix<double> p, Matrix<double> g )
	{
		int r = p.RowCount, c = p.ColumnCount;
		if( g.RowCount != r || g.ColumnCount != c ) throw new ArgumentException( "Gradient shape mismatch." );
		for( int i = 0; i < r; i++ )
			for( int j = 0; j < c; j++ )
				p[ i, j ] -= this.settings.Rate * g[ i, j ];
	}

	/// <summary>
	/// Updates the elements of a vector in place using the corresponding gradients
	/// </summary>
	/// <param name="p">The vector to be updated.</param>
	/// <param name="g">The gradient vector.</param>
	void UpdateVectorInPlace( Vector<double> p, Vector<double> g )
	{
		if( p.Count != g.Count ) throw new ArgumentException( "Gradient length mismatch." );

		for( int i = 0; i < p.Count; i++ ) p[ i ] -= this.settings.Rate * g[ i ];
	}
	
}
