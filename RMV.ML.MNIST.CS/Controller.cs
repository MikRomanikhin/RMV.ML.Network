using System.Diagnostics;

using MathNet.Numerics.LinearAlgebra;

using RMV.ML.Network.Common;

namespace RMV.ML.MNIST.CS;

/// <summary>
/// Neural network training using various optimization algorithms
/// </summary>
class Controller( DataSet trainSet, DataSet testSet, AppSettings settings )
{
	const int LINE = 40;

	/// <summary>
	/// Trains and evaluates a neural network using the configured training and validation data sets.
	/// </summary>
	/// <remarks>
	/// Loads training and test data, initializes the neural network, and performs iterative training. 
	/// Validation is performed at configured intervals, and the best validation results are saved to a file.
	/// </remarks>	
	public void RunML()
	{
		var timer = new Stopwatch();
		timer.Start();

		double maxAccuracy = double.MinValue;

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

			var (trainAcc, _, _) = network.Accuracy( xBatch, tBatch );  // training accuracy for the current batch

			if( i % settings.Print == 0 )
				Console.WriteLine( $"iter:{i} loss:{loss:F4}  accuracy:{trainAcc:F4}  Time={timer.Elapsed:hh\\:mm\\:ss}" );

			if( i % settings.Epoch == 0 ) // validation testing at configured intervals
			{
				var (testAcc, errors, indexes) = network.Accuracy( valX, valT ); // validation accuracy, errors, and indexes

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

	public void RunConv()
	{
		var timer = new Stopwatch();
		timer.Start();

		double maxAccuracy = double.MinValue;

		// Validation matrices required for accuracy tests later
		var valX = Matrix<double>.Build.DenseOfRowArrays( testSet.Source );
		var valT = Matrix<double>.Build.DenseOfRowArrays( testSet.Target );

		var valX4D = ReshapeTo4D( valX, settings.InputDim ); // Convert mathnet 2D to Convolution 4D input

		Console.WriteLine( $"Loaded data sets. Time:{timer.Elapsed.TotalSeconds:f2} sec" );

		var network = new SimpleConvNet( settings );

		int stagnation = 0;
		int bestIter = 0;

		// Adam optimizer state — strongly typed to match ConvNetParams
		double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;

		var p = network.Params;

		var mW1 = new double[ p.W1.GetLength( 0 ), p.W1.GetLength( 1 ), p.W1.GetLength( 2 ), p.W1.GetLength( 3 ) ];
		var vW1 = new double[ p.W1.GetLength( 0 ), p.W1.GetLength( 1 ), p.W1.GetLength( 2 ), p.W1.GetLength( 3 ) ];
		var mb1 = Vector<double>.Build.Dense( p.b1.Count );
		var vb1 = Vector<double>.Build.Dense( p.b1.Count );

		var mW2 = Matrix<double>.Build.Dense( p.W2.RowCount, p.W2.ColumnCount );
		var vW2 = Matrix<double>.Build.Dense( p.W2.RowCount, p.W2.ColumnCount );
		var mb2 = Vector<double>.Build.Dense( p.b2.Count );
		var vb2 = Vector<double>.Build.Dense( p.b2.Count );

		var mW3 = Matrix<double>.Build.Dense( p.W3.RowCount, p.W3.ColumnCount );
		var vW3 = Matrix<double>.Build.Dense( p.W3.RowCount, p.W3.ColumnCount );
		var mb3 = Vector<double>.Build.Dense( p.b3.Count );
		var vb3 = Vector<double>.Build.Dense( p.b3.Count );

		for( int i = 0; i < settings.Iterations; i++ )
		{
			// Gather random validation batch
			var (xBatch, tBatch) = trainSet.GetRandomTrain( settings.Batch );

			// Map batch to 4D to push into Convolution Networks
			var xBatch4D = ReshapeTo4D( xBatch, settings.InputDim );

			var grads = network.Gradient( xBatch4D, tBatch ); // compute gradients via backprop

			// Adam parameter updates
			double t = i + 1;
			double fix1 = 1.0 - Math.Pow( beta1, t );
			double fix2 = 1.0 - Math.Pow( beta2, t );
			double lrT = settings.Rate * Math.Sqrt( fix2 ) / fix1;

			AdamUpdateArray( p.W1, grads.dW1, mW1, vW1, lrT, beta1, beta2, eps );
			AdamUpdateVector( p.b1, grads.db1, mb1, vb1, lrT, beta1, beta2, eps );
			AdamUpdateMatrix( p.W2, grads.dW2, mW2, vW2, lrT, beta1, beta2, eps );
			AdamUpdateVector( p.b2, grads.db2, mb2, vb2, lrT, beta1, beta2, eps );
			AdamUpdateMatrix( p.W3, grads.dW3, mW3, vW3, lrT, beta1, beta2, eps );
			AdamUpdateVector( p.b3, grads.db3, mb3, vb3, lrT, beta1, beta2, eps );

			double loss = network.Loss( xBatch4D, tBatch );

			double trainAcc = network.Accuracy( xBatch4D, tBatch );

			if( i % settings.Print == 0 )
				Console.WriteLine( $"iter:{i} loss:{loss:F4}  accuracy:{trainAcc:F4}  Time={timer.Elapsed:hh\\:mm\\:ss}" );

			if( i % settings.Epoch == 0 ) // validation testing at configured intervals
			{
				double testAcc = network.Accuracy( valX4D, valT );

				Console.WriteLine( $"iter:{i}  validation:{testAcc:F4}  Time={timer.Elapsed:hh\\:mm\\:ss}" );
				Console.WriteLine( new string( '-', LINE ) );

				if( testAcc > maxAccuracy ) // if the accuracy is better than the best so far, save the errors to a file
				{
					maxAccuracy = testAcc;
					stagnation = 0;
					bestIter = i;
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

	#region helpers ------------------------------------------------------------

	#region helpers ------------------------------------------------------------

	static void ApplyGradUpdateArray( object paramObj, object gradObj, double rate )
	{
		var pMap = ( double[,,,] )paramObj;
		var gMap = ( double[,,,] )gradObj;

		int d0 = pMap.GetLength( 0 );
		int d1 = pMap.GetLength( 1 );
		int d2 = pMap.GetLength( 2 );
		int d3 = pMap.GetLength( 3 );

		for( int i = 0; i < d0; i++ )
			for( int j = 0; j < d1; j++ )
				for( int k = 0; k < d2; k++ )
					for( int l = 0; l < d3; l++ )
						pMap[ i, j, k, l ] -= rate * gMap[ i, j, k, l ];
	}

	static void ApplyGradUpdateMatrix( object paramObj, object gradObj, double rate )
	{
		var pMap = ( Matrix<double> )paramObj;
		var gMap = ( Matrix<double> )gradObj;

		pMap.Map2( ( p, g ) => p - ( rate * g ), gMap, pMap );
	}

	static void ApplyGradUpdateVector( object paramObj, object gradObj, double rate )
	{
		var pMap = ( Vector<double> )paramObj;
		var gMap = ( Vector<double> )gradObj;

		pMap.Map2( ( p, g ) => p - ( rate * g ), gMap, pMap );
	}

	static double[,,,] ReshapeTo4D( Matrix<double> matrix, int[] inputDim )
	{
		int batchSize = matrix.RowCount;
		int c = inputDim[ 0 ];
		int h = inputDim[ 1 ];
		int w = inputDim[ 2 ];

		var tensor = new double[ batchSize, c, h, w ];

		for( int n = 0; n < batchSize; n++ )
		{
			for( int i = 0; i < c; i++ )
				for( int j = 0; j < h; j++ )
					for( int k = 0; k < w; k++ )
						tensor[ n, i, j, k ] = matrix[ n, ( i * h * w ) + ( j * w ) + k ];
		}

		return tensor;
	}

	static void AdamUpdateArray( double[,,,] p, double[,,,] g, double[,,,] m, double[,,,] v, double lr, double b1, double b2, double eps )
	{
		int d0 = p.GetLength( 0 ), d1 = p.GetLength( 1 ), d2 = p.GetLength( 2 ), d3 = p.GetLength( 3 );
		for( int i = 0; i < d0; i++ )
			for( int j = 0; j < d1; j++ )
				for( int k = 0; k < d2; k++ )
					for( int l = 0; l < d3; l++ )
					{
						m[ i, j, k, l ] = b1 * m[ i, j, k, l ] + ( 1 - b1 ) * g[ i, j, k, l ];
						v[ i, j, k, l ] = b2 * v[ i, j, k, l ] + ( 1 - b2 ) * g[ i, j, k, l ] * g[ i, j, k, l ];
						p[ i, j, k, l ] -= lr * m[ i, j, k, l ] / ( Math.Sqrt( v[ i, j, k, l ] ) + eps );
					}
	}

	static void AdamUpdateMatrix( Matrix<double> p, Matrix<double> g, Matrix<double> m, Matrix<double> v, double lr, double b1, double b2, double eps )
	{
		for( int i = 0; i < p.RowCount; i++ )
			for( int j = 0; j < p.ColumnCount; j++ )
			{
				m[ i, j ] = b1 * m[ i, j ] + ( 1 - b1 ) * g[ i, j ];
				v[ i, j ] = b2 * v[ i, j ] + ( 1 - b2 ) * g[ i, j ] * g[ i, j ];
				p[ i, j ] -= lr * m[ i, j ] / ( Math.Sqrt( v[ i, j ] ) + eps );
			}
	}

	static void AdamUpdateVector( Vector<double> p, Vector<double> g, Vector<double> m, Vector<double> v, double lr, double b1, double b2, double eps )
	{
		for( int i = 0; i < p.Count; i++ )
		{
			m[ i ] = b1 * m[ i ] + ( 1 - b1 ) * g[ i ];
			v[ i ] = b2 * v[ i ] + ( 1 - b2 ) * g[ i ] * g[ i ];
			p[ i ] -= lr * m[ i ] / ( Math.Sqrt( v[ i ] ) + eps );
		}
	}

	#endregion

	#endregion

}