using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

using RMV.ML.Network.Common;

namespace RMV.ML.MNIST.CS;

/// <summary>
/// Simple two-layer fully connected neural network, supporting forward prediction, 
/// loss computation, accuracy evaluation, and gradient calculation for training.
/// </summary>
/// <remarks>
/// Encapsulates the parameters and layers of a two-layer neural network, including affine transformations, 
/// ReLU activation, and softmax with loss. Provides methods for performing inference, computing loss and accuracy, 
/// and obtaining gradients for backpropagation. All layer parameters are accessible via the Params dictionary. 
/// </remarks>
public class TwoLayerNet
{
	public Dictionary<string, object> Params { get; } = [];

	public Affine Affine1 { get; }
	public Relu Relu1 { get; } = new();
	public Affine Affine2 { get; }
	public SoftmaxWithLoss LastLayer { get; } = new();

	/// <summary>
	/// Construct a two-layer network.
	/// </summary>
	public TwoLayerNet( AppSettings settings )
	{
		double std = 0.01;
		var normalDist = new Normal( 0.0, std );

		// Use MathNet native Random distributions for weight initialization
		Params[ "W1" ] = Matrix<double>.Build.Random( settings.Input, settings.Hidden[ 0 ], normalDist );
		Params[ "b1" ] = Vector<double>.Build.Dense( settings.Hidden[ 0 ], 0.0 );
		Params[ "W2" ] = Matrix<double>.Build.Random( settings.Hidden[ 0 ], settings.Output, normalDist );
		Params[ "b2" ] = Vector<double>.Build.Dense( settings.Output, 0.0 );

		Affine1 = new Affine( ( Matrix<double> )Params[ "W1" ], ( Vector<double> )Params[ "b1" ] );
		Affine2 = new Affine( ( Matrix<double> )Params[ "W2" ], ( Vector<double> )Params[ "b2" ] );
	}

	/// <summary>
	/// Predict: forward pass through layers. 
	/// </summary>
	/// <param name="x">Input data matrix of shape [batch, inputSize].</param>
	/// <returns>Output data matrix of shape [batch, outputSize].</returns>
	Matrix<double> Predict( Matrix<double> x )
	{
		var a1 = Affine1.Forward( x );
		var z1 = Relu1.Forward( a1 ); // Leverages Relu Batch MathNet implementation
		var a2 = Affine2.Forward( z1 );

		return a2;
	}

	/// <summary>
	/// Compute loss given input and target.
	/// Targets t should be one-hot matrix [batch, classes] or label indices [batch,1].
	/// </summary>
	/// <param name="x">Input data matrix of shape [batch, inputSize].</param>
	/// <param name="t">Target data matrix of shape [batch, classes] or [batch,1].</param>
	/// <returns>Loss value as a double.</returns>
	public double Loss( Matrix<double> x, Matrix<double> t )
	{
		var y = Predict( x );

		return LastLayer.Forward( y, t );
	}

	/// <summary>
	/// Compute accuracy. If t is one-hot turn into indices.
	/// </summary>
	/// <param name="x">Input data matrix of shape [batch, inputSize].</param>
	/// <param name="t">Target data matrix of shape [batch, classes] or [batch,1].</param>
	/// <returns>Accuracy as a double between 0 and 1.</returns>
	public double Accuracy( Matrix<double> x, Matrix<double> t )
	{
		var y = Predict( x );
		int batchSize = y.RowCount;
		int correct = 0;

		for( int i = 0; i < batchSize; i++ )
		{			
			int yPred = y.Row( i ).MaximumIndex(); // MathNet MaximumIndex() replaces custom ArgMaxPerRow

			// Extract target index depending on if it's label encoded or one-hot encoded
			int tIndex = t.ColumnCount == 1 ? ( int )t[ i, 0 ] : t.Row( i ).MaximumIndex();

			if( yPred == tIndex ) correct++;			
		}

		return ( double )correct / batchSize;
	}

	/// <summary>
	/// Gradient computed via backprop using existing layer Backward implementations.
	/// Returns gradients dictionary containing W1,b1,W2,b2.	
	/// </summary>
	/// <param name="x">Input data matrix of shape [batch, inputSize].</param>
	/// <param name="t">Target data matrix of shape [batch, classes] or [batch,1].</param>
	/// <returns>Dictionary containing gradients for W1, b1, W2, b2.</returns>
	public Dictionary<string, object> Gradient( Matrix<double> x, Matrix<double> t )
	{
		Loss( x, t ); // forward pass (also stores Y/T in last layer)

		var dout = LastLayer.Backward( 1.0 );
		var d2 = Affine2.Backward( dout );
		var dRelu = Relu1.Backward( d2 );

		Affine1.Backward( dRelu ); // Returned explicitly ignored since it reaches input layer

		return new Dictionary<string, object> {
			[ "W1" ] = Affine1.dW,		[ "b1" ] = Affine1.dB,
			[ "W2" ] = Affine2.dW,		[ "b2" ] = Affine2.dB
		};
	}

	#region obsolete
	//public TwoLayerNet( AppSettings settings )
	//{
	//	Params[ "W1" ] = RandomNormalMatrix( settings.Input, settings.Hidden[0] );
	//	Params[ "b1" ] = new double[ settings.Hidden[0] ];
	//	Params[ "W2" ] = RandomNormalMatrix( settings.Hidden[0], settings.Output );
	//	Params[ "b2" ] = new double[ settings.Output ];

	//	Affine1 = new Affine( ( double[][] )Params[ "W1" ], ( double[] )Params[ "b1" ] );
	//	Affine2 = new Affine( ( double[][] )Params[ "W2" ], ( double[] )Params[ "b2" ] );
	//}


	/// <summary>
	/// Predict: forward pass through layers. Accepts batched input as double[][]: shape [batch][inputSize].
	/// </summary>
	/// <param name="x">Input data of shape [batch, inputSize].</param>
	/// <returns>Output data of shape [batch, outputSize].</returns>	
	//public double[][] Predict( double[][] x )
	//{
	//	var rows = Affine1.Forward( x ); // Affine1 -> Relu (per-row) -> Affine2

	//	// apply Relu per row		
	//	for( int i = 0; i < rows.Length; i++ ) rows[ i ] = Relu1.Forward( rows[ i ] );

	//	var a2 = Affine2.Forward( rows );

	//	return a2;
	//}

	/// <summary>
	/// Compute loss given input and target.
	/// Targets t should be one-hot matrix [batch, classes] or label indices [batch,1].
	/// </summary>
	/// <param name="x">Input data of shape [batch, inputSize].</param>
	/// <param name="t">Target data of shape [batch, classes] or [batch,1].</param>
	/// <returns>Loss value as a double.</returns>
	//public double Loss( double[][] x, double[][] t )
	//{
	//	double[][] y = Predict( x );

	//	return LastLayer.Forward( y, t );
	//}


	/// <summary>
	/// Compute accuracy. If t is one-hot turn into indices.
	/// </summary>
	/// <param name="x">Input data of shape [batch, inputSize].</param>
	/// <param name="t">Target data of shape [batch, classes] or [batch,1].</param>
	/// <returns>Accuracy as a double between 0 and 1.</returns>
	//public double Accuracy( double[][] x, double[][] t )
	//{
	//	double[][] y = Predict( x );
	//	int[] yPred = y.ArgMaxPerRow();

	//	int[] tIndexes = t.Length == 2 && t[0].Length == 1 ? Col0ToIndices( t ) : t.ArgMaxPerRow();

	//	if( yPred.Length != tIndexes.Length ) throw new ArgumentException( "Batch size mismatch between predictions and targets." );

	//	int correct = yPred.Zip( tIndexes, ( pred, target ) => pred == target ? 1 : 0 ).Sum();

	//	return ( double )correct / yPred.Length;
	//}


	/// <summary>
	/// Gradient computed via backprop using existing layer Backward implementations.
	/// Returns gradients dictionary containing W1,b1,W2,b2.	
	/// </summary>
	/// <param name="x">Input data of shape [batch][inputSize].</param>
	/// <param name="t">Target data of shape [batch][classes] or [batch][1].</param>
	/// <returns>Dictionary containing gradients for W1, b1, W2, b2.</returns>
	//public Dictionary<string, object> Gradient( double[][] x, double[][] t )
	//{
	//	Loss( x, t ); // forward (also stores Y/T in last layer)

	//	double[][] dout = LastLayer.Backward( 1 ); // backward
	//	double[][] doutHidden = Affine2.Backward( dout ); // Affine2 backward: returns dout for previous layer

	//	double[][] hiddenRows = doutHidden; // Relu backward: operate per-row
	//	for( int i = 0; i < hiddenRows.Length; i++ ) hiddenRows[ i ] = Relu1.Backward( hiddenRows[ i ] );

	//	Affine1.Backward( hiddenRows ); // we don't need the returned dout for previous layer since it's input layer

	//	var grads = new Dictionary<string, object> {
	//		[ "W1" ] = Affine1.dW!,
	//		[ "b1" ] = Affine1.dB!,
	//		[ "W2" ] = Affine2.dW!,
	//		[ "b2" ] = Affine2.dB!
	//	};

	//	return grads;
	//}
	#endregion

	#region Helpers

	//static double[][] RandomNormalMatrix( int rows, int cols, double std = 0.01 )
	//{
	//	var m = new double[ rows ][];		

	//	for( int i = 0; i < rows; i++ )
	//	{
	//		m[i] = new double[cols];

	//		for( int j = 0; j < cols; j++ )
	//		{
	//			m[ i ][ j ] = BoxMullerNormal() * std;
	//		}
	//	}

	//	return m;
	//}

	/// <summary>
	/// Generates a random number following a standard normal distribution using the Box-Muller transform.
	/// </summary>
	/// <returns>A random number following a standard normal distribution.</returns>
	//static double BoxMullerNormal()
	//{		
	//	double u1 = 1.0 - Random.Shared.NextDouble();
	//	double u2 = 1.0 - Random.Shared.NextDouble();

	//	double r = Math.Sqrt( -2.0 * Math.Log( u1 ) );
	//	double theta = 2.0 * Math.PI * u2;

	//	return r * Math.Cos( theta );
	//}

	/// <summary>
	/// Creates an array of integers by converting the values from the first column of a two-dimensional array of doubles.
	/// </summary>
	/// <remarks>Each value in the first column is cast to an integer, truncating any fractional part.</remarks>
	/// <param name="t">A two-dimensional array of doubles from which to extract and convert the first column values</param>
	/// <returns>An array of integers containing the values from the first column of the input array, converted to integers. 
	/// The length of the returned array matches the number of rows in the input array.</returns>
	//static int[] Col0ToIndices( double[][] t )
	//{
	//	int rows = t.Length;
	//	var result = new int[ rows ];

	//	for( int i = 0; i < rows; i++ ) result[ i ] = ( int )t[ i ][ 0 ];

	//	return result;
	//}

	#endregion
}