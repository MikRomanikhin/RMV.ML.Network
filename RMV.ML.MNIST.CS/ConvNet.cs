using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

using RMV.ML.Network.Common;

namespace RMV.ML.MNIST.CS;

/// <summary>
/// Strongly typed parameter set for a convolutional network layer group.
/// </summary>
public record ConvNetParams(
	double[,,,] W1, Vector<double> b1,
	Matrix<double> W2, Vector<double> b2,
	Matrix<double> W3, Vector<double> b3 );

/// <summary>
/// Strongly typed gradient set mirroring <see cref="ConvNetParams"/>.
/// </summary>
public record ConvNetGrads(
	double[,,,] dW1, Vector<double> db1,
	Matrix<double> dW2, Vector<double> db2,
	Matrix<double> dW3, Vector<double> db3 );

/// <summary>
/// Simple ConvNet
/// conv - relu - pool - affine - relu - affine - softmax
/// </summary>
public class SimpleConvNet
{
	public ConvNetParams Params { get; }

	public Convolution Conv1 { get; }
	public Relu Relu1 { get; } = new();
	public Pooling Pool1 { get; }
	public Affine Affine1 { get; }
	public Relu Relu2 { get; } = new();
	public Affine Affine2 { get; }
	public SoftmaxWithLoss LastLayer { get; } = new();

	// Add these fields alongside existing layers to track the shapes dynamically:
	int a1C, a1H, a1W;
	int p1C, p1H, p1W;

	/// <summary>
	/// Constructs a SimpleConvNet.
	/// </summary>
	/// <param name="inputDim">Tuple or array representing (channels, height, width)</param>
	/// <param name="filterNum">Number of filters</param>
	/// <param name="filterSize">Size of the filter</param>
	/// <param name="pad">Padding</param>
	/// <param name="stride">Stride</param>
	/// <param name="hiddenSize">Size of the hidden layer</param>
	/// <param name="outputSize">Number of output classes</param>
	/// <param name="weightInitStd">Standard deviation for weight initialization</param>
	public SimpleConvNet( AppSettings settings )// int[] inputDim, int filterNum = 30, int filterSize = 5, int pad = 0, int stride = 1, int hiddenSize = 100, int outputSize = 10, double weightInitStd = 0.01 )
	{
		int inputChannels = settings.InputDim[ 0 ];
		int inputHeight = settings.InputDim[ 1 ];
		int inputWidth = settings.InputDim[ 2 ];

		int hiddenSize = settings.Hidden[ 0 ];
		int filterNum = settings.Filters[ 0 ];
		int filterSize = settings.Filters[ 1 ];
		int pad = settings.Filters[ 2 ];
		int stride = settings.Filters[ 3 ];

		int convOutputSize = ( inputHeight - filterSize + 2 * pad ) / stride + 1;
		int poolOutputSize = filterNum * ( convOutputSize / 2 ) * ( convOutputSize / 2 );

		var normalDist = new Normal( 0.0, settings.WeightInitStd );

		// W1 shape: (filter_num, input_channels, filter_size, filter_size)
		var w1 = new double[ filterNum, inputChannels, filterSize, filterSize ];

		for( int fn = 0; fn < filterNum; fn++ )
			for( int c = 0; c < inputChannels; c++ )
				for( int fh = 0; fh < filterSize; fh++ )
					for( int fw = 0; fw < filterSize; fw++ )
						w1[ fn, c, fh, fw ] = normalDist.Sample();

		var b1 = Vector<double>.Build.Dense( filterNum, 0.0 );

		// W2 shape: (pool_output_size, hidden_size)
		var w2 = Matrix<double>.Build.Random( poolOutputSize, hiddenSize, normalDist );
		var b2 = Vector<double>.Build.Dense( hiddenSize, 0.0 );
		// W3 shape: (hidden_size, output_size)
		var w3 = Matrix<double>.Build.Random( hiddenSize, settings.Output, normalDist );
		var b3 = Vector<double>.Build.Dense( settings.Output, 0.0 );

		Params = new ConvNetParams( w1, b1, w2, b2, w3, b3 );

		Conv1 = new Convolution( w1, b1, stride, pad );
		Pool1 = new Pooling( 2, 2, 2, 0 ); // pool_h=2, pool_w=2, stride=2, pad=0
		Affine1 = new Affine( w2, b2 );
		Affine2 = new Affine( w3, b3 );
	}

	/// <summary>
	/// Forward pass through the network to compute output predictions.
	/// </summary>
	public Matrix<double> Predict( double[,,,] x )
	{
		var a1 = Conv1.Forward( x );

		// Capture post-conv structural shape for backward un-flattening routing
		a1C = a1.GetLength( 1 ); a1H = a1.GetLength( 2 ); a1W = a1.GetLength( 3 );

		// Note: Depending on your custom Relu handling in Layer.cs, you'll need to flatten a1 
		// into a 2D Matrix<double> before passing into Relu and Affine layer chains.
		var z1Matrix = Flatten( a1 );
		var r1 = Relu1.Forward( z1Matrix );

		// Reshape back if you have a 4D relu/pool flow, or apply relu manually here. Assuming Pool handles 4D
		var p1 = Pool1.Forward( ReshapeTo4D( r1, x.GetLength( 0 ), a1C, a1H, a1W ) );

		// Capture spatial dimensions downstream of pooling layers
		p1C = p1.GetLength( 1 ); p1H = p1.GetLength( 2 ); p1W = p1.GetLength( 3 );

		var a2 = Affine1.Forward( Flatten( p1 ) );
		var z2 = Relu2.Forward( a2 );
		var a3 = Affine2.Forward( z2 );

		return a3;
	}

	/// <summary>
	/// Compute the loss (cost) of the network's predictions.
	/// </summary>
	public double Loss( double[,,,] x, Matrix<double> t )
	{
		var y = Predict( x );

		return LastLayer.Forward( y, t );
	}

	/// <summary>
	/// Calculate the accuracy of the network's predictions.
	/// </summary>
	public double Accuracy( double[,,,] x, Matrix<double> t )
	{
		var y = Predict( x );
		int correct = 0;
		int total = y.RowCount;

		for( int i = 0; i < total; i++ )
		{
			int yPred = y.Row( i ).MaximumIndex();
			int tIndex = t.ColumnCount == 1 ? ( int )t[ i, 0 ] : t.Row( i ).MaximumIndex();

			if( yPred == tIndex ) correct++;
		}

		return ( double )correct / total;
	}

	/// <summary>
	/// Run backpropagation to compute the gradients of the loss with respect to all weights and biases.
	/// </summary>
	public ConvNetGrads Gradient( double[,,,] x, Matrix<double> t )
	{
		Loss( x, t );  // Forward

		// Backward
		double dout1 = 1.0;
		var dout2 = LastLayer.Backward( dout1 );

		dout2 = Affine2.Backward( dout2 );
		dout2 = Relu2.Backward( dout2 );
		dout2 = Affine1.Backward( dout2 );

		// Reshape 2D back down to 4D to route back through Pool and Convolutions
		var dPool = Pool1.Backward( ReshapeTo4D( dout2, x.GetLength( 0 ), p1C, p1H, p1W ) );

		// Unflatten back down to Matrix to pass back backwards
		var dRelu1 = Relu1.Backward( Flatten( dPool ) );
		var dConv1 = Conv1.Backward( ReshapeTo4D( dRelu1, x.GetLength( 0 ), a1C, a1H, a1W ) );

		return new ConvNetGrads( Conv1.dW!, Conv1.dB!, Affine1.dW!, Affine1.dB!, Affine2.dW!, Affine2.dB! );
	}


	#region helpers ------------------------------------------------------------

	// ----------------------------------------------------
	// Helper methods to transition from 4D arrays to 2D MathNet matrices and vice-versa
	// ----------------------------------------------------

	static Matrix<double> Flatten( double[,,,] tensor )
	{
		int n = tensor.GetLength( 0 );
		int c = tensor.GetLength( 1 );
		int h = tensor.GetLength( 2 );
		int w = tensor.GetLength( 3 );

		var matrix = Matrix<double>.Build.Dense( n, c * h * w );

		for( int i = 0; i < n; i++ )
			for( int j = 0; j < c; j++ )
				for( int k = 0; k < h; k++ )
					for( int l = 0; l < w; l++ )
						matrix[ i, j * h * w + k * w + l ] = tensor[ i, j, k, l ];

		return matrix;
	}

	static double[,,,] ReshapeTo4D( Matrix<double> matrix, int batchSize, int c, int h, int w )
	{
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

	#endregion
}