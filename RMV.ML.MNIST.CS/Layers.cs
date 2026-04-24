using MathNet.Numerics.LinearAlgebra;

using RMV.ML.Network.Common;

namespace RMV.ML.MNIST.CS;

/// <summary>
/// Common interface for layers supporting batch (matrix) forward and backward passes.
/// </summary>
public interface ILayer
{
	Matrix<double> Forward( Matrix<double> x );
	Matrix<double> Backward( Matrix<double> dout );
}

/// <summary>
/// Rectified Linear Unit (ReLU) activation function layer
/// </summary>
public class Relu : ILayer
{
	Matrix<double>? matrixMask;
		
	/// <summary>
	/// Forward pass of the ReLU activation function.
	/// </summary>
	/// <param name="x">Input matrix.</param>
	/// <returns>Output matrix after applying ReLU.</returns>
	public Matrix<double> Forward( Matrix<double> x )
	{		
		this.matrixMask = x.Map( val => val > 0 ? 1d : 0d ); // Map creates a mask: 1.0 if > 0, else 0.0

		// Element-wise multiplication (inherently zeroes out masked values)
		return x.PointwiseMultiply( this.matrixMask );
	}
	
	/// <summary>
	/// Backward pass of the ReLU activation function.
	/// </summary>
	/// <param name="dout">Gradient of the loss with respect to the output.</param>
	/// <returns>Gradient of the loss with respect to the input.</returns>
	public Matrix<double> Backward( Matrix<double> dout )
	{
		if( this.matrixMask is null ) throw new InvalidOperationException( "Forward must be called before Backward." );

		return dout.PointwiseMultiply( this.matrixMask );
	}	
}

/// <summary>
/// Sigmoid activation function layer
/// </summary>
public class Sigmoid : ILayer
{
	Matrix<double>? outMatrix;
	//Vector<double>? outVector;

	static double SigmoidFunc( double x ) => 1.0 / ( 1.0 + Math.Exp( -x ) );

	// --- Batch (Matrix) Support ---

	public Matrix<double> Forward( Matrix<double> x )
	{
		this.outMatrix = x.Map( SigmoidFunc );

		return this.outMatrix;
	}

	public Matrix<double> Backward( Matrix<double> dout )
	{
		if( this.outMatrix is null ) throw new InvalidOperationException( "Forward must be called before Backward." );
				
		// Map y * (1-y) efficiently, then pointwise multiply by dout
		var yOneMinusY = this.outMatrix.Map( y => y * ( 1.0 - y ) ); // dx = dout * y * (1 - y)

		return dout.PointwiseMultiply( yOneMinusY );
	}	
}

/// <summary>
/// Affine (Fully Connected) layer: computes output = input * W + b
/// </summary>
/// <param name="W"></param>
/// <param name="B"></param>
public class Affine( Matrix<double> W, Vector<double> B ) : ILayer
{	
	public Matrix<double>? dW { get; set; }
	public Vector<double>? dB { get; set; }

	Matrix<double>? x;
	
	/// <summary>
	/// Forward pass of the affine (fully connected) layer.
	/// </summary>
	/// <param name="input">Input matrix.</param>
	/// <returns>Output matrix after applying the affine transformation.</returns>
	public Matrix<double> Forward( Matrix<double> input )
	{
		this.x = input; // store input for backward pass
				
		var outMatrix = this.x.Multiply( W ); // Core matrix multiplication
				
		for( int i = 0; i < outMatrix.RowCount; i++ )
		{
			var row = outMatrix.Row( i );
			row.Add( B, row ); // In-place addition of the bias vector to the current row
			outMatrix.SetRow( i, row );
		}

		return outMatrix;
	}

	/// <summary>
	/// Backward pass of the affine (fully connected) layer.
	/// </summary>
	/// <param name="dout">Gradient of the loss with respect to the output.</param>
	/// <returns>Gradient of the loss with respect to the input.</returns>
	public Matrix<double> Backward( Matrix<double> dout )
	{
		if( x is null ) throw new InvalidOperationException( "Forward must be called before Backward." );
				
		var dx = dout.Multiply( W.Transpose() ); // dx = dout * W^T
				
		this.dW = x.TransposeThisAndMultiply( dout ); // dW = x^T * dout (Optimized to avoid explicit transpose allocation)

		this.dB = dout.ColumnSums(); // dB = column-wise sum of dout (Leverages MathNet's internal optimizations)
						
		return dx; // dx is inherently guaranteed to match the original [RowCount, ColumnCount].
	}
}

/// <summary>
/// Softmax activation combined with Cross-Entropy Loss layer. Computes softmax probabilities and loss in the forward pass,
/// </summary>
public class SoftmaxWithLoss
{	
	Matrix<double>? Y; 
	Matrix<double>? T;

	/// <summary>
	/// Compute softmax probabilities and cross-entropy loss for inputs <paramref name="x"/> and targets <paramref name="t"/>.
	/// Supports both one-hot targets and label-index targets (shape [batch,1]).
	/// </summary>
	public double Forward( Matrix<double> x, Matrix<double> t )
	{
		ArgumentNullException.ThrowIfNull( x );
		ArgumentNullException.ThrowIfNull( t );

		this.T = t;
		this.Y = Softmax( x );

		return CrossEntropyError( this.Y, this.T );
	}
	
	/// <summary>
	/// Backward pass of the softmax with loss layer.
	/// </summary>
	/// <param name="dout">Gradient of the loss with respect to the output.</param>
	/// <returns>Gradient of the loss with respect to the input.</returns>
	public Matrix<double> Backward( double dout = 1 )
	{
		if( this.Y is null || this.T is null ) throw new InvalidOperationException( "Forward must be called before Backward." );

		int batchSize = this.T.RowCount;
		int classes = this.Y.ColumnCount;

		var dx = Matrix<double>.Build.Dense( batchSize, classes );

		if( this.T.RowCount == this.Y.RowCount && this.T.ColumnCount == this.Y.ColumnCount )
		{
			dx = ( this.Y - this.T ).Divide( batchSize );
		}
		else // T contains label indices in column 0
		{
			this.Y.CopyTo( dx ); // Start with clone of Y then subtract 1 at the label index

			for( int i = 0; i < batchSize; i++ )
			{
				int label = ( int )this.T[ i, 0 ];
				if( label < 0 || label >= classes ) throw new ArgumentOutOfRangeException( nameof( T ), "Label index out of range." );

				dx[ i, label ] -= 1.0;
			}

			dx = dx.Divide( batchSize ); // scale by batch size			
		}

		if( dout != 1 ) dx = dx.Multiply( dout ); // propagate external scale (dout) if not 1								

		return dx;
	}

	/// <summary>
	/// Computes the softmax probabilities for each row of the input matrix.
	/// </summary>
	/// <param name="x">Input matrix.</param>
	/// <returns>Matrix of softmax probabilities.</returns>
	static Matrix<double> Softmax( Matrix<double> x )
	{
		int batchSize = x.RowCount;
		int classes = x.ColumnCount;
		var result = Matrix<double>.Build.Dense( batchSize, classes );

		for( int i = 0; i < batchSize; i++ )
		{
			var row = x.Row( i );
			double max = row.Maximum();
			
			var expRow = row.Map( val => Math.Exp( val - max ) );

			double sum = expRow.Sum();

			result.SetRow( i, expRow.Divide( sum ) );
		}

		return result;
	}

	/// <summary>
	/// Computes the cross-entropy error between the predicted probabilities and the target labels.
	/// Supports both one-hot encoded targets and label indices.
	/// </summary>
	/// <param name="y">Predicted probabilities (softmax output).</param>
	/// <param name="t">Target labels (one-hot or label indices).</param>
	/// <returns>Average cross-entropy loss over the batch.</returns>
	static double CrossEntropyError( Matrix<double> y, Matrix<double> t )
	{
		int batchSize = y.RowCount;
		const double delta = 1e-7;

		if( t.RowCount == y.RowCount && t.ColumnCount == y.ColumnCount )
		{
			var logY = y.Map( val => Math.Log( val + delta ) ); // For one-hot encoded targets: loss = -sum(t * log(y))

			double sum = t.PointwiseMultiply( logY ).ColumnSums().Sum(); // Pointwise multiply t by log(y) and sum all elements

			return -sum / batchSize; // Average loss over the batch
		}

		double loss = 0;
		for( int i = 0; i < batchSize; i++ )
		{
			int label = ( int )t[ i, 0 ]; // For label indices in the first column of t
			loss -= Math.Log( y[ i, label ] + delta );
		}

		return loss / batchSize;
	}
}


/// <summary>
/// Randomly sets a fraction of input units to zero during training to prevent overfitting. 
/// During inference, scales the output by (1 - dropoutRatio) to maintain expected values.
/// </summary>
/// <param name="ratio">dropout ratio</param>
public class Dropout( double ratio = 0.5 ) : ILayer
{
	Matrix<double>? mask;

	public bool TrainFlag { get; set; } = true;

	public Matrix<double> Forward( Matrix<double> x )
	{
		if( TrainFlag )
		{
			this.mask = x.Map( _ => Random.Shared.NextDouble() > ratio ? 1.0 : 0.0 );

			return x.PointwiseMultiply( this.mask );
		}

		return x.Multiply( 1.0 - ratio );
	}

	public Matrix<double> Backward( Matrix<double> dout )
	{
		if( this.mask is null ) throw new InvalidOperationException( "Forward must be called before Backward." );

		return dout.PointwiseMultiply( this.mask );
	}
}


/// <summary>
/// Convolution layer: applies learned filters over 4D input (N, C, H, W) using im2col-based matrix multiplication.
/// </summary>
/// <param name="W">Filter weights with shape (FN, C, FH, FW)</param>
/// <param name="B">Bias vector of length FN</param>
/// <param name="stride">Convolution stride</param>
/// <param name="pad">Zero-padding size</param>
public class Convolution( double[,,,] W, Vector<double> B, int stride = 1, int pad = 0 )
{
	public double[,,,]? dW { get; set; }
	public Vector<double>? dB { get; set; }

	double[,,,]? x;
	Matrix<double>? col;
	Matrix<double>? colW;

	/// <summary>
	/// Forward pass: convolves input with filters and adds bias.
	/// Input shape: (N, C, H, W). Output shape: (N, FN, outH, outW).
	/// </summary>
	public double[,,,] Forward( double[,,,] input )
	{
		int FN = W.GetLength( 0 ); // number of filters
		int C = W.GetLength( 1 );  // channels
		int FH = W.GetLength( 2 ); // filter height
		int FW = W.GetLength( 3 ); // filter width

		int N = input.GetLength( 0 );   // batch size
		int H = input.GetLength( 2 );   // input height
		int inW = input.GetLength( 3 ); // input width

		int outH = 1 + ( H + 2 * pad - FH ) / stride;   // output height based on input height, filter height, padding, and stride
		int outW = 1 + ( inW + 2 * pad - FW ) / stride; // output width based on input width, filter width, padding, and stride

		this.col = ConvUtils.Im2Col( input, FH, FW, stride, pad ); // col shape: (N*outH*outW, C*FH*FW)

		// Reshape W from (FN, C, FH, FW) to (C*FH*FW, FN) — column-major filter matrix
		this.colW = Matrix<double>.Build.Dense( C * FH * FW, FN );
		for( int fn = 0; fn < FN; fn++ )
		{
			int idx = 0;
			for( int c = 0; c < C; c++ )
				for( int fh = 0; fh < FH; fh++ )
					for( int fw = 0; fw < FW; fw++ )
						this.colW[ idx++, fn ] = W[ fn, c, fh, fw ];
		}

		var outMatrix = this.col.Multiply( this.colW ); // out = col * colW + b, result shape: (N*outH*outW, FN)

		for( int i = 0; i < outMatrix.RowCount; i++ ) // Add bias B to each row of the output matrix
		{
			var row = outMatrix.Row( i );
			row.Add( B, row );
			outMatrix.SetRow( i, row );
		}

		// Reshape (N*outH*outW, FN) → (N, outH, outW, FN) → transpose to (N, FN, outH, outW)
		var result = new double[ N, FN, outH, outW ];
		int r = 0;
		for( int n = 0; n < N; n++ )
			for( int oh = 0; oh < outH; oh++ )
				for( int ow = 0; ow < outW; ow++ )
				{
					for( int fn = 0; fn < FN; fn++ ) result[ n, fn, oh, ow ] = outMatrix[ r, fn ];
					r++;
				}

		this.x = input;

		return result;
	}

	/// <summary>
	/// Backward pass: computes gradients for filters (dW), biases (dB), and input (dx).
	/// </summary>
	public double[,,,] Backward( double[,,,] dout )
	{
		if( this.x is null || this.col is null || this.colW is null )
			throw new InvalidOperationException( "Forward must be called before Backward." );

		int FN = W.GetLength( 0 );
		int C = W.GetLength( 1 );
		int FH = W.GetLength( 2 );
		int FW = W.GetLength( 3 );

		int N = dout.GetLength( 0 );
		int outH = dout.GetLength( 2 );
		int outW = dout.GetLength( 3 );

		// Transpose dout from (N, FN, outH, outW) → reshape to (N*outH*outW, FN)
		var doutMatrix = Matrix<double>.Build.Dense( N * outH * outW, FN );
		int r = 0;
		for( int n = 0; n < N; n++ )
			for( int oh = 0; oh < outH; oh++ )
				for( int ow = 0; ow < outW; ow++ )
				{
					for( int fn = 0; fn < FN; fn++ ) doutMatrix[ r, fn ] = dout[ n, fn, oh, ow ];
					r++;
				}

		this.dB = doutMatrix.ColumnSums(); // dB = sum of dout along axis 0

		var dWMatrix = this.col.TransposeThisAndMultiply( doutMatrix ); // dW = col^T * dout, shape: (C*FH*FW, FN)

		this.dW = new double[ FN, C, FH, FW ]; // Reshape dW from (C*FH*FW, FN) → (FN, C, FH, FW)
		for( int fn = 0; fn < FN; fn++ )
		{
			int idx = 0;
			for( int c = 0; c < C; c++ )
				for( int fh = 0; fh < FH; fh++ )
					for( int fw = 0; fw < FW; fw++ )
						this.dW[ fn, c, fh, fw ] = dWMatrix[ idx++, fn ];
		}

		var dcol = doutMatrix.Multiply( this.colW.Transpose() ); // dcol = dout * colW^T

		int xN = this.x.GetLength( 0 );
		int xC = this.x.GetLength( 1 );
		int xH = this.x.GetLength( 2 );
		int xW = this.x.GetLength( 3 );

		return ConvUtils.Col2Im( dcol, xN, xC, xH, xW, FH, FW, stride, pad );
	}
}

	/// <summary>
	/// Max pooling layer: downsamples 4D input (N, C, H, W) by taking the maximum value in each pooling window.
	/// </summary>
	/// <param name="poolH">Pooling window height</param>
	/// <param name="poolW">Pooling window width</param>
	/// <param name="stride">Pooling stride</param>
	/// <param name="pad">Zero-padding size</param>
	public class Pooling( int poolH, int poolW, int stride = 2, int pad = 0 )
{
	double[,,,]? x;
	int[]? argMax;

	/// <summary>
	/// Forward pass: applies max pooling over each spatial window.
	/// Input shape: (N, C, H, W). Output shape: (N, C, outH, outW).
	/// </summary>
	public double[,,,] Forward( double[,,,] input )
	{
		int N = input.GetLength( 0 );
		int C = input.GetLength( 1 );
		int H = input.GetLength( 2 );
		int W = input.GetLength( 3 );

		int outH = 1 + ( H - poolH ) / stride;
		int outW = 1 + ( W - poolW ) / stride;

		// im2col then reshape each row to pool_size columns
		var col = ConvUtils.Im2Col( input, poolH, poolW, stride, pad );

		int poolSize = poolH * poolW;
		int totalWindows = col.RowCount; // N * outH * outW (per channel, but im2col groups C*poolH*poolW per row)

		// Reshape col: each row has C * poolSize values → split into C groups of poolSize
		int totalPatches = N * C * outH * outW;
		var colReshaped = Matrix<double>.Build.Dense( totalPatches, poolSize );

		// Rearrange from (N*outH*outW, C*poolSize) to (N*C*outH*outW, poolSize)
		int srcRow = 0;
		for( int n = 0; n < N; n++ )
		{
			for( int oh = 0; oh < outH; oh++ )
			{
				for( int ow = 0; ow < outW; ow++ )
				{
					for( int c = 0; c < C; c++ )
					{
						int dstRow = ( ( n * C + c ) * outH + oh ) * outW + ow;
						for( int p = 0; p < poolSize; p++ )
							colReshaped[ dstRow, p ] = col[ srcRow, c * poolSize + p ];
					}
					srcRow++;
				}
			}
		}

		// Compute argmax and max per row
		this.argMax = new int[ totalPatches ];
		var result = new double[ N, C, outH, outW ];

		for( int i = 0; i < totalPatches; i++ )
		{
			double max = double.NegativeInfinity;
			int maxIdx = 0;
			for( int p = 0; p < poolSize; p++ )
			{
				if( colReshaped[ i, p ] > max )
				{
					max = colReshaped[ i, p ];
					maxIdx = p;
				}
			}
			this.argMax[ i ] = maxIdx;

			// Map flat index i back to (n, c, oh, ow)
			int ow2 = i % outW;
			int oh2 = ( i / outW ) % outH;
			int c2 = ( i / ( outW * outH ) ) % C;
			int n2 = i / ( outW * outH * C );

			result[ n2, c2, oh2, ow2 ] = max;
		}

		this.x = input;

		return result;
	}

	/// <summary>
	/// Backward pass: routes gradients back to the positions of the maximum values selected during forward.
	/// </summary>
	public double[,,,] Backward( double[,,,] dout )
	{
		if( this.x is null || this.argMax is null )
			throw new InvalidOperationException( "Forward must be called before Backward." );

		int N = dout.GetLength( 0 );
		int C = dout.GetLength( 1 );
		int outH = dout.GetLength( 2 );
		int outW = dout.GetLength( 3 );

		int poolSize = poolH * poolW;
		int totalPatches = N * C * outH * outW;

		// Build dmax: (totalPatches, poolSize) — sparse, only argMax positions get the gradient
		var dmax = Matrix<double>.Build.Dense( totalPatches, poolSize );
		for( int i = 0; i < totalPatches; i++ )
		{
			int ow2 = i % outW;
			int oh2 = ( i / outW ) % outH;
			int c2 = ( i / ( outW * outH ) ) % C;
			int n2 = i / ( outW * outH * C );

			dmax[ i, this.argMax[ i ] ] = dout[ n2, c2, oh2, ow2 ];
		}

		// Rearrange back from (N*C*outH*outW, poolSize) to (N*outH*outW, C*poolSize)
		var dcol = Matrix<double>.Build.Dense( N * outH * outW, C * poolSize );
		int dstRow = 0;
		for( int n = 0; n < N; n++ )
		{
			for( int oh = 0; oh < outH; oh++ )
			{
				for( int ow = 0; ow < outW; ow++ )
				{
					for( int c = 0; c < C; c++ )
					{
						int srcRow = ( ( n * C + c ) * outH + oh ) * outW + ow;
						for( int p = 0; p < poolSize; p++ )
							dcol[ dstRow, c * poolSize + p ] = dmax[ srcRow, p ];
					}
					dstRow++;
				}
			}
		}

		int xH = this.x.GetLength( 2 );
		int xW = this.x.GetLength( 3 );

		return ConvUtils.Col2Im( dcol, N, C, xH, xW, poolH, poolW, stride, pad );
	}
}
