using MathNet.Numerics.LinearAlgebra;

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
	Vector<double>? vectorMask;

	// --- Batch (Matrix) Support ---

	public Matrix<double> Forward( Matrix<double> x )
	{		
		this.matrixMask = x.Map( val => val > 0 ? 1d : 0d ); // Map creates a mask: 1.0 if > 0, else 0.0

		// Element-wise multiplication (inherently zeroes out masked values)
		return x.PointwiseMultiply( this.matrixMask );
	}

	public Matrix<double> Backward( Matrix<double> dout )
	{
		if( this.matrixMask is null ) throw new InvalidOperationException( "Forward must be called before Backward." );

		return dout.PointwiseMultiply( this.matrixMask );
	}

	// --- Single Entry (Vector) Support ---

	public Vector<double> Forward( Vector<double> x )
	{
		this.vectorMask = x.Map( val => val > 0 ? 1.0 : 0.0 );

		return x.PointwiseMultiply( this.vectorMask );
	}

	public Vector<double> Backward( Vector<double> dout )
	{
		if( this.vectorMask is null ) throw new InvalidOperationException( "Forward must be called before Backward." );

		return dout.PointwiseMultiply( this.vectorMask );
	}
}

/// <summary>
/// Sigmoid activation function layer
/// </summary>
public class Sigmoid : ILayer
{
	Matrix<double>? outMatrix;
	Vector<double>? outVector;

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

	// --- Single Entry (Vector) Support ---

	public Vector<double> Forward( Vector<double> x )
	{
		this.outVector = x.Map( SigmoidFunc );

		return this.outVector;
	}

	public Vector<double> Backward( Vector<double> dout )
	{
		if( this.outVector is null ) throw new InvalidOperationException( "Forward must be called before Backward." );

		var yOneMinusY = this.outVector.Map( y => y * ( 1.0 - y ) );

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
	Matrix<double> W { get; set; } = W;
	//public Vector<double> B { get; set; } = B;
	public Matrix<double>? dW { get; set; }
	public Vector<double>? dB { get; set; }

	Matrix<double>? x;

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

	public Matrix<double> Backward( Matrix<double> dout )
	{
		if( x is null ) throw new InvalidOperationException( "Forward must be called before Backward." );
				
		var dx = dout.Multiply( this.W.Transpose() ); // dx = dout * W^T
				
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
	//double Loss = 0;
	Matrix<double>? Y; //{ get; private set; }
	Matrix<double>? T;// { get; private set; }

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

	static Matrix<double> Softmax( Matrix<double> x )
	{
		int batchSize = x.RowCount;
		int classes = x.ColumnCount;
		var result = Matrix<double>.Build.Dense( batchSize, classes );

		for( int i = 0; i < batchSize; i++ )
		{
			var row = x.Row( i );
			double max = row.Maximum();

			// Subtract max for numerical stability, then exp
			var expRow = row.Map( val => Math.Exp( val - max ) );

			double sum = expRow.Sum();
						
			result.SetRow( i, expRow.Divide( sum ) );
		}

		return result;
	}

	static double CrossEntropyError( Matrix<double> y, Matrix<double> t )
	{
		int batchSize = y.RowCount;
		double delta = 1e-7;

		if( t.RowCount == y.RowCount && t.ColumnCount == y.ColumnCount )
		{
			var logY = y.Map( val => Math.Log( val + delta ) ); // For one-hot encoded targets: loss = -sum(t * log(y))

			double sum = t.PointwiseMultiply( logY ).ColumnSums().Sum(); // Pointwise multiply t by log(y) and sum all elements

			return -sum / batchSize;
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
