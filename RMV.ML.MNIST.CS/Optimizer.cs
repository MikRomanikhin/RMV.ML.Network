using MathNet.Numerics.LinearAlgebra;

namespace RMV.ML.MNIST.CS;

/// <summary>
/// Interface for optimization algorithms used in training neural networks.
/// Arrays use 1-based indexing (index 0 is unused) to match layer numbering.
/// </summary>
interface IOptimizer
{
	void Update( Matrix<double>[] weights, Vector<double>[] biases, Matrix<double>[] dW, Vector<double>[] dB );
}


/// <summary>
/// Stochastic Gradient Descent
/// </summary>
public class SGD( double rate = 0.01 ) : IOptimizer
{
	public void Update( Matrix<double>[] weights, Vector<double>[] biases, Matrix<double>[] dW, Vector<double>[] dB )
	{
		for( int i = 1; i < weights.Length; i++ )
		{
			weights[ i ].Map2( ( p, g ) => p - ( rate * g ), dW[ i ], weights[ i ] );
			biases[ i ].Map2( ( p, g ) => p - ( rate * g ), dB[ i ], biases[ i ] );
		}
	}
}


/// <summary>
/// Momentum SGD
/// </summary>
public class Momentum( double rate = 0.01, double momentum = 0.9 ) : IOptimizer
{
	Matrix<double>[]? vW;
	Vector<double>[]? vB;

	public void Update( Matrix<double>[] weights, Vector<double>[] biases, Matrix<double>[] dW, Vector<double>[] dB )
	{
		if( vW == null )
		{
			vW = new Matrix<double>[ weights.Length ];
			vB = new Vector<double>[ biases.Length ];

			for( int i = 1; i < weights.Length; i++ )
			{
				vW[ i ] = Matrix<double>.Build.Dense( weights[ i ].RowCount, weights[ i ].ColumnCount, 0.0 );
				vB[ i ] = Vector<double>.Build.Dense( biases[ i ].Count, 0.0 );
			}
		}

		for( int i = 1; i < weights.Length; i++ )
		{
			vW[ i ].Map2( ( vi, gi ) => momentum * vi - rate * gi, dW[ i ], vW[ i ] );
			weights[ i ].Map2( ( pi, vi ) => pi + vi, vW[ i ], weights[ i ] );

			vB![ i ].Map2( ( vi, gi ) => momentum * vi - rate * gi, dB[ i ], vB[ i ] );
			biases[ i ].Map2( ( pi, vi ) => pi + vi, vB[ i ], biases[ i ] );
		}
	}
}

/// <summary>
/// Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)
/// </summary>
public class Nesterov( double rate = 0.01, double momentum = 0.9 ) : IOptimizer
{
	Matrix<double>[]? vW;
	Vector<double>[]? vB;

	public void Update( Matrix<double>[] weights, Vector<double>[] biases, Matrix<double>[] dW, Vector<double>[] dB )
	{
		if( vW == null )
		{
			vW = new Matrix<double>[ weights.Length ];
			vB = new Vector<double>[ biases.Length ];

			for( int i = 1; i < weights.Length; i++ )
			{
				vW[ i ] = Matrix<double>.Build.Dense( weights[ i ].RowCount, weights[ i ].ColumnCount, 0.0 );
				vB[ i ] = Vector<double>.Build.Dense( biases[ i ].Count, 0.0 );
			}
		}

		for( int i = 1; i < weights.Length; i++ )
		{
			var vWPrev = vW[ i ].Clone();
			vW[ i ].Map2( ( vi, gi ) => momentum * vi - rate * gi, dW[ i ], vW[ i ] );
			weights[ i ].Map2( ( pi, vpi ) => pi - momentum * vpi, vWPrev, weights[ i ] );
			weights[ i ].Map2( ( pi, vi ) => pi + ( 1 + momentum ) * vi, vW[ i ], weights[ i ] );

			var vBPrev = vB![ i ].Clone();
			vB[ i ].Map2( ( vi, gi ) => momentum * vi - rate * gi, dB[ i ], vB[ i ] );
			biases[ i ].Map2( ( pi, vpi ) => pi - momentum * vpi, vBPrev, biases[ i ] );
			biases[ i ].Map2( ( pi, vi ) => pi + ( 1 + momentum ) * vi, vB[ i ], biases[ i ] );
		}
	}
}

/// <summary>
/// AdaGrad (http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
/// </summary>
public class AdaGrad( double rate = 0.01 ) : IOptimizer
{
	Matrix<double>[]? hW;
	Vector<double>[]? hB;

	public void Update( Matrix<double>[] weights, Vector<double>[] biases, Matrix<double>[] dW, Vector<double>[] dB )
	{
		if( hW == null )
		{
			hW = new Matrix<double>[ weights.Length ];
			hB = new Vector<double>[ biases.Length ];

			for( int i = 1; i < weights.Length; i++ )
			{
				hW[ i ] = Matrix<double>.Build.Dense( weights[ i ].RowCount, weights[ i ].ColumnCount, 0.0 );
				hB[ i ] = Vector<double>.Build.Dense( biases[ i ].Count, 0.0 );
			}
		}

		for( int i = 1; i < weights.Length; i++ )
		{
			hW[ i ].Map2( ( hi, gi ) => hi + gi * gi, dW[ i ], hW[ i ] );
			var stepW = dW[ i ].Map2( ( gi, hi ) => rate * gi / ( Math.Sqrt( hi ) + 1e-7 ), hW[ i ] );
			weights[ i ].Map2( ( pi, si ) => pi - si, stepW, weights[ i ] );

			hB![ i ].Map2( ( hi, gi ) => hi + gi * gi, dB[ i ], hB[ i ] );
			var stepB = dB[ i ].Map2( ( gi, hi ) => rate * gi / ( Math.Sqrt( hi ) + 1e-7 ), hB[ i ] );
			biases[ i ].Map2( ( pi, si ) => pi - si, stepB, biases[ i ] );
		}
	}
}

/// <summary>
/// RMSprop (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
/// </summary>
public class RmsProp( double rate = 0.01, double decay = 0.99 ) : IOptimizer
{
	Matrix<double>[]? hW;
	Vector<double>[]? hB;

	public void Update( Matrix<double>[] weights, Vector<double>[] biases, Matrix<double>[] dW, Vector<double>[] dB )
	{
		if( hW == null )
		{
			hW = new Matrix<double>[ weights.Length ];
			hB = new Vector<double>[ biases.Length ];

			for( int i = 1; i < weights.Length; i++ )
			{
				hW[ i ] = Matrix<double>.Build.Dense( weights[ i ].RowCount, weights[ i ].ColumnCount, 0.0 );
				hB[ i ] = Vector<double>.Build.Dense( biases[ i ].Count, 0.0 );
			}
		}

		for( int i = 1; i < weights.Length; i++ )
		{
			hW[ i ] *= decay;
			hW[ i ] += ( 1d - decay ) * dW[ i ].PointwiseMultiply( dW[ i ] );
			var hWsqrt = hW[ i ].Map( x => Math.Sqrt( x ) + 1e-7 );
			var stepW = dW[ i ].PointwiseDivide( hWsqrt ) * rate;
			weights[ i ] -= stepW;

			hB![ i ] = hB[ i ].Map( h => h * decay ) + ( 1d - decay ) * dB[ i ].PointwiseMultiply( dB[ i ] );
			var hBsqrt = hB[ i ].Map( x => Math.Sqrt( x ) + 1e-7 );
			var stepB = dB[ i ].PointwiseDivide( hBsqrt ) * rate;
			biases[ i ] -= stepB;
		}
	}
}

/// <summary>
/// Adam (http://arxiv.org/abs/1412.6980v8)
/// </summary>
public class Adam( double rate = 0.001, double beta1 = 0.9, double beta2 = 0.999 ) : IOptimizer
{
	int iter = 0;
	Matrix<double>[]? mW, vW;
	Vector<double>[]? mB, vB;

	public void Update( Matrix<double>[] weights, Vector<double>[] biases, Matrix<double>[] dW, Vector<double>[] dB )
	{
		if( mW == null )
		{
			mW = new Matrix<double>[ weights.Length ];
			vW = new Matrix<double>[ weights.Length ];
			mB = new Vector<double>[ biases.Length ];
			vB = new Vector<double>[ biases.Length ];

			for( int i = 1; i < weights.Length; i++ )
			{
				mW[ i ] = Matrix<double>.Build.Dense( weights[ i ].RowCount, weights[ i ].ColumnCount, 0.0 );
				vW[ i ] = Matrix<double>.Build.Dense( weights[ i ].RowCount, weights[ i ].ColumnCount, 0.0 );
				mB[ i ] = Vector<double>.Build.Dense( biases[ i ].Count, 0.0 );
				vB[ i ] = Vector<double>.Build.Dense( biases[ i ].Count, 0.0 );
			}
		}

		iter++;
		double lr_t = rate * Math.Sqrt( 1.0 - Math.Pow( beta2, iter ) ) / ( 1.0 - Math.Pow( beta1, iter ) );

		for( int i = 1; i < weights.Length; i++ )
		{
			// Weights
			mW[ i ].Map2( ( mi, gi ) => beta1 * mi + ( 1 - beta1 ) * gi, dW[ i ], mW[ i ] );
			vW![ i ].Map2( ( vi, gi ) => beta2 * vi + ( 1 - beta2 ) * gi * gi, dW[ i ], vW[ i ] );
			var stepW = mW[ i ].Map2( ( mi, vi ) => lr_t * mi / ( Math.Sqrt( vi ) + 1e-7 ), vW[ i ] );
			weights[ i ].Map2( ( pi, si ) => pi - si, stepW, weights[ i ] );

			// Biases
			mB![ i ].Map2( ( mi, gi ) => beta1 * mi + ( 1 - beta1 ) * gi, dB[ i ], mB[ i ] );
			vB![ i ].Map2( ( vi, gi ) => beta2 * vi + ( 1 - beta2 ) * gi * gi, dB[ i ], vB[ i ] );
			var stepB = mB[ i ].Map2( ( mi, vi ) => lr_t * mi / ( Math.Sqrt( vi ) + 1e-7 ), vB[ i ] );
			biases[ i ].Map2( ( pi, si ) => pi - si, stepB, biases[ i ] );
		}
	}
}
