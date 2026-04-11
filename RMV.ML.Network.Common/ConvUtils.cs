using MathNet.Numerics.LinearAlgebra;

namespace RMV.ML.Network.Common;

/// <summary>
/// Utility methods for converting between 4D image tensors and 2D column matrices,
/// enabling efficient convolution and pooling via matrix multiplication.
/// </summary>
public static class ConvUtils
{
	/// <summary>
	/// Transforms 4D input (N, C, H, W) into a 2D column matrix for convolution/pooling.
	/// Each row corresponds to one receptive field patch unrolled into a vector.
	/// </summary>
	/// <param name="input">4D input tensor of shape (N, C, H, W)</param>
	/// <param name="filterH">Height of the filter</param>
	/// <param name="filterW">Width of the filter</param>
	/// <param name="stride">Stride of the convolution</param>
	/// <param name="pad">Padding size</param>
	/// <returns>2D column matrix</returns>
	public static Matrix<double> Im2Col( double[,,,] input, int filterH, int filterW, int stride, int pad )
	{
		int N = input.GetLength( 0 );
		int C = input.GetLength( 1 );
		int H = input.GetLength( 2 );
		int W = input.GetLength( 3 );

		int outH = 1 + ( H + 2 * pad - filterH ) / stride;
		int outW = 1 + ( W + 2 * pad - filterW ) / stride;
				
		double[,,,] padded; // Pad input if needed: shape (N, C, H + 2*pad, W + 2*pad)
		if( pad > 0 )
		{
			padded = new double[ N, C, H + 2 * pad, W + 2 * pad ];
			for( int n = 0; n < N; n++ )
				for( int c = 0; c < C; c++ )
					for( int h = 0; h < H; h++ )
						for( int w = 0; w < W; w++ )
							padded[ n, c, h + pad, w + pad ] = input[ n, c, h, w ];
		}
		else
		{
			padded = input;
		}

		// Result: (N * outH * outW) rows x (C * filterH * filterW) columns
		var col = Matrix<double>.Build.Dense( N * outH * outW, C * filterH * filterW );

		int row = 0;
		for( int n = 0; n < N; n++ )
		{
			for( int oh = 0; oh < outH; oh++ )
			{
				for( int ow = 0; ow < outW; ow++ )
				{
					int colIdx = 0;
					for( int c = 0; c < C; c++ )
					{
						for( int fh = 0; fh < filterH; fh++ )
						{
							for( int fw = 0; fw < filterW; fw++ )
							{
								col[ row, colIdx++ ] = padded[ n, c, oh * stride + fh, ow * stride + fw ];
							}
						}
					}
					row++;
				}
			}
		}

		return col;
	}

	/// <summary>
	/// Reconstructs a 4D tensor (N, C, H, W) from a 2D column matrix, reversing the Im2Col operation.
	/// Accumulates overlapping patches additively.
	/// </summary>
	/// <param name="col">2D column matrix</param>
	/// <param name="N">Number of samples in the batch</param>
	/// <param name="C">Number of channels</param>
	/// <param name="H">Height of the original input</param>
	/// <param name="W">Width of the original input</param>
	/// <param name="filterH">Height of the filter</param>
	/// <param name="filterW">Width of the filter</param>
	/// <param name="stride">Stride of the convolution</param>
	/// <param name="pad">Padding size</param>
	/// <returns>4D tensor reconstructed from the column matrix</returns>
	public static double[,,,] Col2Im( Matrix<double> col, int N, int C, int H, int W, int filterH, int filterW, int stride, int pad )
	{
		int outH = 1 + ( H + 2 * pad - filterH ) / stride;
		int outW = 1 + ( W + 2 * pad - filterW ) / stride;

		var padded = new double[ N, C, H + 2 * pad, W + 2 * pad ];

		int row = 0;
		for( int n = 0; n < N; n++ )
		{
			for( int oh = 0; oh < outH; oh++ )
			{
				for( int ow = 0; ow < outW; ow++ )
				{
					int colIdx = 0;
					for( int c = 0; c < C; c++ )
					{
						for( int fh = 0; fh < filterH; fh++ )
						{
							for( int fw = 0; fw < filterW; fw++ )
							{
								padded[ n, c, oh * stride + fh, ow * stride + fw ] += col[ row, colIdx++ ];
							}
						}
					}
					row++;
				}
			}
		}

		// Remove padding
		if( pad == 0 ) return padded;

		var result = new double[ N, C, H, W ];
		for( int n = 0; n < N; n++ )
			for( int c = 0; c < C; c++ )
				for( int h = 0; h < H; h++ )
					for( int w = 0; w < W; w++ )
						result[ n, c, h, w ] = padded[ n, c, h + pad, w + pad ];

		return result;
	}
}
