
using System.Numerics;

namespace RMV.ML.MNIST.Matrix;

/// <summary>
/// Extension methods for matrix operations
/// </summary>
public static class MatrixExtensions
{
	/// <summary>
	/// Transposes a matrix
	/// </summary>
	/// <param name="matrix">target matrix</param>
	/// <returns>transposed matrix</returns>
	public static T[,] Transpose<T>( this T[,] matrix )
	{
		int rows = matrix.GetLength( 0 );
		int cols = matrix.GetLength( 1 );
		var result = new T[ cols, rows ];

		for( int i = 0; i < rows; i++ )
			for( int j = 0; j < cols; j++ )
				result[ j, i ] = matrix[ i, j ];

		return result;
	}

	public static T[][] Transpose<T>( this T[][] matrix )
	{
		int rows = matrix.Length;
		int cols = matrix[0].Length;
		var result = new T[ cols ][];

		for( int i = 0; i < cols; i++ )
		{
			result[ i ] = new T[ rows ];
		}
		for( int i = 0; i < rows; i++ )
			for( int j = 0; j < cols; j++ )
				result[ j ][ i ] = matrix[ i ][ j ];

		return result;
	}

	/// <summary>
	/// Reshapes a two-dimensional array into a new array with the specified number of rows and columns.
	/// </summary>	
	/// <param name="array">The source two-dimensional array to reshape. Cannot be null.</param>
	/// <param name="rows">The number of rows for the resulting array. Must be greater than 0.</param>
	/// <param name="cols">The number of columns for the resulting array. Must be greater than 0.</param>
	/// <returns>A new two-dimensional array with the specified dimensions</returns>
	public static T[,] Reshape<T>( this T[,] array, int rows, int cols )
	{		
		T[] flat = [ .. array.Cast<T>() ]; // flatten the 2D array to 1D

		T[,] result = new T[ rows, cols ];
		int idx = 0;
		for( int i = 0; i < rows; i++ )
			for( int j = 0; j < cols; j++ )
				result[ i, j ] = flat[ idx++ ];

		return result;
	}

	/// <summary>
	/// Reshapes a jagged array into a new jagged array with the specified number of rows and columns.
	/// </summary>
	/// <typeparam name="T">The type of the elements in the array.</typeparam>
	/// <param name="array">The source jagged array to reshape.</param>
	/// <param name="rows">The number of rows for the resulting jagged array.</param>
	/// <param name="cols">The number of columns for the resulting jagged array.</param>
	/// <returns>A new jagged array with the specified dimensions</returns>
	public static T[][] Reshape<T>( this T[][] array, int rows, int cols )
	{
		int srcRows = array.Length;
		int srcCols = array[ 0 ].Length;
		int totalElements = srcRows * srcCols;

		// Support -1 to infer dimension (NumPy convention)
		if( rows == -1 && cols == -1 ) throw new ArgumentException( "Only one dimension can be -1." );
		if( rows == -1 ) rows = totalElements / cols;
		if( cols == -1 ) cols = totalElements / rows;
		if( rows * cols != totalElements )	throw new ArgumentException( $"Cannot reshape array of size {totalElements} into shape ({rows}, {cols})." );

		var result = new T[ rows ][];

		int k = 0; // index in the flattened source array

		for( int i = 0; i < rows; i++ )
		{
			result[ i ] = new T[ cols ]; // initialize the row

			for( int j = 0; j < cols; j++ )
			{
				result[ i ][ j ] = array[ k / srcCols ][ k % srcCols ];
				k++;
			}
		}

		return result;
	}
	//public static T[][] Reshape<T>( this T[][] array, int rows, int cols )
	//{		
	//	int srcCols = array[0].Length;
	//	T[][] result = new T[ rows ][];

	//	int k = 0; // index in the flattened source array

	//	for( int i = 0; i < rows; i++ )
	//	{
	//		result[ i ] = new T[ cols ]; // initialize the row

	//		for( int j = 0; j < cols; j++ )
	//		{				
	//			result[ i ][ j ] = array[ k / srcCols ][ k % srcCols ];
	//			k++;
	//		}
	//	}

	//	return result;
	//}

	/// <summary>
	/// Converts a 2D array into a jagged array (array of arrays) where each inner array represents a row.
	/// </summary>
	/// <param name="matrix">The source 2D array.</param>
	/// <returns>A jagged array where each element is a row from the original matrix.</returns>
	public static T[][] ToJagged<T>( this T[,] matrix )
	{
		int rows = matrix.GetLength( 0 );
		int cols = matrix.GetLength( 1 );
		var jagged = new T[ rows ][];

		for( int i = 0; i < rows; i++ )
		{
			jagged[ i ] = new T[ cols ];

			for( int j = 0; j < cols; j++ )
			{
				jagged[ i ][ j ] = matrix[ i, j ];
			}
		}

		return jagged;
	}

	/// <summary>
	/// Converts a jagged array (array of arrays) where each inner array represents a row into a 2D array.
	/// </summary>
	/// <param name="jagged">The jagged array where each element is a row.</param>
	/// <returns>A 2D array representing the same data as the jagged array.</returns>
	/// <exception cref="ArgumentException">Thrown if the rows have inconsistent lengths.</exception>
	public static T[,] FromJagged<T>( this T[][] jagged )
	{
		if( jagged.Length == 0 ) return new T[ 0, 0 ];
		int r = jagged.Length;
		int c = jagged[ 0 ].Length;
		var m = new T[ r, c ];

		for( int i = 0; i < r; i++ )
		{
			if( jagged[ i ].Length != c ) throw new ArgumentException( "Inconsistent row lengths." );
			for( int j = 0; j < c; j++ ) m[ i, j ] = jagged[ i ][ j ];
		}

		return m;
	}

	/// <summary>
	/// Adds a matrix and a vector
	/// </summary>
	/// <param name="matrix">Matrix (m x n)</param>
	/// <param name="vector">Vector (n)</param>
	/// <returns>Resulting matrix (m x n)</returns>
	public static double[,] Add( this double[,] matrix, double[] vector )
	{
		ArgumentNullException.ThrowIfNull( matrix );
		ArgumentNullException.ThrowIfNull( vector );
		int rows = matrix.GetLength( 0 ); int cols = matrix.GetLength( 1 );
		
		if( vector.Length != cols ) throw new ArgumentException( "Matrix columns must match vector length." );

		var result = new double[ rows, cols ];

		for( int i = 0; i < rows; i++ )
			for( int j = 0; j < cols; j++ )
				result[ i, j ] = matrix[ i, j ] + vector[ j ];

		return result;
	}

	public static double[][] Add( this double[][] matrix, double[] vector )
	{
		ArgumentNullException.ThrowIfNull( matrix );
		ArgumentNullException.ThrowIfNull( vector );
		int rows = matrix.Length; int cols = matrix[ 0 ].Length;
		if( vector.Length != cols ) throw new ArgumentException( "Matrix columns must match vector length." );

		var result = new double[ rows ][];
		for( int i = 0; i < rows; i++ )
		{
			result[ i ] = new double[ cols ];
			for( int j = 0; j < cols; j++ )
				result[ i ][ j ] = matrix[ i ][ j ] + vector[ j ];
		}

		return result;
	}	


	/// <summary>
	/// Compute column-wise sums for a 2D array.
	/// </summary>
	/// <param name="matrix">The source matrix.</param>
	/// <returns>An array containing the sum of each column.</returns>
	public static double[] SumColumns( this double[,] matrix )
	{
		ArgumentNullException.ThrowIfNull( matrix );

		int rows = matrix.GetLength( 0 );
		int cols = matrix.GetLength( 1 );
		var sums = new double[ cols ];

		for( int j = 0; j < cols; j++ )
		{
			double sum = 0;
			for( int i = 0; i < rows; i++ )
				sum += matrix[ i, j ];
			sums[ j ] = sum;
		}

		return sums;
	}

	public static double[] SumColumns( this double[][] matrix )
	{
		ArgumentNullException.ThrowIfNull( matrix );

		int rows = matrix.Length;
		int cols = matrix[ 0 ].Length;
		var sums = new double[ cols ];

		for( int i = 0; i < rows; i++ )
		{
			var row = matrix[ i ] ?? throw new ArgumentException( $"Row {i} is null.", nameof( matrix ) );
			if( row.Length != cols ) throw new ArgumentException( "Inconsistent row lengths.", nameof( matrix ) );

			// cache local reference to avoid repeated range checks on matrix
			for( int j = 0; j < cols; j++ )	sums[ j ] += row[ j ];
		}

		return sums;
	}

	/// <summary>
	/// Returns the indices of the maximum values for each row in a matrix.
	/// </summary>
	/// <param name="matrix">The source matrix.</param>
	/// <returns>An array containing the index of the maximum value for each row.</returns>
	public static int[] ArgMaxPerRow( this double[,] matrix )
	{
		int rows = matrix.GetLength( 0 );
		int cols = matrix.GetLength( 1 );
		var result = new int[ rows ];

		for( int i = 0; i < rows; i++ )
		{
			int index = 0;
			double max = matrix[ i, 0 ];

			for( int j = 1; j < cols; j++ )
			{
				if( matrix[ i, j ] > max )
				{
					max = matrix[ i, j ];
					index = j;
				}
			}

			result[ i ] = index;
		}

		return result;
	}

	public static int[] ArgMaxPerRow( this double[][] matrix )
	{
		ArgumentNullException.ThrowIfNull( matrix );

		int rows = matrix.Length;
		int cols = matrix[ 0 ].Length;
		var result = new int[ rows ];

		for( int i = 0; i < rows; i++ )
		{			
			var row = matrix[ i ] ?? throw new ArgumentException( $"Row {i} is null.", nameof( matrix ) );
			if( row.Length != cols ) throw new ArgumentException( "Inconsistent row lengths.", nameof( matrix ) );
			
			double max = row[ 0 ];
			int index = 0;

			for( int j = 1; j < cols; j++ )
			{
				if( row[ j ] > max )
				{
					max = row[ j ];	index = j;
				}
			}

			result[ i ] = index;
		}

		return result;
	}

	/// <summary>
	/// Multiplies two matrices
	/// </summary>
	/// <param name="a">Left matrix</param>
	/// <param name="b">Right matrix</param>
	/// <returns>Product matrix</returns>
	public static double[,] Mult( this double[,] a, double[,] b )
	{
		int aRows = a.GetLength( 0 ); int aCols = a.GetLength( 1 );
		int bRows = b.GetLength( 0 ); int bCols = b.GetLength( 1 );
		var result = new double[ aRows, bCols ];

		if( aCols != bRows ) throw new ArgumentException( "Matrix dimensions are not compatible for multiplication." );

		for( int i = 0; i < aRows; i++ )
			for( int j = 0; j < bCols; j++ )
				for( int k = 0; k < aCols; k++ )
					result[ i, j ] += a[ i, k ] * b[ k, j ];

		return result;
	}

	public static double[][] Mult( this double[][] a, double[][] b )
	{
		int aRows = a.Length; int aCols = a[ 0 ].Length;
		int bRows = b.Length; int bCols = b[ 0 ].Length;
		var result = new double[ aRows][];

		for( int i = 0; i < aRows; i++ ) result[ i ] = new double[ bCols ];

		if( aCols != bRows ) throw new ArgumentException( "Matrix dimensions are not compatible for multiplication." );

		for( int i = 0; i < aRows; i++ )
			for( int j = 0; j < bCols; j++ )
				for( int k = 0; k < aCols; k++ )
					result[ i ][ j ] += a[ i ][ k ] * b[ k ][ j ];

		return result;
	}

	/// <summary>
	/// Multiplies a matrix by a vector
	/// </summary>
	/// <param name="matrix">Matrix (m x n)</param>
	/// <param name="vector">Vector (n)</param>
	/// <returns>Resulting vector (m)</returns>
	public static double[] Mult( this double[,] matrix, double[] vector )
	{
		int rows = matrix.GetLength( 0 ); int cols = matrix.GetLength( 1 );
		var result = new double[ rows ];

		if( vector.Length != cols ) throw new ArgumentException( "Matrix columns must match vector length." );

		for( int i = 0; i < rows; i++ )
		{
			double sum = 0;

			for( int j = 0; j < cols; j++ ) sum += matrix[ i, j ] * vector[ j ];

			result[ i ] = sum;
		}

		return result;
	}

	public static double[] Mult( this double[][] matrix, double[] vector )
	{
		int rows = matrix.Length; int cols = matrix[ 0 ].Length;
		var result = new double[ rows ];

		if( vector.Length != cols ) throw new ArgumentException( "Matrix columns must match vector length." );

		for( int i = 0; i < rows; i++ )
		{
			double sum = 0;

			for( int j = 0; j < cols; j++ ) sum += matrix[ i ][ j ] * vector[ j ];

			result[ i ] = sum;
		}

		return result;
	}

	/// <summary>
	/// Multiplies a vector by a matrix
	/// </summary>
	/// <param name="vector">Vector (length m)</param>
	/// <param name="matrix">Matrix (m x n)</param>
	/// <returns>Resulting vector (length n)</returns>
	public static double[] Mult( this double[] vector, double[,] matrix )
	{
		int m = vector.Length;
		int rows = matrix.GetLength( 0 ); int cols = matrix.GetLength( 1 );
		var result = new double[ cols ];

		if( m != rows ) throw new ArgumentException( "Vector length must match matrix row count." );

		for( int j = 0; j < cols; j++ )
		{
			double sum = 0;

			for( int i = 0; i < m; i++ ) sum += vector[ i ] * matrix[ i, j ];

			result[ j ] = sum;
		}

		return result;
	}

	public static double[] Mult( this double[] vector, double[][] matrix )
	{
		int m = vector.Length;
		int rows = matrix.Length; int cols = matrix[ 0 ].Length;
		var result = new double[ cols ];

		if( m != rows ) throw new ArgumentException( "Vector length must match matrix row count." );

		for( int j = 0; j < cols; j++ )
		{
			double sum = 0;

			for( int i = 0; i < m; i++ ) sum += vector[ i ] * matrix[ i ][ j ];

			result[ j ] = sum;
		}

		return result;
	}

	/// <summary>
	/// Multiplies a matrix by a scalar
	/// </summary>
	/// <param name="matrix">Matrix</param>
	/// <param name="scalar">Scalar value</param>
	/// <returns>Resulting matrix</returns>
	public static double[,] Mult( this double[,] matrix, double scalar )
	{
		int rows = matrix.GetLength( 0 ); int cols = matrix.GetLength( 1 );
		var result = new double[ rows, cols ];

		for( int i = 0; i < rows; i++ )
			for( int j = 0; j < cols; j++ )
				result[ i, j ] = matrix[ i, j ] * scalar;

		return result;
	}

	public static double[][] Mult( this double[][] matrix, double scalar )
	{
		int rows = matrix.Length; int cols = matrix[ 0 ].Length;
		var result = new double[ rows ][];
		for( int i = 0; i < rows; i++ ) result[ i ] = new double[ cols ];

		for( int i = 0; i < rows; i++ )
			for( int j = 0; j < cols; j++ )
				result[ i ][ j ] = matrix[ i ][ j ] * scalar;

		return result;
	}	

	/// <summary>
	/// Multiplies a vector by a scalar
	/// </summary>
	/// <param name="vector">target vector</param>
	/// <param name="scalar">scalar value</param>
	/// <returns>resulting vector</returns>
	public static double[] Mult( this double[] vector, double scalar )
	{
		int length = vector.Length;
		var result = new double[ length ];
		for( int i = 0; i < length; i++ ) result[ i ] = vector[ i ] * scalar;

		return result;
	}

	/// <summary>
	/// Multiplies a vector by a scalar
	/// </summary>
	/// <param name="vector">target vector</param>
	/// <param name="scalar">scalar value</param>
	/// <returns>resulting vector</returns>
	//public static Vector<double> Mult( this Vector<double> vector, double scalar ) => Vector.Multiply( vector, scalar );
	
}
