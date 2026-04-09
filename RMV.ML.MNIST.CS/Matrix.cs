namespace RMV.ML.MNIST.Matrix;

/// <summary>
/// Lightweight wrapper for 2D arrays that provides operator overloads which delegate to the Mult extensions.
/// Implicit conversions are provided so existing code using `double[,]` and `double[]` can interoperate.
/// </summary>
public readonly struct Matrix
{
	readonly double[,] data;

	/// <summary>
	/// Create a wrapper around an existing 2D array. The array reference is used directly (no copy).
	/// </summary>
	/// <param name="data">Underlying 2D array</param>
	public Matrix( double[,] data )
	{
		this.data = data ?? throw new ArgumentNullException( nameof( data ) );
	}

	/// <summary>
	/// Underlying array access
	/// </summary>
	public double[,] Data => this.data;

	/// <summary>
	/// Number of rows
	/// </summary>
	public int Rows => this.data.GetLength( 0 );

	/// <summary>
	/// Number of columns
	/// </summary>
	public int Cols => this.data.GetLength( 1 );

	/// <summary>
	/// Indexer to access elements
	/// </summary>
	public double this[ int r, int c ] => this.data[ r, c ];

	/// <summary>
	/// Implicit conversion from native 2D array to Matrix wrapper.
	/// </summary>
	public static implicit operator Matrix( double[,] array ) => new( array );

	/// <summary>
	/// Implicit conversion from Matrix wrapper to native 2D array.
	/// </summary>
	public static implicit operator double[,]( Matrix m ) => m.data;

	/// <summary>
	/// Multiply Matrix * Matrix
	/// </summary>
	public static Matrix operator *( Matrix a, Matrix b ) => new( a.data.Mult( b.data ) );

	/// <summary>
	/// Multiply Matrix * scalar
	/// </summary>
	public static Matrix operator *( Matrix a, double scalar ) => new( a.data.Mult( scalar ) );

	/// <summary>
	/// Multiply scalar * Matrix
	/// </summary>
	public static Matrix operator *( double scalar, Matrix a ) => new( a.data.Mult( scalar ) );

	/// <summary>
	/// Multiply Matrix * vector => vector
	/// </summary>
	public static double[] operator *( Matrix a, double[] vector ) => a.data.Mult( vector );

	/// <summary>
	/// Multiply vector * Matrix => vector
	/// </summary>
	public static double[] operator *( double[] vector, Matrix a ) => vector.Mult( a.data );

	/// <summary>
	/// Debug friendly representation.
	/// </summary>
	public override string ToString() => $"Matrix({this.Rows}x{this.Cols})";
}
