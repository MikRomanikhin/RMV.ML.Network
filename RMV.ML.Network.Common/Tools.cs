namespace RMV.ML.Network.Common;

/// <summary>
/// Contains frequently used math functions.
/// </summary>
public static class Tools
{	

	///<summary>
	///Generates a random double from -1 to 1 whose absolute value > 0.1.
	///</summary>
	///<returns>generated value</returns>
	public static double GetRandom()
	{
		double ret;

		while( (ret = Random.Shared.NextDouble() ) < 0.1 ) ;		

		return 2d * ret - 1d;
	}

	/// <summary>
	/// Generates a standard normal random value using Box-Muller transform.
	/// </summary>
	public static double GetGaussian()
	{
		double u1 = 1.0 - Random.Shared.NextDouble(); // avoid log(0)
		double u2 = Random.Shared.NextDouble();

		return Math.Sqrt( -2.0 * Math.Log( u1 ) ) * Math.Cos( 2.0 * Math.PI * u2 );
	}

	

	//public static float Percent( int data, int total ) => ((float)data / total) * 100f;
	

	/// <summary>
	/// Reshapes 1d to 2d array
	/// </summary>      
	//public static T[,] Reshape<T>( this T[] data, int rows, int cols )
	//{
	//	var result = new T[ rows, cols ];

	//	int index = 0;

	//	for( int i = 0; i < rows; i++ )
	//	{
	//		for( int j = 0; j < rows; j++ )
	//		{
	//			result[ i, j ] = data[ index++ ];
	//		}
	//	}

	//	return result;
	//}

	/// <summary>
	/// Allocate a 2D array.
	/// </summary>
	/// <typeparam name="T">The type to allocate.</typeparam>
	/// <param name="rows">Rows</param>
	/// <param name="cols">Columns</param>
	/// <returns>The array.</returns>
	//public static T[][] Alloc2D<T>( int rows, int cols )
	//{
	//	var result = new T[ rows ][];

	//	for( int i = 0; i < rows; i++ )
	//	{
	//		result[ i ] = new T[ cols ];
	//	}

	//	return result;
	//}
}
