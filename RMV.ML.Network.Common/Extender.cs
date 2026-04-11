
namespace RMV.ML.Network.Common;

public static class Extender
{
	/// <summary>
	/// Outputs array of doubles
	/// </summary>
	/// <param name="data">target array</param>
	/// <param name="name">array name to print</param>
	public static void Print( this double[] data, string name )
	{
		Console.Write( name );
		Array.ForEach( data, d => Console.Write( d.ToString( "0.00" ) + "  " ) );
		Console.WriteLine();
	}

	/// <summary>
	/// Reshapes 1d to 2d array
	/// </summary>      
	public static T[,] Reshape<T>( this T[] data, int rows, int cols )
	{
		var result = new T[ rows, cols ];

		int index = 0;

		for( int i = 0; i < rows; i++ )
		{
			for( int j = 0; j < rows; j++ )
			{
				result[ i, j ] = data[ index++ ];
			}
		}

		return result;
	}

	/// <summary>
	/// Coverts string collection to string with separators
	/// </summary> 
	/// <summary>	     
	public static string? Join<T>( this IEnumerable<T>? target, char separator = ',' ) =>
		target is not null && target.Any() ? string.Join( separator, target ) : null;


	/// <summary>
	/// string.IsNullOrEmpty extention
	/// </summary>
	/// <param name="target">target string</param>
	/// <returns>indicator</returns>
	public static bool IsNullOrEmpty( this string target ) => string.IsNullOrEmpty( target );
	

	/// <summary>
	/// string.IsNullOrWhiteSpace extention
	/// </summary>
	/// <param name="target">target string</param>
	/// <returns>indicator</returns>
	public static bool IsNullOrWhitespace( this string target ) => string.IsNullOrWhiteSpace( target );
	
}
