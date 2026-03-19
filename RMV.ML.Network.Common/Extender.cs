
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
	/// Standard Deviation
	/// </summary>
	/// <param name="data"></param>
	/// <returns></returns>
	public static double StdDev( this IEnumerable<double> data )
	{
		if( !data.Any() ) return 0;

		double avg = data.Average();

		double sum = data.Sum( d => Math.Pow( d - avg, 2 ) );

		return Math.Sqrt( ( sum ) / ( data.Count() - 1 ) );
	}

	/// <summary>
	/// Coverts string collection to string with separators
	/// </summary> 
	/// <summary>
	/// Converts a collection to a string with the specified separator.
	/// Returns null if the collection is null or empty.
	/// </summary>      
	public static string? Join<T>( this IEnumerable<T>? target, string separator = ", " ) =>
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
