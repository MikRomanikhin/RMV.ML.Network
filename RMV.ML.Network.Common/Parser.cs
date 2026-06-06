namespace RMV.ML.Network.Common;

/// <summary>
/// Parses input data and maps output to binary vectors
/// </summary>
public class Parser( int? outputs = null )
{
	/// <summary>
	/// Builds Train and Test data
	/// </summary>
	/// <param name="value">input</param> 
	public DataSet Run( string[] lines )
	{
		var input = new List<double[]>();
		var output = new List<double[]>();			

		foreach( string line in lines )
		{
			var data = Array.ConvertAll( line.Split( ',' ), double.Parse ); //parse line 

			var buffer = new double[ data.Length - 1 ];			//input data storage

			Array.Copy( data, 1, buffer, 0, data.Length - 1 ); //populate buffer

			input.Add( [ .. Normalize( buffer ) ] );				//add to the input list			

			output.Add( BinaryVector( ( int )data[ 0 ] ) );		//output list
		}

		return new DataSet( input, output );
	}

	/// <summary>
	/// Parses an array of CSV-formatted strings and extracts image data as a list of integer arrays.
	/// </summary>
	/// <remarks>Each input string is expected to contain at least two comma-separated values. The method ignores
	/// the first value in each line, which is used as a label, and returns only the subsequent values as image data.</remarks>
	/// <param name="lines">An array of strings, each representing a line of comma-separated values</param>
	/// <returns>A list of integer arrays, each containing the image data extracted from the corresponding input line</returns>
	public static (List<int[]>, List<int>) GetImages( string[] lines )
	{
		var images = new List<int[]>();
		var labels = new List<int>();

		foreach( string line in lines )
		{
			var data = Array.ConvertAll( line.Split( ',' ), int.Parse ); //parse line
																							 
			var buffer = new int[ data.Length - 1 ];			//input data storage

			Array.Copy( data, 1, buffer, 0, data.Length - 1 ); //populate buffer

			images.Add( buffer );	 //add image	         
			labels.Add( data[ 0 ] ); //add label
		}

		return (images, labels);
	}


	/// <summary>
	/// Normalizes input to [0, 1] range by dividing by a fixed scale (255 for image data).
	/// </summary>	
	static IEnumerable<double> Normalize( IEnumerable<double> data ) => data.Select( d => d / 255d );

	/// <summary>
	/// Normalizes input to have zero mean and unit variance
	/// </summary>
	/// <param name="data">The input data array.</param>
	/// <returns>An enumerable of normalized values.</returns>
	//static IEnumerable<double> NormalizeM( double[] data )
	//{
	//	const double DELTA = 1e-8; // small value to prevent division by zero
	//	double mean = data.Sum() / data.Length; // mean
	//	double variance = data.Sum( d => ( d - mean ) * ( d - mean ) ) / data.Length; // variance
	//	double stdDev = Math.Sqrt( variance ); // standard deviation

	//	return data.Select( d => ( d - mean ) / (stdDev + DELTA) ); // normalize
	//}


	/// <summary>
	/// Binary mapping for output
	/// </summary>
	/// <param name="value">target value</param>      
	/// <param name="min">default value for non-target indices</param>
	/// <param name="max">value for the target index</param>
	/// <returns>corresponding vector</returns>
	double[] BinaryVector( int value, double min = 0.0, double max = 1.0 )
	{
		var result = new double[ ( int )outputs  ];
				
		if( min != 0.0 )	Array.Fill( result, min ); // we only need to fill if min is not 0

		result[ value ] = max; // actual value

		return result;
	}			

}
