using RMV.ML.Network.Common;

namespace RMV.ML.Network.Domain;

/// <summary>
/// Parses input data and maps output to binary vectors
/// </summary>
public class Parser( int outputs )
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
			var data = Array.ConvertAll( line.Split( ',' ), double.Parse ); //parse line to double array

			var buffer = new double[ data.Length - 1 ];  //input data storage

			Array.Copy( data, 1, buffer, 0, data.Length - 1 ); //populate buffer

			input.Add( [ .. Normalize( buffer ) ] );  //add it to the input list

			output.Add( BinaryVector( ( int )data[ 0 ] ) ); //output list
		}

		return new DataSet( input, output );
	}

	//public (double[][], double[][]) RunM( string[] lines )
	//{
	//	var input = new List<double[]>();
	//	var output = new List<double[]>();

	//	foreach( string line in lines )
	//	{
	//		var data = Array.ConvertAll( line.Split( ',' ), double.Parse ); //parse line to double array

	//		var buffer = new double[ data.Length - 1 ];  //input data storage

	//		Array.Copy( data, 1, buffer, 0, data.Length - 1 ); //populate buffer

	//		input.Add( [ .. Normalize( buffer ) ] );  //add it to the input list

	//		output.Add( BinaryVector( ( int )data[ 0 ] ) ); //output list
	//	}

	//	return ([ .. input ], [ .. output ]);
	//}

	/// <summary>
	/// Normalizes input to [0, 1] range by dividing by a fixed scale (255 for image data).
	/// </summary>	
	static IEnumerable<double> Normalize( IEnumerable<double> data ) => data.Select( d => d / 255.0 );


	/// <summary>
	/// Binary mapping for output
	/// </summary>
	/// <param name="value">target value</param>      
	/// <returns>corresponding vector</returns>
	double[] BinaryVector( int value, IActivation? activation = null )
	{
		activation ??= new Relu(); //default activation function
		var result = new double[ outputs ];

		Array.ForEach( result, r => r = activation.Min ); //default value

		result[ value ] = activation.Max; //actual value

		return result;
	}			

}
