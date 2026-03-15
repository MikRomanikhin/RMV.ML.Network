using RMV.ML.Network.Domain;

namespace RMV.ML.Network.Test;

/// <summary>
/// Parses input data and maps output to binary vectors
/// </summary>
public class Parser( int outputs, IActivation? activation = null )
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

			input.Add( [ .. Mapper.Normalize( buffer ) ] );  //add it to the input list

			output.Add( BinaryVector( activation ?? new Relu(), ( int )data[ 0 ] ) ); //output list
		}

		return new DataSet(input, output);
	}
	

	/// <summary>
	/// Binary mapping for output
	/// </summary>
	/// <param name="value">target value</param>      
	/// <returns>corresponding vector</returns>
	double[] BinaryVector( IActivation activation, int value )
	{
		var result = new double[ outputs ];

		Array.ForEach( result, r => r = activation.Min ); //default value

		result[ value ] = activation.Max; //actual value

		return result;
	}			

}
