
namespace RMV.ML.Network.Common;

public class Parser
{
	public Parser( IActivation activation, int outputs )
	{
		this.activation = activation;
		this.outputs = outputs;
	}

	IActivation activation;
	int outputs;

	#region reserved
	/// <summary>
	/// Extracts input and output data from string array
	/// </summary>
	/// <param name="lines">input data</param>
	/// <param name="length"># of output neurons</param>
	/// <returns>scaled Input and Output lists</returns>
	//public (List<double[]>, List<double[]>) Run( string path, int length )
	//{
	//   var input = new List<double[]>();
	//   var output = new List<double[]>();

	//   var table = GetDataTable( path );         

	//   var mapper = new Mapper { Activation = new Sigmoid() };

	//   var tmp = table.AsEnumerable().Select( c => c[ 0 ] ).Cast<int>().ToList();

	//   for( int i = 0; i < tmp.Count(); i++ )
	//   {
	//      output.Add( BinaryVector( tmp[ i ] ) );

	//      var row = table.Rows[ i ].ItemArray.Select( r => double.Parse( r.ToString() ) ).ToList();            

	//      row.RemoveAt( 0 );                 

	//      input.Add( mapper.NormalizeZscore( row ).ToArray() );
	//   }     

	//   return (input, output);
	//}


	//static DataTable GetDataTable( string path, bool isFirstRowHeader = false )
	//{
	//   string header = isFirstRowHeader ? "Yes" : "No";

	//   string sql = "SELECT * FROM [" + Path.GetFileName( path ) + "]";


	//   using( var connection = new OleDbConnection( @"Provider=Microsoft.Jet.OLEDB.4.0;Data Source="
	//      + Path.GetDirectoryName( path ) + ";Extended Properties=\"Text;HDR=" + header + "\"" ) )
	//   using( var command = new OleDbCommand( sql, connection ) )
	//   using( var adapter = new OleDbDataAdapter( command ) )
	//   {
	//      var dataTable = new DataTable();

	//      adapter.Fill( dataTable );

	//      return dataTable;
	//   }
	//}
	#endregion

	/// <summary>
	/// Builds Train and Test data
	/// </summary>
	/// <param name="value">input</param>            
	public (List<double[]>, List<double[]>) Run( string[] lines )
	{
		var input = new List<double[]>();
		var output = new List<double[]>();

		var mapper = new Mapper { Activation = activation };

		foreach( string line in lines )
		{
			var data = Array.ConvertAll( line.Split( ',' ), s => double.Parse( s ) );

			var buffer = new double[ data.Length - 1 ];  //input data storage

			Array.Copy( data, 1, buffer, 0, data.Length - 1 ); //populate buffer

			input.Add( mapper.Normalize( buffer ).ToArray() );  //and add it to the input list

			output.Add( BinaryVector( ( int )data[ 0 ] ) ); //output list
		}

		return (input, output);
	}

	/// <summary>
	/// Binary mapping for output
	/// </summary>
	/// <param name="value">target value</param>      
	/// <returns>corresponding vector</returns>
	double[] BinaryVector( int value )
	{
		var result = new double[ this.outputs ];

		Array.ForEach( result, r => r = this.activation.Min ); //default value

		result[ value ] = this.activation.Max; //actual value

		return result;
	}			

}
